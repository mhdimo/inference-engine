#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_backend.hpp"
#include "../../core/logger.hpp"
#include "../../core/quantization.hpp"
#include <stdexcept>
#include <algorithm>

namespace backend::metal {

static NSString* msl_source = @R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void add_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    C[id] = A[id] + B[id];
}

kernel void add_bias_kernel(
    device float* x [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    constant uint& D [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    x[id] += bias[id % D];
}


kernel void mul_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    C[id] = A[id] * B[id];
}

kernel void scale_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& factor [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = input[id] * factor;
}

kernel void silu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = input[id];
    output[id] = x / (1.0f + exp(-x));
}

kernel void softmax_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    uint row [[thread_position_in_grid]])
{
    uint start = row * N;
    
    // Find max (for numerical stability)
    float max_val = input[start];
    for (uint c = 1; c < N; ++c) {
        float val = input[start + c];
        if (val > max_val) {
            max_val = val;
        }
    }
    
    // Sum exponentials
    float sum = 0.0f;
    for (uint c = 0; c < N; ++c) {
        float val = exp(input[start + c] - max_val);
        output[start + c] = val;
        sum += val;
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    for (uint c = 0; c < N; ++c) {
        output[start + c] *= inv_sum;
    }
}

kernel void rmsnorm_kernel(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    uint start = row * N;
    
    // Sum squares
    float sum_sq = 0.0f;
    for (uint c = 0; c < N; ++c) {
        float val = input[start + c];
        sum_sq += val * val;
    }
    
    float rms = sqrt(sum_sq / static_cast<float>(N) + eps);
    float inv_rms = 1.0f / rms;
    
    // Normalize and scale by gamma
    for (uint c = 0; c < N; ++c) {
        output[start + c] = (input[start + c] * inv_rms) * gamma[c];
    }
}

kernel void matmul_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant bool& transpose_B [[buffer(6)]],
    uint2 pos [[thread_position_in_grid]])
{
    uint row = pos.y;
    uint col = pos.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        if (transpose_B) {
            for (uint k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[col * K + k];
            }
        } else {
            for (uint k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
        }
        C[row * N + col] = sum;
    }
}

kernel void matmul_tiled_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant bool& transpose_B [[buffer(6)]],
    threadgroup float* shared_A [[threadgroup(0)]],
    threadgroup float* shared_B [[threadgroup(1)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint2 t_id [[thread_position_in_threadgroup]],
    uint2 global_id [[thread_position_in_grid]])
{
    uint row = global_id.y;
    uint col = global_id.x;

    float sum = 0.0f;

    for (uint tile_idx = 0; tile_idx < (K + 15) / 16; ++tile_idx) {
        uint a_col = tile_idx * 16 + t_id.x;
        if (row < M && a_col < K) {
            shared_A[t_id.y * 16 + t_id.x] = A[row * K + a_col];
        } else {
            shared_A[t_id.y * 16 + t_id.x] = 0.0f;
        }

        uint b_row = tile_idx * 16 + t_id.y;
        if (transpose_B) {
            if (col < N && b_row < K) {
                shared_B[t_id.y * 16 + t_id.x] = B[col * K + b_row];
            } else {
                shared_B[t_id.y * 16 + t_id.x] = 0.0f;
            }
        } else {
            if (b_row < K && col < N) {
                shared_B[t_id.y * 16 + t_id.x] = B[b_row * N + col];
            } else {
                shared_B[t_id.y * 16 + t_id.x] = 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < 16; ++k) {
            sum += shared_A[t_id.y * 16 + k] * shared_B[k * 16 + t_id.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

struct block_q4_0 {
    half d;
    uchar qs[16];
};

kernel void matmul_q4_0_kernel(
    device const float* A [[buffer(0)]],
    device const block_q4_0* B_q4 [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]])
{
    uint row = pos.y;
    uint col = pos.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        uint num_blocks_per_col = K / 32;

        for (uint b = 0; b < num_blocks_per_col; ++b) {
            device const block_q4_0& block = B_q4[col * num_blocks_per_col + b];
            float d = block.d;

            for (uint l = 0; l < 16; ++l) {
                uchar val = block.qs[l];
                
                float q0 = float(val & 0x0f) - 8.0f;
                float q1 = float(val >> 4) - 8.0f;

                uint k0 = b * 32 + l;
                uint k1 = k0 + 16;

                sum += A[row * K + k0] * q0 * d;
                sum += A[row * K + k1] * q1 * d;
            }
        }
        C[row * N + col] = sum;
    }
}

struct block_q8_0 {
    half d;
    char qs[32];
};

kernel void matmul_q8_0_kernel(
    device const float* A [[buffer(0)]],
    device const block_q8_0* B_q8 [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]])
{
    uint row = pos.y;
    uint col = pos.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        uint num_blocks_per_col = K / 32;

        for (uint b = 0; b < num_blocks_per_col; ++b) {
            device const block_q8_0& block = B_q8[col * num_blocks_per_col + b];
            float d = block.d;

            for (uint l = 0; l < 32; ++l) {
                uint k = b * 32 + l;
                sum += A[row * K + k] * float(block.qs[l]) * d;
            }
        }
        C[row * N + col] = sum;
    }
}


kernel void rope_kernel(
    device float* x [[buffer(0)]],
    constant uint& pos_offset [[buffer(1)]],
    constant uint& n_heads [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant float& freq_base [[buffer(4)]],
    constant uint& rope_dim [[buffer(5)]],
    uint3 pos_in_grid [[thread_position_in_grid]])
{
    uint d = pos_in_grid.x;
    uint h = pos_in_grid.y;
    uint i = pos_in_grid.z;

    uint pos = pos_offset + i;
    float theta = float(pos) / pow(freq_base, float(2 * d) / float(rope_dim));
    float cos_t = cos(theta);
    float sin_t = sin(theta);

    uint idx0 = i * (n_heads * head_dim) + h * head_dim + d;
    uint idx1 = idx0 + rope_dim / 2;

    float v0 = x[idx0];
    float v1 = x[idx1];

    x[idx0] = v0 * cos_t - v1 * sin_t;
    x[idx1] = v0 * sin_t + v1 * cos_t;
}

kernel void gqa_attention_kernel(
    device const float* Q [[buffer(0)]],
    device const float* K_cache [[buffer(1)]],
    device const float* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& L [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& n_heads [[buffer(6)]],
    constant uint& n_kv_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    uint2 pos [[thread_position_in_grid]])
{
    uint h = pos.x;
    uint i = pos.y;

    if (h < n_heads && i < L) {
        uint group_size = n_heads / n_kv_heads;
        uint kv_h = h / group_size;
        float scale = 1.0f / sqrt(float(head_dim));

        // Use thread stack array up to 128 elements capacity
        float scores[128];
        float max_score = -INFINITY;
        uint query_pos = (seq_len - L) + i;

        for (uint t = 0; t < seq_len; ++t) {
            if (t > query_pos) {
                scores[t] = -INFINITY;
                continue;
            }
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += Q[i * (n_heads * head_dim) + (h * head_dim + d)] * K_cache[t * (n_kv_heads * head_dim) + (kv_h * head_dim + d)];
            }
            scores[t] = dot * scale;
            if (scores[t] > max_score) {
                max_score = scores[t];
            }
        }

        float sum_exp = 0.0f;
        for (uint t = 0; t < seq_len; ++t) {
            if (t > query_pos) {
                scores[t] = 0.0f;
            } else {
                scores[t] = exp(scores[t] - max_score);
                sum_exp += scores[t];
            }
        }

        float inv_sum = (sum_exp == 0.0f) ? 0.0f : 1.0f / sum_exp;
        for (uint t = 0; t < seq_len; ++t) {
            scores[t] *= inv_sum;
        }

        for (uint d = 0; d < head_dim; ++d) {
            float sum_v = 0.0f;
            for (uint t = 0; t < seq_len; ++t) {
                sum_v += scores[t] * V_cache[t * (n_kv_heads * head_dim) + (kv_h * head_dim + d)];
            }
            output[i * (n_heads * head_dim) + (h * head_dim + d)] = sum_v;
        }
    }
}
)msl";

class MetalContext::Impl {
public:
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    
    id<MTLComputePipelineState> matmul_pipeline = nil;
    id<MTLComputePipelineState> matmul_tiled_pipeline = nil;
    id<MTLComputePipelineState> matmul_q4_0_pipeline = nil;
    id<MTLComputePipelineState> matmul_q8_0_pipeline = nil;
    id<MTLComputePipelineState> add_pipeline = nil;
    id<MTLComputePipelineState> add_bias_pipeline = nil;
    id<MTLComputePipelineState> silu_pipeline = nil;
    id<MTLComputePipelineState> softmax_pipeline = nil;
    id<MTLComputePipelineState> rmsnorm_pipeline = nil;
    id<MTLComputePipelineState> mul_pipeline = nil;
    id<MTLComputePipelineState> scale_pipeline = nil;
    id<MTLComputePipelineState> rope_pipeline = nil;
    id<MTLComputePipelineState> gqa_attention_pipeline = nil;
};

MetalContext& MetalContext::instance() {
    static MetalContext ctx;
    return ctx;
}

MetalContext::~MetalContext() {
    delete impl_;
}

bool MetalContext::is_available() const {
    return impl_ != nullptr && impl_->device != nil;
}

bool MetalContext::init() {
    if (impl_) {
        return true; // Already initialized
    }

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
        LOG_WARN("No system default Metal device found. Metal backend will be disabled.");
        return false;
    }

    LOG_INFO(std::string("Metal Backend Initialized on GPU: ") + [[dev name] UTF8String]);

    impl_ = new Impl();
    impl_->device = dev;
    impl_->queue = [dev newCommandQueue];

    NSError* error = nil;
    impl_->library = [dev newLibraryWithSource:msl_source options:nil error:&error];
    if (!impl_->library) {
        LOG_ERR(std::string("Failed to compile MSL Shader library: ") + [[error localizedDescription] UTF8String]);
        delete impl_;
        impl_ = nullptr;
        return false;
    }

    // Compile individual compute pipelines
    auto compile_pipeline = [&](NSString* name, id<MTLComputePipelineState> __strong * pipeline) -> bool {
        id<MTLFunction> func = [impl_->library newFunctionWithName:name];
        if (!func) {
            LOG_ERR(std::string("Shader function not found: ") + [name UTF8String]);
            return false;
        }
        NSError* err = nil;
        *pipeline = [impl_->device newComputePipelineStateWithFunction:func error:&err];
        if (!*pipeline) {
            LOG_ERR(std::string("Failed to create pipeline state for ") + [name UTF8String] + ": " + [[err localizedDescription] UTF8String]);
            return false;
        }
        return true;
    };

    if (!compile_pipeline(@"add_kernel", &impl_->add_pipeline) ||
        !compile_pipeline(@"add_bias_kernel", &impl_->add_bias_pipeline) ||
        !compile_pipeline(@"silu_kernel", &impl_->silu_pipeline) ||
        !compile_pipeline(@"softmax_kernel", &impl_->softmax_pipeline) ||
        !compile_pipeline(@"rmsnorm_kernel", &impl_->rmsnorm_pipeline) ||
        !compile_pipeline(@"matmul_kernel", &impl_->matmul_pipeline) ||
        !compile_pipeline(@"matmul_tiled_kernel", &impl_->matmul_tiled_pipeline) ||
        !compile_pipeline(@"matmul_q4_0_kernel", &impl_->matmul_q4_0_pipeline) ||
        !compile_pipeline(@"matmul_q8_0_kernel", &impl_->matmul_q8_0_pipeline) ||
        !compile_pipeline(@"mul_kernel", &impl_->mul_pipeline) ||
        !compile_pipeline(@"scale_kernel", &impl_->scale_pipeline) ||
        !compile_pipeline(@"rope_kernel", &impl_->rope_pipeline) ||
        !compile_pipeline(@"gqa_attention_kernel", &impl_->gqa_attention_pipeline)) {
        delete impl_;
        impl_ = nullptr;
        return false;
    }

    return true;
}

// Helper: wrap raw CPU pointer into an uncopied MTLBuffer using Unified Memory
static id<MTLBuffer> wrap_tensor(id<MTLDevice> device, const tensor::Tensor& t) {
    if (t.nbytes() == 0 || t.data() == nullptr) {
        return nil;
    }
    // Zero-copy wrapping of host virtual memory
    id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:const_cast<void*>(t.data())
                                                  length:t.nbytes()
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
    if (!buf) {
        LOG_ERR("Failed to create zero-copy MTLBuffer for pointer: " + std::to_string(reinterpret_cast<size_t>(t.data())));
        throw std::runtime_error("Failed to create zero-copy MTLBuffer");
    }
    return buf;
}

void MetalContext::add(const tensor::Tensor& A, const tensor::Tensor& B, tensor::Tensor& C) {
    if (!is_available()) throw std::runtime_error("Metal backend not available");

    size_t total_elements = A.numel();
    if (total_elements == 0) return;

    id<MTLBuffer> buf_A = wrap_tensor(impl_->device, A);
    id<MTLBuffer> buf_B = wrap_tensor(impl_->device, B);
    id<MTLBuffer> buf_C = wrap_tensor(impl_->device, C);

    id<MTLCommandBuffer> cmd_buf = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
    
    [encoder setComputePipelineState:impl_->add_pipeline];
    [encoder setBuffer:buf_A offset:0 atIndex:0];
    [encoder setBuffer:buf_B offset:0 atIndex:1];
    [encoder setBuffer:buf_C offset:0 atIndex:2];

    NSUInteger w = impl_->add_pipeline.threadExecutionWidth;
    MTLSize grid_size = MTLSizeMake(total_elements, 1, 1);
    MTLSize group_size = MTLSizeMake(std::min(total_elements, w), 1, 1);

    [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    [encoder endEncoding];

    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];
}

void MetalContext::add_bias(tensor::Tensor& x, const tensor::Tensor& bias) {
    if (!is_available()) throw std::runtime_error("Metal backend not available");

    size_t total_elements = x.numel();
    if (total_elements == 0) return;

    uint32_t D = static_cast<uint32_t>(x.shape()[x.shape().ndim - 1]);

    id<MTLBuffer> buf_x = wrap_tensor(impl_->device, x);
    id<MTLBuffer> buf_bias = wrap_tensor(impl_->device, bias);

    id<MTLCommandBuffer> cmd_buf = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

    [encoder setComputePipelineState:impl_->add_bias_pipeline];
    [encoder setBuffer:buf_x offset:0 atIndex:0];
    [encoder setBuffer:buf_bias offset:0 atIndex:1];
    [encoder setBytes:&D length:sizeof(D) atIndex:2];

    NSUInteger w = impl_->add_bias_pipeline.threadExecutionWidth;
    MTLSize grid_size = MTLSizeMake(total_elements, 1, 1);
    MTLSize group_size = MTLSizeMake(std::min(total_elements, w), 1, 1);

    [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    [encoder endEncoding];

    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];
}


void MetalContext::silu(const tensor::Tensor& input, tensor::Tensor& output) {
    if (!is_available()) throw std::runtime_error("Metal backend not available");

    size_t total_elements = input.numel();
    if (total_elements == 0) return;

    id<MTLBuffer> buf_in = wrap_tensor(impl_->device, input);
    id<MTLBuffer> buf_out = wrap_tensor(impl_->device, output);

    id<MTLCommandBuffer> cmd_buf = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

    [encoder setComputePipelineState:impl_->silu_pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];

    NSUInteger w = impl_->silu_pipeline.threadExecutionWidth;
    MTLSize grid_size = MTLSizeMake(total_elements, 1, 1);
    MTLSize group_size = MTLSizeMake(std::min(total_elements, w), 1, 1);

    [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    [encoder endEncoding];

    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];
}

void MetalContext::softmax(const tensor::Tensor& input, tensor::Tensor& output) {
    if (!is_available()) throw std::runtime_error("Metal backend not available");

    size_t ndim = input.shape().ndim;
    if (ndim == 0) return;

    uint32_t N = static_cast<uint32_t>(input.shape()[ndim - 1]);
    uint32_t num_rows = static_cast<uint32_t>(input.numel() / N);

    id<MTLBuffer> buf_in = wrap_tensor(impl_->device, input);
    id<MTLBuffer> buf_out = wrap_tensor(impl_->device, output);

    id<MTLCommandBuffer> cmd_buf = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

    [encoder setComputePipelineState:impl_->softmax_pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&N length:sizeof(N) atIndex:2];

    NSUInteger w = impl_->softmax_pipeline.threadExecutionWidth;
    MTLSize grid_size = MTLSizeMake(num_rows, 1, 1);
    MTLSize group_size = MTLSizeMake(std::min(static_cast<NSUInteger>(num_rows), w), 1, 1);

    [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    [encoder endEncoding];

    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];
}

void MetalContext::rmsnorm(const tensor::Tensor& input, const tensor::Tensor& gamma, tensor::Tensor& output, float eps) {
    if (!is_available()) throw std::runtime_error("Metal backend not available");

    size_t ndim = input.shape().ndim;
    if (ndim == 0) return;

    uint32_t N = static_cast<uint32_t>(input.shape()[ndim - 1]);
    uint32_t num_rows = static_cast<uint32_t>(input.numel() / N);

    id<MTLBuffer> buf_in = wrap_tensor(impl_->device, input);
    id<MTLBuffer> buf_gam = wrap_tensor(impl_->device, gamma);
    id<MTLBuffer> buf_out = wrap_tensor(impl_->device, output);

    id<MTLCommandBuffer> cmd_buf = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

    [encoder setComputePipelineState:impl_->rmsnorm_pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_gam offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBytes:&N length:sizeof(N) atIndex:3];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:4];

    NSUInteger w = impl_->rmsnorm_pipeline.threadExecutionWidth;
    MTLSize grid_size = MTLSizeMake(num_rows, 1, 1);
    MTLSize group_size = MTLSizeMake(std::min(static_cast<NSUInteger>(num_rows), w), 1, 1);

    [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    [encoder endEncoding];

    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];
}

void MetalContext::matmul(const tensor::Tensor& A, const tensor::Tensor& B, tensor::Tensor& C) {
    if (!is_available()) throw std::runtime_error("Metal backend not available");

    uint32_t M = static_cast<uint32_t>(A.shape()[0]);
    uint32_t K = static_cast<uint32_t>(A.shape()[1]);
    uint32_t N = static_cast<uint32_t>(C.shape()[1]);

    // Check if B is transposed. B is transposed if its strides are non-contiguous [1, K] instead of [N, 1].
    // Our C++ code represented B_transposed as shape {3, 2} with strides {1, 3} (i.e. shape {K, N} with strides {1, K}).
    bool transpose_B = !B.is_contiguous();

    id<MTLBuffer> buf_A = wrap_tensor(impl_->device, A);
    id<MTLBuffer> buf_B = wrap_tensor(impl_->device, B);
    id<MTLBuffer> buf_C = wrap_tensor(impl_->device, C);

    id<MTLCommandBuffer> cmd_buf = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

    if (M == 1) {
        // For matrix-vector multiplication (M=1), tiled shared memory is inefficient.
        // Use the naive element-wise kernel with a 1D grid dispatch.
        [encoder setComputePipelineState:impl_->matmul_pipeline];
        [encoder setBuffer:buf_A offset:0 atIndex:0];
        [encoder setBuffer:buf_B offset:0 atIndex:1];
        [encoder setBuffer:buf_C offset:0 atIndex:2];
        [encoder setBytes:&M length:sizeof(M) atIndex:3];
        [encoder setBytes:&K length:sizeof(K) atIndex:4];
        [encoder setBytes:&N length:sizeof(N) atIndex:5];
        [encoder setBytes:&transpose_B length:sizeof(transpose_B) atIndex:6];

        MTLSize group_size = MTLSizeMake(256, 1, 1);
        MTLSize grid_size = MTLSizeMake(
            (N + group_size.width - 1) / group_size.width * group_size.width,
            1,
            1
        );
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    } else {
        [encoder setComputePipelineState:impl_->matmul_tiled_pipeline];
        [encoder setBuffer:buf_A offset:0 atIndex:0];
        [encoder setBuffer:buf_B offset:0 atIndex:1];
        [encoder setBuffer:buf_C offset:0 atIndex:2];
        [encoder setBytes:&M length:sizeof(M) atIndex:3];
        [encoder setBytes:&K length:sizeof(K) atIndex:4];
        [encoder setBytes:&N length:sizeof(N) atIndex:5];
        [encoder setBytes:&transpose_B length:sizeof(transpose_B) atIndex:6];
        
        // Allocate threadgroup memory size for block caching: 16x16 float shared array for A and B
        [encoder setThreadgroupMemoryLength:16 * 16 * sizeof(float) atIndex:0];
        [encoder setThreadgroupMemoryLength:16 * 16 * sizeof(float) atIndex:1];

        // Threadgroup size of 16x16
        MTLSize group_size = MTLSizeMake(16, 16, 1);
        MTLSize grid_size = MTLSizeMake(
            (N + group_size.width - 1) / group_size.width * group_size.width,
            (M + group_size.height - 1) / group_size.height * group_size.height,
            1
        );
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    }

    [encoder endEncoding];

    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];
}

void MetalContext::mul(const tensor::Tensor& A, const tensor::Tensor& B, tensor::Tensor& C) {
    if (!is_available()) throw std::runtime_error("Metal backend not available");

    size_t total_elements = A.numel();
    if (total_elements == 0) return;

    id<MTLBuffer> buf_A = wrap_tensor(impl_->device, A);
    id<MTLBuffer> buf_B = wrap_tensor(impl_->device, B);
    id<MTLBuffer> buf_C = wrap_tensor(impl_->device, C);

    id<MTLCommandBuffer> cmd_buf = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
    
    [encoder setComputePipelineState:impl_->mul_pipeline];
    [encoder setBuffer:buf_A offset:0 atIndex:0];
    [encoder setBuffer:buf_B offset:0 atIndex:1];
    [encoder setBuffer:buf_C offset:0 atIndex:2];

    NSUInteger w = impl_->mul_pipeline.threadExecutionWidth;
    MTLSize grid_size = MTLSizeMake(total_elements, 1, 1);
    MTLSize group_size = MTLSizeMake(std::min(total_elements, w), 1, 1);

    [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    [encoder endEncoding];

    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];
}

void MetalContext::scale(const tensor::Tensor& input, tensor::Tensor& output, float factor) {
    if (!is_available()) throw std::runtime_error("Metal backend not available");

    size_t total_elements = input.numel();
    if (total_elements == 0) return;

    id<MTLBuffer> buf_in = wrap_tensor(impl_->device, input);
    id<MTLBuffer> buf_out = wrap_tensor(impl_->device, output);

    id<MTLCommandBuffer> cmd_buf = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
    
    [encoder setComputePipelineState:impl_->scale_pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&factor length:sizeof(factor) atIndex:2];

    NSUInteger w = impl_->scale_pipeline.threadExecutionWidth;
    MTLSize grid_size = MTLSizeMake(total_elements, 1, 1);
    MTLSize group_size = MTLSizeMake(std::min(total_elements, w), 1, 1);

    [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    [encoder endEncoding];

    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];
}

void MetalContext::matmul_q4_0(const tensor::Tensor& A, const core::block_q4_0* W_q4, tensor::Tensor& C) {
    if (!is_available()) throw std::runtime_error("Metal backend not available");

    uint32_t M = static_cast<uint32_t>(A.shape()[0]);
    uint32_t K = static_cast<uint32_t>(A.shape()[1]);
    uint32_t N = static_cast<uint32_t>(C.shape()[1]);

    id<MTLBuffer> buf_A = wrap_tensor(impl_->device, A);

    // Map CPU quantized block data directly using unified memory zero-copy wrapping
    size_t bytes_W = N * (K / 32) * sizeof(core::block_q4_0);
    id<MTLBuffer> buf_W = [impl_->device newBufferWithBytesNoCopy:const_cast<core::block_q4_0*>(W_q4)
                                                          length:bytes_W
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
    if (!buf_W) {
        LOG_ERR("Failed to wrap block_q4_0 buffer in Metal");
        throw std::runtime_error("Failed to wrap block_q4_0 buffer in Metal");
    }

    id<MTLBuffer> buf_C = wrap_tensor(impl_->device, C);

    id<MTLCommandBuffer> cmd_buf = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

    [encoder setComputePipelineState:impl_->matmul_q4_0_pipeline];
    [encoder setBuffer:buf_A offset:0 atIndex:0];
    [encoder setBuffer:buf_W offset:0 atIndex:1];
    [encoder setBuffer:buf_C offset:0 atIndex:2];
    [encoder setBytes:&M length:sizeof(M) atIndex:3];
    [encoder setBytes:&K length:sizeof(K) atIndex:4];
    [encoder setBytes:&N length:sizeof(N) atIndex:5];

    MTLSize group_size = MTLSizeMake(16, 16, 1);
    MTLSize grid_size = MTLSizeMake(
        (N + group_size.width - 1) / group_size.width * group_size.width,
        (M + group_size.height - 1) / group_size.height * group_size.height,
        1
    );

    [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    [encoder endEncoding];

    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];
}

void MetalContext::matmul_q8_0(const tensor::Tensor& A, const core::block_q8_0* W_q8, tensor::Tensor& C) {
    if (!is_available()) throw std::runtime_error("Metal backend not available");

    uint32_t M = static_cast<uint32_t>(A.shape()[0]);
    uint32_t K = static_cast<uint32_t>(A.shape()[1]);
    uint32_t N = static_cast<uint32_t>(C.shape()[1]);

    id<MTLBuffer> buf_A = wrap_tensor(impl_->device, A);

    size_t bytes_W = N * (K / 32) * sizeof(core::block_q8_0);
    id<MTLBuffer> buf_W = [impl_->device newBufferWithBytesNoCopy:const_cast<core::block_q8_0*>(W_q8)
                                                          length:bytes_W
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
    if (!buf_W) {
        LOG_ERR("Failed to wrap block_q8_0 buffer in Metal");
        throw std::runtime_error("Failed to wrap block_q8_0 buffer in Metal");
    }

    id<MTLBuffer> buf_C = wrap_tensor(impl_->device, C);

    id<MTLCommandBuffer> cmd_buf = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

    [encoder setComputePipelineState:impl_->matmul_q8_0_pipeline];
    [encoder setBuffer:buf_A offset:0 atIndex:0];
    [encoder setBuffer:buf_W offset:0 atIndex:1];
    [encoder setBuffer:buf_C offset:0 atIndex:2];
    [encoder setBytes:&M length:sizeof(M) atIndex:3];
    [encoder setBytes:&K length:sizeof(K) atIndex:4];
    [encoder setBytes:&N length:sizeof(N) atIndex:5];

    MTLSize group_size = MTLSizeMake(16, 16, 1);
    MTLSize grid_size = MTLSizeMake(
        (N + group_size.width - 1) / group_size.width * group_size.width,
        (M + group_size.height - 1) / group_size.height * group_size.height,
        1
    );

    [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    [encoder endEncoding];

    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];
}


void MetalContext::rope(tensor::Tensor& x, size_t pos_offset, size_t n_heads, size_t head_dim, float freq_base, size_t rope_dim) {
    if (!is_available()) throw std::runtime_error("Metal backend not available");

    if (rope_dim == 0) {
        rope_dim = head_dim;
    }

    uint32_t L = static_cast<uint32_t>(x.shape()[0]);
    uint32_t pos_off = static_cast<uint32_t>(pos_offset);
    uint32_t heads = static_cast<uint32_t>(n_heads);
    uint32_t h_dim = static_cast<uint32_t>(head_dim);
    float fb = freq_base;
    uint32_t r_dim = static_cast<uint32_t>(rope_dim);

    id<MTLBuffer> buf_x = wrap_tensor(impl_->device, x);

    id<MTLCommandBuffer> cmd_buf = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

    [encoder setComputePipelineState:impl_->rope_pipeline];
    [encoder setBuffer:buf_x offset:0 atIndex:0];
    [encoder setBytes:&pos_off length:sizeof(pos_off) atIndex:1];
    [encoder setBytes:&heads length:sizeof(heads) atIndex:2];
    [encoder setBytes:&h_dim length:sizeof(h_dim) atIndex:3];
    [encoder setBytes:&fb length:sizeof(fb) atIndex:4];
    [encoder setBytes:&r_dim length:sizeof(r_dim) atIndex:5];

    // Thread grid represents coord pair index (rope_dim/2), head index (n_heads), and sequence index (L)
    MTLSize grid_size = MTLSizeMake(rope_dim / 2, n_heads, L);
    MTLSize group_size = MTLSizeMake(
        std::min(static_cast<NSUInteger>(rope_dim / 2), impl_->rope_pipeline.threadExecutionWidth),
        1,
        1
    );

    [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    [encoder endEncoding];

    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];
}

void MetalContext::gqa_attention(const tensor::Tensor& Q, const tensor::Tensor& K_cache, const tensor::Tensor& V_cache,
                                 tensor::Tensor& output, size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    if (!is_available()) throw std::runtime_error("Metal backend not available");

    uint32_t L = static_cast<uint32_t>(Q.shape()[0]);
    uint32_t seq_len = static_cast<uint32_t>(K_cache.shape()[0]);
    uint32_t heads = static_cast<uint32_t>(n_heads);
    uint32_t kv_heads = static_cast<uint32_t>(n_kv_heads);
    uint32_t h_dim = static_cast<uint32_t>(head_dim);

    id<MTLBuffer> buf_Q = wrap_tensor(impl_->device, Q);
    id<MTLBuffer> buf_K = wrap_tensor(impl_->device, K_cache);
    id<MTLBuffer> buf_V = wrap_tensor(impl_->device, V_cache);
    id<MTLBuffer> buf_out = wrap_tensor(impl_->device, output);

    id<MTLCommandBuffer> cmd_buf = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

    [encoder setComputePipelineState:impl_->gqa_attention_pipeline];
    [encoder setBuffer:buf_Q offset:0 atIndex:0];
    [encoder setBuffer:buf_K offset:0 atIndex:1];
    [encoder setBuffer:buf_V offset:0 atIndex:2];
    [encoder setBuffer:buf_out offset:0 atIndex:3];
    [encoder setBytes:&L length:sizeof(L) atIndex:4];
    [encoder setBytes:&seq_len length:sizeof(seq_len) atIndex:5];
    [encoder setBytes:&heads length:sizeof(heads) atIndex:6];
    [encoder setBytes:&kv_heads length:sizeof(kv_heads) atIndex:7];
    [encoder setBytes:&h_dim length:sizeof(h_dim) atIndex:8];

    // Thread grid pos.x = query head, pos.y = token index L
    MTLSize grid_size = MTLSizeMake(n_heads, L, 1);
    MTLSize group_size = MTLSizeMake(
        std::min(static_cast<NSUInteger>(n_heads), impl_->gqa_attention_pipeline.threadExecutionWidth),
        1,
        1
    );

    [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    [encoder endEncoding];

    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];
}

} // namespace backend::metal
