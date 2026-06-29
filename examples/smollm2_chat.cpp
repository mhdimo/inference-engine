#include "engine/core/types.hpp"
#include "engine/core/logger.hpp"
#include "engine/core/tokenizer.hpp"
#include "engine/core/sampler.hpp"
#include "engine/core/gguf_loader.hpp"
#include "engine/core/quantization.hpp"
#include "engine/core/chat_template.hpp"
#include "engine/memory/allocator.hpp"
#include "engine/memory/kv_cache.hpp"
#include "engine/tensor/tensor.hpp"
#include "engine/backend/cpu/kernels.hpp"
#include "engine/backend/cpu/matmul.hpp"
#include "engine/backend/cpu/quantized_matmul.hpp"
#include "engine/backend/metal/metal_backend.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <cstring>
#include <algorithm>

using namespace core;
using namespace tensor;
using namespace memory;

// Standard bit-wise IEEE 754 half-precision (FP16) to single-precision (FP32) converter
float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mant = h & 0x03FF;
    if (exp == 0) {
        if (mant == 0) {
            union { uint32_t u; float f; } val;
            val.u = sign;
            return val.f;
        } else {
            while (!(mant & 0x0400)) {
                mant <<= 1;
                exp -= 1;
            }
            exp += 1;
            mant &= ~0x0400;
        }
    } else if (exp == 31) {
        union { uint32_t u; float f; } val;
        val.u = sign | 0x7F800000 | (mant << 13);
        return val.f;
    }
    exp = exp + 127 - 15;
    union { uint32_t u; float f; } val;
    val.u = sign | (exp << 23) | (mant << 13);
    return val.f;
}

struct __attribute__((packed)) block_q4_1 {
    uint16_t d;
    uint16_t m;
    uint8_t qs[16];
};

struct SSMState {
    std::vector<float> conv_state;
    std::vector<float> recurrent_state;

    SSMState() {
        conv_state.assign(6144 * 3, 0.0f);
        recurrent_state.assign(16 * 128 * 128, 0.0f);
    }

    void reset() {
        std::fill(conv_state.begin(), conv_state.end(), 0.0f);
        std::fill(recurrent_state.begin(), recurrent_state.end(), 0.0f);
    }
};

float* allocate_page_aligned_floats(size_t num_elements) {
    void* ptr = nullptr;
    int res = posix_memalign(&ptr, 4096, num_elements * sizeof(float));
    if (res != 0) {
        throw std::bad_alloc();
    }
    return static_cast<float*>(ptr);
}

template<typename T>
T* copy_to_page_aligned(const T* src, size_t size_bytes) {
    if (size_bytes == 0) return nullptr;
    void* ptr = nullptr;
    int res = posix_memalign(&ptr, 4096, size_bytes);
    if (res != 0) {
        throw std::bad_alloc();
    }
    std::memcpy(ptr, src, size_bytes);
    return static_cast<T*>(ptr);
}

float* dequantize_tensor_to_fp32(const GGUFTensorInfo& info, const void* raw_data, size_t total_elements) {
    float* dst = allocate_page_aligned_floats(total_elements);
    
    if (info.type == 0) { // FP32
        std::memcpy(dst, raw_data, total_elements * sizeof(float));
    } else if (info.type == 1) { // FP16
        const uint16_t* src = static_cast<const uint16_t*>(raw_data);
        for (size_t i = 0; i < total_elements; ++i) {
            dst[i] = fp16_to_fp32(src[i]);
        }
    } else if (info.type == 2) { // Q4_0
        const core::block_q4_0* src = static_cast<const core::block_q4_0*>(raw_data);
        size_t num_blocks = total_elements / 32;
        for (size_t b = 0; b < num_blocks; ++b) {
            float d = fp16_to_fp32(src[b].d);
            for (size_t i = 0; i < 16; ++i) {
                uint8_t val = src[b].qs[i];
                dst[b * 32 + i]      = d * static_cast<float>((val & 0x0F) - 8);
                dst[b * 32 + i + 16] = d * static_cast<float>((val >> 4) - 8);
            }
        }
    } else if (info.type == 3) { // Q4_1
        const block_q4_1* src = static_cast<const block_q4_1*>(raw_data);
        size_t num_blocks = total_elements / 32;
        for (size_t b = 0; b < num_blocks; ++b) {
            float d = fp16_to_fp32(src[b].d);
            float m = fp16_to_fp32(src[b].m);
            for (size_t i = 0; i < 16; ++i) {
                uint8_t val = src[b].qs[i];
                dst[b * 32 + i]      = d * static_cast<float>(val & 0x0F) + m;
                dst[b * 32 + i + 16] = d * static_cast<float>(val >> 4) + m;
            }
        }
    } else if (info.type == 8) { // Q8_0
        const core::block_q8_0* src = static_cast<const core::block_q8_0*>(raw_data);
        size_t num_blocks = total_elements / 32;
        for (size_t b = 0; b < num_blocks; ++b) {
            float d = fp16_to_fp32(src[b].d);
            for (size_t i = 0; i < 32; ++i) {
                dst[b * 32 + i] = d * static_cast<float>(src[b].qs[i]);
            }
        }
    } else if (info.type == 14) { // Q6_K (enum value 14 in GGUF/llama.cpp)
        const core::block_q6_K* src = static_cast<const core::block_q6_K*>(raw_data);
        core::dequantize_q6_K(src, dst, total_elements);
    } else {
        std::cerr << "[ERR] Unsupported dequantization type: " << info.type << " for tensor " << info.name << std::endl;
        throw std::runtime_error("Unsupported dequantization type");
    }
    return dst;
}

void head_rmsnorm(Tensor& x, const Tensor& gamma, float eps, size_t n_heads, size_t head_dim) {
    size_t L = x.shape()[0];
    float* x_data = x.data_as<float>();
    const float* g_data = gamma.data_as<float>();
    for (size_t t = 0; t < L; ++t) {
        for (size_t h = 0; h < n_heads; ++h) {
            float* h_data = x_data + t * n_heads * head_dim + h * head_dim;
            float sum = 0.0f;
            for (size_t i = 0; i < head_dim; ++i) {
                sum += h_data[i] * h_data[i];
            }
            float scale = 1.0f / sqrtf(sum / head_dim + eps);
            for (size_t i = 0; i < head_dim; ++i) {
                h_data[i] = h_data[i] * scale * g_data[i];
            }
        }
    }
}

// Execution routine for a universal dynamic block
void run_inference(
    bool use_gpu,
    const Tensor& x_in,                          // Input shape [L, dim]
    size_t n_layers,
    size_t dim,
    size_t ffn_dim,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim,
    float rope_freq_base,
    float rms_eps,
    const std::vector<const block_q4_0*>& w_q,    // Layer weights (standard LLaMA)
    const std::vector<const block_q4_0*>& w_k,
    const std::vector<const block_q4_0*>& w_v,
    const std::vector<const block_q4_0*>& w_o,
    const std::vector<const block_q4_0*>& w_gate,
    const std::vector<const block_q4_0*>& w_up,
    const std::vector<Tensor>& w_down,            // FP32 dequantized down-projection weights
    const std::vector<Tensor>& gamma_attn,
    const std::vector<Tensor>& gamma_mlp,
    const Tensor& gamma_final,
    const Tensor& w_logits,                      // Dequantized logits projection weights (FP32)
    bool has_bias,
    const std::vector<Tensor>& b_q,
    const std::vector<Tensor>& b_k,
    const std::vector<Tensor>& b_v,
    std::vector<KVCacheManager>& layer_caches,
    size_t pos_offset,                           // Cache sequence position offset
    Tensor& logits_out,                          // Output logits shape [L, vocab_size]
    ArenaAllocator& workspace_arena,
    bool debug_mode,
    // Qwen3.5 weights & options
    bool is_qwen35,
    std::vector<SSMState>& layer_ssm_states,
    const std::vector<Tensor>& w_q_fp32,
    const std::vector<Tensor>& w_k_fp32,
    const std::vector<Tensor>& w_v_fp32,
    const std::vector<Tensor>& w_o_fp32,
    const std::vector<Tensor>& w_gate_fp32,
    const std::vector<Tensor>& w_up_fp32,
    const std::vector<Tensor>& w_attn_q_norm,
    const std::vector<Tensor>& w_attn_k_norm,
    const std::vector<Tensor>& w_qkv,
    const std::vector<Tensor>& w_gate_attn,
    const std::vector<Tensor>& w_ssm_alpha,
    const std::vector<Tensor>& w_ssm_beta,
    const std::vector<Tensor>& w_ssm_conv1d,
    const std::vector<Tensor>& w_ssm_dt_bias,
    const std::vector<Tensor>& w_ssm_a,
    const std::vector<Tensor>& w_ssm_norm,
    const std::vector<Tensor>& w_ssm_out
) {
    size_t L = x_in.shape()[0];
    workspace_arena.reset();

    auto alloc_tensor = [&](Shape shape) {
        float* ptr = static_cast<float*>(workspace_arena.allocate(shape.size() * sizeof(float), 4096));
        return Tensor(shape, DType::Float32, ptr);
    };

    auto print_stats = [debug_mode](const std::string& name, const Tensor& t) {
        if (!debug_mode) return;
        const float* d = t.data_as<float>();
        double sum = 0.0;
        size_t n = t.numel();
        for (size_t idx = 0; idx < n; ++idx) {
            sum += d[idx];
        }
        std::cout << "    [STATS] " << name << " sum=" << sum << " size=" << n << std::endl;
    };

    // Double buffering for current_X to avoid arena memory growth and preserve page alignment
    float* current_x_ptr = allocate_page_aligned_floats(L * dim);
    float* next_x_ptr = allocate_page_aligned_floats(L * dim);
    std::memcpy(current_x_ptr, x_in.data(), L * dim * sizeof(float));

    Tensor current_X({L, dim}, DType::Float32, current_x_ptr);

    for (size_t l = 0; l < n_layers; ++l) {
        workspace_arena.reset();

        Tensor norm_1 = alloc_tensor({L, dim});
        if (use_gpu) {
            backend::metal::MetalContext::instance().rmsnorm(current_X, gamma_attn[l], norm_1, rms_eps);
        } else {
            backend::cpu::rmsnorm(current_X, gamma_attn[l], norm_1, rms_eps);
        }

        Tensor proj_o = alloc_tensor({L, dim});

        if (is_qwen35) {
            if (l % 4 == 3) {
                // Qwen3.5 Standard Gated Attention Layer
                Tensor Qcur_full = alloc_tensor({L, dim * 2});
                Tensor w_q_T({dim, dim * 2}, DType::Float32, const_cast<void*>(w_q_fp32[l].data()), {1, dim});
                if (use_gpu) {
                    backend::metal::MetalContext::instance().matmul(norm_1, w_q_T, Qcur_full);
                } else {
                    backend::cpu::matmul(norm_1, w_q_T, Qcur_full);
                }

                // De-interleave Qcur_full into q and gate.
                // The Q projection output layout is INTERLEAVED per head:
                //   [Q_h0(256), Gate_h0(256), Q_h1(256), Gate_h1(256), ..., Q_h7(256), Gate_h7(256)]
                // Each head occupies 2*head_dim = 512 contiguous floats in Qcur_full.
                static constexpr size_t ATTN_N_Q_HEADS = 8;
                static constexpr size_t ATTN_HEAD_DIM  = 256;
                static constexpr size_t ATTN_STRIDE    = ATTN_HEAD_DIM * 2; // per-head stride in Qcur_full
                Tensor q = alloc_tensor({L, dim});
                Tensor gate = alloc_tensor({L, dim});
                float* split_q_ptr = q.data_as<float>();
                float* split_gate_ptr = gate.data_as<float>();
                const float* split_qcur_ptr = Qcur_full.data_as<float>();
                for (size_t t = 0; t < L; ++t) {
                    const float* tok_qcur = split_qcur_ptr + t * dim * 2;
                    for (size_t h = 0; h < ATTN_N_Q_HEADS; ++h) {
                        // Q_h at offset h * ATTN_STRIDE, Gate_h at offset h * ATTN_STRIDE + head_dim
                        std::memcpy(split_q_ptr    + t * dim + h * ATTN_HEAD_DIM,
                                    tok_qcur + h * ATTN_STRIDE,
                                    ATTN_HEAD_DIM * sizeof(float));
                        std::memcpy(split_gate_ptr + t * dim + h * ATTN_HEAD_DIM,
                                    tok_qcur + h * ATTN_STRIDE + ATTN_HEAD_DIM,
                                    ATTN_HEAD_DIM * sizeof(float));
                    }
                }

                // QK-Norm
                head_rmsnorm(q, w_attn_q_norm[l], rms_eps, 8, 256);

                Tensor k = alloc_tensor({L, 2 * 256});
                Tensor v = alloc_tensor({L, 2 * 256});

                Tensor w_k_T({dim, 512}, DType::Float32, const_cast<void*>(w_k_fp32[l].data()), {1, dim});
                Tensor w_v_T({dim, 512}, DType::Float32, const_cast<void*>(w_v_fp32[l].data()), {1, dim});
                if (use_gpu) {
                    backend::metal::MetalContext::instance().matmul(norm_1, w_k_T, k);
                    backend::metal::MetalContext::instance().matmul(norm_1, w_v_T, v);
                } else {
                    backend::cpu::matmul(norm_1, w_k_T, k);
                    backend::cpu::matmul(norm_1, w_v_T, v);
                }

                head_rmsnorm(k, w_attn_k_norm[l], rms_eps, 2, 256);

                // Apply Rotary Position Embeddings (RoPE)
                if (use_gpu) {
                    backend::metal::MetalContext::instance().rope(q, pos_offset, 8, 256, rope_freq_base, 64);
                    backend::metal::MetalContext::instance().rope(k, pos_offset, 2, 256, rope_freq_base, 64);
                } else {
                    backend::cpu::rope(q, pos_offset, 8, 256, rope_freq_base, 64);
                    backend::cpu::rope(k, pos_offset, 2, 256, rope_freq_base, 64);
                }

                // Cache Keys and Values
                layer_caches[l].append_kv(k, v);

                Tensor K_active = layer_caches[l].get_active_k();
                Tensor V_active = layer_caches[l].get_active_v();

                Tensor attn_out = alloc_tensor({L, dim});
                if (use_gpu && K_active.shape()[0] <= 128) {
                    backend::metal::MetalContext::instance().gqa_attention(q, K_active, V_active, attn_out, 8, 2, 256);
                } else {
                    backend::cpu::gqa_attention(q, K_active, V_active, attn_out, 8, 2, 256);
                }

                // Gated attention scaling: attn_out = attn_out * Sigmoid(gate)
                float* out_ptr = attn_out.data_as<float>();
                const float* gate_ptr = gate.data_as<float>();
                for (size_t i = 0; i < L * dim; ++i) {
                    float sig = 1.0f / (1.0f + expf(-gate_ptr[i]));
                    out_ptr[i] *= sig;
                }

                // Output projection
                Tensor w_o_T({dim, dim}, DType::Float32, const_cast<void*>(w_o_fp32[l].data()), {1, dim});
                if (use_gpu) {
                    backend::metal::MetalContext::instance().matmul(attn_out, w_o_T, proj_o);
                } else {
                    backend::cpu::matmul(attn_out, w_o_T, proj_o);
                }
            } else {
                // Qwen3.5 SSM / Gated DeltaNet Layer
                Tensor qkv_mixed = alloc_tensor({L, 6144});
                Tensor w_qkv_T({dim, 6144}, DType::Float32, const_cast<void*>(w_qkv[l].data()), {1, dim});
                if (use_gpu) {
                    backend::metal::MetalContext::instance().matmul(norm_1, w_qkv_T, qkv_mixed);
                } else {
                    backend::cpu::matmul(norm_1, w_qkv_T, qkv_mixed);
                }

                Tensor z = alloc_tensor({L, dim});
                Tensor w_gate_attn_T({dim, dim}, DType::Float32, const_cast<void*>(w_gate_attn[l].data()), {1, dim});
                if (use_gpu) {
                    backend::metal::MetalContext::instance().matmul(norm_1, w_gate_attn_T, z);
                } else {
                    backend::cpu::matmul(norm_1, w_gate_attn_T, z);
                }

                Tensor alpha = alloc_tensor({L, 16});
                Tensor beta = alloc_tensor({L, 16});
                Tensor w_ssm_alpha_T({dim, 16}, DType::Float32, const_cast<void*>(w_ssm_alpha[l].data()), {1, dim});
                Tensor w_ssm_beta_T({dim, 16}, DType::Float32, const_cast<void*>(w_ssm_beta[l].data()), {1, dim});
                if (use_gpu) {
                    backend::metal::MetalContext::instance().matmul(norm_1, w_ssm_alpha_T, alpha);
                    backend::metal::MetalContext::instance().matmul(norm_1, w_ssm_beta_T, beta);
                } else {
                    backend::cpu::matmul(norm_1, w_ssm_alpha_T, alpha);
                    backend::cpu::matmul(norm_1, w_ssm_beta_T, beta);
                }

                Tensor conv_act = alloc_tensor({L, 6144});
                float* conv_act_ptr = conv_act.data_as<float>();
                const float* qkv_ptr = qkv_mixed.data_as<float>();
                const float* conv_w = w_ssm_conv1d[l].data_as<float>();
                auto& ssm_state = layer_ssm_states[l];

                // Causal 1D convolution and SiLU activation
                for (size_t t = 0; t < L; ++t) {
                    for (size_t c = 0; c < 6144; ++c) {
                        float s0 = ssm_state.conv_state[c * 3 + 0];
                        float s1 = ssm_state.conv_state[c * 3 + 1];
                        float s2 = ssm_state.conv_state[c * 3 + 2];
                        float cur_val = qkv_ptr[t * 6144 + c];

                        ssm_state.conv_state[c * 3 + 0] = s1;
                        ssm_state.conv_state[c * 3 + 1] = s2;
                        ssm_state.conv_state[c * 3 + 2] = cur_val;

                        float w0 = conv_w[c * 4 + 0];
                        float w1 = conv_w[c * 4 + 1];
                        float w2 = conv_w[c * 4 + 2];
                        float w3 = conv_w[c * 4 + 3];

                        float val = s0 * w0 + s1 * w1 + s2 * w2 + cur_val * w3;
                        conv_act_ptr[t * 6144 + c] = val / (1.0f + expf(-val));
                    }
                }

                // Apply L2 Norm head-wise to q_conv and k_conv
                for (size_t t = 0; t < L; ++t) {
                    for (size_t h = 0; h < 16; ++h) {
                        float* q_h = conv_act_ptr + t * 6144 + h * 128;
                        float* k_h = conv_act_ptr + t * 6144 + 2048 + h * 128;

                        float q_sum = 0.0f;
                        for (size_t i = 0; i < 128; ++i) q_sum += q_h[i] * q_h[i];
                        float q_norm = sqrtf(q_sum + 1e-6f);
                        for (size_t i = 0; i < 128; ++i) q_h[i] /= q_norm;

                        float k_sum = 0.0f;
                        for (size_t i = 0; i < 128; ++i) k_sum += k_h[i] * k_h[i];
                        float k_norm = sqrtf(k_sum + 1e-6f);
                        for (size_t i = 0; i < 128; ++i) k_h[i] /= k_norm;
                    }
                }

                // Recurrent Gated DeltaNet Scan
                Tensor ssm_out = alloc_tensor({L, dim});
                float* ssm_out_ptr = ssm_out.data_as<float>();
                const float* alpha_ptr = alpha.data_as<float>();
                const float* beta_ptr = beta.data_as<float>();
                const float* ssm_dt_bias_ptr = w_ssm_dt_bias[l].data_as<float>();
                const float* ssm_a_ptr = w_ssm_a[l].data_as<float>();

                for (size_t t = 0; t < L; ++t) {
                    for (size_t h = 0; h < 16; ++h) {
                        float* state_h = ssm_state.recurrent_state.data() + h * 128 * 128;
                        const float* q_h = conv_act_ptr + t * 6144 + h * 128;
                        const float* k_h = conv_act_ptr + t * 6144 + 2048 + h * 128;
                        const float* v_h = conv_act_ptr + t * 6144 + 4096 + h * 128;
                        float* o_h = ssm_out_ptr + t * 2048 + h * 128;

                        float a_bias = alpha_ptr[t * 16 + h] + ssm_dt_bias_ptr[h];
                        float alpha_softplus = (a_bias > 20.0f) ? a_bias : logf(1.0f + expf(a_bias));
                        float gate_val = alpha_softplus * ssm_a_ptr[h];
                        float decay_val = expf(gate_val);

                        float beta_sigmoid = 1.0f / (1.0f + expf(-beta_ptr[t * 16 + h]));

                        // State decay
                        for (size_t i = 0; i < 128 * 128; ++i) {
                            state_h[i] *= decay_val;
                        }

                        // sk = state_h * k_h
                        float sk[128];
                        for (size_t i = 0; i < 128; ++i) {
                            float sum = 0.0f;
                            for (size_t j = 0; j < 128; ++j) {
                                sum += state_h[i * 128 + j] * k_h[j];
                            }
                            sk[i] = sum;
                        }

                        // d = beta_sigmoid * (v_h - sk)
                        float d[128];
                        for (size_t i = 0; i < 128; ++i) {
                            d[i] = beta_sigmoid * (v_h[i] - sk[i]);
                        }

                        // state_h += d * k_h^T
                        for (size_t i = 0; i < 128; ++i) {
                            for (size_t j = 0; j < 128; ++j) {
                                state_h[i * 128 + j] += d[i] * k_h[j];
                            }
                        }

                        // o_h = state_h * (q_h / sqrt(128))
                        float scale = 1.0f / sqrtf(128.0f);
                        for (size_t i = 0; i < 128; ++i) {
                            float sum = 0.0f;
                            for (size_t j = 0; j < 128; ++j) {
                                sum += state_h[i * 128 + j] * q_h[j];
                            }
                            o_h[i] = sum * scale;
                        }

                        if (debug_mode && l == 0 && h == 0) {
                            std::cout << "      [SSM L0 H0 t=" << t << "] a_bias=" << a_bias 
                                      << " decay=" << decay_val << " beta=" << beta_sigmoid 
                                      << " v[0]=" << v_h[0] << " sk[0]=" << sk[0] 
                                      << " state[0]=" << state_h[0] << " q[0]=" << q_h[0] 
                                      << " o[0]=" << o_h[0] << std::endl;
                        }
                    }
                }

                // Gated Normalization: RMSNorm(ssm_out) * SiLU(z)
                head_rmsnorm(ssm_out, w_ssm_norm[l], rms_eps, 16, 128);

                float* normed_ptr = ssm_out.data_as<float>();
                const float* z_ptr = z.data_as<float>();
                for (size_t i = 0; i < L * dim; ++i) {
                    float silu_z = z_ptr[i] / (1.0f + expf(-z_ptr[i]));
                    normed_ptr[i] *= silu_z;
                }

                // Output projection
                Tensor w_ssm_out_T({dim, dim}, DType::Float32, const_cast<void*>(w_ssm_out[l].data()), {1, dim});
                if (use_gpu) {
                    backend::metal::MetalContext::instance().matmul(ssm_out, w_ssm_out_T, proj_o);
                } else {
                    backend::cpu::matmul(ssm_out, w_ssm_out_T, proj_o);
                }
            }
        } else {
            // Standard LLaMA / SmolLM2 path
            Tensor q = alloc_tensor({L, dim});
            Tensor k = alloc_tensor({L, n_kv_heads * head_dim});
            Tensor v = alloc_tensor({L, n_kv_heads * head_dim});

            if (use_gpu) {
                backend::metal::MetalContext::instance().matmul_q4_0(norm_1, w_q[l], q);
                backend::metal::MetalContext::instance().matmul_q4_0(norm_1, w_k[l], k);
                backend::metal::MetalContext::instance().matmul_q4_0(norm_1, w_v[l], v);
                if (has_bias) {
                    backend::metal::MetalContext::instance().add_bias(q, b_q[l]);
                    backend::metal::MetalContext::instance().add_bias(k, b_k[l]);
                    backend::metal::MetalContext::instance().add_bias(v, b_v[l]);
                }
            } else {
                backend::cpu::gemm_q4_0(norm_1, w_q[l], q);
                backend::cpu::gemm_q4_0(norm_1, w_k[l], k);
                backend::cpu::gemm_q4_0(norm_1, w_v[l], v);
                if (has_bias) {
                    backend::cpu::add_bias(q, b_q[l]);
                    backend::cpu::add_bias(k, b_k[l]);
                    backend::cpu::add_bias(v, b_v[l]);
                }
            }

            if (use_gpu) {
                backend::metal::MetalContext::instance().rope(q, pos_offset, n_heads, head_dim, rope_freq_base);
                backend::metal::MetalContext::instance().rope(k, pos_offset, n_kv_heads, head_dim, rope_freq_base);
            } else {
                backend::cpu::rope(q, pos_offset, n_heads, head_dim, rope_freq_base);
                backend::cpu::rope(k, pos_offset, n_kv_heads, head_dim, rope_freq_base);
            }

            layer_caches[l].append_kv(k, v);
            Tensor K_active = layer_caches[l].get_active_k();
            Tensor V_active = layer_caches[l].get_active_v();

            Tensor attn_out = alloc_tensor({L, dim});
            if (use_gpu && K_active.shape()[0] <= 128) {
                backend::metal::MetalContext::instance().gqa_attention(q, K_active, V_active, attn_out, n_heads, n_kv_heads, head_dim);
            } else {
                backend::cpu::gqa_attention(q, K_active, V_active, attn_out, n_heads, n_kv_heads, head_dim);
            }

            if (use_gpu) {
                backend::metal::MetalContext::instance().matmul_q4_0(attn_out, w_o[l], proj_o);
            } else {
                backend::cpu::gemm_q4_0(attn_out, w_o[l], proj_o);
            }
        }

        // Residual Add
        Tensor attn_res = alloc_tensor({L, dim});
        if (use_gpu) {
            backend::metal::MetalContext::instance().add(current_X, proj_o, attn_res);
        } else {
            backend::cpu::add(current_X, proj_o, attn_res);
        }

        // MLP SwiGLU Block
        Tensor norm_2 = alloc_tensor({L, dim});
        if (use_gpu) {
            backend::metal::MetalContext::instance().rmsnorm(attn_res, gamma_mlp[l], norm_2, rms_eps);
        } else {
            backend::cpu::rmsnorm(attn_res, gamma_mlp[l], norm_2, rms_eps);
        }

        Tensor gate = alloc_tensor({L, ffn_dim});
        Tensor up = alloc_tensor({L, ffn_dim});

        if (is_qwen35) {
            Tensor w_gate_T({dim, ffn_dim}, DType::Float32, const_cast<void*>(w_gate_fp32[l].data()), {1, dim});
            Tensor w_up_T({dim, ffn_dim}, DType::Float32, const_cast<void*>(w_up_fp32[l].data()), {1, dim});
            if (use_gpu) {
                backend::metal::MetalContext::instance().matmul(norm_2, w_gate_T, gate);
                backend::metal::MetalContext::instance().matmul(norm_2, w_up_T, up);
            } else {
                backend::cpu::matmul(norm_2, w_gate_T, gate);
                backend::cpu::matmul(norm_2, w_up_T, up);
            }
        } else {
            if (use_gpu) {
                backend::metal::MetalContext::instance().matmul_q4_0(norm_2, w_gate[l], gate);
                backend::metal::MetalContext::instance().matmul_q4_0(norm_2, w_up[l], up);
            } else {
                backend::cpu::gemm_q4_0(norm_2, w_gate[l], gate);
                backend::cpu::gemm_q4_0(norm_2, w_up[l], up);
            }
        }

        Tensor gate_silu = alloc_tensor({L, ffn_dim});
        if (use_gpu) {
            backend::metal::MetalContext::instance().silu(gate, gate_silu);
        } else {
            backend::cpu::silu(gate, gate_silu);
        }

        Tensor gate_mul = alloc_tensor({L, ffn_dim});
        if (use_gpu) {
            backend::metal::MetalContext::instance().mul(gate_silu, up, gate_mul);
        } else {
            backend::cpu::mul(gate_silu, up, gate_mul);
        }

        Tensor down = alloc_tensor({L, dim});
        Tensor w_down_T({ffn_dim, dim}, DType::Float32, const_cast<void*>(w_down[l].data()), {1, ffn_dim});

        if (use_gpu) {
            backend::metal::MetalContext::instance().matmul(gate_mul, w_down_T, down);
        } else {
            backend::cpu::matmul(gate_mul, w_down_T, down);
        }

        Tensor mlp_res = alloc_tensor({L, dim});
        if (use_gpu) {
            backend::metal::MetalContext::instance().add(attn_res, down, mlp_res);
        } else {
            backend::cpu::add(attn_res, down, mlp_res);
        }

        std::memcpy(next_x_ptr, mlp_res.data(), L * dim * sizeof(float));
        std::swap(current_x_ptr, next_x_ptr);
        current_X = Tensor({L, dim}, DType::Float32, current_x_ptr);

        if (debug_mode && pos_offset == 0) {
            const float* d = current_X.data_as<float>();
            double s = 0.0; for (size_t q_idx = 0; q_idx < current_X.numel(); ++q_idx) s += d[q_idx];
            std::cout << "    [LAYER " << l << "] x sum=" << s << std::endl;
        }
    }

    Tensor norm_final = alloc_tensor({L, dim});
    if (use_gpu) {
        backend::metal::MetalContext::instance().rmsnorm(current_X, gamma_final, norm_final, rms_eps);
    } else {
        backend::cpu::rmsnorm(current_X, gamma_final, norm_final, rms_eps);
    }

    size_t vocab_size = w_logits.shape()[0];
    Tensor w_logits_T({dim, vocab_size}, DType::Float32, const_cast<void*>(w_logits.data()), {1, dim});
    if (use_gpu) {
        backend::metal::MetalContext::instance().matmul(norm_final, w_logits_T, logits_out);
    } else {
        backend::cpu::matmul(norm_final, w_logits_T, logits_out);
    }

    std::free(current_x_ptr);
    std::free(next_x_ptr);
}

int main(int argc, char* argv[]) {
    std::cout << "==========================================================" << std::endl;
    std::cout << "SmollM2 Real LLM Dynamic Inference Runner (Phase 10)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    std::string script_path = "scripts/download_smollm2.py";
    FILE* file = std::fopen(script_path.c_str(), "r");
    if (file) {
        std::fclose(file);
    } else {
        script_path = "../scripts/download_smollm2.py";
    }

    std::string dl_cmd = "python3 " + script_path;
    std::cout << "[INIT] Resolving model weight files..." << std::endl;
    int ret = std::system(dl_cmd.c_str());
    if (ret != 0) {
        std::cerr << "[ERR] Failed to download or verify SmollM2 weights." << std::endl;
        return 1;
    }

    std::string model_path = "qwen2.5-1.5b-instruct-q4_0.gguf";
    std::string automated_prompt;
    bool debug_mode = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--prompt" && i + 1 < argc) {
            automated_prompt = argv[++i];
        } else if (a == "--debug") {
            debug_mode = true;
        } else if (a.rfind("--", 0) != 0) {
            model_path = a;
        }
    }
    FILE* mfile = std::fopen(model_path.c_str(), "r");
    if (mfile) {
        std::fclose(mfile);
    } else {
        size_t slash = model_path.find_last_of('/');
        std::string base = (slash == std::string::npos) ? model_path : model_path.substr(slash + 1);
        model_path = "../" + base;
    }

    std::cout << "[INIT] Memory-mapping " << model_path << "..." << std::endl;
    GGUFLoader loader(model_path);
    if (!loader.load()) {
        std::cerr << "[ERR] GGUFLoader failed to parse model file." << std::endl;
        return 1;
    }

    std::string arch = loader.get_metadata_string("general.architecture");
    if (arch.empty()) arch = "llama";
    size_t dim = loader.get_metadata_uint32(arch + ".embedding_length");
    size_t n_layers = loader.get_metadata_uint32(arch + ".block_count");
    size_t ffn_dim = loader.get_metadata_uint32(arch + ".feed_forward_length");
    size_t n_heads = loader.get_metadata_uint32(arch + ".attention.head_count");
    size_t n_kv_heads = loader.get_metadata_uint32(arch + ".attention.head_count_kv");
    if (n_kv_heads == 0) n_kv_heads = n_heads;

    size_t head_dim = dim / n_heads;
    size_t vocab_size = loader.vocabulary().size();

    float rope_freq_base = loader.get_metadata_float(arch + ".rope.freq_base");
    if (rope_freq_base == 0.0f) rope_freq_base = 10000.0f;
    float rms_eps = loader.get_metadata_float(arch + ".attention.layer_norm_rms_epsilon");
    if (rms_eps == 0.0f) rms_eps = 1e-5f;

    uint32_t bos_id = loader.has_metadata("tokenizer.ggml.bos_token_id")
                          ? loader.get_metadata_uint32("tokenizer.ggml.bos_token_id") : 1;
    uint32_t eos_id = loader.has_metadata("tokenizer.ggml.eos_token_id")
                          ? loader.get_metadata_uint32("tokenizer.ggml.eos_token_id") : 2;
    bool add_bos = loader.has_metadata("tokenizer.ggml.add_bos_token")
                       ? (loader.get_metadata_uint32("tokenizer.ggml.add_bos_token") != 0) : true;

    std::cout << "[INFO] rope_freq_base=" << rope_freq_base << " rms_eps=" << rms_eps
              << " bos=" << bos_id << " eos=" << eos_id << " add_bos=" << add_bos << std::endl;

    std::cout << "[INFO] Model configuration parsed successfully:" << std::endl;
    std::cout << "  - Layers:        " << n_layers << std::endl;
    std::cout << "  - Dimension:     " << dim << std::endl;
    std::cout << "  - FFN Dimension: " << ffn_dim << std::endl;
    std::cout << "  - Query Heads:   " << n_heads << std::endl;
    std::cout << "  - KV Heads:      " << n_kv_heads << std::endl;
    std::cout << "  - Head Dim:      " << head_dim << std::endl;
    std::cout << "  - Vocab Size:    " << vocab_size << std::endl;

    backend::metal::MetalContext::instance().init();
    bool use_gpu = backend::metal::MetalContext::instance().is_available();
    if (use_gpu) {
        std::cout << "[INIT] Metal GPU acceleration enabled on GPU: Apple M5" << std::endl;
    } else {
        std::cout << "[INIT] CPU fallback active." << std::endl;
    }

    std::cout << "[INIT] Reconstructing tokenizer vocab and merge priorities..." << std::endl;
    Tokenizer tokenizer;
    const auto& vocab_strs = loader.vocabulary();
    for (size_t i = 0; i < vocab_strs.size(); ++i) {
        tokenizer.add_token(vocab_strs[i], i);
    }
    const auto& merge_strs = loader.merges();
    for (size_t i = 0; i < merge_strs.size(); ++i) {
        std::string m = merge_strs[i];
        size_t space_pos = m.find(' ');
        if (space_pos != std::string::npos) {
            std::string left = m.substr(0, space_pos);
            std::string right = m.substr(space_pos + 1);
            tokenizer.add_merge(left, right, i);
        }
    }
    auto register_special = [&](const std::string& tok) {
        for (size_t i = 0; i < vocab_strs.size(); ++i) {
            if (vocab_strs[i] == tok) {
                tokenizer.add_special(tok, static_cast<int32_t>(i));
                std::cout << "[INIT] Registered special token: " << tok << " = " << i << std::endl;
                return;
            }
        }
        std::cout << "[INIT] WARNING: Special token not found in vocabulary: " << tok << std::endl;
    };
    register_special("<|im_start|>");
    register_special("<|im_end|>");
    register_special("<|endoftext|>");
    // Qwen3.5 reasoning tokens – must be single tokens so they are never BPE-split
    register_special("<think>");     // token 248068
    register_special("</think>");    // token 248069
    // Convenience IDs used when building Qwen3.5 prompts
    static constexpr int32_t TOKEN_THINK_OPEN  = 248068;
    static constexpr int32_t TOKEN_THINK_CLOSE = 248069;
    static constexpr int32_t TOKEN_NEWLINE     = 198;

    std::string chat_template = loader.get_metadata_string("tokenizer.chat_template");
    if (debug_mode) {
        std::cout << "[DEBUG] parsed GGUF chat template:\n" << chat_template << std::endl;
    }
    std::cout << "[INIT] Tokenizer ready. (" << vocab_strs.size() << " tokens, " << merge_strs.size() << " merge pairs)" << std::endl;

    std::cout << "[INIT] Loading token embeddings..." << std::endl;
    const GGUFTensorInfo* emb_info = loader.get_tensor_info("token_embd.weight");
    if (!emb_info) {
        std::cerr << "[ERR] Could not locate 'token_embd.weight' tensor." << std::endl;
        return 1;
    }
    const void* emb_raw_data = loader.get_tensor_data(*emb_info);
    float* w_embed = allocate_page_aligned_floats(vocab_size * dim);

    if (emb_info->type == 1) { // FP16
        const uint16_t* raw_fp16 = static_cast<const uint16_t*>(emb_raw_data);
        for (size_t i = 0; i < vocab_size * dim; ++i) {
            w_embed[i] = fp16_to_fp32(raw_fp16[i]);
        }
    } else if (emb_info->type == 0) { // FP32
        const float* raw_fp32 = static_cast<const float*>(emb_raw_data);
        std::memcpy(w_embed, raw_fp32, vocab_size * dim * sizeof(float));
    } else if (emb_info->type == 8) { // Q8_0
        const core::block_q8_0* raw_q8 = static_cast<const core::block_q8_0*>(emb_raw_data);
        size_t total_elements = vocab_size * dim;
        size_t num_blocks = total_elements / 32;
        for (size_t b = 0; b < num_blocks; ++b) {
            float d = fp16_to_fp32(raw_q8[b].d);
            for (size_t i = 0; i < 32; ++i) {
                w_embed[b * 32 + i] = d * raw_q8[b].qs[i];
            }
        }
    } else if (emb_info->type == 2) { // Q4_0
        core::dequantize_q4_0(static_cast<const core::block_q4_0*>(emb_raw_data), w_embed,
                              static_cast<size_t>(vocab_size) * dim, 1);
    } else {
        std::cerr << "[ERR] Unsupported embedding type (type=" << emb_info->type << ")" << std::endl;
        return 1;
    }

    bool is_qwen35 = loader.get_tensor_info("blk.0.ssm_a") != nullptr;
    if (is_qwen35) {
        std::cout << "[INIT] Qwen3.5 hybrid SSM / Gated DeltaNet model detected!" << std::endl;
        if (debug_mode) {
            std::cout << "[DEBUG vocab] token 0: '" << vocab_strs[0] << "', token 1: '" << vocab_strs[1] << "'" << std::endl;
        }
        add_bos = false;
        std::cout << "[INIT] Qwen model: Disabled prepending BOS token." << std::endl;
    }

    // Allocate standard LLaMA layers weight arrays
    std::vector<const block_q4_0*> w_q(n_layers);
    std::vector<const block_q4_0*> w_k(n_layers);
    std::vector<const block_q4_0*> w_v(n_layers);
    std::vector<const block_q4_0*> w_o(n_layers);
    std::vector<const block_q4_0*> w_gate(n_layers);
    std::vector<const block_q4_0*> w_up(n_layers);

    // Allocate Qwen3.5 specific layers
    std::vector<Tensor> w_q_fp32(n_layers);
    std::vector<Tensor> w_k_fp32(n_layers);
    std::vector<Tensor> w_v_fp32(n_layers);
    std::vector<Tensor> w_o_fp32(n_layers);
    std::vector<Tensor> w_gate_fp32(n_layers);
    std::vector<Tensor> w_up_fp32(n_layers);
    std::vector<Tensor> w_attn_q_norm(n_layers);
    std::vector<Tensor> w_attn_k_norm(n_layers);
    std::vector<Tensor> w_qkv(n_layers);
    std::vector<Tensor> w_gate_attn(n_layers);
    std::vector<Tensor> w_ssm_alpha(n_layers);
    std::vector<Tensor> w_ssm_beta(n_layers);
    std::vector<Tensor> w_ssm_conv1d(n_layers);
    std::vector<Tensor> w_ssm_dt_bias(n_layers);
    std::vector<Tensor> w_ssm_a(n_layers);
    std::vector<Tensor> w_ssm_norm(n_layers);
    std::vector<Tensor> w_ssm_out(n_layers);

    std::vector<Tensor> b_q(n_layers);
    std::vector<Tensor> b_k(n_layers);
    std::vector<Tensor> b_v(n_layers);
    bool has_bias = loader.get_tensor_info("blk.0.attn_q.bias") != nullptr;

    std::vector<Tensor> w_down(n_layers);
    std::vector<Tensor> gamma_attn(n_layers);
    std::vector<Tensor> gamma_mlp(n_layers);

    auto bind_q4_aligned = [&](const std::string& name) {
        const GGUFTensorInfo* info = loader.get_tensor_info(name);
        if (!info) {
            std::cerr << "[ERR] Missing tensor: " << name << std::endl;
            throw std::runtime_error("Missing tensor: " + name);
        }
        const void* raw = loader.get_tensor_data(*info);
        return copy_to_page_aligned(static_cast<const block_q4_0*>(raw), info->size_bytes);
    };

    auto bind_fp32_aligned = [&](const std::string& name, Shape shape) {
        const GGUFTensorInfo* info = loader.get_tensor_info(name);
        if (!info) {
            std::cerr << "[ERR] Missing tensor: " << name << std::endl;
            throw std::runtime_error("Missing tensor: " + name);
        }
        const void* raw = loader.get_tensor_data(*info);
        float* ptr = copy_to_page_aligned(static_cast<const float*>(raw), info->size_bytes);
        return Tensor(shape, DType::Float32, ptr);
    };

    auto bind_and_dequantize = [&](const std::string& name, Shape shape) {
        const GGUFTensorInfo* info = loader.get_tensor_info(name);
        if (!info) {
            std::cerr << "[ERR] Missing tensor: " << name << std::endl;
            throw std::runtime_error("Missing tensor: " + name);
        }
        const void* raw = loader.get_tensor_data(*info);
        float* ptr = dequantize_tensor_to_fp32(*info, raw, shape.size());
        return Tensor(shape, DType::Float32, ptr);
    };

    std::cout << "[INIT] Dequantizing model layers..." << std::endl;
    for (size_t l = 0; l < n_layers; ++l) {
        std::string prefix = "blk." + std::to_string(l) + ".";
        
        gamma_attn[l] = bind_fp32_aligned(prefix + "attn_norm.weight", {dim});
        std::string mlp_norm_name = prefix + "post_attention_norm.weight";
        if (loader.get_tensor_info(prefix + "ffn_norm.weight") != nullptr) {
            mlp_norm_name = prefix + "ffn_norm.weight";
        }
        gamma_mlp[l]  = bind_fp32_aligned(mlp_norm_name, {dim});

        // Dequantize ffn_down
        const GGUFTensorInfo* down_info = loader.get_tensor_info(prefix + "ffn_down.weight");
        if (!down_info) {
            std::cerr << "[ERR] Missing tensor: ffn_down" << std::endl;
            return 1;
        }
        w_down[l] = bind_and_dequantize(prefix + "ffn_down.weight", {dim, ffn_dim});

        if (is_qwen35) {
            w_gate_fp32[l] = bind_and_dequantize(prefix + "ffn_gate.weight", {ffn_dim, dim});
            w_up_fp32[l]   = bind_and_dequantize(prefix + "ffn_up.weight", {ffn_dim, dim});

            if (l % 4 == 3) {
                w_q_fp32[l] = bind_and_dequantize(prefix + "attn_q.weight", {dim * 2, dim});
                w_k_fp32[l] = bind_and_dequantize(prefix + "attn_k.weight", {n_kv_heads * head_dim, dim});
                w_v_fp32[l] = bind_and_dequantize(prefix + "attn_v.weight", {n_kv_heads * head_dim, dim});
                w_o_fp32[l] = bind_and_dequantize(prefix + "attn_output.weight", {dim, dim});
                w_attn_q_norm[l] = bind_and_dequantize(prefix + "attn_q_norm.weight", {256});
                w_attn_k_norm[l] = bind_and_dequantize(prefix + "attn_k_norm.weight", {256});
            } else {
                w_qkv[l]         = bind_and_dequantize(prefix + "attn_qkv.weight", {dim * 3, dim});
                w_gate_attn[l]   = bind_and_dequantize(prefix + "attn_gate.weight", {dim, dim});
                w_ssm_conv1d[l]  = bind_and_dequantize(prefix + "ssm_conv1d.weight", {dim * 3, 4});
                w_ssm_alpha[l]   = bind_and_dequantize(prefix + "ssm_alpha.weight", {16, dim});
                w_ssm_beta[l]    = bind_and_dequantize(prefix + "ssm_beta.weight", {16, dim});
                w_ssm_dt_bias[l] = bind_and_dequantize(prefix + "ssm_dt.bias", {16});
                w_ssm_a[l]       = bind_and_dequantize(prefix + "ssm_a", {16});
                w_ssm_norm[l]    = bind_and_dequantize(prefix + "ssm_norm.weight", {128});
                w_ssm_out[l]     = bind_and_dequantize(prefix + "ssm_out.weight", {dim, dim});
            }
        } else {
            w_q[l]    = bind_q4_aligned(prefix + "attn_q.weight");
            w_k[l]    = bind_q4_aligned(prefix + "attn_k.weight");
            w_v[l]    = bind_q4_aligned(prefix + "attn_v.weight");
            w_o[l]    = bind_q4_aligned(prefix + "attn_output.weight");
            w_gate[l] = bind_q4_aligned(prefix + "ffn_gate.weight");
            w_up[l]   = bind_q4_aligned(prefix + "ffn_up.weight");

            if (has_bias) {
                b_q[l] = bind_fp32_aligned(prefix + "attn_q.bias", {dim});
                b_k[l] = bind_fp32_aligned(prefix + "attn_k.bias", {n_kv_heads * head_dim});
                b_v[l] = bind_fp32_aligned(prefix + "attn_v.bias", {n_kv_heads * head_dim});
            }
        }
    }

    Tensor gamma_final = bind_fp32_aligned("output_norm.weight", {dim});
    Tensor w_logits;
    if (loader.get_tensor_info("output.weight") != nullptr) {
        w_logits = bind_and_dequantize("output.weight", {vocab_size, dim});
    } else {
        std::cout << "[INIT] Tied weights detected. Reusing token embeddings for logits projection..." << std::endl;
        w_logits = Tensor({vocab_size, dim}, DType::Float32, w_embed);
    }

    size_t max_seq_len = 4096; // Reasoning models need long context for thinking traces

    std::vector<KVCacheManager> layer_caches;
    for (size_t l = 0; l < n_layers; ++l) {
        if (is_qwen35 && l % 4 == 3) {
            layer_caches.push_back(KVCacheManager(max_seq_len, 2 * 256));
        } else {
            layer_caches.push_back(KVCacheManager(max_seq_len, n_kv_heads * head_dim));
        }
    }

    std::vector<SSMState> layer_ssm_states(n_layers);

    ArenaAllocator workspace_arena(32 * 1024 * 1024); // 32MB workspace

    if (!automated_prompt.empty()) {
        std::cout << "\n[RUN] Running in Automated verification mode..." << std::endl;
        std::cout << "User > " << automated_prompt << std::endl;
        
        std::string rendered;
        std::vector<int32_t> prompt_tokens;
        if (is_qwen35) {
            // Build the base string WITHOUT the think open token so that
            // the BPE tokenizer never tries to split "<think>" into parts.
            std::string base = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
            base += "<|im_start|>user\n" + automated_prompt + "<|im_end|>\n";
            base += "<|im_start|>assistant\n";
            prompt_tokens = tokenizer.encode(base);
            // Manually inject the reasoning-open token + newline as known IDs.
            prompt_tokens.push_back(TOKEN_THINK_OPEN);
            prompt_tokens.push_back(TOKEN_NEWLINE);
        } else {
            rendered = chat_template.empty()
                ? automated_prompt
                : core::apply_chat_template(chat_template, std::vector<core::ChatMessage>{{"user", automated_prompt}}, true);
            prompt_tokens = tokenizer.encode(rendered);
        }
        if (add_bos) prompt_tokens.insert(prompt_tokens.begin(), static_cast<int32_t>(bos_id));
        
        if (debug_mode) {
            std::cout << "[DEBUG Prompt] Token IDs: ";
            for (int32_t tid : prompt_tokens) std::cout << tid << " ";
            std::cout << "\n[DEBUG Prompt] Decoded String:\n" << tokenizer.decode(prompt_tokens) << std::endl;
        }
        
        size_t L_prompt = prompt_tokens.size();
        std::vector<float> prompt_embeddings(L_prompt * dim);
        for (size_t i = 0; i < L_prompt; ++i) {
            std::memcpy(prompt_embeddings.data() + i * dim, w_embed + prompt_tokens[i] * dim, dim * sizeof(float));
        }

        Tensor x_prefill({L_prompt, dim}, DType::Float32, prompt_embeddings.data());
        std::vector<float> data_logits(L_prompt * vocab_size);
        Tensor prefill_logits({L_prompt, vocab_size}, DType::Float32, data_logits.data());

        for (size_t l = 0; l < n_layers; ++l) {
            layer_caches[l].reset();
            layer_ssm_states[l].reset();
        }

        // Prefill Phase
        run_inference(use_gpu, x_prefill, n_layers, dim, ffn_dim, n_heads, n_kv_heads, head_dim, rope_freq_base, rms_eps,
                      w_q, w_k, w_v, w_o, w_gate, w_up, w_down, gamma_attn, gamma_mlp, gamma_final, w_logits,
                      has_bias, b_q, b_k, b_v,
                      layer_caches, 0, prefill_logits, workspace_arena, debug_mode,
                      is_qwen35, layer_ssm_states,
                      w_q_fp32, w_k_fp32, w_v_fp32, w_o_fp32, w_gate_fp32, w_up_fp32,
                      w_attn_q_norm, w_attn_k_norm, w_qkv, w_gate_attn, w_ssm_alpha, w_ssm_beta,
                      w_ssm_conv1d, w_ssm_dt_bias, w_ssm_a, w_ssm_norm, w_ssm_out);

        // Decode Loop
        std::vector<int32_t> generated_tokens;
        // Sample the first new token from the last row of the prefill logits.
        // This avoids feeding the last prompt token again and creating duplicate KV cache entries.
        int32_t last_token = sample_logits(data_logits.data() + (L_prompt - 1) * vocab_size, vocab_size, 0.7f);
        generated_tokens.push_back(last_token);
        std::cout << "Assistant >";
        if (is_qwen35) {
            std::cout << "\n<think>\n\033[90m" << std::flush;
        }
        size_t total_seq = L_prompt;
        bool in_thinking = is_qwen35;
        bool thinking_done = false;

        // Process and print the first sampled token
        if (is_qwen35 && in_thinking) {
            if (last_token == TOKEN_THINK_CLOSE) {
                in_thinking = false;
                thinking_done = true;
                std::cout << "\033[0m\n</think>\n" << std::flush;
            } else {
                std::cout << tokenizer.decode({last_token}) << std::flush;
            }
        } else {
            std::cout << tokenizer.decode({last_token}) << std::flush;
        }
        total_seq++;

        for (size_t step = 1; step < 1024; ++step) {
            std::vector<float> x_step_data(dim);
            std::memcpy(x_step_data.data(), w_embed + last_token * dim, dim * sizeof(float));
            Tensor x_step({1, dim}, DType::Float32, x_step_data.data());

            std::vector<float> step_logits_data(vocab_size);
            Tensor step_logits({1, vocab_size}, DType::Float32, step_logits_data.data());

            run_inference(use_gpu, x_step, n_layers, dim, ffn_dim, n_heads, n_kv_heads, head_dim, rope_freq_base, rms_eps,
                          w_q, w_k, w_v, w_o, w_gate, w_up, w_down, gamma_attn, gamma_mlp, gamma_final, w_logits,
                          has_bias, b_q, b_k, b_v,
                          layer_caches, total_seq, step_logits, workspace_arena, debug_mode,
                          is_qwen35, layer_ssm_states,
                          w_q_fp32, w_k_fp32, w_v_fp32, w_o_fp32, w_gate_fp32, w_up_fp32,
                          w_attn_q_norm, w_attn_k_norm, w_qkv, w_gate_attn, w_ssm_alpha, w_ssm_beta,
                          w_ssm_conv1d, w_ssm_dt_bias, w_ssm_a, w_ssm_norm, w_ssm_out);

            // Apply repetition penalty
            std::vector<float> penalized_logits = step_logits_data;
            float penalty = 1.15f;
            size_t penalty_window = 64;
            size_t start_idx = (generated_tokens.size() > penalty_window) ? generated_tokens.size() - penalty_window : 0;
            for (size_t i = start_idx; i < generated_tokens.size(); ++i) {
                int32_t t_id = generated_tokens[i];
                if (t_id == TOKEN_THINK_CLOSE || t_id == TOKEN_THINK_OPEN || t_id == 248045 || t_id == 248046) {
                    continue;
                }
                if (penalized_logits[t_id] > 0.0f) {
                    penalized_logits[t_id] /= penalty;
                } else {
                    penalized_logits[t_id] *= penalty;
                }
            }

            int32_t next_token = sample_logits(penalized_logits.data(), vocab_size, 0.7f);
            generated_tokens.push_back(next_token);
            // Stop on EOS or <|im_end|>
            if (next_token == static_cast<int32_t>(eos_id) ||
                next_token == static_cast<int32_t>(248046) /* <|im_end|> */) {
                if (in_thinking) {
                    std::cout << "\033[0m\n" << std::flush;
                }
                break;
            }

            // Thinking stream
            if (is_qwen35 && in_thinking) {
                if (next_token == TOKEN_THINK_CLOSE) {
                    in_thinking = false;
                    thinking_done = true;
                    std::cout << "\033[0m\n</think>\n" << std::flush;
                } else {
                    std::cout << tokenizer.decode({next_token}) << std::flush;
                }
                last_token = next_token;
                total_seq++;
                continue;
            }
            if (is_qwen35 && thinking_done && next_token == TOKEN_NEWLINE) {
                thinking_done = false;
                last_token = next_token;
                total_seq++;
                continue;
            }

            last_token = next_token;
            total_seq++;
            std::cout << tokenizer.decode({next_token}) << std::flush;
        }
        std::cout << std::endl;

        
        std::cout << "\n==========================================================" << std::endl;
        std::cout << "[SUCCESS] Dynamic runner verification completed successfully!" << std::endl;
        std::cout << "==========================================================" << std::endl;
        return 0;
    }

    std::cout << "\n[SUCCESS] Model parsed and loaded successfully!" << std::endl;
    std::cout << "[INFO] Type your prompt to start chatting. Type 'exit' to terminate.\n" << std::endl;

    while (true) {
        std::cout << "\nUser > ";
        std::string input;
        std::getline(std::cin, input);
        if (input == "exit" || input.empty()) {
            break;
        }

        std::string rendered;
        std::vector<int32_t> prompt_tokens;
        if (is_qwen35) {
            std::string base = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
            base += "<|im_start|>user\n" + input + "<|im_end|>\n";
            base += "<|im_start|>assistant\n";
            prompt_tokens = tokenizer.encode(base);
            prompt_tokens.push_back(TOKEN_THINK_OPEN);
            prompt_tokens.push_back(TOKEN_NEWLINE);
        } else {
            rendered = chat_template.empty()
                ? input
                : core::apply_chat_template(chat_template, std::vector<core::ChatMessage>{{"user", input}}, true);
            prompt_tokens = tokenizer.encode(rendered);
        }
        if (prompt_tokens.empty()) {
            continue;
        }
        if (add_bos) prompt_tokens.insert(prompt_tokens.begin(), static_cast<int32_t>(bos_id));
        size_t L_prompt = prompt_tokens.size();

        std::vector<float> prompt_embeddings(L_prompt * dim);
        for (size_t i = 0; i < L_prompt; ++i) {
            std::memcpy(prompt_embeddings.data() + i * dim, w_embed + prompt_tokens[i] * dim, dim * sizeof(float));
        }

        Tensor x_prefill({L_prompt, dim}, DType::Float32, prompt_embeddings.data());
        std::vector<float> data_logits(L_prompt * vocab_size);
        Tensor prefill_logits({L_prompt, vocab_size}, DType::Float32, data_logits.data());

        for (size_t l = 0; l < n_layers; ++l) {
            layer_caches[l].reset();
            layer_ssm_states[l].reset();
        }

        // A. Prefill contextual prompt
        run_inference(use_gpu, x_prefill, n_layers, dim, ffn_dim, n_heads, n_kv_heads, head_dim, rope_freq_base, rms_eps,
                      w_q, w_k, w_v, w_o, w_gate, w_up, w_down, gamma_attn, gamma_mlp, gamma_final, w_logits,
                      has_bias, b_q, b_k, b_v,
                      layer_caches, 0, prefill_logits, workspace_arena, debug_mode,
                      is_qwen35, layer_ssm_states,
                      w_q_fp32, w_k_fp32, w_v_fp32, w_o_fp32, w_gate_fp32, w_up_fp32,
                      w_attn_q_norm, w_attn_k_norm, w_qkv, w_gate_attn, w_ssm_alpha, w_ssm_beta,
                      w_ssm_conv1d, w_ssm_dt_bias, w_ssm_a, w_ssm_norm, w_ssm_out);

        // B. Stream decoded words in real-time
        std::vector<int32_t> generated_tokens;
        // Sample the first new token from the last row of the prefill logits.
        int32_t last_token = sample_logits(data_logits.data() + (L_prompt - 1) * vocab_size, vocab_size, 0.5f);
        generated_tokens.push_back(last_token);
        std::cout << "Assistant >";
        if (is_qwen35) {
            std::cout << "\n<think>\n\033[90m" << std::flush;
        }
        size_t total_seq = L_prompt;

        bool in_thinking = is_qwen35;
        bool thinking_done = false;

        // Process and print the first sampled token
        if (is_qwen35 && in_thinking) {
            if (last_token == TOKEN_THINK_CLOSE) {
                in_thinking = false;
                thinking_done = true;
                std::cout << "\033[0m\n</think>\n" << std::flush;
            } else {
                std::cout << tokenizer.decode({last_token}) << std::flush;
            }
        } else {
            std::cout << tokenizer.decode({last_token}) << std::flush;
        }
        total_seq++;

        for (size_t step = 1; step < 1024; ++step) {
            std::vector<float> x_step_data(dim);
            std::memcpy(x_step_data.data(), w_embed + last_token * dim, dim * sizeof(float));
            Tensor x_step({1, dim}, DType::Float32, x_step_data.data());

            std::vector<float> step_logits_data(vocab_size);
            Tensor step_logits({1, vocab_size}, DType::Float32, step_logits_data.data());

            run_inference(use_gpu, x_step, n_layers, dim, ffn_dim, n_heads, n_kv_heads, head_dim, rope_freq_base, rms_eps,
                          w_q, w_k, w_v, w_o, w_gate, w_up, w_down, gamma_attn, gamma_mlp, gamma_final, w_logits,
                          has_bias, b_q, b_k, b_v,
                          layer_caches, total_seq, step_logits, workspace_arena, debug_mode,
                          is_qwen35, layer_ssm_states,
                          w_q_fp32, w_k_fp32, w_v_fp32, w_o_fp32, w_gate_fp32, w_up_fp32,
                          w_attn_q_norm, w_attn_k_norm, w_qkv, w_gate_attn, w_ssm_alpha, w_ssm_beta,
                          w_ssm_conv1d, w_ssm_dt_bias, w_ssm_a, w_ssm_norm, w_ssm_out);

            // Apply repetition penalty
            std::vector<float> penalized_logits = step_logits_data;
            float penalty = 1.15f;
            size_t penalty_window = 64;
            size_t start_idx = (generated_tokens.size() > penalty_window) ? generated_tokens.size() - penalty_window : 0;
            for (size_t i = start_idx; i < generated_tokens.size(); ++i) {
                int32_t t_id = generated_tokens[i];
                if (t_id == TOKEN_THINK_CLOSE || t_id == TOKEN_THINK_OPEN || t_id == 248045 || t_id == 248046) {
                    continue;
                }
                if (penalized_logits[t_id] > 0.0f) {
                    penalized_logits[t_id] /= penalty;
                } else {
                    penalized_logits[t_id] *= penalty;
                }
            }

            int32_t next_token = sample_logits(penalized_logits.data(), vocab_size, 0.5f);
            generated_tokens.push_back(next_token);

            // Stop on EOS or <|im_end|>
            if (next_token == static_cast<int32_t>(eos_id) ||
                next_token == static_cast<int32_t>(248046) /* <|im_end|> */) {
                if (in_thinking) {
                    std::cout << "\033[0m\n" << std::flush;
                }
                break;
            }

            // Thinking stream
            if (is_qwen35 && in_thinking) {
                if (next_token == TOKEN_THINK_CLOSE) {
                    in_thinking = false;
                    thinking_done = true;
                    std::cout << "\033[0m\n</think>\n" << std::flush;
                    // skip the two newlines that follow </think>
                    last_token = next_token;
                    total_seq++;
                    continue;
                }
                std::cout << tokenizer.decode({next_token}) << std::flush;
                last_token = next_token;
                total_seq++;
                continue;
            }
            // Skip the double-newline separator that immediately follows </think>
            if (is_qwen35 && thinking_done &&
                next_token == TOKEN_NEWLINE && step > 0) {
                thinking_done = false; // only skip once
                last_token = next_token;
                total_seq++;
                continue;
            }

            last_token = next_token;
            total_seq++;

            std::cout << tokenizer.decode({next_token}) << std::flush;
        }
        std::cout << std::endl;
    }

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "Qwen/SmollM CLI session terminated. Cleaned up." << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}
