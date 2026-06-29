#pragma once
#include "../../tensor/tensor.hpp"
#include <string>

namespace core {
    struct block_q4_0;
    struct block_q8_0;
}

namespace backend::metal {

enum class BackendType {
    CPU,
    Metal
};

class MetalContext {
public:
    static MetalContext& instance();

    bool is_available() const;
    
    // Initialize Metal device, queue, and compile the MSL shader source code
    bool init();

    // Metal kernel runners (expect inputs and outputs to reside in page-aligned buffers)
    void matmul(const tensor::Tensor& A, const tensor::Tensor& B, tensor::Tensor& C);
    void add(const tensor::Tensor& A, const tensor::Tensor& B, tensor::Tensor& C);
    void silu(const tensor::Tensor& input, tensor::Tensor& output);
    void softmax(const tensor::Tensor& input, tensor::Tensor& output);
    void rmsnorm(const tensor::Tensor& input, const tensor::Tensor& gamma, tensor::Tensor& output, float eps);
    void mul(const tensor::Tensor& A, const tensor::Tensor& B, tensor::Tensor& C);
    void scale(const tensor::Tensor& input, tensor::Tensor& output, float factor);
    
    // DeepSeek-V4 & Q4_0 Quantization: Quantized matrix-vector multiplication
    void matmul_q4_0(const tensor::Tensor& A, const core::block_q4_0* W_q4, tensor::Tensor& C);
    void matmul_q8_0(const tensor::Tensor& A, const core::block_q8_0* W_q8, tensor::Tensor& C);

    // Universal LLaMA-Architecture (RoPE & GQA)
    void rope(tensor::Tensor& x, size_t pos_offset, size_t n_heads, size_t head_dim, float freq_base = 10000.0f, size_t rope_dim = 0);
    void gqa_attention(const tensor::Tensor& Q, const tensor::Tensor& K_cache, const tensor::Tensor& V_cache,
                       tensor::Tensor& output, size_t n_heads, size_t n_kv_heads, size_t head_dim);
    void add_bias(tensor::Tensor& x, const tensor::Tensor& bias);

private:
    MetalContext() = default;
    ~MetalContext();

    // Disable copying
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    class Impl;
    Impl* impl_ = nullptr;
};

} // namespace backend::metal
