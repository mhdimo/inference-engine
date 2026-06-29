#pragma once
#include "../../tensor/tensor.hpp"

namespace backend::cpu {

// Computes C = A + B element-wise.
// A, B, and C must have matching shape, strides, and Float32 type.
void add(const tensor::Tensor& A, const tensor::Tensor& B, tensor::Tensor& C);

// Computes output = SiLU(input) element-wise.
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
// input and output must have matching shape and Float32 type.
void silu(const tensor::Tensor& input, tensor::Tensor& output);

// Computes output = Softmax(input) along the last dimension.
// input and output must have matching shape and Float32 type.
void softmax(const tensor::Tensor& input, tensor::Tensor& output);

// Computes output = RMSNorm(input) * gamma along the last dimension.
// input and output must have matching shape and Float32 type.
// gamma must be a 1D parameter tensor with size matching the last dimension of input.
void rmsnorm(const tensor::Tensor& input, const tensor::Tensor& gamma, tensor::Tensor& output, float eps = 1e-5f);

// Computes C = A * B element-wise.
// A, B, and C must have matching shape and Float32 type.
void mul(const tensor::Tensor& A, const tensor::Tensor& B, tensor::Tensor& C);

// Computes output = input * factor element-wise.
// input and output must have matching shape and Float32 type.
void scale(const tensor::Tensor& input, tensor::Tensor& output, float factor);

// Applies RoPE rotations to Q or K tensors in-place based on token position offset
void rope(tensor::Tensor& x, size_t pos_offset, size_t n_heads, size_t head_dim, float freq_base = 10000.0f, size_t rope_dim = 0);

// Computes attention using Grouped-Query Attention (GQA) directly on cache buffers
void gqa_attention(const tensor::Tensor& Q, const tensor::Tensor& K_cache, const tensor::Tensor& V_cache,
                   tensor::Tensor& output, size_t n_heads, size_t n_kv_heads, size_t head_dim);

// Adds a 1D bias vector to a 2D/3D tensor along the last dimension.
void add_bias(tensor::Tensor& x, const tensor::Tensor& bias);

} // namespace backend::cpu

