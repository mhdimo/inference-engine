#pragma once
#include "../../tensor/tensor.hpp"
#include "../../core/quantization.hpp"

namespace backend::cpu {

// Computes C = A * W_q4 (Matrix Multiplication).
// A: FP32 tensor of shape [M, K]
// W_q4: Quantized Q4_0 blocks of shape [K, N] (total N * (K / 32) blocks)
// C: FP32 output tensor of shape [M, N]
void gemm_q4_0(const tensor::Tensor& A, const core::block_q4_0* W_q4, tensor::Tensor& C);
void gemm_q8_0(const tensor::Tensor& A, const core::block_q8_0* W_q8, tensor::Tensor& C);

} // namespace backend::cpu
