#pragma once
#include "../../tensor/tensor.hpp"

namespace backend::cpu {

// Computes C = A * B.
// C must be pre-allocated and have shape [A.shape[0], B.shape[1]].
// Inner dimension of A and B must match: A.shape[1] == B.shape[0].
// Currently supports Float32 data type.
void matmul(const tensor::Tensor& A, const tensor::Tensor& B, tensor::Tensor& C);

} // namespace backend::cpu
