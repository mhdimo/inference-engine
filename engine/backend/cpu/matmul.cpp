#include "matmul.hpp"
#include "../../core/logger.hpp"
#include <stdexcept>

namespace backend::cpu {

void matmul(const tensor::Tensor& A, const tensor::Tensor& B, tensor::Tensor& C) {
    if (A.shape().ndim != 2 || B.shape().ndim != 2 || C.shape().ndim != 2) {
        LOG_ERR("matmul requires 2D tensors");
        throw std::invalid_argument("matmul requires 2D tensors");
    }

    size_t M = A.shape()[0];
    size_t K = A.shape()[1];
    size_t K_b = B.shape()[0];
    size_t N = B.shape()[1];

    size_t M_c = C.shape()[0];
    size_t N_c = C.shape()[1];

    if (K != K_b) {
        LOG_ERR("matmul inner dimensions mismatch: A inner is " + std::to_string(K) + 
                ", B outer is " + std::to_string(K_b));
        throw std::invalid_argument("matmul inner dimensions mismatch");
    }
    if (M != M_c || N != N_c) {
        LOG_ERR("matmul output dimension mismatch: expected " + std::to_string(M) + "x" + std::to_string(N) +
                ", got " + std::to_string(M_c) + "x" + std::to_string(N_c));
        throw std::invalid_argument("matmul output dimension mismatch");
    }

    if (A.dtype() != core::DType::Float32 || B.dtype() != core::DType::Float32 || C.dtype() != core::DType::Float32) {
        LOG_ERR("matmul only supports Float32 at this time");
        throw std::invalid_argument("matmul only supports Float32");
    }

    const float* data_A = A.data_as<float>();
    const float* data_B = B.data_as<float>();
    float* data_C = C.data_as<float>();

    size_t stride_A_M = A.strides()[0];
    size_t stride_A_K = A.strides()[1];
    size_t stride_B_K = B.strides()[0];
    size_t stride_B_N = B.strides()[1];
    size_t stride_C_M = C.strides()[0];
    size_t stride_C_N = C.strides()[1];

    // Compute C = A * B using naive loops with strides
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float val_A = data_A[i * stride_A_M + k * stride_A_K];
                float val_B = data_B[k * stride_B_K + j * stride_B_N];
                sum += val_A * val_B;
            }
            data_C[i * stride_C_M + j * stride_C_N] = sum;
        }
    }
}

} // namespace backend::cpu
