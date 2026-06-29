#include "quantized_matmul.hpp"
#include "../../core/logger.hpp"
#include <stdexcept>
#include <iostream>

namespace backend::cpu {

inline float dequantize_fp16(uint16_t h) {
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

inline int8_t sign_extend_4bit(uint8_t val) {
    return (val & 0x08) ? static_cast<int8_t>(val | 0xF0) : static_cast<int8_t>(val & 0x0F);
}

void gemm_q4_0(const tensor::Tensor& A, const core::block_q4_0* W_q4, tensor::Tensor& C) {
    if (A.shape().ndim != 2 || C.shape().ndim != 2) {
        LOG_ERR("gemm_q4_0 requires 2D tensors");
        throw std::invalid_argument("gemm_q4_0 requires 2D tensors");
    }

    size_t M = A.shape()[0];
    size_t K = A.shape()[1];
    size_t N = C.shape()[1];

    if (C.shape()[0] != M) {
        LOG_ERR("gemm_q4_0 shape mismatch");
        throw std::invalid_argument("gemm_q4_0 shape mismatch");
    }

    if (K % core::QK4_0 != 0) {
        LOG_ERR("gemm_q4_0 inner dimension K must be multiple of 32");
        throw std::invalid_argument("gemm_q4_0 inner dimension K must be multiple of 32");
    }

    const float* data_A = A.data_as<float>();
    float* data_C = C.data_as<float>();

    size_t stride_A_0 = A.strides()[0];
    size_t stride_A_1 = A.strides()[1];
    size_t stride_C_0 = C.strides()[0];
    size_t stride_C_1 = C.strides()[1];

    size_t num_blocks_per_col = K / core::QK4_0;

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t b = 0; b < num_blocks_per_col; ++b) {
                // Fetch the quantized block for column j
                const core::block_q4_0& block = W_q4[j * num_blocks_per_col + b];
                float d = dequantize_fp16(block.d);



                // ggml Q4_0 convention: value = (nibble - 8) * d, BLOCKED layout
                // (low nibbles at block positions 0..15, high nibbles at 16..31).
                for (size_t l = 0; l < core::QK4_0 / 2; ++l) {
                    uint8_t val = block.qs[l];

                    int q0 = (val & 0x0F) - 8;
                    int q1 = (val >> 4) - 8;

                    size_t k0 = b * core::QK4_0 + l;
                    size_t k1 = b * core::QK4_0 + l + core::QK4_0 / 2;

                    sum += data_A[i * stride_A_0 + k0 * stride_A_1] * static_cast<float>(q0) * d;
                    sum += data_A[i * stride_A_0 + k1 * stride_A_1] * static_cast<float>(q1) * d;
                }
            }
            data_C[i * stride_C_0 + j * stride_C_1] = sum;
        }
    }
}

void gemm_q8_0(const tensor::Tensor& A, const core::block_q8_0* W_q8, tensor::Tensor& C) {
    if (A.shape().ndim != 2 || C.shape().ndim != 2) {
        LOG_ERR("gemm_q8_0 requires 2D tensors");
        throw std::invalid_argument("gemm_q8_0 requires 2D tensors");
    }

    size_t M = A.shape()[0];
    size_t K = A.shape()[1];
    size_t N = C.shape()[1];

    if (C.shape()[0] != M) {
        LOG_ERR("gemm_q8_0 shape mismatch");
        throw std::invalid_argument("gemm_q8_0 shape mismatch");
    }

    if (K % core::QK8_0 != 0) {
        LOG_ERR("gemm_q8_0 inner dimension K must be multiple of 32");
        throw std::invalid_argument("gemm_q8_0 inner dimension K must be multiple of 32");
    }

    const float* data_A = A.data_as<float>();
    float* data_C = C.data_as<float>();

    size_t stride_A_0 = A.strides()[0];
    size_t stride_A_1 = A.strides()[1];
    size_t stride_C_0 = C.strides()[0];
    size_t stride_C_1 = C.strides()[1];

    size_t num_blocks_per_col = K / core::QK8_0;

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t b = 0; b < num_blocks_per_col; ++b) {
                // Fetch the quantized block for column j
                const core::block_q8_0& block = W_q8[j * num_blocks_per_col + b];
                float d = dequantize_fp16(block.d);

                for (size_t l = 0; l < core::QK8_0; ++l) {
                    size_t k = b * core::QK8_0 + l;
                    sum += data_A[i * stride_A_0 + k * stride_A_1] * static_cast<float>(block.qs[l]) * d;
                }
            }
            data_C[i * stride_C_0 + j * stride_C_1] = sum;
        }
    }
}

} // namespace backend::cpu

