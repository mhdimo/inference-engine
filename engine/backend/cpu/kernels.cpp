#include "kernels.hpp"
#include "../../core/logger.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace backend::cpu {

// Helper to compute physical offset of a flat index in a stride-aware manner.
inline size_t get_physical_offset(size_t flat_idx, const core::Shape& shape, const core::Shape& strides) {
    size_t offset = 0;
    size_t rem = flat_idx;
    for (size_t i = shape.ndim; i > 0; --i) {
        size_t d = i - 1;
        size_t idx = rem % shape[d];
        offset += idx * strides[d];
        rem /= shape[d];
    }
    return offset;
}

// Helper to compute the start physical offset of a given row when processing along the last dimension.
inline size_t get_row_start_offset(size_t row_idx, const core::Shape& shape, const core::Shape& strides) {
    if (shape.ndim <= 1) return 0;
    size_t offset = 0;
    size_t rem = row_idx;
    for (size_t i = shape.ndim - 1; i > 0; --i) {
        size_t d = i - 1;
        size_t idx = rem % shape[d];
        offset += idx * strides[d];
        rem /= shape[d];
    }
    return offset;
}

void add(const tensor::Tensor& A, const tensor::Tensor& B, tensor::Tensor& C) {
    if (A.shape() != B.shape() || A.shape() != C.shape()) {
        LOG_ERR("Add shapes mismatch");
        throw std::invalid_argument("Add shapes mismatch");
    }
    if (A.dtype() != core::DType::Float32 || B.dtype() != core::DType::Float32 || C.dtype() != core::DType::Float32) {
        LOG_ERR("Add only supports Float32");
        throw std::invalid_argument("Add only supports Float32");
    }

    const float* data_A = A.data_as<float>();
    const float* data_B = B.data_as<float>();
    float* data_C = C.data_as<float>();

    size_t total_elements = A.numel();

    if (A.is_contiguous() && B.is_contiguous() && C.is_contiguous()) {
        for (size_t i = 0; i < total_elements; ++i) {
            data_C[i] = data_A[i] + data_B[i];
        }
    } else {
        for (size_t i = 0; i < total_elements; ++i) {
            size_t off_A = get_physical_offset(i, A.shape(), A.strides());
            size_t off_B = get_physical_offset(i, B.shape(), B.strides());
            size_t off_C = get_physical_offset(i, C.shape(), C.strides());
            data_C[off_C] = data_A[off_A] + data_B[off_B];
        }
    }
}

void silu(const tensor::Tensor& input, tensor::Tensor& output) {
    if (input.shape() != output.shape()) {
        LOG_ERR("SiLU shapes mismatch");
        throw std::invalid_argument("SiLU shapes mismatch");
    }
    if (input.dtype() != core::DType::Float32 || output.dtype() != core::DType::Float32) {
        LOG_ERR("SiLU only supports Float32");
        throw std::invalid_argument("SiLU only supports Float32");
    }

    const float* in_data = input.data_as<float>();
    float* out_data = output.data_as<float>();
    size_t total_elements = input.numel();

    if (input.is_contiguous() && output.is_contiguous()) {
        for (size_t i = 0; i < total_elements; ++i) {
            float x = in_data[i];
            out_data[i] = x / (1.0f + std::exp(-x));
        }
    } else {
        for (size_t i = 0; i < total_elements; ++i) {
            size_t off_in = get_physical_offset(i, input.shape(), input.strides());
            size_t off_out = get_physical_offset(i, output.shape(), output.strides());
            float x = in_data[off_in];
            out_data[off_out] = x / (1.0f + std::exp(-x));
        }
    }
}

void softmax(const tensor::Tensor& input, tensor::Tensor& output) {
    if (input.shape() != output.shape()) {
        LOG_ERR("Softmax shapes mismatch");
        throw std::invalid_argument("Softmax shapes mismatch");
    }
    if (input.dtype() != core::DType::Float32 || output.dtype() != core::DType::Float32) {
        LOG_ERR("Softmax only supports Float32");
        throw std::invalid_argument("Softmax only supports Float32");
    }
    if (input.shape().ndim == 0) {
        return;
    }

    size_t ndim = input.shape().ndim;
    size_t N = input.shape()[ndim - 1]; // last dimension size
    size_t num_rows = input.numel() / N;

    size_t stride_in_last = input.strides()[ndim - 1];
    size_t stride_out_last = output.strides()[ndim - 1];

    const float* in_ptr = input.data_as<float>();
    float* out_ptr = output.data_as<float>();

    for (size_t r = 0; r < num_rows; ++r) {
        size_t in_start = get_row_start_offset(r, input.shape(), input.strides());
        size_t out_start = get_row_start_offset(r, output.shape(), output.strides());

        // Find max (for numerical stability)
        float max_val = in_ptr[in_start];
        for (size_t c = 1; c < N; ++c) {
            float val = in_ptr[in_start + c * stride_in_last];
            if (val > max_val) {
                max_val = val;
            }
        }

        // Sum exponentiation
        float sum = 0.0f;
        for (size_t c = 0; c < N; ++c) {
            float val = std::exp(in_ptr[in_start + c * stride_in_last] - max_val);
            out_ptr[out_start + c * stride_out_last] = val;
            sum += val;
        }

        // Normalize
        float inv_sum = 1.0f / sum;
        for (size_t c = 0; c < N; ++c) {
            out_ptr[out_start + c * stride_out_last] *= inv_sum;
        }
    }
}

void rmsnorm(const tensor::Tensor& input, const tensor::Tensor& gamma, tensor::Tensor& output, float eps) {
    if (input.shape() != output.shape()) {
        LOG_ERR("RMSNorm input and output shape mismatch");
        throw std::invalid_argument("RMSNorm input and output shape mismatch");
    }
    if (gamma.shape().ndim != 1) {
        LOG_ERR("RMSNorm gamma must be 1D parameter");
        throw std::invalid_argument("RMSNorm gamma must be 1D parameter");
    }
    size_t ndim = input.shape().ndim;
    size_t N = input.shape()[ndim - 1];
    if (gamma.shape()[0] != N) {
        LOG_ERR("RMSNorm gamma dimension mismatch: expected " + std::to_string(N) + ", got " + std::to_string(gamma.shape()[0]));
        throw std::invalid_argument("RMSNorm gamma dimension mismatch");
    }

    if (input.dtype() != core::DType::Float32 || gamma.dtype() != core::DType::Float32 || output.dtype() != core::DType::Float32) {
        LOG_ERR("RMSNorm only supports Float32");
        throw std::invalid_argument("RMSNorm only supports Float32");
    }

    size_t num_rows = input.numel() / N;
    size_t stride_in_last = input.strides()[ndim - 1];
    size_t stride_out_last = output.strides()[ndim - 1];
    size_t stride_gamma = gamma.strides()[0];

    const float* in_ptr = input.data_as<float>();
    const float* gamma_ptr = gamma.data_as<float>();
    float* out_ptr = output.data_as<float>();

    for (size_t r = 0; r < num_rows; ++r) {
        size_t in_start = get_row_start_offset(r, input.shape(), input.strides());
        size_t out_start = get_row_start_offset(r, output.shape(), output.strides());

        // Compute mean of squared values
        float sum_sq = 0.0f;
        for (size_t c = 0; c < N; ++c) {
            float val = in_ptr[in_start + c * stride_in_last];
            sum_sq += val * val;
        }

        float rms = std::sqrt(sum_sq / static_cast<float>(N) + eps);
        float inv_rms = 1.0f / rms;

        // Apply normalization and scale by gamma
        for (size_t c = 0; c < N; ++c) {
            float val = in_ptr[in_start + c * stride_in_last];
            float g = gamma_ptr[c * stride_gamma];
            out_ptr[out_start + c * stride_out_last] = val * inv_rms * g;
        }
    }
}

void mul(const tensor::Tensor& A, const tensor::Tensor& B, tensor::Tensor& C) {
    if (A.shape() != B.shape() || A.shape() != C.shape()) {
        LOG_ERR("Mul shapes mismatch");
        throw std::invalid_argument("Mul shapes mismatch");
    }
    if (A.dtype() != core::DType::Float32 || B.dtype() != core::DType::Float32 || C.dtype() != core::DType::Float32) {
        LOG_ERR("Mul only supports Float32");
        throw std::invalid_argument("Mul only supports Float32");
    }

    const float* data_A = A.data_as<float>();
    const float* data_B = B.data_as<float>();
    float* data_C = C.data_as<float>();

    size_t total_elements = A.numel();

    if (A.is_contiguous() && B.is_contiguous() && C.is_contiguous()) {
        for (size_t i = 0; i < total_elements; ++i) {
            data_C[i] = data_A[i] * data_B[i];
        }
    } else {
        for (size_t i = 0; i < total_elements; ++i) {
            size_t off_A = get_physical_offset(i, A.shape(), A.strides());
            size_t off_B = get_physical_offset(i, B.shape(), B.strides());
            size_t off_C = get_physical_offset(i, C.shape(), C.strides());
            data_C[off_C] = data_A[off_A] * data_B[off_B];
        }
    }
}

void scale(const tensor::Tensor& input, tensor::Tensor& output, float factor) {
    if (input.shape() != output.shape()) {
        LOG_ERR("Scale shapes mismatch");
        throw std::invalid_argument("Scale shapes mismatch");
    }
    if (input.dtype() != core::DType::Float32 || output.dtype() != core::DType::Float32) {
        LOG_ERR("Scale only supports Float32");
        throw std::invalid_argument("Scale only supports Float32");
    }

    const float* in_data = input.data_as<float>();
    float* out_data = output.data_as<float>();
    size_t total_elements = input.numel();

    if (input.is_contiguous() && output.is_contiguous()) {
        for (size_t i = 0; i < total_elements; ++i) {
            out_data[i] = in_data[i] * factor;
        }
    } else {
        for (size_t i = 0; i < total_elements; ++i) {
            size_t off_in = get_physical_offset(i, input.shape(), input.strides());
            size_t off_out = get_physical_offset(i, output.shape(), output.strides());
            out_data[off_out] = in_data[off_in] * factor;
        }
    }
}

void rope(tensor::Tensor& x, size_t pos_offset, size_t n_heads, size_t head_dim, float freq_base, size_t rope_dim) {
    if (x.dtype() != core::DType::Float32) {
        LOG_ERR("RoPE only supports Float32");
        throw std::invalid_argument("RoPE only supports Float32");
    }

    if (rope_dim == 0) {
        rope_dim = head_dim;
    }

    float* data = x.data_as<float>();
    size_t L = x.shape()[0];
    size_t stride_0 = x.strides()[0];
    size_t stride_1 = x.strides()[1];

    for (size_t i = 0; i < L; ++i) {
        size_t pos = pos_offset + i;
        for (size_t h = 0; h < n_heads; ++h) {
            for (size_t d = 0; d < rope_dim / 2; ++d) {
                float theta = static_cast<float>(pos) / std::pow(freq_base, static_cast<float>(2 * d) / static_cast<float>(rope_dim));
                float cos_t = std::cos(theta);
                float sin_t = std::sin(theta);

                size_t idx0 = i * stride_0 + (h * head_dim + d) * stride_1;
                size_t idx1 = i * stride_0 + (h * head_dim + d + rope_dim / 2) * stride_1;

                float v0 = data[idx0];
                float v1 = data[idx1];

                data[idx0] = v0 * cos_t - v1 * sin_t;
                data[idx1] = v0 * sin_t + v1 * cos_t;
            }
        }
    }
}

void gqa_attention(const tensor::Tensor& Q, const tensor::Tensor& K_cache, const tensor::Tensor& V_cache,
                   tensor::Tensor& output, size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    if (Q.dtype() != core::DType::Float32 || K_cache.dtype() != core::DType::Float32 || V_cache.dtype() != core::DType::Float32 || output.dtype() != core::DType::Float32) {
        LOG_ERR("GQA only supports Float32");
        throw std::invalid_argument("GQA only supports Float32");
    }

    size_t L = Q.shape()[0];
    size_t seq_len = K_cache.shape()[0];
    size_t group_size = n_heads / n_kv_heads;

    const float* data_Q = Q.data_as<float>();
    const float* data_K = K_cache.data_as<float>();
    const float* data_V = V_cache.data_as<float>();
    float* data_out = output.data_as<float>();

    size_t stride_Q_0 = Q.strides()[0];
    size_t stride_Q_1 = Q.strides()[1];
    
    size_t stride_K_0 = K_cache.strides()[0];
    size_t stride_K_1 = K_cache.strides()[1];
    
    size_t stride_V_0 = V_cache.strides()[0];
    size_t stride_V_1 = V_cache.strides()[1];
    
    size_t stride_out_0 = output.strides()[0];
    size_t stride_out_1 = output.strides()[1];

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Allocate thread-local scores buffer to avoid dynamic memory allocation in loop
    std::vector<float> scores(seq_len);

    for (size_t i = 0; i < L; ++i) {
        for (size_t h = 0; h < n_heads; ++h) {
            size_t kv_h = h / group_size;

            // 1. Compute dot product attention scores
            float max_score = -INFINITY;
            size_t query_pos = (seq_len - L) + i;
            for (size_t t = 0; t < seq_len; ++t) {
                if (t > query_pos) {
                    scores[t] = -INFINITY;
                    continue;
                }
                float sum = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    size_t idx_q = i * stride_Q_0 + (h * head_dim + d) * stride_Q_1;
                    size_t idx_k = t * stride_K_0 + (kv_h * head_dim + d) * stride_K_1;
                    sum += data_Q[idx_q] * data_K[idx_k];
                }
                scores[t] = sum * scale;
                if (scores[t] > max_score) {
                    max_score = scores[t];
                }
            }

            // 2. Softmax
            float sum_exp = 0.0f;
            for (size_t t = 0; t < seq_len; ++t) {
                if (t > query_pos) {
                    scores[t] = 0.0f;
                } else {
                    scores[t] = std::exp(scores[t] - max_score);
                    sum_exp += scores[t];
                }
            }
            float inv_sum = (sum_exp == 0.0f) ? 0.0f : 1.0f / sum_exp;
            for (size_t t = 0; t < seq_len; ++t) {
                scores[t] *= inv_sum; // scores[t] is now the attention probability
            }

            // 3. Weighted sum over values
            for (size_t d = 0; d < head_dim; ++d) {
                float sum_v = 0.0f;
                for (size_t t = 0; t < seq_len; ++t) {
                    size_t idx_v = t * stride_V_0 + (kv_h * head_dim + d) * stride_V_1;
                    sum_v += scores[t] * data_V[idx_v];
                }
                size_t idx_out = i * stride_out_0 + (h * head_dim + d) * stride_out_1;
                data_out[idx_out] = sum_v;
            }
        }
    }
}

void add_bias(tensor::Tensor& x, const tensor::Tensor& bias) {
    if (x.dtype() != core::DType::Float32 || bias.dtype() != core::DType::Float32) {
        LOG_ERR("add_bias only supports Float32");
        throw std::invalid_argument("add_bias only supports Float32");
    }
    float* data_x = x.data_as<float>();
    const float* data_b = bias.data_as<float>();
    size_t L = x.shape()[0];
    size_t D = x.shape()[1];
    size_t stride_x_0 = x.strides()[0];
    size_t stride_x_1 = x.strides()[1];
    size_t stride_b = bias.strides()[0];

    for (size_t i = 0; i < L; ++i) {
        for (size_t j = 0; j < D; ++j) {
            data_x[i * stride_x_0 + j * stride_x_1] += data_b[j * stride_b];
        }
    }
}

} // namespace backend::cpu

