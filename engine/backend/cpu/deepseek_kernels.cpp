#include "deepseek_kernels.hpp"
#include "../../core/logger.hpp"
#include <stdexcept>
#include <algorithm>
#include <utility>

namespace backend::cpu {

void seq_compress(const tensor::Tensor& input, tensor::Tensor& output, size_t factor) {
    if (input.shape().ndim != 2 || output.shape().ndim != 2) {
        LOG_ERR("seq_compress requires 2D tensors");
        throw std::invalid_argument("seq_compress requires 2D tensors");
    }
    size_t seq_len = input.shape()[0];
    size_t dim = input.shape()[1];
    size_t comp_len = output.shape()[0];
    size_t comp_dim = output.shape()[1];

    if (comp_len != seq_len / factor || comp_dim != dim) {
        LOG_ERR("seq_compress output dimensions mismatch");
        throw std::invalid_argument("seq_compress output dimensions mismatch");
    }

    const float* in_data = input.data_as<float>();
    float* out_data = output.data_as<float>();

    size_t stride_in_0 = input.strides()[0];
    size_t stride_in_1 = input.strides()[1];
    size_t stride_out_0 = output.strides()[0];
    size_t stride_out_1 = output.strides()[1];

    for (size_t g = 0; g < comp_len; ++g) {
        for (size_t d = 0; d < dim; ++d) {
            float sum = 0.0f;
            for (size_t f = 0; f < factor; ++f) {
                size_t in_idx = (g * factor + f) * stride_in_0 + d * stride_in_1;
                sum += in_data[in_idx];
            }
            size_t out_idx = g * stride_out_0 + d * stride_out_1;
            out_data[out_idx] = sum / static_cast<float>(factor);
        }
    }
}

void topk_route(const tensor::Tensor& query, const tensor::Tensor& keys, tensor::Tensor& output_indices, size_t k) {
    if (query.shape().ndim != 2 || keys.shape().ndim != 2) {
        LOG_ERR("topk_route requires 2D tensors");
        throw std::invalid_argument("topk_route requires 2D tensors");
    }
    size_t dim = query.shape()[1];
    size_t num_groups = keys.shape()[0];
    if (keys.shape()[1] != dim) {
        LOG_ERR("topk_route query and keys dimension mismatch");
        throw std::invalid_argument("topk_route query and keys dimension mismatch");
    }
    if (output_indices.shape().ndim != 1 || output_indices.shape()[0] != k) {
        LOG_ERR("topk_route output_indices shape must be [k]");
        throw std::invalid_argument("topk_route output_indices shape must be [k]");
    }
    if (output_indices.dtype() != core::DType::Int32) {
        LOG_ERR("topk_route output_indices must be Int32 type");
        throw std::invalid_argument("topk_route output_indices must be Int32 type");
    }

    const float* q_data = query.data_as<float>();
    const float* k_data = keys.data_as<float>();
    int32_t* out_data = output_indices.data_as<int32_t>();

    size_t stride_q_1 = query.strides()[1];
    size_t stride_k_0 = keys.strides()[0];
    size_t stride_k_1 = keys.strides()[1];

    // Restrict group sorting to stack buffers for zero-heap-allocation compliance
    if (num_groups > 64) {
        LOG_ERR("topk_route num_groups exceeds max support limit of 64");
        throw std::runtime_error("topk_route num_groups exceeds limit");
    }

    float group_scores[64] = {0.0f};
    int32_t indices[64];

    for (size_t g = 0; g < num_groups; ++g) {
        indices[g] = static_cast<int32_t>(g);
        float sum = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            sum += q_data[d * stride_q_1] * k_data[g * stride_k_0 + d * stride_k_1];
        }
        group_scores[g] = sum;
    }

    // Sort indices in-place based on scores (descending)
    for (size_t i = 0; i < num_groups - 1; ++i) {
        for (size_t j = i + 1; j < num_groups; ++j) {
            if (group_scores[indices[j]] > group_scores[indices[i]]) {
                std::swap(indices[i], indices[j]);
            }
        }
    }

    // Write the top K selected expert indices
    for (size_t i = 0; i < k; ++i) {
        if (i < num_groups) {
            out_data[i] = indices[i];
        } else {
            out_data[i] = -1; // Padding
        }
    }
}

void route_gather(const tensor::Tensor& src, const tensor::Tensor& indices, tensor::Tensor& dst) {
    if (src.shape().ndim != 2 || dst.shape().ndim != 2 || indices.shape().ndim != 1) {
        LOG_ERR("route_gather shape rank mismatch");
        throw std::invalid_argument("route_gather shape rank mismatch");
    }
    size_t dim = src.shape()[1];
    size_t num_indices = indices.shape()[0];
    if (dst.shape()[0] != num_indices || dst.shape()[1] != dim) {
        LOG_ERR("route_gather dst dimensions mismatch");
        throw std::invalid_argument("route_gather dst dimensions mismatch");
    }
    if (indices.dtype() != core::DType::Int32) {
        LOG_ERR("route_gather indices must be Int32");
        throw std::invalid_argument("route_gather indices must be Int32");
    }

    const float* src_ptr = src.data_as<float>();
    const int32_t* idx_ptr = indices.data_as<int32_t>();
    float* dst_ptr = dst.data_as<float>();

    size_t stride_src_0 = src.strides()[0];
    size_t stride_src_1 = src.strides()[1];
    size_t stride_dst_0 = dst.strides()[0];
    size_t stride_dst_1 = dst.strides()[1];

    for (size_t i = 0; i < num_indices; ++i) {
        int32_t idx = idx_ptr[i];
        if (idx < 0 || idx >= static_cast<int32_t>(src.shape()[0])) {
            // Fill with zeros for invalid/padding index
            for (size_t d = 0; d < dim; ++d) {
                dst_ptr[i * stride_dst_0 + d * stride_dst_1] = 0.0f;
            }
        } else {
            for (size_t d = 0; d < dim; ++d) {
                dst_ptr[i * stride_dst_0 + d * stride_dst_1] = src_ptr[idx * stride_src_0 + d * stride_src_1];
            }
        }
    }
}

void concat2(const tensor::Tensor& A, const tensor::Tensor& B, tensor::Tensor& C) {
    if (A.shape().ndim != 2 || B.shape().ndim != 2 || C.shape().ndim != 2) {
        LOG_ERR("concat2 requires 2D tensors");
        throw std::invalid_argument("concat2 requires 2D tensors");
    }
    size_t dim = A.shape()[1];
    size_t M_A = A.shape()[0];
    size_t M_B = B.shape()[0];

    if (B.shape()[1] != dim || C.shape()[1] != dim || C.shape()[0] != M_A + M_B) {
        LOG_ERR("concat2 dimension mismatch");
        throw std::invalid_argument("concat2 dimension mismatch");
    }

    const float* ptr_A = A.data_as<float>();
    const float* ptr_B = B.data_as<float>();
    float* ptr_C = C.data_as<float>();

    size_t stride_A = A.strides()[0];
    size_t stride_B = B.strides()[0];
    size_t stride_C = C.strides()[0];

    size_t stride_A_1 = A.strides()[1];
    size_t stride_B_1 = B.strides()[1];
    size_t stride_C_1 = C.strides()[1];

    for (size_t i = 0; i < M_A; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            ptr_C[i * stride_C + d * stride_C_1] = ptr_A[i * stride_A + d * stride_A_1];
        }
    }
    for (size_t i = 0; i < M_B; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            ptr_C[(M_A + i) * stride_C + d * stride_C_1] = ptr_B[i * stride_B + d * stride_B_1];
        }
    }
}

} // namespace backend::cpu
