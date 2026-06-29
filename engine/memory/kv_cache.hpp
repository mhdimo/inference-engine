#pragma once
#include "../tensor/tensor.hpp"
#include "../core/logger.hpp"
#include <vector>
#include <cstring>
#include <stdexcept>

namespace memory {

class KVCacheManager {
public:
    KVCacheManager(size_t max_seq_len, size_t dim) 
        : max_len_(max_seq_len), dim_(dim), curr_len_(0) {
        k_data_.resize(max_seq_len * dim, 0.0f);
        v_data_.resize(max_seq_len * dim, 0.0f);
    }

    void reset() {
        curr_len_ = 0;
    }

    void append_kv(const tensor::Tensor& K_new, const tensor::Tensor& V_new) {
        if (K_new.shape().ndim != 2 || V_new.shape().ndim != 2) {
            LOG_ERR("K/V new tensors must be 2D");
            throw std::invalid_argument("K/V new tensors must be 2D");
        }
        size_t L_new = K_new.shape()[0];
        size_t d_k = K_new.shape()[1];

        if (d_k != dim_ || V_new.shape()[1] != dim_ || V_new.shape()[0] != L_new) {
            LOG_ERR("K/V new tensor dimension mismatch");
            throw std::invalid_argument("K/V new tensor dimension mismatch");
        }

        if (curr_len_ + L_new > max_len_) {
            LOG_ERR("K/V cache capacity exceeded!");
            throw std::runtime_error("K/V cache capacity exceeded");
        }

        const float* src_k = K_new.data_as<float>();
        const float* src_v = V_new.data_as<float>();

        size_t stride_k = K_new.strides()[0];
        size_t stride_v = V_new.strides()[0];

        // Copy K_new and V_new into persistent cache arrays
        for (size_t i = 0; i < L_new; ++i) {
            std::memcpy(k_data_.data() + (curr_len_ + i) * dim_, src_k + i * stride_k, dim_ * sizeof(float));
            std::memcpy(v_data_.data() + (curr_len_ + i) * dim_, src_v + i * stride_v, dim_ * sizeof(float));
        }

        curr_len_ += L_new;
    }

    tensor::Tensor get_active_k() const {
        if (curr_len_ == 0) {
            return tensor::Tensor(core::Shape{0, dim_}, core::DType::Float32, const_cast<float*>(k_data_.data()));
        }
        return tensor::Tensor(core::Shape{curr_len_, dim_}, core::DType::Float32, const_cast<float*>(k_data_.data()));
    }

    tensor::Tensor get_active_v() const {
        if (curr_len_ == 0) {
            return tensor::Tensor(core::Shape{0, dim_}, core::DType::Float32, const_cast<float*>(v_data_.data()));
        }
        return tensor::Tensor(core::Shape{curr_len_, dim_}, core::DType::Float32, const_cast<float*>(v_data_.data()));
    }

    size_t current_len() const { return curr_len_; }
    size_t max_len() const { return max_len_; }
    size_t dim() const { return dim_; }

private:
    size_t max_len_;
    size_t dim_;
    size_t curr_len_ = 0;
    std::vector<float> k_data_;
    std::vector<float> v_data_;
};

} // namespace memory
