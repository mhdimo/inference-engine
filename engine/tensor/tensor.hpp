#pragma once
#include "../core/types.hpp"

namespace tensor {

class Tensor {
public:
    Tensor() = default;
    
    // Constructor. If strides is empty, it will compute contiguous strides automatically.
    Tensor(core::Shape shape, core::DType dtype, void* data, core::Shape strides = {});

    const core::Shape& shape() const { return shape_; }
    const core::Shape& strides() const { return strides_; }
    core::DType dtype() const { return dtype_; }
    void* data() { return data_; }
    const void* data() const { return data_; }

    template<typename T>
    T* data_as() { return static_cast<T*>(data_); }

    template<typename T>
    const T* data_as() const { return static_cast<const T*>(data_); }

    size_t numel() const { return shape_.size(); }
    size_t nbytes() const { return numel() * core::dtype_size(dtype_); }
    bool is_contiguous() const;

private:
    core::Shape shape_;
    core::Shape strides_;
    core::DType dtype_ = core::DType::Float32;
    void* data_ = nullptr;
};

} // namespace tensor