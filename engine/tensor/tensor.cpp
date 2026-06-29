#include "tensor.hpp"

namespace tensor {

Tensor::Tensor(core::Shape shape, core::DType dtype, void* data, core::Shape strides)
    : shape_(shape), dtype_(dtype), data_(data) {
    if (strides.ndim == 0 && shape.ndim > 0) {
        strides_ = core::compute_contiguous_strides(shape);
    } else {
        strides_ = strides;
    }
}

bool Tensor::is_contiguous() const {
    core::Shape expected = core::compute_contiguous_strides(shape_);
    return strides_ == expected;
}

} // namespace tensor
