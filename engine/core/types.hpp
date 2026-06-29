#pragma once
#include <cstddef>
#include <initializer_list>

namespace core {

enum class DType {
    Float32,
    Float16,
    Int32
};

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::Int32:   return 4;
    }
    return 0;
}

inline const char* dtype_to_string(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "Float32";
        case DType::Float16: return "Float16";
        case DType::Int32:   return "Int32";
    }
    return "Unknown";
}

struct Shape {
    static constexpr size_t MAX_DIMS = 4;
    size_t dims[MAX_DIMS] = {0};
    size_t ndim = 0;

    Shape() = default;
    Shape(std::initializer_list<size_t> list) {
        for (auto val : list) {
            if (ndim >= MAX_DIMS) break;
            dims[ndim++] = val;
        }
    }

    size_t size() const {
        if (ndim == 0) return 0;
        size_t total = 1;
        for (size_t i = 0; i < ndim; ++i) {
            total *= dims[i];
        }
        return total;
    }

    bool operator==(const Shape& other) const {
        if (ndim != other.ndim) return false;
        for (size_t i = 0; i < ndim; ++i) {
            if (dims[i] != other.dims[i]) return false;
        }
        return true;
    }

    bool operator!=(const Shape& other) const {
        return !(*this == other);
    }

    size_t operator[](size_t idx) const { return dims[idx]; }
    size_t& operator[](size_t idx) { return dims[idx]; }
};

inline Shape compute_contiguous_strides(const Shape& shape) {
    Shape strides;
    strides.ndim = shape.ndim;
    if (shape.ndim == 0) return strides;
    
    size_t current_stride = 1;
    for (size_t i = shape.ndim; i > 0; --i) {
        strides[i - 1] = current_stride;
        current_stride *= shape[i - 1];
    }
    return strides;
}

} // namespace core
