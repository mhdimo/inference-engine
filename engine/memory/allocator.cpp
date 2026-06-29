#include "allocator.hpp"
#include "../core/logger.hpp"
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace memory {

ArenaAllocator::ArenaAllocator(size_t capacity_bytes)
    : capacity_(capacity_bytes), offset_(0) {
    if (capacity_bytes == 0) {
        return;
    }
    // Allocate 16KB (page) aligned memory for macOS Metal zero-copy compatibility
    int res = posix_memalign(reinterpret_cast<void**>(&buffer_), 16384, capacity_bytes);
    if (res != 0) {
        LOG_ERR("Failed to allocate page-aligned memory of size: " + std::to_string(capacity_bytes));
        throw std::bad_alloc();
    }
}

ArenaAllocator::~ArenaAllocator() {
    if (buffer_) {
        std::free(buffer_);
    }
}

ArenaAllocator::ArenaAllocator(ArenaAllocator&& other) noexcept
    : buffer_(other.buffer_), capacity_(other.capacity_), offset_(other.offset_) {
    other.buffer_ = nullptr;
    other.capacity_ = 0;
    other.offset_ = 0;
}

ArenaAllocator& ArenaAllocator::operator=(ArenaAllocator&& other) noexcept {
    if (this != &other) {
        if (buffer_) {
            std::free(buffer_);
        }
        buffer_ = other.buffer_;
        capacity_ = other.capacity_;
        offset_ = other.offset_;
        other.buffer_ = nullptr;
        other.capacity_ = 0;
        other.offset_ = 0;
    }
    return *this;
}

void* ArenaAllocator::allocate(size_t size_bytes, size_t alignment) {
    if (capacity_ == 0 || buffer_ == nullptr) {
        LOG_ERR("Attempted to allocate from an uninitialized or moved ArenaAllocator");
        throw std::bad_alloc();
    }

    // Ensure alignment is a power of 2
    if ((alignment & (alignment - 1)) != 0 || alignment == 0) {
        LOG_ERR("Alignment must be a non-zero power of 2");
        throw std::invalid_argument("Alignment must be a non-zero power of 2");
    }

    // Round up the current offset to the next multiple of alignment
    size_t current_ptr = reinterpret_cast<size_t>(buffer_ + offset_);
    size_t aligned_ptr = (current_ptr + alignment - 1) & ~(alignment - 1);
    size_t new_offset = (aligned_ptr - reinterpret_cast<size_t>(buffer_)) + size_bytes;

    if (new_offset > capacity_) {
        LOG_ERR("ArenaAllocator out of memory! Capacity: " + std::to_string(capacity_) + 
                ", Required: " + std::to_string(new_offset));
        throw std::bad_alloc();
    }

    offset_ = new_offset;
    return reinterpret_cast<void*>(aligned_ptr);
}

void ArenaAllocator::reset() {
    offset_ = 0;
}

} // namespace memory
