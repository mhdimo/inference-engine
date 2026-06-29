#pragma once
#include <cstddef>
#include <cstdint>

namespace memory {

class ArenaAllocator {
public:
    explicit ArenaAllocator(size_t capacity_bytes);
    ~ArenaAllocator();

    // Disable copy semantics to prevent duplicate ownership of memory
    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    // Allow move semantics
    ArenaAllocator(ArenaAllocator&& other) noexcept;
    ArenaAllocator& operator=(ArenaAllocator&& other) noexcept;

    // Allocates memory of size_bytes, aligned to alignment (default 64 for SIMD/caches)
    void* allocate(size_t size_bytes, size_t alignment = 64);
    
    // Resets the offset pointer back to zero without freeing memory (for activation reuse)
    void reset();

    size_t used_bytes() const { return offset_; }
    size_t capacity_bytes() const { return capacity_; }
    uint8_t* raw_buffer() { return buffer_; }
    const uint8_t* raw_buffer() const { return buffer_; }

private:
    uint8_t* buffer_ = nullptr;
    size_t capacity_ = 0;
    size_t offset_ = 0;
};

} // namespace memory
