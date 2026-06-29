#pragma once
#include <cstdint>
#include <cstddef>

namespace core {

// Q4_0 block size is 32 elements.
constexpr size_t QK4_0 = 32;

struct __attribute__((packed)) block_q4_0 {
    uint16_t d;        // Block scale factor (FP16, 2 bytes)
    uint8_t qs[16];    // 32 packed 4-bit signed integers (stored in 16 bytes)
};

// Q8_0 block size is 32 elements.
constexpr size_t QK8_0 = 32;

struct __attribute__((packed)) block_q8_0 {
    uint16_t d;        // Block scale factor (FP16, 2 bytes)
    int8_t qs[32];     // 32 8-bit signed integers (32 bytes)
};

// K-quant super-block size is 256 elements.
constexpr size_t QK_K = 256;

// Q6_K block: 256 elements packed in 210 bytes (matches ggml block_q6_K).
struct __attribute__((packed)) block_q6_K {
    uint8_t  ql[QK_K / 2];      // 128 bytes: lower 4 bits of each 6-bit quant
    uint8_t  qh[QK_K / 4];      //  64 bytes: upper 2 bits of each 6-bit quant
    int8_t   scales[QK_K / 16]; //  16 bytes: per-subblock (16 elements) 8-bit scales
    uint16_t d;                 //   2 bytes: fp16 super-block scale
};

// Dequantizes Q6_K blocks into a flat float array (row-major, `elements` total, multiple of 256).
void dequantize_q6_K(const block_q6_K* src, float* dst, size_t elements);

// Quantizes a float weight matrix of shape [K, N] (K must be multiple of 32) into Q4_0 blocks.
void quantize_q4_0(const float* src, block_q4_0* dst, size_t K, size_t N);

// Dequantizes Q4_0 blocks back into a float weight matrix of shape [K, N].
void dequantize_q4_0(const block_q4_0* src, float* dst, size_t K, size_t N);

} // namespace core
