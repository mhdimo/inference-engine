#include "quantization.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace core {

// Helper to sign-extend a 4-bit signed integer
inline int8_t sign_extend_4bit(uint8_t val) {
    return (val & 0x08) ? static_cast<int8_t>(val | 0xF0) : static_cast<int8_t>(val & 0x0F);
}

inline uint16_t fp32_to_fp16(float f) {
    union { float f; uint32_t u; } val;
    val.f = f;
    uint32_t i = val.u;
    uint32_t sign = (i >> 16) & 0x8000;
    int32_t exp = ((i >> 23) & 0xFF) - 127;
    uint32_t mant = i & 0x7FFFFF;
    if (exp <= -15) {
        return sign;
    }
    if (exp >= 16) {
        return sign | 0x7C00;
    }
    return sign | ((exp + 15) << 10) | (mant >> 13);
}

inline float fp16_to_fp32(uint16_t h) {
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

void quantize_q4_0(const float* src, block_q4_0* dst, size_t K, size_t N) {
    if (K % QK4_0 != 0) {
        throw std::invalid_argument("quantize_q4_0 requires K dimension to be multiple of 32");
    }

    size_t num_blocks_per_col = K / QK4_0;

    for (size_t j = 0; j < N; ++j) {
        for (size_t b = 0; b < num_blocks_per_col; ++b) {
            float max_val = 0.0f;
            float block_src[QK4_0];

            // 1. Gather column elements for the block
            for (size_t l = 0; l < QK4_0; ++l) {
                block_src[l] = src[(b * QK4_0 + l) * N + j];
                max_val = std::max(max_val, std::abs(block_src[l]));
            }

            // 2. Compute block scale factor (d)
            float d = max_val / 8.0f; 
            dst[j * num_blocks_per_col + b].d = fp32_to_fp16(d);

            // 3. Quantize and pack
            float id = (d == 0.0f) ? 0.0f : 1.0f / d;
            for (size_t i = 0; i < QK4_0 / 2; ++i) {
                float v0 = block_src[2 * i] * id;
                float v1 = block_src[2 * i + 1] * id;

                int8_t q0 = static_cast<int8_t>(std::clamp(std::round(v0), -8.0f, 7.0f));
                int8_t q1 = static_cast<int8_t>(std::clamp(std::round(v1), -8.0f, 7.0f));

                uint8_t n0 = static_cast<uint8_t>(q0) & 0x0F;
                uint8_t n1 = static_cast<uint8_t>(q1) & 0x0F;

                dst[j * num_blocks_per_col + b].qs[i] = n0 | (n1 << 4);
            }
        }
    }
}

void dequantize_q4_0(const block_q4_0* src, float* dst, size_t K, size_t N) {
    if (K % QK4_0 != 0) {
        throw std::invalid_argument("dequantize_q4_0 requires K dimension to be multiple of 32");
    }

    size_t num_blocks_per_col = K / QK4_0;

    for (size_t j = 0; j < N; ++j) {
        for (size_t b = 0; b < num_blocks_per_col; ++b) {
            const block_q4_0& block = src[j * num_blocks_per_col + b];
            float d = fp16_to_fp32(block.d);

            // ggml Q4_0 convention: value = (nibble - 8) * d, BLOCKED layout
            // (low nibbles at block positions 0..15, high nibbles at 16..31).
            for (size_t i = 0; i < QK4_0 / 2; ++i) {
                uint8_t val = block.qs[i];

                int q0 = (val & 0x0F) - 8;
                int q1 = (val >> 4) - 8;

                dst[(b * QK4_0 + i) * N + j] = static_cast<float>(q0) * d;
                dst[(b * QK4_0 + i + QK4_0 / 2) * N + j] = static_cast<float>(q1) * d;
            }
        }
    }
}

// Scalar dequantization of Q6_K, mirroring ggml's dequantize_row_q6_K exactly.
void dequantize_q6_K(const block_q6_K* src, float* dst, size_t elements) {
    if (elements % QK_K != 0) {
        throw std::invalid_argument("dequantize_q6_K requires elements to be a multiple of 256");
    }
    const size_t nb = elements / QK_K;

    for (size_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(src[i].d);

        const uint8_t* ql = src[i].ql;
        const uint8_t* qh = src[i].qh;
        const int8_t*  sc = src[i].scales;
        float* y = dst + i * QK_K;

        for (int n = 0; n < (int)QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = static_cast<int8_t>((ql[l]      & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = static_cast<int8_t>( (ql[l]      >> 4)     | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = static_cast<int8_t>( (ql[l + 32] >> 4)     | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * static_cast<float>(q1);
                y[l + 32] = d * sc[is + 2] * static_cast<float>(q2);
                y[l + 64] = d * sc[is + 4] * static_cast<float>(q3);
                y[l + 96] = d * sc[is + 6] * static_cast<float>(q4);
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

} // namespace core
