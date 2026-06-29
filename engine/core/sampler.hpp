#pragma once
#include <cstdint>
#include <cstddef>

namespace core {

// Samples the next token index from logits.
// If temperature <= 0.0f, performs greedy decoding (argmax).
// Otherwise, scales by temperature, applies softmax, and samples randomly.
int32_t sample_logits(const float* logits, size_t size, float temperature);

} // namespace core
