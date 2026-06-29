#include "sampler.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

namespace core {

int32_t sample_logits(const float* logits, size_t size, float temperature) {
    if (size == 0) return -1;

    if (temperature <= 0.0f) {
        // Greedy decoding: find argmax index
        float max_val = logits[0];
        int32_t argmax = 0;
        for (size_t i = 1; i < size; ++i) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                argmax = static_cast<int32_t>(i);
            }
        }
        return argmax;
    }

    // Scaled softmax sampling
    std::vector<float> probs(size);
    
    // Stability offset (subtract max)
    float max_logit = logits[0];
    for (size_t i = 1; i < size; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    float sum_exp = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        probs[i] = std::exp((logits[i] - max_logit) / temperature);
        sum_exp += probs[i];
    }

    for (size_t i = 0; i < size; ++i) {
        probs[i] /= sum_exp;
    }

    // Cumulative distribution sampling
    static thread_local std::mt19937 gen(42); // Fixed seed for test verification
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    float r = dis(gen);

    float cumulative = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        cumulative += probs[i];
        if (r <= cumulative) {
            return static_cast<int32_t>(i);
        }
    }

    return static_cast<int32_t>(size - 1);
}

} // namespace core
