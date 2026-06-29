#include "engine/core/types.hpp"
#include "engine/core/logger.hpp"
#include "engine/core/quantization.hpp"
#include "engine/tensor/tensor.hpp"
#include "engine/backend/cpu/matmul.hpp"
#include "engine/backend/cpu/quantized_matmul.hpp"
#include "engine/backend/metal/metal_backend.hpp"

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

void fill_random_weights(std::vector<float>& vec, float min_val, float max_val) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (auto& val : vec) {
        val = dis(gen);
    }
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "Q4_0 Block-wise 4-Bit Quantization Demonstration (Phase 6)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    using namespace core;
    using namespace tensor;
    using namespace backend::cpu;
    using namespace backend::metal;

    size_t M = 4;   // Input sequence length
    size_t K = 64;  // Hidden dimension (must be multiple of 32)
    size_t N = 64;  // Projection dimension

    // 1. Initialize input activation and weight matrix (FP32)
    std::vector<float> data_A(M * K);
    std::vector<float> data_W(K * N); // Weight matrix W

    fill_random_weights(data_A, -0.5f, 0.5f);
    fill_random_weights(data_W, -0.2f, 0.2f); // Weights are typically smaller

    Tensor A({M, K}, DType::Float32, data_A.data());
    Tensor W({K, N}, DType::Float32, data_W.data());

    // 2. Perform baseline Float32 Matrix Multiplication: C_baseline = A * W
    std::vector<float> data_C_baseline(M * N, 0.0f);
    Tensor C_baseline({M, N}, DType::Float32, data_C_baseline.data());
    matmul(A, W, C_baseline);

    // 3. Perform Q4_0 block-wise quantization on weight matrix W
    size_t num_blocks = N * (K / QK4_0);
    std::vector<block_q4_0> W_q4(num_blocks);

    std::cout << "[QUANT] Compressing " << K << "x" << N << " FP32 weight matrix into Q4_0 format..." << std::endl;
    quantize_q4_0(data_W.data(), W_q4.data(), K, N);

    // Calculate memory stats
    size_t fp32_bytes = K * N * sizeof(float);
    size_t q4_bytes = num_blocks * sizeof(block_q4_0);
    double ratio = static_cast<double>(fp32_bytes) / q4_bytes;

    std::cout << "[MEM] FP32 Weight Size: " << fp32_bytes << " bytes" << std::endl;
    std::cout << "[MEM] Q4_0 Weight Size: " << q4_bytes << " bytes" << std::endl;
    std::cout << "[MEM] COMPRESSION RATIO: " << std::fixed << std::setprecision(2) << ratio << "x reduction!" << std::endl;

    // 4. Run quantized matrix multiplication on CPU
    std::vector<float> data_C_cpu(M * N, 0.0f);
    Tensor C_cpu({M, N}, DType::Float32, data_C_cpu.data());

    std::cout << "[RUN] Running quantized GEMM on CPU..." << std::endl;
    gemm_q4_0(A, W_q4.data(), C_cpu);

    // Calculate quantization accuracy loss (compared to FP32 baseline)
    float max_error = 0.0f;
    float sum_squared_error = 0.0f;
    for (size_t i = 0; i < M * N; ++i) {
        float err = std::abs(data_C_cpu[i] - data_C_baseline[i]);
        max_error = std::max(max_error, err);
        sum_squared_error += err * err;
    }
    float rmse = std::sqrt(sum_squared_error / (M * N));

    std::cout << "[ERR] Max error (FP32 vs Q4_0 CPU): " << max_error << std::endl;
    std::cout << "[ERR] Root Mean Squared Error (RMSE): " << rmse << std::endl;

    // 5. Run quantized matrix multiplication on GPU (Metal)
    std::cout << "[RUN] Initializing Metal..." << std::endl;
    bool has_gpu = MetalContext::instance().init();
    if (!has_gpu) {
        std::cout << "[ERR] Metal GPU not available. Concluding." << std::endl;
        return 0;
    }

    std::vector<float> data_C_gpu(M * N, 0.0f);
    Tensor C_gpu({M, N}, DType::Float32, data_C_gpu.data());

    std::cout << "[RUN] Running quantized GEMM on GPU (Metal)..." << std::endl;
    MetalContext::instance().matmul_q4_0(A, W_q4.data(), C_gpu);

    // 6. Verify CPU vs GPU equivalence
    float max_gpu_diff = 0.0f;
    bool matched = true;
    for (size_t i = 0; i < M * N; ++i) {
        float diff = std::abs(data_C_gpu[i] - data_C_cpu[i]);
        max_gpu_diff = std::max(max_gpu_diff, diff);
        if (diff > 1e-4f) {
            matched = false;
        }
    }

    std::cout << "==========================================================" << std::endl;
    if (matched) {
        std::cout << "[SUCCESS] CPU and GPU quantized output matched perfectly!" << std::endl;
        std::cout << "[INFO] Maximum difference between CPU and GPU: " << max_gpu_diff << std::endl;
    } else {
        std::cout << "[FAILED] GPU quantized output does not match CPU! Max diff: " << max_gpu_diff << std::endl;
        return 1;
    }

    // Print a few baseline vs CPU vs GPU output values
    std::cout << "[INFO] First 4 elements of token 0 projection logits:" << std::endl;
    std::cout << "  FP32 Baseline: [ ";
    for (size_t i = 0; i < 4; ++i) std::cout << data_C_baseline[i] << " ";
    std::cout << "]" << std::endl;
    std::cout << "  Q4_0 CPU:      [ ";
    for (size_t i = 0; i < 4; ++i) std::cout << data_C_cpu[i] << " ";
    std::cout << "]" << std::endl;
    std::cout << "  Q4_0 Metal:    [ ";
    for (size_t i = 0; i < 4; ++i) std::cout << data_C_gpu[i] << " ";
    std::cout << "]" << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}
