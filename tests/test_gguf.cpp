#include "engine/core/types.hpp"
#include "engine/core/logger.hpp"
#include "engine/core/quantization.hpp"
#include "engine/core/gguf_loader.hpp"
#include "engine/tensor/tensor.hpp"
#include "engine/backend/cpu/quantized_matmul.hpp"
#include "engine/backend/metal/metal_backend.hpp"

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdio>

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "GGUF Weight Loader Integration Test (Phase 7)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Generate the mock GGUF file using the Python script
    std::cout << "[TEST] Running Python mock GGUF generator script..." << std::endl;
    int ret = std::system("python3 scripts/create_mock_gguf.py mock_model.gguf");
    if (ret != 0) {
        std::cerr << "[ERR] Failed to execute mock generator script" << std::endl;
        return 1;
    }

    // 2. Open GGUF file using our C++ Memory-Mapped GGUFLoader
    std::cout << "[TEST] Instantiating GGUFLoader and memory-mapping file..." << std::endl;
    core::GGUFLoader loader("mock_model.gguf");
    bool ok = loader.load();
    if (!ok) {
        std::cerr << "[ERR] GGUFLoader failed to load file" << std::endl;
        std::remove("mock_model.gguf");
        return 1;
    }

    // 3. Verify Metadata extraction
    std::string model_name = loader.get_metadata_string("general.name");
    uint32_t block_count = loader.get_metadata_uint32("llama.block_count");
    uint32_t alignment = loader.get_metadata_uint32("general.alignment");

    std::cout << "[META] Model Name:      " << model_name << " (Expected: Mock-LLaMA)" << std::endl;
    std::cout << "[META] Block Count:     " << block_count << " (Expected: 1)" << std::endl;
    std::cout << "[META] Alignment Size:  " << alignment << " (Expected: 32)" << std::endl;

    assert(model_name == "Mock-LLaMA");
    assert(block_count == 1);
    assert(alignment == 32);

    // 4. Verify Tensors extraction
    std::cout << "[TEST] Checking loaded tensor descriptors..." << std::endl;
    const auto& tensors = loader.tensors();
    assert(tensors.size() == 2);

    const auto* w_q_info = loader.get_tensor_info("w_q");
    const auto* w_o_info = loader.get_tensor_info("w_o");

    assert(w_q_info != nullptr);
    assert(w_o_info != nullptr);

    std::cout << "[TENS] Tensor 'w_q': shape=[" << w_q_info->dims[0] << ", " << w_q_info->dims[1] 
              << "], type=" << w_q_info->type << " (Expected: 2=Q4_0), size=" << w_q_info->size_bytes << " bytes" << std::endl;
    std::cout << "[TENS] Tensor 'w_o': shape=[" << w_o_info->dims[0] << ", " << w_o_info->dims[1] 
              << "], type=" << w_o_info->type << " (Expected: 0=FP32), size=" << w_o_info->size_bytes << " bytes" << std::endl;

    assert(w_q_info->dims[0] == 64 && w_q_info->dims[1] == 64);
    assert(w_q_info->type == 2); // Q4_0
    assert(w_o_info->dims[0] == 64 && w_o_info->dims[1] == 64);
    assert(w_o_info->type == 0); // FP32

    // 5. Verify direct memory-mapped tensor values
    const void* raw_wq = loader.get_tensor_data(*w_q_info);
    const void* raw_wo = loader.get_tensor_data(*w_o_info);

    const float* data_wo = static_cast<const float*>(raw_wo);
    const core::block_q4_0* data_wq = static_cast<const core::block_q4_0*>(raw_wq);

    // Verify FP32 values
    for (size_t i = 0; i < 64 * 64; ++i) {
        if (data_wo[i] != 1.5f) {
            std::cerr << "[ERR] FP32 value mismatch at index " << i << ": got " << data_wo[i] << " (Expected: 1.5f)" << std::endl;
            std::remove("mock_model.gguf");
            return 1;
        }
    }
    std::cout << "[SUCCESS] FP32 weights verified successfully (all elements are 1.5f)." << std::endl;

    // Verify Q4_0 block values
    size_t num_blocks = 64 * (64 / 32);
    for (size_t b = 0; b < num_blocks; ++b) {
        if (data_wq[b].d != 0x3c00) {
            std::cerr << "[ERR] Q4_0 scale mismatch at block " << b << ": got " << data_wq[b].d << " (expected 15360/0x3c00)" << std::endl;
            std::remove("mock_model.gguf");
            return 1;
        }
        for (size_t i = 0; i < 16; ++i) {
            if (data_wq[b].qs[i] != 0x33) {
                std::cerr << "[ERR] Q4_0 qs mismatch at block " << b << ", index " << i << ": got " << (int)data_wq[b].qs[i] << std::endl;
                std::remove("mock_model.gguf");
                return 1;
            }
        }
    }
    std::cout << "[SUCCESS] Q4_0 weights verified successfully (all block scales are 1.0, qs are 0x33)." << std::endl;

    // 6. Execute CPU quantized GEMM using loaded weights
    size_t M = 4;
    size_t K = 64;
    size_t N = 64;
    std::vector<float> data_A(M * K, 0.5f);
    tensor::Tensor A({M, K}, core::DType::Float32, data_A.data());

    std::vector<float> data_C_cpu(M * N, 0.0f);
    tensor::Tensor C_cpu({M, N}, core::DType::Float32, data_C_cpu.data());

    std::cout << "[RUN] Running quantized GEMM on CPU using memory-mapped weights..." << std::endl;
    backend::cpu::gemm_q4_0(A, data_wq, C_cpu);

    // 7. Verify GPU quantized GEMM using loaded weights
    std::cout << "[RUN] Initializing Metal..." << std::endl;
    bool has_gpu = backend::metal::MetalContext::instance().init();
    if (has_gpu) {
        std::vector<float> data_C_gpu(M * N, 0.0f);
        tensor::Tensor C_gpu({M, N}, core::DType::Float32, data_C_gpu.data());

        std::cout << "[RUN] Running quantized GEMM on GPU (Metal)..." << std::endl;
        backend::metal::MetalContext::instance().matmul_q4_0(A, data_wq, C_gpu);

        // Check matching
        float max_diff = 0.0f;
        for (size_t i = 0; i < M * N; ++i) {
            float diff = std::abs(data_C_gpu[i] - data_C_cpu[i]);
            if (diff > max_diff) max_diff = diff;
        }

        std::cout << "[TEST] Maximum absolute difference CPU vs GPU: " << max_diff << std::endl;
        assert(max_diff < 1e-4f);
        std::cout << "[SUCCESS] GGUF mapped quantized GPU matches CPU perfectly!" << std::endl;
    }

    // Clean up
    std::remove("mock_model.gguf");
    std::cout << "==========================================================" << std::endl;
    std::cout << "[SUCCESS] GGUF loader integration test PASSED!" << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}
