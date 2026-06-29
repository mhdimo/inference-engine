#include "engine/core/types.hpp"
#include "engine/core/logger.hpp"
#include "engine/memory/allocator.hpp"
#include "engine/tensor/tensor.hpp"
#include "engine/graph/graph.hpp"
#include "engine/backend/metal/metal_backend.hpp"

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

// Helper to fill a vector with random values
void fill_random(std::vector<float>& vec, float min_val, float max_val) {
    std::mt19937 gen(42); // Seed for reproducibility
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (auto& val : vec) {
        val = dis(gen);
    }
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "LLaMA Transformer Block Demonstration (Phase 4)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // hidden dimensions
    size_t seq_len = 4;
    size_t dim = 64;
    size_t head_dim = 16;
    size_t mlp_dim = 128;

    using namespace graph;
    using namespace core;
    using namespace tensor;
    using namespace backend::metal;

    Graph g;

    // 1. Nodes definition
    // Inputs
    size_t x_id = g.add_input("x", {seq_len, dim}, DType::Float32);
    size_t residual_id = g.add_input("residual_dummy", {seq_len, dim}, DType::Float32); // Used to bind prompt input

    // Parameters (weights)
    size_t gamma1_id = g.add_parameter("gamma1", {dim}, DType::Float32);
    size_t w_q_id = g.add_parameter("w_q", {dim, dim}, DType::Float32);
    size_t w_k_id = g.add_parameter("w_k", {dim, dim}, DType::Float32);
    size_t w_v_id = g.add_parameter("w_v", {dim, dim}, DType::Float32);
    size_t w_o_id = g.add_parameter("w_o", {dim, dim}, DType::Float32);

    size_t gamma2_id = g.add_parameter("gamma2", {dim}, DType::Float32);
    size_t w_gate_id = g.add_parameter("w_gate", {dim, mlp_dim}, DType::Float32);
    size_t w_up_id = g.add_parameter("w_up", {dim, mlp_dim}, DType::Float32);
    size_t w_down_id = g.add_parameter("w_down", {mlp_dim, dim}, DType::Float32);

    // Intermediate activations
    size_t x_norm_id = g.add_intermediate({seq_len, dim}, DType::Float32);
    size_t q_id = g.add_intermediate({seq_len, dim}, DType::Float32);
    size_t k_id = g.add_intermediate({seq_len, dim}, DType::Float32);
    size_t v_id = g.add_intermediate({seq_len, dim}, DType::Float32);

    size_t q_scaled_id = g.add_intermediate({seq_len, dim}, DType::Float32);
    size_t scores_id = g.add_intermediate({seq_len, seq_len}, DType::Float32);
    size_t probs_id = g.add_intermediate({seq_len, seq_len}, DType::Float32);
    size_t attn_out_id = g.add_intermediate({seq_len, dim}, DType::Float32);
    size_t proj_id = g.add_intermediate({seq_len, dim}, DType::Float32);

    size_t x_attn_id = g.add_intermediate({seq_len, dim}, DType::Float32);

    size_t x_norm2_id = g.add_intermediate({seq_len, dim}, DType::Float32);
    size_t gate_id = g.add_intermediate({seq_len, mlp_dim}, DType::Float32);
    size_t up_id = g.add_intermediate({seq_len, mlp_dim}, DType::Float32);
    size_t gate_act_id = g.add_intermediate({seq_len, mlp_dim}, DType::Float32);
    size_t swiglu_id = g.add_intermediate({seq_len, mlp_dim}, DType::Float32);
    size_t down_id = g.add_intermediate({seq_len, dim}, DType::Float32);

    size_t y_id = g.add_intermediate({seq_len, dim}, DType::Float32); // Final output of transformer block

    // 2. Add operations (LLaMA Block sequence)
    // --- Causal/Self Attention Prefill ---
    g.add_rmsnorm(x_id, gamma1_id, x_norm_id, 1e-5f);
    g.add_matmul(x_norm_id, w_q_id, q_id);
    g.add_matmul(x_norm_id, w_k_id, k_id);
    g.add_matmul(x_norm_id, w_v_id, v_id);

    float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));
    g.add_scale(q_id, q_scaled_id, scale_factor);

    // scores = Q_scaled * K^T
    g.add_matmul(q_scaled_id, k_id, scores_id, true); // transpose_b = true

    g.add_softmax(scores_id, probs_id);

    // attn_out = probs * V
    g.add_matmul(probs_id, v_id, attn_out_id);

    // proj = attn_out * W_o
    g.add_matmul(attn_out_id, w_o_id, proj_id);
    g.add_add(x_id, proj_id, x_attn_id); // residual connection

    // --- SwiGLU MLP ---
    g.add_rmsnorm(x_attn_id, gamma2_id, x_norm2_id, 1e-5f);
    g.add_matmul(x_norm2_id, w_gate_id, gate_id);
    g.add_matmul(x_norm2_id, w_up_id, up_id);
    g.add_silu(gate_id, gate_act_id);
    g.add_mul(gate_act_id, up_id, swiglu_id);
    g.add_matmul(swiglu_id, w_down_id, down_id);

    g.add_add(x_attn_id, down_id, y_id); // residual connection

    // 3. Compile and Plan Memory
    memory::ArenaAllocator activation_arena(1024 * 1024); // 1MB activation memory
    
    size_t arena_used_before = activation_arena.used_bytes();
    g.compile(activation_arena);
    size_t peak_memory_planned = activation_arena.used_bytes() - arena_used_before;

    // Calculate sum of individual intermediate sizes (what naive allocations would take)
    size_t sum_individual_sizes = 0;
    auto add_size = [&](const Shape& sh) {
        size_t bytes = sh.size() * sizeof(float);
        return (bytes + 63) & ~63; // 64-byte rounded
    };
    sum_individual_sizes += add_size({seq_len, dim});       // x_norm
    sum_individual_sizes += add_size({seq_len, dim});       // q
    sum_individual_sizes += add_size({seq_len, dim});       // k
    sum_individual_sizes += add_size({seq_len, dim});       // v
    sum_individual_sizes += add_size({seq_len, dim});       // q_scaled
    sum_individual_sizes += add_size({seq_len, seq_len});   // scores
    sum_individual_sizes += add_size({seq_len, seq_len});   // probs
    sum_individual_sizes += add_size({seq_len, dim});       // attn_out
    sum_individual_sizes += add_size({seq_len, dim});       // proj
    sum_individual_sizes += add_size({seq_len, dim});       // x_attn
    sum_individual_sizes += add_size({seq_len, dim});       // x_norm2
    sum_individual_sizes += add_size({seq_len, mlp_dim});   // gate
    sum_individual_sizes += add_size({seq_len, mlp_dim});   // up
    sum_individual_sizes += add_size({seq_len, mlp_dim});   // gate_act
    sum_individual_sizes += add_size({seq_len, mlp_dim});   // swiglu
    sum_individual_sizes += add_size({seq_len, dim});       // down
    sum_individual_sizes += add_size({seq_len, dim});       // y

    double saving_pct = (1.0 - (double)peak_memory_planned / sum_individual_sizes) * 100.0;

    std::cout << "[MEM] Individual Tensors (rounded): " << sum_individual_sizes << " bytes" << std::endl;
    std::cout << "[MEM] Planner Allocated Buffer:   " << peak_memory_planned << " bytes" << std::endl;
    std::cout << "[MEM] STATIC MEMORY SAVED:         " << std::fixed << std::setprecision(1) << saving_pct << "%" << std::endl;

    // 4. Initialize random buffers
    std::vector<float> x_data(seq_len * dim);
    std::vector<float> residual_data(seq_len * dim);
    std::vector<float> gamma1_data(dim);
    std::vector<float> w_q_data(dim * dim);
    std::vector<float> w_k_data(dim * dim);
    std::vector<float> w_v_data(dim * dim);
    std::vector<float> w_o_data(dim * dim);
    std::vector<float> gamma2_data(dim);
    std::vector<float> w_gate_data(dim * mlp_dim);
    std::vector<float> w_up_data(dim * mlp_dim);
    std::vector<float> w_down_data(mlp_dim * dim);

    fill_random(x_data, -0.5f, 0.5f);
    fill_random(residual_data, -0.1f, 0.1f);
    fill_random(gamma1_data, 0.9f, 1.1f);
    fill_random(w_q_data, -0.1f, 0.1f);
    fill_random(w_k_data, -0.1f, 0.1f);
    fill_random(w_v_data, -0.1f, 0.1f);
    fill_random(w_o_data, -0.1f, 0.1f);
    fill_random(gamma2_data, 0.9f, 1.1f);
    fill_random(w_gate_data, -0.1f, 0.1f);
    fill_random(w_up_data, -0.1f, 0.1f);
    fill_random(w_down_data, -0.1f, 0.1f);

    // Bind inputs/weights
    g.bind_data(x_id, x_data.data());
    g.bind_data(residual_id, residual_data.data());
    g.bind_data(gamma1_id, gamma1_data.data());
    g.bind_data(w_q_id, w_q_data.data());
    g.bind_data(w_k_id, w_k_data.data());
    g.bind_data(w_v_id, w_v_data.data());
    g.bind_data(w_o_id, w_o_data.data());
    g.bind_data(gamma2_id, gamma2_data.data());
    g.bind_data(w_gate_id, w_gate_data.data());
    g.bind_data(w_up_id, w_up_data.data());
    g.bind_data(w_down_id, w_down_data.data());

    // 5. Execute on CPU
    std::cout << "[RUN] Executing on CPU..." << std::endl;
    g.execute(BackendType::CPU);

    std::vector<float> cpu_output(seq_len * dim);
    std::memcpy(cpu_output.data(), g.get_tensor(y_id).data(), cpu_output.size() * sizeof(float));

    // Zero out activations to ensure Metal doesn't copy from CPU
    std::memset(g.get_tensor(y_id).data(), 0, seq_len * dim * sizeof(float));

    // 6. Initialize Metal and Execute
    std::cout << "[RUN] Initializing Metal..." << std::endl;
    bool has_gpu = MetalContext::instance().init();
    if (!has_gpu) {
        std::cout << "[ERR] Metal not available. Cannot execute on GPU." << std::endl;
        return 1;
    }

    std::cout << "[RUN] Executing on GPU (Metal)..." << std::endl;
    g.execute(BackendType::Metal);

    // 7. Verify Results
    float* metal_res = g.get_tensor(y_id).data_as<float>();
    bool matched = true;
    float max_diff = 0.0f;
    for (size_t i = 0; i < cpu_output.size(); ++i) {
        float diff = std::abs(metal_res[i] - cpu_output[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > 1e-4f) {
            matched = false;
        }
    }

    std::cout << "==========================================================" << std::endl;
    if (matched) {
        std::cout << "[SUCCESS] CPU and GPU outputs MATCH PERFECTLY!" << std::endl;
        std::cout << "[INFO] Maximum absolute difference: " << max_diff << std::endl;
    } else {
        std::cout << "[FAILED] CPU and GPU output mismatch! Max diff: " << max_diff << std::endl;
        return 1;
    }

    // Print first token outputs
    std::cout << "[INFO] First 4 elements of token 0 output:" << std::endl;
    std::cout << "  CPU:   [ ";
    for(size_t i=0; i<4; ++i) std::cout << cpu_output[i] << " ";
    std::cout << "]" << std::endl;
    std::cout << "  Metal: [ ";
    for(size_t i=0; i<4; ++i) std::cout << metal_res[i] << " ";
    std::cout << "]" << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}
