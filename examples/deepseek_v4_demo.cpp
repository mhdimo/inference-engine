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
    std::mt19937 gen(1337); // Seed for reproducibility
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (auto& val : vec) {
        val = dis(gen);
    }
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "DeepSeek-V4 Hybrid Attention Demonstration" << std::endl;
    std::cout << "==========================================================" << std::endl;

    using namespace graph;
    using namespace core;
    using namespace tensor;
    using namespace backend::metal;

    Graph g;

    // Dimension parameters
    size_t hist_len = 12; // 12 past tokens
    size_t curr_len = 4;  // 4 current tokens (local sliding window)
    size_t dim = 64;
    size_t head_dim = 16;
    size_t factor = 4;    // Compression factor C = 4
    size_t k_select = 2;  // Select Top-2 compressed blocks

    size_t comp_len = hist_len / factor; // 12 / 4 = 3 compressed groups
    size_t active_len = k_select + curr_len; // 2 selected + 4 current = 6 active tokens

    // 1. Nodes definition
    // Inputs & parameters
    size_t x_hist_id = g.add_input("x_hist", {hist_len, dim}, DType::Float32);
    size_t x_curr_id = g.add_input("x_curr", {curr_len, dim}, DType::Float32);
    size_t q_last_id = g.add_input("q_last", {1, dim}, DType::Float32);

    size_t w_q_id = g.add_parameter("w_q", {dim, dim}, DType::Float32);
    size_t w_k_id = g.add_parameter("w_k", {dim, dim}, DType::Float32);
    size_t w_v_id = g.add_parameter("w_v", {dim, dim}, DType::Float32);
    size_t w_o_id = g.add_parameter("w_o", {dim, dim}, DType::Float32);

    // Intermediate activations
    // Sequence compression
    size_t x_hist_comp_id = g.add_intermediate({comp_len, dim}, DType::Float32);

    // Lightning indexer Top-K scoring & routing
    size_t route_idx_id = g.add_intermediate({k_select}, DType::Int32); // Holds selected group indices

    // Gather selected compressed context & merge with sliding window
    size_t x_hist_routed_id = g.add_intermediate({k_select, dim}, DType::Float32);
    size_t x_active_id = g.add_intermediate({active_len, dim}, DType::Float32);

    // Attention projections
    size_t q_proj_id = g.add_intermediate({1, dim}, DType::Float32);
    size_t k_active_id = g.add_intermediate({active_len, dim}, DType::Float32);
    size_t v_active_id = g.add_intermediate({active_len, dim}, DType::Float32);

    // Attention scaling & scoring
    size_t q_scaled_id = g.add_intermediate({1, dim}, DType::Float32);
    size_t scores_id = g.add_intermediate({1, active_len}, DType::Float32);
    size_t probs_id = g.add_intermediate({1, active_len}, DType::Float32);
    size_t attn_out_id = g.add_intermediate({1, dim}, DType::Float32);

    size_t proj_id = g.add_intermediate({1, dim}, DType::Float32);
    size_t y_id = g.add_intermediate({1, dim}, DType::Float32); // Final routed attention output token

    // 2. Add operations
    // - Compress sequence
    g.add_seq_compress(x_hist_id, x_hist_comp_id, factor);

    // - Route Top-K blocks based on last token query
    g.add_topk_route(q_last_id, x_hist_comp_id, route_idx_id, k_select);

    // - Gather selected blocks and concatenate with uncompressed sliding window
    g.add_route_gather(x_hist_comp_id, route_idx_id, x_hist_routed_id);
    g.add_concat2(x_hist_routed_id, x_curr_id, x_active_id);

    // - Attention query, key, value projections
    g.add_matmul(q_last_id, w_q_id, q_proj_id);
    g.add_matmul(x_active_id, w_k_id, k_active_id);
    g.add_matmul(x_active_id, w_v_id, v_active_id);

    // - Softmax causal attention scoring
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));
    g.add_scale(q_proj_id, q_scaled_id, scale_factor);
    g.add_matmul(q_scaled_id, k_active_id, scores_id, true); // transpose_b = true
    g.add_softmax(scores_id, probs_id);

    // - Attention weighted sum & output projection
    g.add_matmul(probs_id, v_active_id, attn_out_id);
    g.add_matmul(attn_out_id, w_o_id, proj_id);
    g.add_add(q_last_id, proj_id, y_id); // residual add

    // 3. Compile and Plan Memory
    memory::ArenaAllocator activation_arena(1024 * 1024); // 1MB allocation buffer
    
    size_t arena_used_before = activation_arena.used_bytes();
    g.compile(activation_arena);
    size_t peak_memory_planned = activation_arena.used_bytes() - arena_used_before;

    std::cout << "[MEM] Memory Planner Allocated Size: " << peak_memory_planned << " bytes" << std::endl;

    // 4. Initialize random buffers
    std::vector<float> x_hist_data(hist_len * dim);
    std::vector<float> x_curr_data(curr_len * dim);
    std::vector<float> q_last_data(dim);

    std::vector<float> w_q_data(dim * dim);
    std::vector<float> w_k_data(dim * dim);
    std::vector<float> w_v_data(dim * dim);
    std::vector<float> w_o_data(dim * dim);

    fill_random(x_hist_data, -0.5f, 0.5f);
    fill_random(x_curr_data, -0.5f, 0.5f);
    fill_random(q_last_data, -0.2f, 0.2f);
    fill_random(w_q_data, -0.1f, 0.1f);
    fill_random(w_k_data, -0.1f, 0.1f);
    fill_random(w_v_data, -0.1f, 0.1f);
    fill_random(w_o_data, -0.1f, 0.1f);

    // Bind inputs/weights
    g.bind_data(x_hist_id, x_hist_data.data());
    g.bind_data(x_curr_id, x_curr_data.data());
    g.bind_data(q_last_id, q_last_data.data());
    g.bind_data(w_q_id, w_q_data.data());
    g.bind_data(w_k_id, w_k_data.data());
    g.bind_data(w_v_id, w_v_data.data());
    g.bind_data(w_o_id, w_o_data.data());

    // 5. Execute on CPU
    std::cout << "[RUN] Executing DeepSeek-V4 block on CPU..." << std::endl;
    g.execute(BackendType::CPU);

    std::vector<float> cpu_output(dim);
    std::memcpy(cpu_output.data(), g.get_tensor(y_id).data(), cpu_output.size() * sizeof(float));

    // Print routed indices
    int32_t* route_indices = g.get_tensor(route_idx_id).data_as<int32_t>();
    std::cout << "[ROUTE] Top-2 Selected Historical Compressed Blocks: [ ";
    for (size_t i = 0; i < k_select; ++i) {
        std::cout << route_indices[i] << " ";
    }
    std::cout << "]" << std::endl;

    // Zero out activation output
    std::memset(g.get_tensor(y_id).data(), 0, dim * sizeof(float));

    // 6. Initialize Metal and Execute (using hybrid fallback execution)
    std::cout << "[RUN] Initializing Metal..." << std::endl;
    bool has_gpu = MetalContext::instance().init();
    if (!has_gpu) {
        std::cout << "[ERR] Metal not available. Cannot execute on GPU." << std::endl;
        return 1;
    }

    std::cout << "[RUN] Executing on GPU (Metal with Unified Memory fallbacks)..." << std::endl;
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
    std::cout << "[INFO] First 4 elements of routed attention output:" << std::endl;
    std::cout << "  CPU:   [ ";
    for(size_t i=0; i<4; ++i) std::cout << cpu_output[i] << " ";
    std::cout << "]" << std::endl;
    std::cout << "  Metal: [ ";
    for(size_t i=0; i<4; ++i) std::cout << metal_res[i] << " ";
    std::cout << "]" << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}
