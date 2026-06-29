#include "engine/core/types.hpp"
#include "engine/core/logger.hpp"
#include "engine/core/sampler.hpp"
#include "engine/memory/allocator.hpp"
#include "engine/memory/kv_cache.hpp"
#include "engine/tensor/tensor.hpp"
#include "engine/graph/graph.hpp"
#include "engine/backend/metal/metal_backend.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <random>

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "Auto-Regressive Text Generation Demonstration (Phase 5)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // Simple vocabulary dictionary
    const std::vector<std::string> vocab = {
        " The",      // 0
        " robot",    // 1
        " learns",   // 2
        " to",       // 3
        " code",     // 4
        " faster",   // 5
        " than",     // 6
        " humans"    // 7
    };
    size_t vocab_size = vocab.size();

    // Model dimensions
    size_t dim = 64;
    size_t head_dim = 16;
    size_t max_seq_len = 16;

    using namespace graph;
    using namespace core;
    using namespace tensor;
    using namespace backend::metal;
    using namespace memory;

    // 1. Initialize KV Cache Manager
    KVCacheManager kv_cache(max_seq_len, dim);

    // Prompt tokens: " The robot learns to" -> token IDs: [0, 1, 2, 3]
    std::vector<int32_t> prompt = {0, 1, 2, 3};
    std::cout << "[PROMPT] Original Tokens: [ ";
    for (int32_t t : prompt) {
        std::cout << t << " (" << vocab[t] << ") ";
    }
    std::cout << "]" << std::endl;

    // Embeddings mapping (one-hot weights simulation)
    // w_embed maps token ID to a unique vector of dimension 64
    std::vector<float> w_embed(vocab_size * dim);
    for (size_t i = 0; i < vocab_size; ++i) {
        std::mt19937 gen(i * 100);
        std::uniform_real_distribution<float> dis(-0.2f, 0.2f);
        for (size_t d = 0; d < dim; ++d) {
            w_embed[i * dim + d] = dis(gen);
        }
    }

    // Weight matrices for projections
    std::vector<float> w_q_data(dim * dim);
    std::vector<float> w_k_data(dim * dim);
    std::vector<float> w_v_data(dim * dim);
    std::vector<float> w_o_data(dim * dim);
    std::vector<float> w_logits_data(dim * vocab_size);

    std::mt19937 w_gen(1337);
    std::uniform_real_distribution<float> w_dis(-0.1f, 0.1f);
    auto fill_w = [&](std::vector<float>& w) {
        for (auto& val : w) val = w_dis(w_gen);
    };
    fill_w(w_q_data);
    fill_w(w_k_data);
    fill_w(w_v_data);
    fill_w(w_o_data);
    fill_w(w_logits_data);

    // Arena for activations
    ArenaAllocator activation_arena(1024 * 1024); // 1MB activation memory

    // A. PREFILL PHASE: Process the initial prompt sequence (length = 4)
    size_t prefill_len = prompt.size();
    std::vector<float> prefill_embeddings(prefill_len * dim);
    for (size_t i = 0; i < prefill_len; ++i) {
        std::memcpy(prefill_embeddings.data() + i * dim, w_embed.data() + prompt[i] * dim, dim * sizeof(float));
    }

    std::cout << "\n[RUN] Starting Prefill Phase..." << std::endl;
    {
        Graph prefill_graph;
        
        // Input sequence
        size_t x_id = prefill_graph.add_input("x", {prefill_len, dim}, DType::Float32);

        // Parameters
        size_t w_q_id = prefill_graph.add_parameter("w_q", {dim, dim}, DType::Float32);
        size_t w_k_id = prefill_graph.add_parameter("w_k", {dim, dim}, DType::Float32);
        size_t w_v_id = prefill_graph.add_parameter("w_v", {dim, dim}, DType::Float32);
        size_t w_o_id = prefill_graph.add_parameter("w_o", {dim, dim}, DType::Float32);
        size_t w_logits_id = prefill_graph.add_parameter("w_logits", {dim, vocab_size}, DType::Float32);

        // Intermediate projection nodes
        size_t q_id = prefill_graph.add_intermediate({prefill_len, dim}, DType::Float32);
        size_t k_id = prefill_graph.add_intermediate({prefill_len, dim}, DType::Float32);
        size_t v_id = prefill_graph.add_intermediate({prefill_len, dim}, DType::Float32);

        size_t q_scaled_id = prefill_graph.add_intermediate({prefill_len, dim}, DType::Float32);
        size_t scores_id = prefill_graph.add_intermediate({prefill_len, prefill_len}, DType::Float32);
        size_t probs_id = prefill_graph.add_intermediate({prefill_len, prefill_len}, DType::Float32);
        size_t attn_out_id = prefill_graph.add_intermediate({prefill_len, dim}, DType::Float32);

        size_t proj_id = prefill_graph.add_intermediate({prefill_len, dim}, DType::Float32);
        size_t out_id = prefill_graph.add_intermediate({prefill_len, dim}, DType::Float32);
        size_t logits_id = prefill_graph.add_intermediate({prefill_len, vocab_size}, DType::Float32);

        // Connect operations
        prefill_graph.add_matmul(x_id, w_q_id, q_id);
        prefill_graph.add_matmul(x_id, w_k_id, k_id);
        prefill_graph.add_matmul(x_id, w_v_id, v_id);

        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        prefill_graph.add_scale(q_id, q_scaled_id, scale);
        prefill_graph.add_matmul(q_scaled_id, k_id, scores_id, true);
        prefill_graph.add_softmax(scores_id, probs_id);

        prefill_graph.add_matmul(probs_id, v_id, attn_out_id);
        prefill_graph.add_matmul(attn_out_id, w_o_id, proj_id);
        prefill_graph.add_add(x_id, proj_id, out_id);
        prefill_graph.add_matmul(out_id, w_logits_id, logits_id);

        // Bind data
        prefill_graph.bind_data(x_id, prefill_embeddings.data());
        prefill_graph.bind_data(w_q_id, w_q_data.data());
        prefill_graph.bind_data(w_k_id, w_k_data.data());
        prefill_graph.bind_data(w_v_id, w_v_data.data());
        prefill_graph.bind_data(w_o_id, w_o_data.data());
        prefill_graph.bind_data(w_logits_id, w_logits_data.data());

        // Compile & Execute
        activation_arena.reset();
        prefill_graph.compile(activation_arena);
        prefill_graph.execute(BackendType::CPU);

        // Populate KV cache with prefilled keys/values
        kv_cache.append_kv(prefill_graph.get_tensor(k_id), prefill_graph.get_tensor(v_id));
        std::cout << "[RUN] Prefill completed. Cached " << kv_cache.current_len() << " tokens." << std::endl;
    }

    // B. DECODING LOOP: Generate tokens auto-regressively
    std::cout << "\n[RUN] Starting Decoding Phase..." << std::endl;

    std::vector<int32_t> generated_tokens;
    std::vector<float> last_token_embedding(dim);
    int32_t last_token_id = prompt.back();

    // Configure generation settings
    float temperature = 0.7f;
    size_t tokens_to_generate = 4; // Generate 4 words

    for (size_t step = 0; step < tokens_to_generate; ++step) {
        // Embed the last token
        std::memcpy(last_token_embedding.data(), w_embed.data() + last_token_id * dim, dim * sizeof(float));

        size_t current_len = kv_cache.current_len();

        // Build active cache tensors
        Tensor k_active_view = kv_cache.get_active_k();
        Tensor v_active_view = kv_cache.get_active_v();

        Graph decode_graph;

        // Inputs
        size_t x_step_id = decode_graph.add_input("x_step", {1, dim}, DType::Float32);
        size_t k_cache_id = decode_graph.add_input("k_cache", {current_len, dim}, DType::Float32);
        size_t v_cache_id = decode_graph.add_input("v_cache", {current_len, dim}, DType::Float32);

        // Parameters
        size_t w_q_id = decode_graph.add_parameter("w_q", {dim, dim}, DType::Float32);
        size_t w_k_id = decode_graph.add_parameter("w_k", {dim, dim}, DType::Float32);
        size_t w_v_id = decode_graph.add_parameter("w_v", {dim, dim}, DType::Float32);
        size_t w_o_id = decode_graph.add_parameter("w_o", {dim, dim}, DType::Float32);
        size_t w_logits_id = decode_graph.add_parameter("w_logits", {dim, vocab_size}, DType::Float32);

        // Projections
        size_t q_id = decode_graph.add_intermediate({1, dim}, DType::Float32);
        size_t k_new_id = decode_graph.add_intermediate({1, dim}, DType::Float32);
        size_t v_new_id = decode_graph.add_intermediate({1, dim}, DType::Float32);

        // Persistent KV Cache update nodes (concatenated cache)
        size_t k_updated_id = decode_graph.add_intermediate({current_len + 1, dim}, DType::Float32);
        size_t v_updated_id = decode_graph.add_intermediate({current_len + 1, dim}, DType::Float32);

        // Attention execution
        size_t q_scaled_id = decode_graph.add_intermediate({1, dim}, DType::Float32);
        size_t scores_id = decode_graph.add_intermediate({1, current_len + 1}, DType::Float32);
        size_t probs_id = decode_graph.add_intermediate({1, current_len + 1}, DType::Float32);
        size_t attn_out_id = decode_graph.add_intermediate({1, dim}, DType::Float32);

        size_t proj_id = decode_graph.add_intermediate({1, dim}, DType::Float32);
        size_t out_id = decode_graph.add_intermediate({1, dim}, DType::Float32);
        size_t logits_id = decode_graph.add_intermediate({1, vocab_size}, DType::Float32);

        // Schedules
        decode_graph.add_matmul(x_step_id, w_q_id, q_id);
        decode_graph.add_matmul(x_step_id, w_k_id, k_new_id);
        decode_graph.add_matmul(x_step_id, w_v_id, v_new_id);

        // Concat cached K and V with the new step's key and value
        decode_graph.add_concat2(k_cache_id, k_new_id, k_updated_id);
        decode_graph.add_concat2(v_cache_id, v_new_id, v_updated_id);

        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        decode_graph.add_scale(q_id, q_scaled_id, scale);
        // Matmul against updated full KV Cache
        decode_graph.add_matmul(q_scaled_id, k_updated_id, scores_id, true);
        decode_graph.add_softmax(scores_id, probs_id);

        decode_graph.add_matmul(probs_id, v_updated_id, attn_out_id);
        decode_graph.add_matmul(attn_out_id, w_o_id, proj_id);
        decode_graph.add_add(x_step_id, proj_id, out_id);
        decode_graph.add_matmul(out_id, w_logits_id, logits_id);

        // Bind data (including active views from our cache manager)
        decode_graph.bind_data(x_step_id, last_token_embedding.data());
        decode_graph.bind_data(k_cache_id, k_active_view.data());
        decode_graph.bind_data(v_cache_id, v_active_view.data());
        decode_graph.bind_data(w_q_id, w_q_data.data());
        decode_graph.bind_data(w_k_id, w_k_data.data());
        decode_graph.bind_data(w_v_id, w_v_data.data());
        decode_graph.bind_data(w_o_id, w_o_data.data());
        decode_graph.bind_data(w_logits_id, w_logits_data.data());

        // Compile and run (using tiled matmul in GPU / CPU)
        activation_arena.reset();
        decode_graph.compile(activation_arena);
        
        // Execute on CPU for sampling
        decode_graph.execute(BackendType::CPU);

        // Extract generated logits and sample
        float* logits_ptr = decode_graph.get_tensor(logits_id).data_as<float>();
        int32_t next_token = sample_logits(logits_ptr, vocab_size, temperature);

        generated_tokens.push_back(next_token);
        last_token_id = next_token;

        // Append generated key and value to persistent KV cache
        kv_cache.append_kv(decode_graph.get_tensor(k_new_id), decode_graph.get_tensor(v_new_id));

        std::cout << "[GEN] Step " << step << ": Token ID = " << next_token 
                  << ", Word = \"" << vocab[next_token] << "\"" << std::endl;
    }

    // Print generated phrase
    std::cout << "==========================================================" << std::endl;
    std::cout << "[OUTPUT] Generated Phrase:" << std::endl;
    std::cout << "  ";
    for (int32_t t : prompt) std::cout << vocab[t];
    for (int32_t t : generated_tokens) std::cout << vocab[t];
    std::cout << std::endl;
    std::cout << "==========================================================" << std::endl;

    // GPU verification of one decode step
    std::cout << "[RUN] Verifying Tiled Metal GPU compilation..." << std::endl;
    bool has_gpu = MetalContext::instance().init();
    if (has_gpu) {
        // Build decode graph for verification
        size_t current_len = kv_cache.current_len() - 1; // back to previous step
        Tensor k_active_view = kv_cache.get_active_k();
        Tensor v_active_view = kv_cache.get_active_v();

        Graph verify_graph;
        size_t x_step_id = verify_graph.add_input("x_step", {1, dim}, DType::Float32);
        size_t k_cache_id = verify_graph.add_input("k_cache", {current_len, dim}, DType::Float32);
        size_t v_cache_id = verify_graph.add_input("v_cache", {current_len, dim}, DType::Float32);
        size_t w_q_id = verify_graph.add_parameter("w_q", {dim, dim}, DType::Float32);
        size_t w_k_id = verify_graph.add_parameter("w_k", {dim, dim}, DType::Float32);
        size_t w_v_id = verify_graph.add_parameter("w_v", {dim, dim}, DType::Float32);
        size_t w_o_id = verify_graph.add_parameter("w_o", {dim, dim}, DType::Float32);
        size_t w_logits_id = verify_graph.add_parameter("w_logits", {dim, vocab_size}, DType::Float32);

        size_t q_id = verify_graph.add_intermediate({1, dim}, DType::Float32);
        size_t k_new_id = verify_graph.add_intermediate({1, dim}, DType::Float32);
        size_t v_new_id = verify_graph.add_intermediate({1, dim}, DType::Float32);
        size_t k_updated_id = verify_graph.add_intermediate({current_len + 1, dim}, DType::Float32);
        size_t v_updated_id = verify_graph.add_intermediate({current_len + 1, dim}, DType::Float32);

        size_t q_scaled_id = verify_graph.add_intermediate({1, dim}, DType::Float32);
        size_t scores_id = verify_graph.add_intermediate({1, current_len + 1}, DType::Float32);
        size_t probs_id = verify_graph.add_intermediate({1, current_len + 1}, DType::Float32);
        size_t attn_out_id = verify_graph.add_intermediate({1, dim}, DType::Float32);
        size_t proj_id = verify_graph.add_intermediate({1, dim}, DType::Float32);
        size_t out_id = verify_graph.add_intermediate({1, dim}, DType::Float32);
        size_t logits_id = verify_graph.add_intermediate({1, vocab_size}, DType::Float32);

        verify_graph.add_matmul(x_step_id, w_q_id, q_id);
        verify_graph.add_matmul(x_step_id, w_k_id, k_new_id);
        verify_graph.add_matmul(x_step_id, w_v_id, v_new_id);
        verify_graph.add_concat2(k_cache_id, k_new_id, k_updated_id);
        verify_graph.add_concat2(v_cache_id, v_new_id, v_updated_id);

        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        verify_graph.add_scale(q_id, q_scaled_id, scale);
        verify_graph.add_matmul(q_scaled_id, k_updated_id, scores_id, true);
        verify_graph.add_softmax(scores_id, probs_id);
        verify_graph.add_matmul(probs_id, v_updated_id, attn_out_id);
        verify_graph.add_matmul(attn_out_id, w_o_id, proj_id);
        verify_graph.add_add(x_step_id, proj_id, out_id);
        verify_graph.add_matmul(out_id, w_logits_id, logits_id);

        // Bind data
        verify_graph.bind_data(x_step_id, last_token_embedding.data());
        verify_graph.bind_data(k_cache_id, k_active_view.data());
        verify_graph.bind_data(v_cache_id, v_active_view.data());
        verify_graph.bind_data(w_q_id, w_q_data.data());
        verify_graph.bind_data(w_k_id, w_k_data.data());
        verify_graph.bind_data(w_v_id, w_v_data.data());
        verify_graph.bind_data(w_o_id, w_o_data.data());
        verify_graph.bind_data(w_logits_id, w_logits_data.data());

        activation_arena.reset();
        verify_graph.compile(activation_arena);
        
        // Execute CPU
        verify_graph.execute(BackendType::CPU);
        std::vector<float> cpu_logits(vocab_size);
        std::memcpy(cpu_logits.data(), verify_graph.get_tensor(logits_id).data(), vocab_size * sizeof(float));

        // Execute Metal GPU (with optimized tiled GEMM)
        verify_graph.execute(BackendType::Metal);
        float* metal_logits = verify_graph.get_tensor(logits_id).data_as<float>();

        bool matched = true;
        float max_diff = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            float diff = std::abs(metal_logits[i] - cpu_logits[i]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-4f) matched = false;
        }

        if (matched) {
            std::cout << "[SUCCESS] Optimized Tiled Metal GPU output matches CPU perfectly! Max diff: " << max_diff << std::endl;
        } else {
            std::cout << "[FAILED] GPU output mismatch! Max diff: " << max_diff << std::endl;
            return 1;
        }
    }
    std::cout << "==========================================================" << std::endl;

    return 0;
}
