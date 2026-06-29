#include "engine/core/types.hpp"
#include "engine/core/logger.hpp"
#include "engine/core/tokenizer.hpp"
#include "engine/core/sampler.hpp"
#include "engine/core/gguf_loader.hpp"
#include "engine/memory/allocator.hpp"
#include "engine/memory/kv_cache.hpp"
#include "engine/tensor/tensor.hpp"
#include "engine/backend/cpu/kernels.hpp"
#include "engine/backend/cpu/matmul.hpp"
#include "engine/backend/cpu/quantized_matmul.hpp"
#include "engine/backend/metal/metal_backend.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <random>

using namespace core;
using namespace tensor;
using namespace memory;

// Execution routine for our 2-layer stacked model block
void run_inference(
    bool use_gpu,
    const Tensor& x_in,                          // Input shape [L, dim]
    const std::vector<const block_q4_0*>& w_q,    // Layer weights
    const std::vector<const block_q4_0*>& w_k,
    const std::vector<const block_q4_0*>& w_v,
    const std::vector<const block_q4_0*>& w_o,
    const std::vector<const block_q4_0*>& w_gate,
    const std::vector<const block_q4_0*>& w_up,
    const std::vector<const block_q4_0*>& w_down,
    const std::vector<Tensor>& gamma_attn,
    const std::vector<Tensor>& gamma_mlp,
    const Tensor& gamma_final,
    const Tensor& w_logits,                      // FP32 weight matrix [dim, vocab_size]
    std::vector<KVCacheManager>& layer_caches,
    Tensor& logits_out,                          // Output logits shape [L, vocab_size]
    ArenaAllocator& workspace_arena
) {
    size_t L = x_in.shape()[0];
    size_t dim = 64;
    size_t head_dim = 16;

    workspace_arena.reset();

    // Contiguous workspace allocator using non-owning stack descriptors
    auto alloc_tensor = [&](Shape shape) {
        float* ptr = static_cast<float*>(workspace_arena.allocate(shape.size() * sizeof(float)));
        return Tensor(shape, DType::Float32, ptr);
    };

    Tensor current_X = x_in;

    for (size_t l = 0; l < 2; ++l) {
        Tensor norm_1 = alloc_tensor({L, dim});
        Tensor q = alloc_tensor({L, dim});
        Tensor k = alloc_tensor({L, dim});
        Tensor v = alloc_tensor({L, dim});

        // 1. RMSNorm
        if (use_gpu) {
            backend::metal::MetalContext::instance().rmsnorm(current_X, gamma_attn[l], norm_1, 1e-5f);
        } else {
            backend::cpu::rmsnorm(current_X, gamma_attn[l], norm_1, 1e-5f);
        }

        // 2. Quantized Projections
        if (use_gpu) {
            backend::metal::MetalContext::instance().matmul_q4_0(norm_1, w_q[l], q);
            backend::metal::MetalContext::instance().matmul_q4_0(norm_1, w_k[l], k);
            backend::metal::MetalContext::instance().matmul_q4_0(norm_1, w_v[l], v);
        } else {
            backend::cpu::gemm_q4_0(norm_1, w_q[l], q);
            backend::cpu::gemm_q4_0(norm_1, w_k[l], k);
            backend::cpu::gemm_q4_0(norm_1, w_v[l], v);
        }

        // 3. Cache Keys and Values
        layer_caches[l].append_kv(k, v);

        // 4. Retrieve Active KV Context Views
        Tensor K_active = layer_caches[l].get_active_k();
        Tensor V_active = layer_caches[l].get_active_v();
        size_t seq_len = K_active.shape()[0];

        // 5. Causal Scaled Attention
        Tensor q_scaled = alloc_tensor({L, dim});
        float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));
        if (use_gpu) {
            backend::metal::MetalContext::instance().scale(q, q_scaled, scale_factor);
        } else {
            backend::cpu::scale(q, q_scaled, scale_factor);
        }

        Tensor scores = alloc_tensor({L, seq_len});
        // Create a transposed view of K_active
        Tensor K_active_T({dim, seq_len}, DType::Float32, K_active.data(), {1, seq_len});

        if (use_gpu) {
            backend::metal::MetalContext::instance().matmul(q_scaled, K_active_T, scores);
        } else {
            backend::cpu::matmul(q_scaled, K_active_T, scores);
        }

        Tensor probs = alloc_tensor({L, seq_len});
        if (use_gpu) {
            backend::metal::MetalContext::instance().softmax(scores, probs);
        } else {
            backend::cpu::softmax(scores, probs);
        }

        Tensor attn_out = alloc_tensor({L, dim});
        if (use_gpu) {
            backend::metal::MetalContext::instance().matmul(probs, V_active, attn_out);
        } else {
            backend::cpu::matmul(probs, V_active, attn_out);
        }

        // 6. Attention Out projection
        Tensor proj_o = alloc_tensor({L, dim});
        if (use_gpu) {
            backend::metal::MetalContext::instance().matmul_q4_0(attn_out, w_o[l], proj_o);
        } else {
            backend::cpu::gemm_q4_0(attn_out, w_o[l], proj_o);
        }

        // 7. Residual Add
        Tensor attn_res = alloc_tensor({L, dim});
        if (use_gpu) {
            backend::metal::MetalContext::instance().add(current_X, proj_o, attn_res);
        } else {
            backend::cpu::add(current_X, proj_o, attn_res);
        }

        // 8. MLP SwiGLU Block
        Tensor norm_2 = alloc_tensor({L, dim});
        if (use_gpu) {
            backend::metal::MetalContext::instance().rmsnorm(attn_res, gamma_mlp[l], norm_2, 1e-5f);
        } else {
            backend::cpu::rmsnorm(attn_res, gamma_mlp[l], norm_2, 1e-5f);
        }

        Tensor gate = alloc_tensor({L, dim});
        Tensor up = alloc_tensor({L, dim});
        if (use_gpu) {
            backend::metal::MetalContext::instance().matmul_q4_0(norm_2, w_gate[l], gate);
            backend::metal::MetalContext::instance().matmul_q4_0(norm_2, w_up[l], up);
        } else {
            backend::cpu::gemm_q4_0(norm_2, w_gate[l], gate);
            backend::cpu::gemm_q4_0(norm_2, w_up[l], up);
        }

        Tensor gate_silu = alloc_tensor({L, dim});
        if (use_gpu) {
            backend::metal::MetalContext::instance().silu(gate, gate_silu);
        } else {
            backend::cpu::silu(gate, gate_silu);
        }

        Tensor gate_mul = alloc_tensor({L, dim});
        if (use_gpu) {
            backend::metal::MetalContext::instance().mul(gate_silu, up, gate_mul);
        } else {
            backend::cpu::mul(gate_silu, up, gate_mul);
        }

        Tensor down = alloc_tensor({L, dim});
        if (use_gpu) {
            backend::metal::MetalContext::instance().matmul_q4_0(gate_mul, w_down[l], down);
        } else {
            backend::cpu::gemm_q4_0(gate_mul, w_down[l], down);
        }

        Tensor mlp_res = alloc_tensor({L, dim});
        if (use_gpu) {
            backend::metal::MetalContext::instance().add(attn_res, down, mlp_res);
        } else {
            backend::cpu::add(attn_res, down, mlp_res);
        }

        current_X = mlp_res;
    }

    // 9. Final Normalization
    Tensor norm_final = alloc_tensor({L, dim});
    if (use_gpu) {
        backend::metal::MetalContext::instance().rmsnorm(current_X, gamma_final, norm_final, 1e-5f);
    } else {
        backend::cpu::rmsnorm(current_X, gamma_final, norm_final, 1e-5f);
    }

    // 10. Output logit scores
    if (use_gpu) {
        backend::metal::MetalContext::instance().matmul(norm_final, w_logits, logits_out);
    } else {
        backend::cpu::matmul(norm_final, w_logits, logits_out);
    }
}

int main(int argc, char* argv[]) {
    std::cout << "==========================================================" << std::endl;
    std::cout << "DeepChat-2L Stacked Inference CLI Application (Phase 9)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    std::string script_path = "scripts/create_chat_gguf.py";
    FILE* file = std::fopen(script_path.c_str(), "r");
    if (file) {
        std::fclose(file);
    } else {
        script_path = "../scripts/create_chat_gguf.py";
    }

    std::string cmd = "python3 " + script_path + " chat_model.gguf";
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "[ERR] Failed to execute Python create_chat_gguf script (tried local and parent fallbacks)" << std::endl;
        return 1;
    }

    // 2. Load model metadata and mapped pointers using GGUFLoader
    std::cout << "[INIT] Memory-mapping chat_model.gguf file..." << std::endl;
    GGUFLoader loader("chat_model.gguf");
    if (!loader.load()) {
        std::cerr << "[ERR] GGUFLoader failed to parse file" << std::endl;
        std::remove("chat_model.gguf");
        return 1;
    }

    // 3. Initialize Tokenizer and load vocabulary directly from the GGUF file
    std::cout << "[INIT] Loading tokenizer vocab from GGUF metadata..." << std::endl;
    Tokenizer tokenizer;
    const auto& gguf_vocab = loader.vocabulary();
    for (size_t i = 0; i < gguf_vocab.size(); ++i) {
        tokenizer.add_token(gguf_vocab[i], i);
    }

    // Set BPE merge ranks for words compilation
    tokenizer.add_merge("r", "o", 0);
    tokenizer.add_merge("b", "o", 1);
    tokenizer.add_merge("bo", "t", 2);
    tokenizer.add_merge("ro", "bot", 3);
    tokenizer.add_merge(" ", "l", 4);
    tokenizer.add_merge(" l", "e", 5);
    tokenizer.add_merge(" le", "a", 6);
    tokenizer.add_merge(" lea", "r", 7);
    tokenizer.add_merge(" lear", "n", 8);
    tokenizer.add_merge(" learn", "s", 9);
    tokenizer.add_merge(" ", "t", 10);
    tokenizer.add_merge(" t", "o", 11);
    tokenizer.add_merge(" ", "c", 12);
    tokenizer.add_merge(" c", "o", 13);
    tokenizer.add_merge(" co", "d", 14);
    tokenizer.add_merge(" cod", "e", 15);
    // Extra merges for awesome
    tokenizer.add_merge(" ", "a", 16);
    tokenizer.add_merge(" a", "w", 17);
    tokenizer.add_merge(" aw", "e", 18);
    tokenizer.add_merge(" awe", "s", 19);
    tokenizer.add_merge(" awes", "o", 20);
    tokenizer.add_merge(" aweso", "m", 21);
    tokenizer.add_merge(" awesom", "e", 22);
    // Merges for is
    tokenizer.add_merge(" ", "i", 23);
    tokenizer.add_merge(" i", "s", 24);

    // 4. Bind parameters mapped pointers from GGUF Loader
    size_t dim = 64;
    size_t vocab_size = 16;
    size_t max_seq_len = 16;

    std::vector<const block_q4_0*> w_q(2);
    std::vector<const block_q4_0*> w_k(2);
    std::vector<const block_q4_0*> w_v(2);
    std::vector<const block_q4_0*> w_o(2);
    std::vector<const block_q4_0*> w_gate(2);
    std::vector<const block_q4_0*> w_up(2);
    std::vector<const block_q4_0*> w_down(2);
    std::vector<Tensor> gamma_attn(2);
    std::vector<Tensor> gamma_mlp(2);

    auto bind_q4 = [&](const std::string& name) {
        return static_cast<const block_q4_0*>(loader.get_tensor_data(*loader.get_tensor_info(name)));
    };

    auto bind_fp32 = [&](const std::string& name, Shape shape) {
        float* ptr = const_cast<float*>(static_cast<const float*>(loader.get_tensor_data(*loader.get_tensor_info(name))));
        return Tensor(shape, DType::Float32, ptr);
    };

    for (size_t l = 0; l < 2; ++l) {
        std::string suffix = "_" + std::to_string(l);
        w_q[l] = bind_q4("w_q" + suffix);
        w_k[l] = bind_q4("w_k" + suffix);
        w_v[l] = bind_q4("w_v" + suffix);
        w_o[l] = bind_q4("w_o" + suffix);
        w_gate[l] = bind_q4("w_gate" + suffix);
        w_up[l] = bind_q4("w_up" + suffix);
        w_down[l] = bind_q4("w_down" + suffix);
        gamma_attn[l] = bind_fp32("gamma_attn" + suffix, {dim});
        gamma_mlp[l] = bind_fp32("gamma_mlp" + suffix, {dim});
    }

    Tensor gamma_final = bind_fp32("gamma_final", {dim});
    Tensor w_logits = bind_fp32("w_logits", {dim, vocab_size});

    // 5. Initialize device contexts & select GPU/CPU
    bool use_gpu = backend::metal::MetalContext::instance().init();
    if (use_gpu) {
        std::cout << "[INIT] Metal GPU backend active." << std::endl;
    } else {
        std::cout << "[INIT] CPU backend active (fallback)." << std::endl;
    }

    // Embeddings vocabulary weights
    std::vector<float> w_embed(vocab_size * dim);
    for (size_t i = 0; i < vocab_size; ++i) {
        std::mt19937 gen(i * 100);
        std::uniform_real_distribution<float> dis(-0.2f, 0.2f);
        for (size_t d = 0; d < dim; ++d) {
            w_embed[i * dim + d] = dis(gen);
        }
    }

    // Allocate workspaces & caches
    ArenaAllocator workspace_arena(1024 * 1024); // 1MB workspace
    std::vector<KVCacheManager> layer_caches;
    layer_caches.push_back(KVCacheManager(max_seq_len, dim));
    layer_caches.push_back(KVCacheManager(max_seq_len, dim));

    // Check if running in automated test mode
    std::string automated_prompt = "";
    if (argc >= 3 && std::string(argv[1]) == "--prompt") {
        automated_prompt = argv[2];
    }

    if (!automated_prompt.empty()) {
        std::cout << "\n[RUN] Running in Automated verification mode..." << std::endl;
        std::cout << "User > " << automated_prompt << std::endl;
        
        std::vector<int32_t> prompt_tokens = tokenizer.encode(automated_prompt);
        size_t L_prompt = prompt_tokens.size();

        std::vector<float> prompt_embeddings(L_prompt * dim);
        for (size_t i = 0; i < L_prompt; ++i) {
            std::memcpy(prompt_embeddings.data() + i * dim, w_embed.data() + prompt_tokens[i] * dim, dim * sizeof(float));
        }

        Tensor x_prefill({L_prompt, dim}, DType::Float32, prompt_embeddings.data());
        std::vector<float> data_logits(L_prompt * vocab_size);
        Tensor prefill_logits({L_prompt, vocab_size}, DType::Float32, data_logits.data());

        // Prefill Phase
        run_inference(use_gpu, x_prefill, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, gamma_attn, gamma_mlp, gamma_final, w_logits, layer_caches, prefill_logits, workspace_arena);

        // Decode Loop
        int32_t last_token = prompt_tokens.back();
        std::cout << "Assistant >";
        for (size_t step = 0; step < 4; ++step) {
            std::vector<float> x_step_data(dim);
            std::memcpy(x_step_data.data(), w_embed.data() + last_token * dim, dim * sizeof(float));
            Tensor x_step({1, dim}, DType::Float32, x_step_data.data());

            std::vector<float> step_logits_data(vocab_size);
            Tensor step_logits({1, vocab_size}, DType::Float32, step_logits_data.data());

            run_inference(use_gpu, x_step, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, gamma_attn, gamma_mlp, gamma_final, w_logits, layer_caches, step_logits, workspace_arena);

            int32_t next_token = sample_logits(step_logits_data.data(), vocab_size, 0.7f);
            last_token = next_token;
            std::cout << tokenizer.decode({next_token}) << std::flush;
        }
        std::cout << std::endl;
        
        std::remove("chat_model.gguf");
        std::cout << "\n==========================================================" << std::endl;
        std::cout << "[SUCCESS] Automated CLI verification completed successfully!" << std::endl;
        std::cout << "==========================================================" << std::endl;
        return 0;
    }

    // 6. Interactive Chat Loop
    std::cout << "\n[SUCCESS] Model parsed and loaded successfully!" << std::endl;
    std::cout << "[INFO] Type your prompt to start chatting. Type 'exit' to terminate.\n" << std::endl;

    while (true) {
        std::cout << "\nUser > ";
        std::string input;
        std::getline(std::cin, input);
        if (input == "exit" || input.empty()) {
            break;
        }

        std::vector<int32_t> prompt_tokens = tokenizer.encode(input);
        if (prompt_tokens.empty()) {
            continue;
        }
        size_t L_prompt = prompt_tokens.size();

        std::vector<float> prompt_embeddings(L_prompt * dim);
        for (size_t i = 0; i < L_prompt; ++i) {
            std::memcpy(prompt_embeddings.data() + i * dim, w_embed.data() + prompt_tokens[i] * dim, dim * sizeof(float));
        }

        Tensor x_prefill({L_prompt, dim}, DType::Float32, prompt_embeddings.data());
        std::vector<float> data_logits(L_prompt * vocab_size);
        Tensor prefill_logits({L_prompt, vocab_size}, DType::Float32, data_logits.data());

        // A. Prefill contextual prompt
        run_inference(use_gpu, x_prefill, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, gamma_attn, gamma_mlp, gamma_final, w_logits, layer_caches, prefill_logits, workspace_arena);

        // B. Dynamic decode loops with real-time character streaming
        std::cout << "Assistant >";
        int32_t last_token = prompt_tokens.back();

        for (size_t step = 0; step < 5; ++step) {
            std::vector<float> x_step_data(dim);
            std::memcpy(x_step_data.data(), w_embed.data() + last_token * dim, dim * sizeof(float));
            Tensor x_step({1, dim}, DType::Float32, x_step_data.data());

            std::vector<float> step_logits_data(vocab_size);
            Tensor step_logits({1, vocab_size}, DType::Float32, step_logits_data.data());

            run_inference(use_gpu, x_step, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, gamma_attn, gamma_mlp, gamma_final, w_logits, layer_caches, step_logits, workspace_arena);

            // Sample token
            int32_t next_token = sample_logits(step_logits_data.data(), vocab_size, 0.7f);
            last_token = next_token;

            // Stream word to stdout
            std::cout << tokenizer.decode({next_token}) << std::flush;
        }
        std::cout << std::endl;
    }

    std::remove("chat_model.gguf");
    std::cout << "\n==========================================================" << std::endl;
    std::cout << "DeepChat CLI session terminated. Cleaned up." << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}
