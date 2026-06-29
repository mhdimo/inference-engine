# Inference Engine

> A from-scratch C++20 + Metal LLM inference engine for macOS — no ML frameworks, just the math.

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue)
![macOS / Metal](https://img.shields.io/badge/platform-macOS%20%2F%20Metal-silver)
![CMake](https://img.shields.io/badge/build-CMake%20%E2%89%A53.18-064F8C)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

This project is **not** a wrapper around PyTorch, ONNX Runtime, llama.cpp, ggml, or any other ML
library. It is a ground-up reimplementation of the pieces needed to run a real transformer — tensor
representation, a compute-graph executor with lifetime-based memory planning, an arena allocator, a
mmap'd GGUF loader, a GPT-2 byte-level BPE tokenizer, a Jinja2-subset chat-template renderer, and two
hand-written backends (scalar CPU and Metal).

It loads real GGUF models and produces coherent text. The default target is
**Qwen2.5-1.5B-Instruct (Q4_0)**, and it also runs **SmolLM2-135M-Instruct**.

The goal is to understand, in detail, how modern inference systems actually work.

---

## ✨ What works today

| Subsystem | Status |
|---|---|
| Tensor runtime (shape, strides, dtype, non-owning views) | ✅ |
| Arena allocator + activation reset / reuse | ✅ |
| Compute graph (Kahn topological sort + 1D lifetime memory planning) | ✅ |
| mmap GGUF v2/v3 loader (FP32, FP16, Q4_0, Q4_1, Q8_0, Q6_K) | ✅ |
| GPT-2 byte-level BPE tokenizer (merges, special tokens, byte fallback) | ✅ |
| Jinja2-subset chat-template engine (for/if/set, filters, trim markers) | ✅ |
| Greedy + temperature sampling | ✅ |
| Single-stream KV cache | ✅ |
| **CPU backend** — matmul, RoPE, GQA attention, RMSNorm, SwiGLU, softmax, Q4_0/Q8_0 GEMM | ✅ |
| **Metal backend** — zero-copy unified-memory kernels, 16×16 tiled matmul, 13 pipelines | ✅ |
| LLaMA / `qwen2` architecture end-to-end inference + chat | ✅ |
| Qwen3.5 hybrid SSM / Gated-DeltaNet path | 🧪 experimental |

Every graph example cross-checks CPU vs. Metal output to ≤1e-4 and prints `[SUCCESS]`.

---

## 🚀 Quick start

Requirements: **macOS** with a Metal-capable GPU, a C++20 compiler, CMake ≥ 3.18, and `python3`
(used by the model-download and mock-GGUF helper scripts).

```bash
# 1. Build
cmake -S . -B build && cmake --build build -j

# 2. Chat with Qwen2.5-1.5B-Instruct.
#    First run auto-downloads the ~1 GB Q4_0 weights via scripts/download_smollm2.py.
./build/smollm2_chat

# One-shot generation:
./build/smollm2_chat --prompt "Explain attention in one sentence."

# Or point it at any supported gguf:
./build/smollm2_chat path/to/model.gguf --debug
```

`smollm2_chat` auto-detects the architecture from tensor names, selects Metal when available
(falling back to CPU when needed), and streams tokens as they decode.

---

## 🧪 Tests & examples

Built executables live in `build/`.

**Tests** (hand-rolled, no framework):

```bash
./build/test_tensor      # engine unit tests: shapes, allocator, kernels, graph sort + memplan + e2e, Metal
./build/test_gguf        # GGUF parse + CPU/Metal quantized GEMM (generates a mock gguf via python3)
./build/test_tokenizer   # BPE encode/decode round-trip + byte fallback
```

`tests/test_tokenizer_bpe.cpp` and `tests/test_chat_template.cpp` are standalone (each has a
one-line build command in its header comment) and include oracle checks against the real SmolLM2
weights and Qwen2.5/SmolLM2 ChatML templates.

**Examples:**

| Target | What it does |
|---|---|
| `smollm2_chat` | The real runner — chat with Qwen2.5-1.5B / SmolLM2 via BPE + Jinja template, CPU or Metal |
| `chat` | Interactive REPL on a tiny toy 2-layer model generated on the fly (`scripts/create_chat_gguf.py`) |
| `generate` | Autoregressive generation demo with KV cache + temperature sampling |
| `llama_block` | One LLaMA transformer block built as a `Graph`, CPU-vs-Metal correctness check |
| `quant_demo` | Quantize a weight matrix to Q4_0, compare FP32 vs CPU vs Metal GEMM (compression + RMSE) |
| `deepseek_v4_demo` | DeepSeek-V4 hybrid-attention routing concept (seq-compress, top-K route, gather) on random data |

---

## 📐 Architecture

```
        GGUF model (mmap)
              │
   Tokenizer (BPE) ── Chat template (Jinja subset)
              │
        Compute Graph   ── topological sort + lifetime memory planning
              │
      Backend dispatch  ── CPU kernels  /  Metal kernels (zero-copy, unified memory)
              │
      KV cache + sampler ── next token
              │
          detokenize → text
```

**Backends**

- **CPU** — correctness-first scalar kernels. No SIMD and no threading (intentional: readable
  reference implementations).
- **Metal** — compiles an inline MSL source string into 13 compute pipelines (matmul, 16×16 tiled
  matmul, Q4_0/Q8_0 matmul, add, add_bias, silu, softmax, rmsnorm, mul, scale, rope, gqa_attention),
  using `newBufferWithBytesNoCopy` for zero-copy on unified memory (hence the 16 KB page-aligned
  arena). Falls back gracefully if no GPU is present.

---

## 📁 Project structure

```
engine/
  core/         types, logger, quantization (Q4_0/Q8_0/Q6_K), sampler,
                gguf_loader, tokenizer (BPE), chat_template (Jinja subset)
  memory/       arena allocator, KV cache
  tensor/       non-owning tensor descriptor
  graph/        op graph, topological sort, lifetime memory planning
  backend/
    cpu/        matmul, kernels (RoPE, GQA, norms, activations), quantized matmul, deepseek kernels
    metal/      Metal backend (zero-copy, 13 MSL kernels)
  scheduler/    (placeholder — not yet implemented)
examples/       llama_block, deepseek_v4_demo, generate, quant_demo, chat, smollm2_chat
tests/          test_tensor, test_gguf, test_tokenizer (+ standalone BPE & chat-template tests)
scripts/        download_smollm2.py, create_chat_gguf.py, create_mock_gguf.py
benchmarks/     (placeholder)
docs/           design specs
```

Model weights (`.gguf`) are gitignored — download them via the helper scripts or supply your own.

---

## 🧱 Limitations & roadmap

Honest about what this is **not** yet:

- **Scalar CPU kernels** — no SIMD (no Accelerate/AMX) and no multithreading. Correct, not fast.
- **Single-stream KV cache** — no continuous batching, no paged attention, no prefix caching.
- **Metal GQA** uses a fixed-size thread-local buffer, so contexts longer than ~128 tokens fall back
  to CPU.
- **Quantization** — only Q4_0 can be *produced* in-engine; Q6_K and Q4_1 have dequantizers only.
- **Qwen3.5 hybrid SSM path** is experimental (hard-coded dims); the `qwen2`/LLaMA path is the
  supported one.
- **Stubs** — `engine/scheduler/`, the `cpu_backend` abstraction layer, and `benchmarks/` are empty
  placeholders.
- **Sampler** uses a fixed seed for reproducibility across runs.

Planned: SIMD/tiled CPU kernels, multi-threading, paged KV cache + batching, larger Metal attention
contexts, and a real scheduler.

---

## 📜 License

MIT — see [LICENSE](LICENSE). Built for learning; pull requests welcome.
