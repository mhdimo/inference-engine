#include "engine/core/types.hpp"
#include "engine/core/logger.hpp"
#include "engine/memory/allocator.hpp"
#include "engine/tensor/tensor.hpp"
#include "engine/backend/cpu/matmul.hpp"
#include "engine/backend/cpu/kernels.hpp"
#include "engine/backend/metal/metal_backend.hpp"
#include "engine/graph/graph.hpp"

#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#define RUN_TEST(test_func) \
    do { \
        std::cout << "[RUNNING] " << #test_func << "..." << std::endl; \
        try { \
            test_func(); \
            std::cout << "[SUCCESS] " << #test_func << std::endl; \
        } catch (const std::exception& e) { \
            std::cerr << "[FAILED] " << #test_func << ": " << e.what() << std::endl; \
            std::exit(1); \
        } \
    } while (0)

// Check if two floats are close
inline bool float_close(float a, float b, float tol = 1e-4f) {
    return std::abs(a - b) < tol;
}

// ----------------------------------------------------
// Original Phase 1 Tests
// ----------------------------------------------------

void test_shape_and_strides() {
    using namespace core;
    
    Shape s1{2, 3, 4};
    assert(s1.ndim == 3);
    assert(s1[0] == 2);
    assert(s1[1] == 3);
    assert(s1[2] == 4);
    assert(s1.size() == 24);

    Shape strides = compute_contiguous_strides(s1);
    assert(strides.ndim == 3);
    assert(strides[0] == 12);
    assert(strides[1] == 4);
    assert(strides[2] == 1);

    Shape s2{2, 3, 4};
    assert(s1 == s2);

    Shape s3{2, 3, 5};
    assert(s1 != s3);
}

void test_arena_allocator() {
    using namespace memory;

    size_t capacity = 1024 * 1024; // 1MB
    ArenaAllocator allocator(capacity);

    assert(allocator.capacity_bytes() == capacity);
    assert(allocator.used_bytes() == 0);

    void* ptr1 = allocator.allocate(100, 64);
    size_t addr1 = reinterpret_cast<size_t>(ptr1);
    assert(addr1 % 64 == 0);
    assert(allocator.used_bytes() >= 100);

    void* ptr2 = allocator.allocate(50, 32);
    size_t addr2 = reinterpret_cast<size_t>(ptr2);
    assert(addr2 % 32 == 0);
    assert(addr2 >= addr1 + 100);

    allocator.reset();
    assert(allocator.used_bytes() == 0);

    void* ptr3 = allocator.allocate(100, 64);
    assert(ptr3 == ptr1);

    bool threw = false;
    try {
        allocator.allocate(capacity + 1);
    } catch (const std::bad_alloc&) {
        threw = true;
    }
    assert(threw);
}

void test_tensor_basics() {
    using namespace core;
    using namespace tensor;
    using namespace memory;

    ArenaAllocator allocator(1024);
    Shape shape{2, 3};
    
    float* raw_data = static_cast<float*>(allocator.allocate(shape.size() * sizeof(float)));
    for (size_t i = 0; i < shape.size(); ++i) {
        raw_data[i] = static_cast<float>(i);
    }

    Tensor t(shape, DType::Float32, raw_data);
    assert(t.shape() == shape);
    assert(t.dtype() == DType::Float32);
    assert(t.numel() == 6);
    assert(t.nbytes() == 24);
    assert(t.is_contiguous());

    assert(t.strides()[0] == 3);
    assert(t.strides()[1] == 1);

    const float* typed_data = t.data_as<float>();
    assert(typed_data == raw_data);
    assert(typed_data[4] == 4.0f);
}

void test_matmul_naive() {
    using namespace core;
    using namespace tensor;
    using namespace memory;
    using namespace backend::cpu;

    ArenaAllocator allocator(1024 * 1024);

    float* data_A = static_cast<float*>(allocator.allocate(6 * sizeof(float)));
    float* data_B = static_cast<float*>(allocator.allocate(6 * sizeof(float)));
    float* data_C = static_cast<float*>(allocator.allocate(4 * sizeof(float)));

    data_A[0] = 1.0f; data_A[1] = 2.0f; data_A[2] = 3.0f;
    data_A[3] = 4.0f; data_A[4] = 5.0f; data_A[5] = 6.0f;

    data_B[0] = 7.0f;  data_B[1] = 8.0f;
    data_B[2] = 9.0f;  data_B[3] = 10.0f;
    data_B[4] = 11.0f; data_B[5] = 12.0f;

    Tensor A({2, 3}, DType::Float32, data_A);
    Tensor B({3, 2}, DType::Float32, data_B);
    Tensor C({2, 2}, DType::Float32, data_C);

    matmul(A, B, C);

    float* res = C.data_as<float>();
    assert(float_close(res[0], 58.0f));
    assert(float_close(res[1], 64.0f));
    assert(float_close(res[2], 139.0f));
    assert(float_close(res[3], 154.0f));
}

void test_matmul_transposed() {
    using namespace core;
    using namespace tensor;
    using namespace memory;
    using namespace backend::cpu;

    ArenaAllocator allocator(1024 * 1024);

    float* data_A = static_cast<float*>(allocator.allocate(6 * sizeof(float)));
    float* data_B = static_cast<float*>(allocator.allocate(6 * sizeof(float)));
    float* data_C = static_cast<float*>(allocator.allocate(4 * sizeof(float)));

    data_A[0] = 1.0f; data_A[1] = 2.0f; data_A[2] = 3.0f;
    data_A[3] = 4.0f; data_A[4] = 5.0f; data_A[5] = 6.0f;

    data_B[0] = 7.0f; data_B[1] = 9.0f; data_B[2] = 11.0f;
    data_B[3] = 8.0f; data_B[4] = 10.0f; data_B[5] = 12.0f;

    Tensor A({2, 3}, DType::Float32, data_A);
    Tensor B_transposed({3, 2}, DType::Float32, data_B, Shape{1, 3});
    Tensor C({2, 2}, DType::Float32, data_C);

    assert(!B_transposed.is_contiguous());

    matmul(A, B_transposed, C);

    float* res = C.data_as<float>();
    assert(float_close(res[0], 58.0f));
    assert(float_close(res[1], 64.0f));
    assert(float_close(res[2], 139.0f));
    assert(float_close(res[3], 154.0f));
}

// ----------------------------------------------------
// Original Phase 2 Tests
// ----------------------------------------------------

void test_add_kernel() {
    using namespace core;
    using namespace tensor;
    using namespace memory;
    using namespace backend::cpu;

    ArenaAllocator allocator(1024);

    float* data_A = static_cast<float*>(allocator.allocate(4 * sizeof(float)));
    float* data_B = static_cast<float*>(allocator.allocate(4 * sizeof(float)));
    float* data_C = static_cast<float*>(allocator.allocate(4 * sizeof(float)));

    data_A[0] = 1.0f; data_A[1] = 2.0f; data_A[2] = 3.0f; data_A[3] = 4.0f;
    data_B[0] = 10.0f; data_B[1] = 20.0f; data_B[2] = 30.0f; data_B[3] = 40.0f;

    Tensor A({2, 2}, DType::Float32, data_A);
    Tensor B({2, 2}, DType::Float32, data_B);
    Tensor C({2, 2}, DType::Float32, data_C);

    add(A, B, C);

    float* res = C.data_as<float>();
    assert(float_close(res[0], 11.0f));
    assert(float_close(res[1], 22.0f));
    assert(float_close(res[2], 33.0f));
    assert(float_close(res[3], 44.0f));
}

void test_silu_kernel() {
    using namespace core;
    using namespace tensor;
    using namespace memory;
    using namespace backend::cpu;

    ArenaAllocator allocator(1024);

    float* data_in = static_cast<float*>(allocator.allocate(3 * sizeof(float)));
    float* data_out = static_cast<float*>(allocator.allocate(3 * sizeof(float)));

    data_in[0] = -1.0f;
    data_in[1] = 0.0f;
    data_in[2] = 1.0f;

    Tensor input({3}, DType::Float32, data_in);
    Tensor output({3}, DType::Float32, data_out);

    silu(input, output);

    float* res = output.data_as<float>();
    assert(float_close(res[0], -0.268941f));
    assert(float_close(res[1], 0.0f));
    assert(float_close(res[2], 0.731058f));
}

void test_softmax_kernel() {
    using namespace core;
    using namespace tensor;
    using namespace memory;
    using namespace backend::cpu;

    ArenaAllocator allocator(1024);

    float* data_in = static_cast<float*>(allocator.allocate(6 * sizeof(float)));
    float* data_out = static_cast<float*>(allocator.allocate(6 * sizeof(float)));

    data_in[0] = 1.0f; data_in[1] = 2.0f; data_in[2] = 3.0f;
    data_in[3] = 0.0f; data_in[4] = 0.0f; data_in[5] = 0.0f;

    Tensor input({2, 3}, DType::Float32, data_in);
    Tensor output({2, 3}, DType::Float32, data_out);

    softmax(input, output);

    float* res = output.data_as<float>();
    assert(float_close(res[0], 0.09003f));
    assert(float_close(res[1], 0.24472f));
    assert(float_close(res[2], 0.66524f));
    assert(float_close(res[3], 0.33333f));
    assert(float_close(res[4], 0.33333f));
    assert(float_close(res[5], 0.33333f));
}

void test_rmsnorm_kernel() {
    using namespace core;
    using namespace tensor;
    using namespace memory;
    using namespace backend::cpu;

    ArenaAllocator allocator(1024);

    float* data_in = static_cast<float*>(allocator.allocate(4 * sizeof(float)));
    float* data_gamma = static_cast<float*>(allocator.allocate(4 * sizeof(float)));
    float* data_out = static_cast<float*>(allocator.allocate(4 * sizeof(float)));

    data_in[0] = 1.0f; data_in[1] = 2.0f; data_in[2] = 3.0f; data_in[3] = 4.0f;
    data_gamma[0] = 1.0f; data_gamma[1] = 1.5f; data_gamma[2] = 2.0f; data_gamma[3] = 2.5f;

    Tensor input({1, 4}, DType::Float32, data_in);
    Tensor gamma({4}, DType::Float32, data_gamma);
    Tensor output({1, 4}, DType::Float32, data_out);

    rmsnorm(input, gamma, output, 1e-5f);

    float* res = output.data_as<float>();
    assert(float_close(res[0], 0.365148f));
    assert(float_close(res[1], 1.095445f));
    assert(float_close(res[2], 2.190890f));
    assert(float_close(res[3], 3.651483f));
}

void test_graph_topological_sort() {
    using namespace graph;
    using namespace core;

    Graph g;

    size_t in0 = g.add_input("in0", {1}, DType::Float32);
    size_t param0 = g.add_parameter("param0", {1}, DType::Float32);
    
    size_t inter0 = g.add_intermediate({1}, DType::Float32);
    size_t inter1 = g.add_intermediate({1}, DType::Float32);
    size_t inter2 = g.add_intermediate({1}, DType::Float32);

    g.add_add(inter1, param0, inter2);
    g.add_silu(in0, inter0);
    g.add_silu(inter0, inter1);

    memory::ArenaAllocator arena(1024 * 1024);
    g.compile(arena);

    float val_in0 = 1.0f;
    float val_param0 = 10.0f;
    g.bind_data(in0, &val_in0);
    g.bind_data(param0, &val_param0);

    g.execute();

    float* res = g.get_tensor(inter2).data_as<float>();
    assert(float_close(*res, 10.4935f));
}

void test_graph_memory_planning() {
    using namespace graph;
    using namespace core;

    Graph g;

    size_t in = g.add_input("in", {4}, DType::Float32);
    
    size_t inter0 = g.add_intermediate({4}, DType::Float32);
    size_t inter1 = g.add_intermediate({4}, DType::Float32);
    size_t inter2 = g.add_intermediate({4}, DType::Float32);

    g.add_silu(in, inter0);
    g.add_silu(inter0, inter1);
    g.add_silu(inter1, inter2);

    memory::ArenaAllocator arena(1024 * 1024);
    g.compile(arena);

    const auto& tensor0 = g.get_tensor(inter0);
    const auto& tensor2 = g.get_tensor(inter2);

    assert(tensor0.data() == tensor2.data()); // VERIFY MEMORY REUSE!
    
    const auto& tensor1 = g.get_tensor(inter1);
    assert(tensor1.data() != tensor0.data());
}

void test_graph_execution_e2e() {
    using namespace graph;
    using namespace core;
    using namespace tensor;

    Graph g;

    size_t x_id = g.add_input("x", {2, 3}, DType::Float32);
    size_t gamma_id = g.add_parameter("gamma", {3}, DType::Float32);
    size_t w_id = g.add_parameter("w", {3, 2}, DType::Float32);
    size_t residual_id = g.add_input("residual", {2, 2}, DType::Float32);

    size_t x_norm_id = g.add_intermediate({2, 3}, DType::Float32);
    size_t w_x_id = g.add_intermediate({2, 2}, DType::Float32);
    size_t act_id = g.add_intermediate({2, 2}, DType::Float32);
    size_t y_id = g.add_intermediate({2, 2}, DType::Float32);

    g.add_rmsnorm(x_id, gamma_id, x_norm_id, 1e-5f);
    g.add_matmul(x_norm_id, w_id, w_x_id);
    g.add_silu(w_x_id, act_id);
    g.add_add(act_id, residual_id, y_id);

    memory::ArenaAllocator arena(1024 * 1024);
    g.compile(arena);

    std::vector<float> x_data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    std::vector<float> gamma_data = {1.0f, 1.2f, 1.5f};
    std::vector<float> w_data = {
        0.5f, -0.2f,
        0.1f, 0.8f,
        -0.4f, 0.3f
    };
    std::vector<float> residual_data = {
        1.0f, 2.0f,
        -1.0f, 0.5f
    };

    g.bind_data(x_id, x_data.data());
    g.bind_data(gamma_id, gamma_data.data());
    g.bind_data(w_id, w_data.data());
    g.bind_data(residual_id, residual_data.data());

    g.execute();

    float* res = g.get_tensor(y_id).data_as<float>();
    assert(float_close(res[0], 0.813673f, 1e-3f));
    assert(float_close(res[1], 3.144750f, 1e-3f));
    assert(float_close(res[2], -1.088983f, 1e-3f));
    assert(float_close(res[3], 1.544217f, 1e-3f));
}

// ----------------------------------------------------
// New Phase 3 Tests
// ----------------------------------------------------

void test_metal_backend_e2e() {
    using namespace graph;
    using namespace core;
    using namespace tensor;
    using namespace backend::metal;

    Graph g;

    size_t x_id = g.add_input("x", {2, 3}, DType::Float32);
    size_t gamma_id = g.add_parameter("gamma", {3}, DType::Float32);
    size_t w_id = g.add_parameter("w", {3, 2}, DType::Float32);
    size_t residual_id = g.add_input("residual", {2, 2}, DType::Float32);

    size_t x_norm_id = g.add_intermediate({2, 3}, DType::Float32);
    size_t w_x_id = g.add_intermediate({2, 2}, DType::Float32);
    size_t act_id = g.add_intermediate({2, 2}, DType::Float32);
    size_t y_id = g.add_intermediate({2, 2}, DType::Float32);

    g.add_rmsnorm(x_id, gamma_id, x_norm_id, 1e-5f);
    g.add_matmul(x_norm_id, w_id, w_x_id);
    g.add_silu(w_x_id, act_id);
    g.add_add(act_id, residual_id, y_id);

    memory::ArenaAllocator arena(1024 * 1024);
    g.compile(arena);

    // Initialize inputs
    std::vector<float> x_data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    std::vector<float> gamma_data = {1.0f, 1.2f, 1.5f};
    std::vector<float> w_data = {
        0.5f, -0.2f,
        0.1f, 0.8f,
        -0.4f, 0.3f
    };
    std::vector<float> residual_data = {
        1.0f, 2.0f,
        -1.0f, 0.5f
    };

    g.bind_data(x_id, x_data.data());
    g.bind_data(gamma_id, gamma_data.data());
    g.bind_data(w_id, w_data.data());
    g.bind_data(residual_id, residual_data.data());

    // 1. Execute on CPU first and capture output values
    g.execute(BackendType::CPU);
    
    std::vector<float> cpu_output(4);
    std::memcpy(cpu_output.data(), g.get_tensor(y_id).data(), 4 * sizeof(float));

    // Zero out output and activation tensors to verify that the GPU actually writes to them
    std::memset(g.get_tensor(y_id).data(), 0, 4 * sizeof(float));
    std::memset(g.get_tensor(x_norm_id).data(), 0, 6 * sizeof(float));
    std::memset(g.get_tensor(w_x_id).data(), 0, 4 * sizeof(float));
    std::memset(g.get_tensor(act_id).data(), 0, 4 * sizeof(float));

    // 2. Execute on Metal (GPU)
    g.execute(BackendType::Metal);

    // 3. Verify correctness (assert GPU output matches CPU output)
    float* metal_res = g.get_tensor(y_id).data_as<float>();
    for (size_t i = 0; i < 4; ++i) {
        assert(float_close(metal_res[i], cpu_output[i], 1e-4f));
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Starting Inference Engine Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    // Phase 1 Tests
    RUN_TEST(test_shape_and_strides);
    RUN_TEST(test_arena_allocator);
    RUN_TEST(test_tensor_basics);
    RUN_TEST(test_matmul_naive);
    RUN_TEST(test_matmul_transposed);

    // Phase 2 Tests
    RUN_TEST(test_add_kernel);
    RUN_TEST(test_silu_kernel);
    RUN_TEST(test_softmax_kernel);
    RUN_TEST(test_rmsnorm_kernel);
    RUN_TEST(test_graph_topological_sort);
    RUN_TEST(test_graph_memory_planning);
    RUN_TEST(test_graph_execution_e2e);

    // Phase 3 Tests
    if (backend::metal::MetalContext::instance().init()) {
        RUN_TEST(test_metal_backend_e2e);
    } else {
        std::cout << "[INFO] Skipping Metal backend tests (Metal GPU not available)" << std::endl;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "All Tests Completed Successfully!" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
