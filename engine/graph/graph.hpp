#pragma once
#include "../core/types.hpp"
#include "../tensor/tensor.hpp"
#include "../memory/allocator.hpp"
#include "../backend/metal/metal_backend.hpp"
#include "operation.hpp"
#include <string>
#include <vector>

namespace graph {

class Graph {
public:
    Graph() = default;

    // Node registration
    size_t add_input(const std::string& name, core::Shape shape, core::DType dtype);
    size_t add_parameter(const std::string& name, core::Shape shape, core::DType dtype);
    size_t add_intermediate(core::Shape shape, core::DType dtype);

    // Operation scheduling
    void add_matmul(size_t input_a, size_t input_b, size_t output, bool transpose_b = false);
    void add_add(size_t input_a, size_t input_b, size_t output);
    void add_rmsnorm(size_t input, size_t gamma, size_t output, float eps = 1e-5f);
    void add_silu(size_t input, size_t output);
    void add_softmax(size_t input, size_t output);
    void add_mul(size_t input_a, size_t input_b, size_t output);
    void add_scale(size_t input, size_t output, float factor);
    void add_seq_compress(size_t input, size_t output, size_t factor);
    void add_topk_route(size_t query, size_t keys, size_t output_indices, size_t k);
    void add_route_gather(size_t src, size_t indices, size_t dst);
    void add_concat2(size_t input_a, size_t input_b, size_t output);

    // Compiles graph, performs topological sorting, plans intermediate memories, and allocates from arena
    void compile(memory::ArenaAllocator& activation_arena);

    // Bind raw pointers to Input/Parameter nodes
    void bind_data(size_t node_id, void* data_ptr);

    // Execute all operations in topological order using the specified backend
    void execute(backend::metal::BackendType backend = backend::metal::BackendType::CPU);

    // Access tensors
    tensor::Tensor& get_tensor(size_t node_id);
    const tensor::Tensor& get_tensor(size_t node_id) const;

private:
    std::vector<size_t> topological_sort() const;
    void plan_memory(memory::ArenaAllocator& activation_arena, const std::vector<size_t>& sorted_ops);

    enum class NodeType {
        Input,
        Parameter,
        Intermediate
    };

    struct NodeInfo {
        size_t id;
        std::string name;
        core::Shape shape;
        core::DType dtype;
        NodeType type;
        void* external_data = nullptr;
        tensor::Tensor tensor;

        // Lifetime parameters (measured in topological op index)
        int first_used = -1;
        int last_used = -1;
        size_t memory_offset = 0;
    };

    struct OpInfo {
        OpType type;
        std::vector<size_t> inputs;
        size_t output;
        float float_param = 0.0f; // eps for RMSNorm
        bool transpose_b = false;  // Whether operand B is transposed
    };

    std::vector<NodeInfo> nodes_;
    std::vector<OpInfo> ops_;
    std::vector<size_t> execution_order_;
    bool compiled_ = false;
};

} // namespace graph
