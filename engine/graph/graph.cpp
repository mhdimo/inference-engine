#include "graph.hpp"
#include "../core/logger.hpp"
#include "../backend/cpu/matmul.hpp"
#include "../backend/cpu/kernels.hpp"
#include "../backend/cpu/deepseek_kernels.hpp"
#include <algorithm>
#include <stdexcept>

namespace graph {

size_t Graph::add_input(const std::string& name, core::Shape shape, core::DType dtype) {
    size_t id = nodes_.size();
    NodeInfo info;
    info.id = id;
    info.name = name;
    info.shape = shape;
    info.dtype = dtype;
    info.type = NodeType::Input;
    nodes_.push_back(std::move(info));
    return id;
}

size_t Graph::add_parameter(const std::string& name, core::Shape shape, core::DType dtype) {
    size_t id = nodes_.size();
    NodeInfo info;
    info.id = id;
    info.name = name;
    info.shape = shape;
    info.dtype = dtype;
    info.type = NodeType::Parameter;
    nodes_.push_back(std::move(info));
    return id;
}

size_t Graph::add_intermediate(core::Shape shape, core::DType dtype) {
    size_t id = nodes_.size();
    NodeInfo info;
    info.id = id;
    info.name = "intermediate_" + std::to_string(id);
    info.shape = shape;
    info.dtype = dtype;
    info.type = NodeType::Intermediate;
    nodes_.push_back(std::move(info));
    return id;
}

void Graph::add_matmul(size_t input_a, size_t input_b, size_t output, bool transpose_b) {
    OpInfo op;
    op.type = OpType::MatMul;
    op.inputs = {input_a, input_b};
    op.output = output;
    op.transpose_b = transpose_b;
    ops_.push_back(std::move(op));
}

void Graph::add_add(size_t input_a, size_t input_b, size_t output) {
    OpInfo op;
    op.type = OpType::Add;
    op.inputs = {input_a, input_b};
    op.output = output;
    ops_.push_back(std::move(op));
}

void Graph::add_rmsnorm(size_t input, size_t gamma, size_t output, float eps) {
    OpInfo op;
    op.type = OpType::RMSNorm;
    op.inputs = {input, gamma};
    op.output = output;
    op.float_param = eps;
    ops_.push_back(std::move(op));
}

void Graph::add_silu(size_t input, size_t output) {
    OpInfo op;
    op.type = OpType::Silu;
    op.inputs = {input};
    op.output = output;
    ops_.push_back(std::move(op));
}

void Graph::add_softmax(size_t input, size_t output) {
    OpInfo op;
    op.type = OpType::Softmax;
    op.inputs = {input};
    op.output = output;
    ops_.push_back(std::move(op));
}

void Graph::add_mul(size_t input_a, size_t input_b, size_t output) {
    OpInfo op;
    op.type = OpType::Mul;
    op.inputs = {input_a, input_b};
    op.output = output;
    ops_.push_back(std::move(op));
}

void Graph::add_scale(size_t input, size_t output, float factor) {
    OpInfo op;
    op.type = OpType::Scale;
    op.inputs = {input};
    op.output = output;
    op.float_param = factor;
    ops_.push_back(std::move(op));
}

void Graph::add_seq_compress(size_t input, size_t output, size_t factor) {
    OpInfo op;
    op.type = OpType::SeqCompress;
    op.inputs = {input};
    op.output = output;
    op.float_param = static_cast<float>(factor); // Reuse float_param as factor
    ops_.push_back(std::move(op));
}

void Graph::add_topk_route(size_t query, size_t keys, size_t output_indices, size_t k) {
    OpInfo op;
    op.type = OpType::TopKRoute;
    op.inputs = {query, keys};
    op.output = output_indices;
    op.float_param = static_cast<float>(k); // Reuse float_param as k
    ops_.push_back(std::move(op));
}

void Graph::add_route_gather(size_t src, size_t indices, size_t dst) {
    OpInfo op;
    op.type = OpType::RouteGather;
    op.inputs = {src, indices};
    op.output = dst;
    ops_.push_back(std::move(op));
}

void Graph::add_concat2(size_t input_a, size_t input_b, size_t output) {
    OpInfo op;
    op.type = OpType::Concat2;
    op.inputs = {input_a, input_b};
    op.output = output;
    ops_.push_back(std::move(op));
}

std::vector<size_t> Graph::topological_sort() const {
    // 1. Determine creator operation for each node.
    std::vector<int> node_creator(nodes_.size(), -1);
    for (size_t i = 0; i < ops_.size(); ++i) {
        node_creator[ops_[i].output] = static_cast<int>(i);
    }

    // 2. Build dependency graph of operations.
    // Operation B depends on A if B has an input node created by A (A -> B).
    std::vector<std::vector<size_t>> adj(ops_.size());
    std::vector<size_t> in_degree(ops_.size(), 0);

    for (size_t b = 0; b < ops_.size(); ++b) {
        for (size_t input_id : ops_[b].inputs) {
            int creator = node_creator[input_id];
            if (creator != -1) {
                adj[creator].push_back(b);
                in_degree[b]++;
            }
        }
    }

    // 3. Kahn's topological sorting algorithm (non-recursive)
    std::vector<size_t> sorted_ops;
    std::vector<size_t> queue;
    for (size_t i = 0; i < ops_.size(); ++i) {
        if (in_degree[i] == 0) {
            queue.push_back(i);
        }
    }

    size_t head = 0;
    while (head < queue.size()) {
        size_t u = queue[head++];
        sorted_ops.push_back(u);
        for (size_t v : adj[u]) {
            in_degree[v]--;
            if (in_degree[v] == 0) {
                queue.push_back(v);
            }
        }
    }

    if (sorted_ops.size() != ops_.size()) {
        LOG_ERR("Dependency cycle detected in graph operations!");
        throw std::runtime_error("Dependency cycle detected in graph operations");
    }

    return sorted_ops;
}

void Graph::plan_memory(memory::ArenaAllocator& activation_arena, const std::vector<size_t>& sorted_ops) {
    // Reset lifetime parameters
    for (auto& node : nodes_) {
        node.first_used = -1;
        node.last_used = -1;
    }

    // Determine birth and death indices of each intermediate node
    for (size_t step = 0; step < sorted_ops.size(); ++step) {
        size_t op_idx = sorted_ops[step];
        const auto& op = ops_[op_idx];

        // Creator step
        if (nodes_[op.output].type == NodeType::Intermediate) {
            if (nodes_[op.output].first_used == -1) {
                nodes_[op.output].first_used = static_cast<int>(step);
            }
            nodes_[op.output].last_used = std::max(nodes_[op.output].last_used, static_cast<int>(step));
        }

        // Consumer step
        for (size_t input_id : op.inputs) {
            if (nodes_[input_id].type == NodeType::Intermediate) {
                if (nodes_[input_id].first_used == -1) {
                    LOG_ERR("Intermediate node used before created: " + nodes_[input_id].name);
                    throw std::runtime_error("Intermediate node used before created");
                }
                nodes_[input_id].last_used = std::max(nodes_[input_id].last_used, static_cast<int>(step));
            }
        }
    }

    // Greedy 1D packing of intermediate buffers
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].type != NodeType::Intermediate) {
            continue;
        }

        size_t size_i = nodes_[i].shape.size() * core::dtype_size(nodes_[i].dtype);
        // Round up size to multiple of 64 bytes to preserve SIMD alignment
        size_t size_i_rounded = (size_i + 63) & ~63;

        size_t Y = 0;
        while (true) {
            bool collision = false;
            for (size_t j = 0; j < i; ++j) {
                if (nodes_[j].type != NodeType::Intermediate) {
                    continue;
                }

                // Check lifetime overlap
                bool overlap_lifetime = !(nodes_[i].last_used < nodes_[j].first_used || 
                                          nodes_[j].last_used < nodes_[i].first_used);
                if (!overlap_lifetime) {
                    continue;
                }

                size_t size_j = nodes_[j].shape.size() * core::dtype_size(nodes_[j].dtype);
                size_t size_j_rounded = (size_j + 63) & ~63;

                // Check memory interval overlap
                bool overlap_memory = !(Y + size_i_rounded <= nodes_[j].memory_offset || 
                                        nodes_[j].memory_offset + size_j_rounded <= Y);
                if (overlap_memory) {
                    collision = true;
                    break;
                }
            }

            if (!collision) {
                break;
            }
            Y += 64; // Increment by 64 bytes to align next candidate offset
        }

        nodes_[i].memory_offset = Y;
    }

    // Find the total memory footprint needed
    size_t total_memory_required = 0;
    for (const auto& node : nodes_) {
        if (node.type == NodeType::Intermediate) {
            size_t size_i = node.shape.size() * core::dtype_size(node.dtype);
            size_t size_i_rounded = (size_i + 63) & ~63;
            total_memory_required = std::max(total_memory_required, node.memory_offset + size_i_rounded);
        }
    }

    // Allocate the bulk activation buffer from the arena
    void* base_ptr = nullptr;
    if (total_memory_required > 0) {
        // Request 16KB aligned block from the arena
        base_ptr = activation_arena.allocate(total_memory_required, 16384);
    }

    // Bind views of the shared memory buffer to the intermediate node tensors
    for (auto& node : nodes_) {
        if (node.type == NodeType::Intermediate) {
            void* node_data = static_cast<uint8_t*>(base_ptr) + node.memory_offset;
            node.tensor = tensor::Tensor(node.shape, node.dtype, node_data);
        }
    }
}

void Graph::compile(memory::ArenaAllocator& activation_arena) {
    execution_order_ = topological_sort();
    plan_memory(activation_arena, execution_order_);
    compiled_ = true;
}

void Graph::bind_data(size_t node_id, void* data_ptr) {
    if (node_id >= nodes_.size()) {
        LOG_ERR("Invalid node id: " + std::to_string(node_id));
        throw std::invalid_argument("Invalid node id");
    }
    if (nodes_[node_id].type == NodeType::Intermediate) {
        LOG_ERR("Cannot bind external data to intermediate node: " + nodes_[node_id].name);
        throw std::invalid_argument("Cannot bind external data to intermediate node");
    }
    nodes_[node_id].external_data = data_ptr;
    nodes_[node_id].tensor = tensor::Tensor(nodes_[node_id].shape, nodes_[node_id].dtype, data_ptr);
}

void Graph::execute(backend::metal::BackendType backend) {
    if (!compiled_) {
        LOG_ERR("Graph must be compiled before execution");
        throw std::runtime_error("Graph must be compiled before execution");
    }

    // Validate that inputs and params are fully bound
    for (const auto& node : nodes_) {
        if (node.type != NodeType::Intermediate && node.external_data == nullptr) {
            LOG_ERR("Unbound input/parameter node before execution: " + node.name);
            throw std::runtime_error("Unbound input/parameter node before execution: " + node.name);
        }
    }

    // Dispatch operations sequentially in scheduled topological order
    for (size_t op_idx : execution_order_) {
        const auto& op = ops_[op_idx];
        
        std::vector<const tensor::Tensor*> inputs;
        inputs.reserve(op.inputs.size());
        for (size_t input_id : op.inputs) {
            inputs.push_back(&nodes_[input_id].tensor);
        }
        tensor::Tensor& output = nodes_[op.output].tensor;

        if (backend == backend::metal::BackendType::CPU) {
            switch (op.type) {
                case OpType::MatMul: {
                    if (op.transpose_b) {
                        const tensor::Tensor& B = *inputs[1];
                        core::Shape shape_trans = {B.shape()[1], B.shape()[0]};
                        core::Shape strides_trans = {B.strides()[1], B.strides()[0]};
                        tensor::Tensor B_trans(shape_trans, B.dtype(), const_cast<void*>(B.data()), strides_trans);
                        backend::cpu::matmul(*inputs[0], B_trans, output);
                    } else {
                        backend::cpu::matmul(*inputs[0], *inputs[1], output);
                    }
                    break;
                }
                case OpType::Add:
                    backend::cpu::add(*inputs[0], *inputs[1], output);
                    break;
                case OpType::RMSNorm:
                    backend::cpu::rmsnorm(*inputs[0], *inputs[1], output, op.float_param);
                    break;
                case OpType::Silu:
                    backend::cpu::silu(*inputs[0], output);
                    break;
                case OpType::Softmax:
                    backend::cpu::softmax(*inputs[0], output);
                    break;
                case OpType::Mul:
                    backend::cpu::mul(*inputs[0], *inputs[1], output);
                    break;
                case OpType::Scale:
                    backend::cpu::scale(*inputs[0], output, op.float_param);
                    break;
                case OpType::SeqCompress:
                    backend::cpu::seq_compress(*inputs[0], output, static_cast<size_t>(op.float_param));
                    break;
                case OpType::TopKRoute:
                    backend::cpu::topk_route(*inputs[0], *inputs[1], output, static_cast<size_t>(op.float_param));
                    break;
                case OpType::RouteGather:
                    backend::cpu::route_gather(*inputs[0], *inputs[1], output);
                    break;
                case OpType::Concat2:
                    backend::cpu::concat2(*inputs[0], *inputs[1], output);
                    break;
            }
        } else if (backend == backend::metal::BackendType::Metal) {
            auto& ctx = backend::metal::MetalContext::instance();
            if (!ctx.is_available()) {
                throw std::runtime_error("Metal execution requested but Metal backend is not available");
            }
            switch (op.type) {
                case OpType::MatMul: {
                    if (op.transpose_b) {
                        const tensor::Tensor& B = *inputs[1];
                        core::Shape shape_trans = {B.shape()[1], B.shape()[0]};
                        core::Shape strides_trans = {B.strides()[1], B.strides()[0]};
                        tensor::Tensor B_trans(shape_trans, B.dtype(), const_cast<void*>(B.data()), strides_trans);
                        ctx.matmul(*inputs[0], B_trans, output);
                    } else {
                        ctx.matmul(*inputs[0], *inputs[1], output);
                    }
                    break;
                }
                case OpType::Add:
                    ctx.add(*inputs[0], *inputs[1], output);
                    break;
                case OpType::RMSNorm:
                    ctx.rmsnorm(*inputs[0], *inputs[1], output, op.float_param);
                    break;
                case OpType::Silu:
                    ctx.silu(*inputs[0], output);
                    break;
                case OpType::Softmax:
                    ctx.softmax(*inputs[0], output);
                    break;
                case OpType::Mul:
                    ctx.mul(*inputs[0], *inputs[1], output);
                    break;
                case OpType::Scale:
                    ctx.scale(*inputs[0], output, op.float_param);
                    break;
                case OpType::SeqCompress:
                    backend::cpu::seq_compress(*inputs[0], output, static_cast<size_t>(op.float_param));
                    break;
                case OpType::TopKRoute:
                    backend::cpu::topk_route(*inputs[0], *inputs[1], output, static_cast<size_t>(op.float_param));
                    break;
                case OpType::RouteGather:
                    backend::cpu::route_gather(*inputs[0], *inputs[1], output);
                    break;
                case OpType::Concat2:
                    backend::cpu::concat2(*inputs[0], *inputs[1], output);
                    break;
            }
        }
    }
}

tensor::Tensor& Graph::get_tensor(size_t node_id) {
    if (node_id >= nodes_.size()) {
        throw std::invalid_argument("Invalid node id");
    }
    return nodes_[node_id].tensor;
}

const tensor::Tensor& Graph::get_tensor(size_t node_id) const {
    if (node_id >= nodes_.size()) {
        throw std::invalid_argument("Invalid node id");
    }
    return nodes_[node_id].tensor;
}

} // namespace graph
