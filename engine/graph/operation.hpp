#pragma once

namespace graph {

enum class OpType {
    MatMul,
    Add,      // Element-wise addition (matching shapes)
    RMSNorm,  // Root Mean Square Normalization
    Silu,     // Sigmoid Linear Unit
    Softmax,  // Softmax along the last dimension
    Mul,      // Element-wise multiplication (matching shapes)
    Scale,    // Element-wise scalar multiplication (factor)
    SeqCompress, // DeepSeek-V4: sequence pooling compression along seq_len dim
    TopKRoute,   // DeepSeek-V4: topk indices routing via Q * K^T score calculation
    RouteGather, // DeepSeek-V4: gather routed vectors from compressed cache
    Concat2      // Concatenate two 2D tensors along sequence dimension (dim 0)
};

} // namespace graph
