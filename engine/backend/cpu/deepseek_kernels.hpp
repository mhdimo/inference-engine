#pragma once
#include "../../tensor/tensor.hpp"

namespace backend::cpu {

// Compresses the input tensor along the sequence dimension (dim 0).
// input shape: [seq_len, dim]
// output shape: [seq_len / factor, dim]
// factor: compression window size (e.g. 4 for CSA)
// Computes the average of each group.
void seq_compress(const tensor::Tensor& input, tensor::Tensor& output, size_t factor);

// Computes the dot product of query against keys, sorts them, and writes the Top-K indices.
// query: shape [1, dim]
// keys: shape [num_groups, dim]
// output_indices: shape [k] (DType::Int32)
// k: number of top indices to select
void topk_route(const tensor::Tensor& query, const tensor::Tensor& keys, tensor::Tensor& output_indices, size_t k);

// Gathers vectors from src according to indices, writing them into dst.
// src: shape [num_groups, dim]
// indices: shape [k] (DType::Int32)
// dst: shape [k, dim]
void route_gather(const tensor::Tensor& src, const tensor::Tensor& indices, tensor::Tensor& dst);

// Concatenates two 2D tensors A and B along the sequence dimension (dim 0).
// A: shape [M_A, dim]
// B: shape [M_B, dim]
// C: shape [M_A + M_B, dim] (output)
void concat2(const tensor::Tensor& A, const tensor::Tensor& B, tensor::Tensor& C);

} // namespace backend::cpu
