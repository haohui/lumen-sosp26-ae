#pragma once

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>

hipError_t ksearch_launch_build_expert_buckets(
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    hipStream_t stream,
    const int32_t* topk_ids,
    const float* topk_weights,
    int32_t* expert_counts,
    int32_t* expert_token_indices,
    float* expert_route_weights,
    int tokens,
    int topk,
    int num_experts,
    int max_tokens_per_expert);

hipError_t ksearch_launch_zero_f32(
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    hipStream_t stream,
    float* data,
    int64_t n);

hipError_t ksearch_launch_fused_expert_bucket_compute(
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    hipStream_t stream,
    const float* input_f,
    const float* w1_f,
    const float* w2_f,
    const int32_t* expert_counts,
    const int32_t* expert_token_indices,
    const float* expert_route_weights,
    float* output_f,
    int tokens,
    int max_tokens_per_expert,
    int num_experts);