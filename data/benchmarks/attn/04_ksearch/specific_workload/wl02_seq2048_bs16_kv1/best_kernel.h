#ifndef DENSE_QKV_PREFILL_CAUSAL_H8_KV1_D128_BS16_KERNEL_H_
#define DENSE_QKV_PREFILL_CAUSAL_H8_KV1_D128_BS16_KERNEL_H_

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstddef>
#include <cstdint>

__global__ void online_row_streaming_causal_attention_kernel(
    const hip_bfloat16* __restrict__ q,
    const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v,
    hip_bfloat16* __restrict__ out,
    int seq_len,
    float sm_scale);

hipError_t ksearch_launch_online_row_streaming_causal_attention(
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    hipStream_t stream,
    const hip_bfloat16* q,
    const hip_bfloat16* k,
    const hip_bfloat16* v,
    hip_bfloat16* out,
    int seq_len,
    float sm_scale);

#endif  // DENSE_QKV_PREFILL_CAUSAL_H8_KV1_D128_BS16_KERNEL_H_