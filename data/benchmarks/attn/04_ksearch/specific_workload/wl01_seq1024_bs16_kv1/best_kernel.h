#ifndef KSEARCH_DENSE_ATTENTION_KERNEL_H_
#define KSEARCH_DENSE_ATTENTION_KERNEL_H_

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstddef>
#include <cstdint>

__global__ void dense_qkv_prefill_causal_h8_kv1_d128_bs16_single_pass_kernel(
    const hip_bfloat16* q,
    const hip_bfloat16* k,
    const hip_bfloat16* v,
    hip_bfloat16* out,
    int seq_len,
    float sm_scale);

hipError_t ksearch_launch_dense_qkv_prefill_causal_h8_kv1_d128_bs16_single_pass(
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

#endif  // KSEARCH_DENSE_ATTENTION_KERNEL_H_