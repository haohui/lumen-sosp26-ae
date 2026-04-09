#ifndef KERNEL_H_
#define KERNEL_H_

#include <cstddef>
#include <cstdint>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

hipError_t ksearch_launch_dense_qkv_prefill_causal_h8_kv1_d128_bs16_shared_kv_kernel(
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

#endif  // KERNEL_H_