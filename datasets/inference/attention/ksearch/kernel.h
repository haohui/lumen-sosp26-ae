#ifndef KSEARCH_KERNEL_H_
#define KSEARCH_KERNEL_H_

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

hipError_t ksearch_launch_dense_qkv_prefill_causal_h8_kv1or8_d128(
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    hipStream_t stream,
    const hip_bfloat16* q,
    const hip_bfloat16* k,
    const hip_bfloat16* v,
    hip_bfloat16* out,
    int B,
    int S,
    int Hq,
    int Hkv,
    int D,
    float sm_scale);

#endif  // KSEARCH_KERNEL_H_