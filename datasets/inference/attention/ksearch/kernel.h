#ifndef DENSE_QKV_PREFILL_CAUSAL_H8_KV1OR8_D128_KERNEL_H_
#define DENSE_QKV_PREFILL_CAUSAL_H8_KV1OR8_D128_KERNEL_H_

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>

hipError_t ksearch_launch_dense_qkv_prefill_causal_h8_kv1or8_d128(
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    hipStream_t stream,
    const uint16_t* q,
    const uint16_t* k,
    const uint16_t* v,
    uint16_t* out,
    int seq_len,
    int num_kv_heads,
    float sm_scale);

#endif  // DENSE_QKV_PREFILL_CAUSAL_H8_KV1OR8_D128_KERNEL_H_