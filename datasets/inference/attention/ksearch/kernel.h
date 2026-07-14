#pragma once

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

hipError_t ksearch_launch_dense_qkv_prefill_causal_h8_kv1or8_d128(
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    hipStream_t stream,
    const hip_bfloat16* q,
    const hip_bfloat16* k,
    const hip_bfloat16* v,
    float sm_scale,
    int batch_size,
    int seq_len,
    int num_kv_heads,
    hip_bfloat16* out);