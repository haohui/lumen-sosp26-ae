#ifndef KSEARCH_DENSE_QKV_PREFILL_CAUSAL_H8_KV1OR8_D128_KERNEL_H
#define KSEARCH_DENSE_QKV_PREFILL_CAUSAL_H8_KV1OR8_D128_KERNEL_H

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstddef>
#include <cstdint>

__global__ void dense_qkv_prefill_causal_h8_kv1or8_d128_online_kernel(
    const hip_bfloat16* q,
    const hip_bfloat16* k,
    const hip_bfloat16* v,
    hip_bfloat16* out,
    int B,
    int S,
    int HQ,
    int HK,
    float sm_scale);

__global__ void dense_qkv_prefill_causal_h8_kv1_d128_online_fused_hk1_kernel(
    const hip_bfloat16* q,
    const hip_bfloat16* k,
    const hip_bfloat16* v,
    hip_bfloat16* out,
    int B,
    int S,
    float sm_scale);

__global__ void dense_qkv_prefill_causal_h8_kv8_d128_online_fused_hk8_kernel(
    const hip_bfloat16* q,
    const hip_bfloat16* k,
    const hip_bfloat16* v,
    hip_bfloat16* out,
    int B,
    int S,
    float sm_scale);

hipError_t ksearch_launch_dense_qkv_prefill_causal_h8_kv1or8_d128_online(
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
    int HQ,
    int HK,
    float sm_scale);

#endif
