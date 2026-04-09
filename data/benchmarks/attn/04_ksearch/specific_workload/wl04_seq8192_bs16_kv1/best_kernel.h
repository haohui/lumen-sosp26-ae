#ifndef DENSE_QKV_PREFILL_CAUSAL_H8_KV1_D128_BS16_KERNEL_H_
#define DENSE_QKV_PREFILL_CAUSAL_H8_KV1_D128_BS16_KERNEL_H_

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstddef>
#include <cstdint>

static constexpr int kFixedBatchSize = 16;
static constexpr int kNumQHeads = 8;
static constexpr int kNumKVHeads = 1;
static constexpr int kHeadDim = 128;
static constexpr int kTileK = 32;
static constexpr int kBlockThreads = 256;
static constexpr int kQueriesPerBlock = 2;

hipError_t ksearch_launch_dense_qkv_prefill_causal_h8_kv1_d128_bs16(
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

hipError_t launch_dense_qkv_prefill_causal_h8_kv1_d128_bs16(
    const hip_bfloat16* q,
    const hip_bfloat16* k,
    const hip_bfloat16* v,
    hip_bfloat16* out,
    int seq_len,
    float sm_scale,
    hipStream_t stream);

#endif  // DENSE_QKV_PREFILL_CAUSAL_H8_KV1_D128_BS16_KERNEL_H_