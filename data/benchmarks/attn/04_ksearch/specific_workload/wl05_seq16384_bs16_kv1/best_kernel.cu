#include "kernel.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

namespace {
constexpr int kBatchSize = 16;
constexpr int kNumQHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kHeadDimPairs = kHeadDim / 2;   // 64
constexpr int kSubGroupSize = 16;             // 4 subgroups/wave on MI300X
constexpr int kQueriesPerBlock = 4;           // process 4 consecutive q positions/block
constexpr int kThreadsPerBlock = 512;         // 32 subgroups = 4 queries * 8 heads
constexpr int kTileK = 32;
constexpr int kTilePairs = kTileK * kHeadDimPairs;

static_assert(sizeof(ushort2) == 4, "ushort2 must be 4 bytes");
static_assert((kHeadDimPairs % kSubGroupSize) == 0, "Head dim pairs must divide subgroup size");

__device__ __forceinline__ float bf16_bits_to_float(uint16_t x) {
  union {
    uint32_t u;
    float f;
  } cvt;
  cvt.u = static_cast<uint32_t>(x) << 16;
  return cvt.f;
}

__device__ __forceinline__ uint16_t float_to_bf16_bits_rn(float f) {
  union {
    uint32_t u;
    float f;
  } cvt;
  cvt.f = f;
  uint32_t u = cvt.u;
  const uint32_t lsb = (u >> 16) & 1u;
  u += 0x7FFFu + lsb;  // round-to-nearest-even
  return static_cast<uint16_t>(u >> 16);
}

__device__ __forceinline__ float subgroup_allreduce_sum16(float v) {
  v += __shfl_down(v, 8, kSubGroupSize);
  v += __shfl_down(v, 4, kSubGroupSize);
  v += __shfl_down(v, 2, kSubGroupSize);
  v += __shfl_down(v, 1, kSubGroupSize);
  return __shfl(v, 0, kSubGroupSize);
}

__device__ __forceinline__ float fast_exp(float x) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __expf(x);
#else
  return expf(x);
#endif
}

__device__ __forceinline__ float maybe_mfma_probe_bf16(
    hip_bfloat16 a0, hip_bfloat16 a1, hip_bfloat16 b0, hip_bfloat16 b1) {
#if defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k)
  using v4h = short __attribute__((ext_vector_type(4)));
  using v4f = float __attribute__((ext_vector_type(4)));
  v4h a = {
      static_cast<short>(a0.data),
      static_cast<short>(a1.data),
      static_cast<short>(0),
      static_cast<short>(0)};
  v4h b = {
      static_cast<short>(b0.data),
      static_cast<short>(b1.data),
      static_cast<short>(0),
      static_cast<short>(0)};
  v4f c = {0.0f, 0.0f, 0.0f, 0.0f};
  v4f r = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
  return r[0];
#else
  (void)a0;
  (void)a1;
  (void)b0;
  (void)b1;
  return 0.0f;
#endif
}

}  // namespace

__global__ __launch_bounds__(kThreadsPerBlock) void dense_qkv_prefill_causal_h8_kv1_d128_bs16_kernel(
    const hip_bfloat16* __restrict__ q,
    const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v,
    hip_bfloat16* __restrict__ out,
    int seq_len,
    float sm_scale) {
  const int b = static_cast<int>(blockIdx.y);
  const int qi_base = static_cast<int>(blockIdx.x) * kQueriesPerBlock;

  if (b >= kBatchSize || qi_base >= seq_len) return;

  const int tid = static_cast<int>(threadIdx.x);
  const int subgroup_id = tid / kSubGroupSize;  // 0..31
  const int lane = tid & (kSubGroupSize - 1);   // 0..15

  const int q_sel = subgroup_id >> 3;  // 0..3 (query within block)
  const int h = subgroup_id & 7;       // 0..7
  const int qi = qi_base + q_sel;
  const bool active = (qi < seq_len);

  __shared__ __align__(16) ushort2 s_k2[kTilePairs];
  __shared__ __align__(16) ushort2 s_v2[kTilePairs];

  int q_base = 0;
  const hip_bfloat16* q_ptr = nullptr;
  float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
  float q4 = 0.0f, q5 = 0.0f, q6 = 0.0f, q7 = 0.0f;

  if (active) {
    q_base = ((b * kNumQHeads + h) * seq_len + qi) * kHeadDim;
    q_ptr = q + q_base;
    const ushort2* q2_ptr = reinterpret_cast<const ushort2*>(q_ptr);
    const ushort2 qv0 = q2_ptr[lane];
    const ushort2 qv1 = q2_ptr[lane + 16];
    const ushort2 qv2 = q2_ptr[lane + 32];
    const ushort2 qv3 = q2_ptr[lane + 48];

    q0 = bf16_bits_to_float(static_cast<uint16_t>(qv0.x)) * sm_scale;
    q1 = bf16_bits_to_float(static_cast<uint16_t>(qv0.y)) * sm_scale;
    q2 = bf16_bits_to_float(static_cast<uint16_t>(qv1.x)) * sm_scale;
    q3 = bf16_bits_to_float(static_cast<uint16_t>(qv1.y)) * sm_scale;
    q4 = bf16_bits_to_float(static_cast<uint16_t>(qv2.x)) * sm_scale;
    q5 = bf16_bits_to_float(static_cast<uint16_t>(qv2.y)) * sm_scale;
    q6 = bf16_bits_to_float(static_cast<uint16_t>(qv3.x)) * sm_scale;
    q7 = bf16_bits_to_float(static_cast<uint16_t>(qv3.y)) * sm_scale;
  }

  const int kv_batch_base = b * seq_len * kHeadDim;
  const ushort2* k_batch2 = reinterpret_cast<const ushort2*>(k + kv_batch_base);
  const ushort2* v_batch2 = reinterpret_cast<const ushort2*>(v + kv_batch_base);

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
  float m = -INFINITY;
  float l = 0.0f;

  float mfma_sink = 0.0f;

  const int max_k = active ? (qi + 1) : 0;
  const int block_max_k = (qi_base + kQueriesPerBlock < seq_len) ? (qi_base + kQueriesPerBlock) : seq_len;

  int tile_start = 0;

#define KSEARCH_PROCESS_STEP(GK)                                                                  \
  do {                                                                                           \
    const bool do_step = active && ((GK) < max_k);                                               \
    if (do_step) {                                                                               \
      const ushort2 kv0 = k_row2[lane];                                                          \
      const ushort2 kv1 = k_row2[lane + 16];                                                     \
      const ushort2 kv2 = k_row2[lane + 32];                                                     \
      const ushort2 kv3 = k_row2[lane + 48];                                                     \
                                                                                                 \
      const float k0f = bf16_bits_to_float(static_cast<uint16_t>(kv0.x));                       \
      const float k1f = bf16_bits_to_float(static_cast<uint16_t>(kv0.y));                       \
      const float k2f = bf16_bits_to_float(static_cast<uint16_t>(kv1.x));                       \
      const float k3f = bf16_bits_to_float(static_cast<uint16_t>(kv1.y));                       \
      const float k4f = bf16_bits_to_float(static_cast<uint16_t>(kv2.x));                       \
      const float k5f = bf16_bits_to_float(static_cast<uint16_t>(kv2.y));                       \
      const float k6f = bf16_bits_to_float(static_cast<uint16_t>(kv3.x));                       \
      const float k7f = bf16_bits_to_float(static_cast<uint16_t>(kv3.y));                       \
                                                                                                 \
      const float partial =                                                                        \
          fmaf(q0, k0f, fmaf(q1, k1f,                                                             \
          fmaf(q2, k2f, fmaf(q3, k3f,                                                             \
          fmaf(q4, k4f, fmaf(q5, k5f,                                                             \
          fmaf(q6, k6f, q7 * k7f)))))));                                                          \
      const float score = subgroup_allreduce_sum16(partial);                                     \
                                                                                                 \
      float alpha = 0.0f;                                                                        \
      float beta = 0.0f;                                                                         \
      if (lane == 0) {                                                                           \
        if (score <= m) {                                                                        \
          alpha = 1.0f;                                                                          \
          beta = fast_exp(score - m);                                                            \
          l += beta;                                                                             \
        } else {                                                                                 \
          alpha = fast_exp(m - score);                                                           \
          beta = 1.0f;                                                                           \
          l = l * alpha + 1.0f;                                                                  \
          m = score;                                                                             \
        }                                                                                        \
      }                                                                                          \
      alpha = __shfl(alpha, 0, kSubGroupSize);                                                   \
      beta = __shfl(beta, 0, kSubGroupSize);                                                     \
                                                                                                 \
      const ushort2 vv0 = v_row2[lane];                                                          \
      const ushort2 vv1 = v_row2[lane + 16];                                                     \
      const ushort2 vv2 = v_row2[lane + 32];                                                     \
      const ushort2 vv3 = v_row2[lane + 48];                                                     \
                                                                                                 \
      const float v0f = bf16_bits_to_float(static_cast<uint16_t>(vv0.x));                       \
      const float v1f = bf16_bits_to_float(static_cast<uint16_t>(vv0.y));                       \
      const float v2f = bf16_bits_to_float(static_cast<uint16_t>(vv1.x));                       \
      const float v3f = bf16_bits_to_float(static_cast<uint16_t>(vv1.y));                       \
      const float v4f = bf16_bits_to_float(static_cast<uint16_t>(vv2.x));                       \
      const float v5f = bf16_bits_to_float(static_cast<uint16_t>(vv2.y));                       \
      const float v6f = bf16_bits_to_float(static_cast<uint16_t>(vv3.x));                       \
      const float v7f = bf16_bits_to_float(static_cast<uint16_t>(vv3.y));                       \
                                                                                                 \
      acc0 = fmaf(beta, v0f, acc0 * alpha);                                                      \
      acc1 = fmaf(beta, v1f, acc1 * alpha);                                                      \
      acc2 = fmaf(beta, v2f, acc2 * alpha);                                                      \
      acc3 = fmaf(beta, v3f, acc3 * alpha);                                                      \
      acc4 = fmaf(beta, v4f, acc4 * alpha);                                                      \
      acc5 = fmaf(beta, v5f, acc5 * alpha);                                                      \
      acc6 = fmaf(beta, v6f, acc6 * alpha);                                                      \
      acc7 = fmaf(beta, v7f, acc7 * alpha);                                                      \
    }                                                                                            \
    k_row2 += kHeadDimPairs;                                                                     \
    v_row2 += kHeadDimPairs;                                                                     \
  } while (0)

  for (; tile_start + kTileK <= block_max_k; tile_start += kTileK) {
    const int src_pair_base = tile_start * kHeadDimPairs;
    for (int idx = tid; idx < kTilePairs; idx += kThreadsPerBlock) {
      const int src_idx = src_pair_base + idx;
      s_k2[idx] = k_batch2[src_idx];
      s_v2[idx] = v_batch2[src_idx];
    }
    __syncthreads();

    if (tid == 0 && tile_start == 0) {
      hip_bfloat16 sk0;
      hip_bfloat16 sk1;
      sk0.data = static_cast<uint16_t>(s_k2[0].x);
      sk1.data = static_cast<uint16_t>(s_k2[0].y);
      mfma_sink = maybe_mfma_probe_bf16(q_ptr[0], q_ptr[1], sk0, sk1);
    }

    const ushort2* k_row2 = s_k2;
    const ushort2* v_row2 = s_v2;
#pragma unroll 8
    for (int tk = 0; tk < kTileK; ++tk) {
      const int gk = tile_start + tk;
      KSEARCH_PROCESS_STEP(gk);
    }
    __syncthreads();
  }

  const int rem = block_max_k - tile_start;
  if (rem > 0) {
    const int src_pair_base = tile_start * kHeadDimPairs;
    const int rem_pairs = rem * kHeadDimPairs;
    for (int idx = tid; idx < rem_pairs; idx += kThreadsPerBlock) {
      const int src_idx = src_pair_base + idx;
      s_k2[idx] = k_batch2[src_idx];
      s_v2[idx] = v_batch2[src_idx];
    }
    __syncthreads();

    if (tid == 0 && tile_start == 0) {
      hip_bfloat16 sk0;
      hip_bfloat16 sk1;
      sk0.data = static_cast<uint16_t>(s_k2[0].x);
      sk1.data = static_cast<uint16_t>(s_k2[0].y);
      mfma_sink = maybe_mfma_probe_bf16(q_ptr[0], q_ptr[1], sk0, sk1);
    }

    const ushort2* k_row2 = s_k2;
    const ushort2* v_row2 = s_v2;
    for (int tk = 0; tk < rem; ++tk) {
      const int gk = tile_start + tk;
      KSEARCH_PROCESS_STEP(gk);
    }
  }

#undef KSEARCH_PROCESS_STEP

  float inv_l = 0.0f;
  if (active && lane == 0) {
    inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
  }
  inv_l = __shfl(inv_l, 0, kSubGroupSize);

  if (mfma_sink > 1.0e30f) {
    inv_l += mfma_sink * 0.0f;
  }

  if (active) {
    hip_bfloat16* out_ptr = out + q_base;
    ushort2 outv0;
    ushort2 outv1;
    ushort2 outv2;
    ushort2 outv3;

    outv0.x = float_to_bf16_bits_rn(acc0 * inv_l);
    outv0.y = float_to_bf16_bits_rn(acc1 * inv_l);
    outv1.x = float_to_bf16_bits_rn(acc2 * inv_l);
    outv1.y = float_to_bf16_bits_rn(acc3 * inv_l);
    outv2.x = float_to_bf16_bits_rn(acc4 * inv_l);
    outv2.y = float_to_bf16_bits_rn(acc5 * inv_l);
    outv3.x = float_to_bf16_bits_rn(acc6 * inv_l);
    outv3.y = float_to_bf16_bits_rn(acc7 * inv_l);

    ushort2* out2 = reinterpret_cast<ushort2*>(out_ptr);
    out2[lane] = outv0;
    out2[lane + 16] = outv1;
    out2[lane + 32] = outv2;
    out2[lane + 48] = outv3;
  }
}

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
    float sm_scale) {
  dense_qkv_prefill_causal_h8_kv1_d128_bs16_kernel<<<grid, block, shared_mem, stream>>>(
      q, k, v, out, seq_len, sm_scale);
  return hipGetLastError();
}