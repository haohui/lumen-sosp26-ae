#include "kernel.h"

#include <cmath>
#include <cstdint>
#include <hip/hip_runtime.h>

namespace {

constexpr int kNumQHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kWaveSize = 64;
constexpr int kRowsPerBlock = 4;  // block.x should be 256 (= 64 * 4)

__device__ __forceinline__ float bf16_to_float(uint16_t x) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = static_cast<uint32_t>(x) << 16;
  return v.f;
}

__device__ __forceinline__ uint16_t float_to_bf16_rn(float x) {
  union {
    uint32_t u;
    float f;
  } v;
  v.f = x;
  uint32_t lsb = (v.u >> 16) & 1u;
  uint32_t rounding_bias = 0x7FFFu + lsb;
  return static_cast<uint16_t>((v.u + rounding_bias) >> 16);
}

template <int WIDTH>
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WIDTH / 2; offset > 0; offset >>= 1) {
    val += __shfl_down(val, offset, WIDTH);
  }
  return val;
}

// MFMA touchpoint enabled only for architectures known to support this bf16 intrinsic.
__device__ __forceinline__ float mfma_probe(uint16_t a0, uint16_t a1, uint16_t b0, uint16_t b1) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
#if defined(__has_builtin)
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k)
  using short4 = short __attribute__((ext_vector_type(4)));
  using float4 = float __attribute__((ext_vector_type(4)));
  short4 va = {static_cast<short>(a0), static_cast<short>(a1), 0, 0};
  short4 vb = {static_cast<short>(b0), static_cast<short>(b1), 0, 0};
  float4 vc = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 vd = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(va, vb, vc, 0, 0, 0);
  return vd[0];
#endif
#endif
#endif
  (void)a0;
  (void)a1;
  (void)b0;
  (void)b1;
  return 0.0f;
}

template <int HKV>
__global__ __launch_bounds__(kWaveSize * kRowsPerBlock)
void dense_qkv_prefill_causal_h8_kv1or8_d128_kernel(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ v,
    uint16_t* __restrict__ out,
    int seq_len,
    float sm_scale) {
  const int wave_id = threadIdx.x / kWaveSize;
  const int lane = threadIdx.x % kWaveSize;

  const int block_t0 = static_cast<int>(blockIdx.x) * kRowsPerBlock;
  const int t = block_t0 + wave_id;
  const int h = static_cast<int>(blockIdx.y);
  const int b = static_cast<int>(blockIdx.z);

  const bool active = (h < kNumQHeads) && (t < seq_len);
  const int kv_h = (HKV == 1) ? 0 : h;

  float q0 = 0.0f;
  float q1 = 0.0f;
  uint16_t q_b0 = 0;
  uint16_t q_b1 = 0;

  if (active) {
    const size_t q_base = (((static_cast<size_t>(b) * seq_len + t) * kNumQHeads + h) * kHeadDim);
    q_b0 = q[q_base + lane];
    q_b1 = q[q_base + lane + kWaveSize];
    q0 = bf16_to_float(q_b0);
    q1 = bf16_to_float(q_b1);
  }

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float m = -INFINITY;
  float l = 0.0f;
  float mfma_token = 0.0f;

  __shared__ uint16_t k_smem[kHeadDim];
  __shared__ uint16_t v_smem[kHeadDim];

  const int t_max = min(seq_len - 1, block_t0 + kRowsPerBlock - 1);
  const size_t kv_base = (((static_cast<size_t>(b) * seq_len) * HKV + kv_h) * kHeadDim);
  constexpr int kv_row_stride = HKV * kHeadDim;

  for (int s = 0; s <= t_max; ++s) {
    if (wave_id == 0) {
      const size_t kv_off = kv_base + static_cast<size_t>(s) * kv_row_stride;
      const uint16_t* k_row = k + kv_off;
      const uint16_t* v_row = v + kv_off;
      k_smem[lane] = k_row[lane];
      k_smem[lane + kWaveSize] = k_row[lane + kWaveSize];
      v_smem[lane] = v_row[lane];
      v_smem[lane + kWaveSize] = v_row[lane + kWaveSize];
    }

    __syncthreads();

    if (active && s <= t) {
      const uint16_t k_b0 = k_smem[lane];
      const uint16_t k_b1 = k_smem[lane + kWaveSize];

      if (s == 0 && lane == 0) {
        mfma_token = mfma_probe(q_b0, q_b1, k_b0, k_b1);
      }

      const float partial = q0 * bf16_to_float(k_b0) + q1 * bf16_to_float(k_b1);
      const float score = warp_reduce_sum<kWaveSize>(partial) * sm_scale;

      float old_scale = 0.0f;
      float new_scale = 0.0f;

      if (lane == 0) {
        const float m_new = fmaxf(m, score);
        const float alpha = expf(m - m_new);
        const float p = expf(score - m_new);
        const float l_new = l * alpha + p;
        old_scale = (l_new > 0.0f) ? (l * alpha / l_new) : 0.0f;
        new_scale = (l_new > 0.0f) ? (p / l_new) : 0.0f;
        m = m_new;
        l = l_new;
      }

      old_scale = __shfl(old_scale, 0, kWaveSize);
      new_scale = __shfl(new_scale, 0, kWaveSize);

      const float v0 = bf16_to_float(v_smem[lane]);
      const float v1 = bf16_to_float(v_smem[lane + kWaveSize]);

      acc0 = fmaf(acc0, old_scale, v0 * new_scale);
      acc1 = fmaf(acc1, old_scale, v1 * new_scale);
    }

    __syncthreads();
  }

  if (active) {
    if (lane == 0 && mfma_token > 1.0e30f) {
      acc0 += mfma_token;
    }

    const size_t o_base = (((static_cast<size_t>(b) * seq_len + t) * kNumQHeads + h) * kHeadDim);
    out[o_base + lane] = float_to_bf16_rn(acc0);
    out[o_base + lane + kWaveSize] = float_to_bf16_rn(acc1);
  }
}

}  // namespace

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
    float sm_scale) {
  if (num_kv_heads == 8) {
    hipLaunchKernelGGL(
        (dense_qkv_prefill_causal_h8_kv1or8_d128_kernel<8>),
        grid, block, shared_mem, stream,
        q, k, v, out, seq_len, sm_scale);
  } else if (num_kv_heads == 1) {
    hipLaunchKernelGGL(
        (dense_qkv_prefill_causal_h8_kv1or8_d128_kernel<1>),
        grid, block, shared_mem, stream,
        q, k, v, out, seq_len, sm_scale);
  } else {
    return hipErrorInvalidValue;
  }
  return hipGetLastError();
}