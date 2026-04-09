#include "kernel.h"

#include <cstdint>

namespace {

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;

using int16x4_t = short __attribute__((ext_vector_type(4)));
using float32x4_t = float __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float ksearch_bf16_to_float(hip_bfloat16 v) {
  union {
    hip_bfloat16 b;
    uint16_t u16;
  } in_bits;
  union {
    uint32_t u32;
    float f32;
  } out_bits;
  in_bits.b = v;
  out_bits.u32 = static_cast<uint32_t>(in_bits.u16) << 16;
  return out_bits.f32;
}

__device__ __forceinline__ hip_bfloat16 ksearch_float_to_bf16_rn(float x) {
  union {
    float f32;
    uint32_t u32;
  } in_bits;
  union {
    uint16_t u16;
    hip_bfloat16 b;
  } out_bits;

  in_bits.f32 = x;
  const uint32_t lsb = (in_bits.u32 >> 16) & 1u;
  in_bits.u32 += 0x7FFFu + lsb;  // round-to-nearest-even
  out_bits.u16 = static_cast<uint16_t>(in_bits.u32 >> 16);
  return out_bits.b;
}

__global__ __launch_bounds__(64) void gemm_bf16_var_mnk_balanced_kernel(
    const hip_bfloat16* __restrict__ A,
    const hip_bfloat16* __restrict__ B,
    hip_bfloat16* __restrict__ C,
    int M,
    int N,
    int K) {
  const int tid = static_cast<int>(threadIdx.x) +
                  static_cast<int>(blockDim.x) * static_cast<int>(threadIdx.y);
  if (tid >= 64) return;

  const int lane_i = tid & 15;   // 0..15
  const int lane_g = tid >> 4;   // 0..3

  const int block_row = static_cast<int>(blockIdx.y) * TILE_M;
  const int block_col = static_cast<int>(blockIdx.x) * TILE_N;

  const int out_row = block_row + lane_i;
  const int out_col_base = block_col + lane_g * 4;

#if defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx90a__) || defined(__gfx908__))
  const uint16_t* __restrict__ A_bits = reinterpret_cast<const uint16_t*>(A);
  const uint16_t* __restrict__ B_bits = reinterpret_cast<const uint16_t*>(B);

  float32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int k0 = 0; k0 < K; k0 += TILE_K) {
    const int k_base = k0 + lane_g * 4;

    uint16_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    if (out_row < M) {
      if (k_base + 0 < K) a0 = A_bits[out_row * K + (k_base + 0)];
      if (k_base + 1 < K) a1 = A_bits[out_row * K + (k_base + 1)];
      if (k_base + 2 < K) a2 = A_bits[out_row * K + (k_base + 2)];
      if (k_base + 3 < K) a3 = A_bits[out_row * K + (k_base + 3)];
    }

    const int b_row = block_col + lane_i;
    uint16_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    if (b_row < N) {
      if (k_base + 0 < K) b0 = B_bits[b_row * K + (k_base + 0)];
      if (k_base + 1 < K) b1 = B_bits[b_row * K + (k_base + 1)];
      if (k_base + 2 < K) b2 = B_bits[b_row * K + (k_base + 2)];
      if (k_base + 3 < K) b3 = B_bits[b_row * K + (k_base + 3)];
    }

    const int16x4_t a_frag = {
        static_cast<short>(a0),
        static_cast<short>(a1),
        static_cast<short>(a2),
        static_cast<short>(a3)};
    const int16x4_t b_frag = {
        static_cast<short>(b0),
        static_cast<short>(b1),
        static_cast<short>(b2),
        static_cast<short>(b3)};

    acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b_frag, acc, 0, 0, 0);
  }

  if (out_row < M) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int out_col = out_col_base + i;
      if (out_col < N) {
        C[out_row * N + out_col] = ksearch_float_to_bf16_rn(acc[i]);
      }
    }
  }
#else
  if (out_row < M) {
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int out_col = out_col_base + jj;
      if (out_col >= N) continue;
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += ksearch_bf16_to_float(A[out_row * K + k]) *
               ksearch_bf16_to_float(B[out_col * K + k]);
      }
      C[out_row * N + out_col] = ksearch_float_to_bf16_rn(sum);
    }
  }
#endif
}

}  // namespace

hipError_t ksearch_launch_gemm_bf16_var_mnk_balanced(
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    hipStream_t stream,
    const hip_bfloat16* A,
    const hip_bfloat16* B,
    hip_bfloat16* C,
    int M,
    int N,
    int K) {
  (void)shared_mem;
  gemm_bf16_var_mnk_balanced_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  return hipGetLastError();
}