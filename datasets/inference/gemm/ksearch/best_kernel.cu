#include "kernel.h"

#include <cstdint>
#include <hip/hip_runtime.h>

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k)
#define KSEARCH_HAS_MFMA_BF16 1
#else
#define KSEARCH_HAS_MFMA_BF16 0
#endif

namespace {
constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;
constexpr int TILE_ELEMS = TILE_N * TILE_K;
constexpr int WAVE_SIZE = 64;

constexpr int TILES_N_PER_BLOCK_BASE = 4;
constexpr int BLOCK_N_BASE = TILE_N * TILES_N_PER_BLOCK_BASE;
constexpr int BLOCK_THREADS_BASE = WAVE_SIZE;

constexpr int WAVES_PER_BLOCK_BALANCED = 2;
constexpr int TILES_N_PER_WAVE = 4;
constexpr int TILES_N_PER_BLOCK_BALANCED = WAVES_PER_BLOCK_BALANCED * TILES_N_PER_WAVE;
constexpr int BLOCK_N_BALANCED = TILE_N * TILES_N_PER_BLOCK_BALANCED;
constexpr int BLOCK_THREADS_BALANCED = WAVE_SIZE * WAVES_PER_BLOCK_BALANCED;

using fp32x4 = float __attribute__((ext_vector_type(4)));
using i16x4 = int16_t __attribute__((ext_vector_type(4)));

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
  uint32_t bias = 0x7fffu + lsb;
  v.u += bias;
  return static_cast<uint16_t>(v.u >> 16);
}

__device__ __forceinline__ void scalar_fallback_tile(
    const uint16_t* __restrict__ A,
    const uint16_t* __restrict__ B,
    uint16_t* __restrict__ C,
    int64_t M,
    int64_t N,
    int64_t K,
    int lane,
    int64_t block_m,
    int64_t block_n) {
  const int row = lane & 15;
  const int col_group = lane >> 4;

  const int64_t gm = block_m + row;
  const int64_t gn0 = block_n + col_group * 4 + 0;
  const int64_t gn1 = block_n + col_group * 4 + 1;
  const int64_t gn2 = block_n + col_group * 4 + 2;
  const int64_t gn3 = block_n + col_group * 4 + 3;

  float s0 = 0.0f;
  float s1 = 0.0f;
  float s2 = 0.0f;
  float s3 = 0.0f;

  if (gm < M) {
    for (int64_t k = 0; k < K; ++k) {
      const float a = bf16_to_float(A[gm * K + k]);
      if (gn0 < N) s0 += a * bf16_to_float(B[gn0 * K + k]);
      if (gn1 < N) s1 += a * bf16_to_float(B[gn1 * K + k]);
      if (gn2 < N) s2 += a * bf16_to_float(B[gn2 * K + k]);
      if (gn3 < N) s3 += a * bf16_to_float(B[gn3 * K + k]);
    }
  }

  if (gm < M) {
    if (gn0 < N) C[gm * N + gn0] = float_to_bf16_rn(s0);
    if (gn1 < N) C[gm * N + gn1] = float_to_bf16_rn(s1);
    if (gn2 < N) C[gm * N + gn2] = float_to_bf16_rn(s2);
    if (gn3 < N) C[gm * N + gn3] = float_to_bf16_rn(s3);
  }
}
}  // namespace

__global__ __launch_bounds__(BLOCK_THREADS_BASE) void gemm_bf16_var_mnk_kernel(
    const uint16_t* __restrict__ A,
    const uint16_t* __restrict__ B,
    uint16_t* __restrict__ C,
    int64_t M,
    int64_t N,
    int64_t K) {
  const int lane = static_cast<int>(threadIdx.x);
  if (lane >= WAVE_SIZE) return;

  const int row = lane & 15;
  const int k_group = lane >> 4;
  const int base = row * TILE_K + k_group * 4;

  const int64_t block_m = static_cast<int64_t>(blockIdx.y) * TILE_M;
  const int64_t block_n = static_cast<int64_t>(blockIdx.x) * BLOCK_N_BASE;

#if KSEARCH_HAS_MFMA_BF16
  __shared__ uint16_t As[TILE_M * TILE_K];
  __shared__ uint16_t Bs0[TILE_N * TILE_K];
  __shared__ uint16_t Bs1[TILE_N * TILE_K];
  __shared__ uint16_t Bs2[TILE_N * TILE_K];
  __shared__ uint16_t Bs3[TILE_N * TILE_K];

  i16x4 a_cal;
  i16x4 b_cal;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int kk = k_group * 4 + i;
    const uint16_t a_bits = (row == kk) ? static_cast<uint16_t>(0x3f80u) : static_cast<uint16_t>(0u);
    const uint16_t b_bits = float_to_bf16_rn(static_cast<float>(kk * 16 + row));
    a_cal[i] = static_cast<int16_t>(a_bits);
    b_cal[i] = static_cast<int16_t>(b_bits);
  }

  fp32x4 map_acc = {0.0f, 0.0f, 0.0f, 0.0f};
  map_acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_cal, b_cal, map_acc, 0, 0, 0);

  int out_m[4];
  int out_n[4];
  bool local_valid = true;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int v = __float2int_rn(map_acc[i]);
    if (static_cast<unsigned>(v) >= 256u) {
      local_valid = false;
    }
    out_m[i] = v >> 4;
    out_n[i] = v & 15;
  }

  if (!__all(local_valid ? 1 : 0)) {
    scalar_fallback_tile(A, B, C, M, N, K, lane, block_m, block_n + 0 * TILE_N);
    scalar_fallback_tile(A, B, C, M, N, K, lane, block_m, block_n + 1 * TILE_N);
    scalar_fallback_tile(A, B, C, M, N, K, lane, block_m, block_n + 2 * TILE_N);
    scalar_fallback_tile(A, B, C, M, N, K, lane, block_m, block_n + 3 * TILE_N);
    return;
  }

  fp32x4 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
  fp32x4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
  fp32x4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
  fp32x4 acc3 = {0.0f, 0.0f, 0.0f, 0.0f};

  const bool full_mn = (block_m + TILE_M <= M) && (block_n + BLOCK_N_BASE <= N);
  const int64_t k_full = (K / TILE_K) * TILE_K;

  if (full_mn) {
    for (int64_t k0 = 0; k0 < k_full; k0 += TILE_K) {
#pragma unroll
      for (int t = lane; t < TILE_M * TILE_K; t += WAVE_SIZE) {
        const int r = t >> 4;
        const int kk = t & 15;
        const int64_t gk = k0 + kk;
        As[t] = A[(block_m + r) * K + gk];
        Bs0[t] = B[(block_n + r) * K + gk];
        Bs1[t] = B[(block_n + TILE_N + r) * K + gk];
        Bs2[t] = B[(block_n + 2 * TILE_N + r) * K + gk];
        Bs3[t] = B[(block_n + 3 * TILE_N + r) * K + gk];
      }

      __syncthreads();

      i16x4 a_frag = {
          static_cast<int16_t>(As[base + 0]),
          static_cast<int16_t>(As[base + 1]),
          static_cast<int16_t>(As[base + 2]),
          static_cast<int16_t>(As[base + 3])};

      i16x4 b0_frag = {
          static_cast<int16_t>(Bs0[base + 0]),
          static_cast<int16_t>(Bs0[base + 1]),
          static_cast<int16_t>(Bs0[base + 2]),
          static_cast<int16_t>(Bs0[base + 3])};

      i16x4 b1_frag = {
          static_cast<int16_t>(Bs1[base + 0]),
          static_cast<int16_t>(Bs1[base + 1]),
          static_cast<int16_t>(Bs1[base + 2]),
          static_cast<int16_t>(Bs1[base + 3])};

      i16x4 b2_frag = {
          static_cast<int16_t>(Bs2[base + 0]),
          static_cast<int16_t>(Bs2[base + 1]),
          static_cast<int16_t>(Bs2[base + 2]),
          static_cast<int16_t>(Bs2[base + 3])};

      i16x4 b3_frag = {
          static_cast<int16_t>(Bs3[base + 0]),
          static_cast<int16_t>(Bs3[base + 1]),
          static_cast<int16_t>(Bs3[base + 2]),
          static_cast<int16_t>(Bs3[base + 3])};

      acc0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b0_frag, acc0, 0, 0, 0);
      acc1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b1_frag, acc1, 0, 0, 0);
      acc2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b2_frag, acc2, 0, 0, 0);
      acc3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b3_frag, acc3, 0, 0, 0);
    }

    if (k_full < K) {
#pragma unroll
      for (int t = lane; t < TILE_M * TILE_K; t += WAVE_SIZE) {
        const int r = t >> 4;
        const int kk = t & 15;
        const int64_t gk = k_full + kk;

        uint16_t av = 0;
        uint16_t b0v = 0;
        uint16_t b1v = 0;
        uint16_t b2v = 0;
        uint16_t b3v = 0;
        if (gk < K) {
          av = A[(block_m + r) * K + gk];
          b0v = B[(block_n + r) * K + gk];
          b1v = B[(block_n + TILE_N + r) * K + gk];
          b2v = B[(block_n + 2 * TILE_N + r) * K + gk];
          b3v = B[(block_n + 3 * TILE_N + r) * K + gk];
        }
        As[t] = av;
        Bs0[t] = b0v;
        Bs1[t] = b1v;
        Bs2[t] = b2v;
        Bs3[t] = b3v;
      }

      __syncthreads();

      i16x4 a_frag = {
          static_cast<int16_t>(As[base + 0]),
          static_cast<int16_t>(As[base + 1]),
          static_cast<int16_t>(As[base + 2]),
          static_cast<int16_t>(As[base + 3])};

      i16x4 b0_frag = {
          static_cast<int16_t>(Bs0[base + 0]),
          static_cast<int16_t>(Bs0[base + 1]),
          static_cast<int16_t>(Bs0[base + 2]),
          static_cast<int16_t>(Bs0[base + 3])};

      i16x4 b1_frag = {
          static_cast<int16_t>(Bs1[base + 0]),
          static_cast<int16_t>(Bs1[base + 1]),
          static_cast<int16_t>(Bs1[base + 2]),
          static_cast<int16_t>(Bs1[base + 3])};

      i16x4 b2_frag = {
          static_cast<int16_t>(Bs2[base + 0]),
          static_cast<int16_t>(Bs2[base + 1]),
          static_cast<int16_t>(Bs2[base + 2]),
          static_cast<int16_t>(Bs2[base + 3])};

      i16x4 b3_frag = {
          static_cast<int16_t>(Bs3[base + 0]),
          static_cast<int16_t>(Bs3[base + 1]),
          static_cast<int16_t>(Bs3[base + 2]),
          static_cast<int16_t>(Bs3[base + 3])};

      acc0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b0_frag, acc0, 0, 0, 0);
      acc1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b1_frag, acc1, 0, 0, 0);
      acc2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b2_frag, acc2, 0, 0, 0);
      acc3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b3_frag, acc3, 0, 0, 0);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t gm = block_m + out_m[i];
      const int64_t gn0 = block_n + out_n[i];
      C[gm * N + (gn0 + 0 * TILE_N)] = float_to_bf16_rn(acc0[i]);
      C[gm * N + (gn0 + 1 * TILE_N)] = float_to_bf16_rn(acc1[i]);
      C[gm * N + (gn0 + 2 * TILE_N)] = float_to_bf16_rn(acc2[i]);
      C[gm * N + (gn0 + 3 * TILE_N)] = float_to_bf16_rn(acc3[i]);
    }
  } else {
    for (int64_t k0 = 0; k0 < k_full; k0 += TILE_K) {
#pragma unroll
      for (int t = lane; t < TILE_M * TILE_K; t += WAVE_SIZE) {
        const int r = t >> 4;
        const int kk = t & 15;

        const int64_t gm = block_m + r;
        const int64_t gn0 = block_n + r;
        const int64_t gn1 = block_n + TILE_N + r;
        const int64_t gn2 = block_n + 2 * TILE_N + r;
        const int64_t gn3 = block_n + 3 * TILE_N + r;
        const int64_t gk = k0 + kk;

        uint16_t av = 0;
        if (gm < M) av = A[gm * K + gk];
        As[t] = av;

        uint16_t b0v = 0;
        if (gn0 < N) b0v = B[gn0 * K + gk];
        Bs0[t] = b0v;

        uint16_t b1v = 0;
        if (gn1 < N) b1v = B[gn1 * K + gk];
        Bs1[t] = b1v;

        uint16_t b2v = 0;
        if (gn2 < N) b2v = B[gn2 * K + gk];
        Bs2[t] = b2v;

        uint16_t b3v = 0;
        if (gn3 < N) b3v = B[gn3 * K + gk];
        Bs3[t] = b3v;
      }

      __syncthreads();

      i16x4 a_frag = {
          static_cast<int16_t>(As[base + 0]),
          static_cast<int16_t>(As[base + 1]),
          static_cast<int16_t>(As[base + 2]),
          static_cast<int16_t>(As[base + 3])};

      i16x4 b0_frag = {
          static_cast<int16_t>(Bs0[base + 0]),
          static_cast<int16_t>(Bs0[base + 1]),
          static_cast<int16_t>(Bs0[base + 2]),
          static_cast<int16_t>(Bs0[base + 3])};

      i16x4 b1_frag = {
          static_cast<int16_t>(Bs1[base + 0]),
          static_cast<int16_t>(Bs1[base + 1]),
          static_cast<int16_t>(Bs1[base + 2]),
          static_cast<int16_t>(Bs1[base + 3])};

      i16x4 b2_frag = {
          static_cast<int16_t>(Bs2[base + 0]),
          static_cast<int16_t>(Bs2[base + 1]),
          static_cast<int16_t>(Bs2[base + 2]),
          static_cast<int16_t>(Bs2[base + 3])};

      i16x4 b3_frag = {
          static_cast<int16_t>(Bs3[base + 0]),
          static_cast<int16_t>(Bs3[base + 1]),
          static_cast<int16_t>(Bs3[base + 2]),
          static_cast<int16_t>(Bs3[base + 3])};

      acc0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b0_frag, acc0, 0, 0, 0);
      acc1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b1_frag, acc1, 0, 0, 0);
      acc2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b2_frag, acc2, 0, 0, 0);
      acc3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b3_frag, acc3, 0, 0, 0);
    }

    if (k_full < K) {
#pragma unroll
      for (int t = lane; t < TILE_M * TILE_K; t += WAVE_SIZE) {
        const int r = t >> 4;
        const int kk = t & 15;

        const int64_t gm = block_m + r;
        const int64_t gn0 = block_n + r;
        const int64_t gn1 = block_n + TILE_N + r;
        const int64_t gn2 = block_n + 2 * TILE_N + r;
        const int64_t gn3 = block_n + 3 * TILE_N + r;
        const int64_t gk = k_full + kk;

        uint16_t av = 0;
        if (gm < M && gk < K) av = A[gm * K + gk];
        As[t] = av;

        uint16_t b0v = 0;
        if (gn0 < N && gk < K) b0v = B[gn0 * K + gk];
        Bs0[t] = b0v;

        uint16_t b1v = 0;
        if (gn1 < N && gk < K) b1v = B[gn1 * K + gk];
        Bs1[t] = b1v;

        uint16_t b2v = 0;
        if (gn2 < N && gk < K) b2v = B[gn2 * K + gk];
        Bs2[t] = b2v;

        uint16_t b3v = 0;
        if (gn3 < N && gk < K) b3v = B[gn3 * K + gk];
        Bs3[t] = b3v;
      }

      __syncthreads();

      i16x4 a_frag = {
          static_cast<int16_t>(As[base + 0]),
          static_cast<int16_t>(As[base + 1]),
          static_cast<int16_t>(As[base + 2]),
          static_cast<int16_t>(As[base + 3])};

      i16x4 b0_frag = {
          static_cast<int16_t>(Bs0[base + 0]),
          static_cast<int16_t>(Bs0[base + 1]),
          static_cast<int16_t>(Bs0[base + 2]),
          static_cast<int16_t>(Bs0[base + 3])};

      i16x4 b1_frag = {
          static_cast<int16_t>(Bs1[base + 0]),
          static_cast<int16_t>(Bs1[base + 1]),
          static_cast<int16_t>(Bs1[base + 2]),
          static_cast<int16_t>(Bs1[base + 3])};

      i16x4 b2_frag = {
          static_cast<int16_t>(Bs2[base + 0]),
          static_cast<int16_t>(Bs2[base + 1]),
          static_cast<int16_t>(Bs2[base + 2]),
          static_cast<int16_t>(Bs2[base + 3])};

      i16x4 b3_frag = {
          static_cast<int16_t>(Bs3[base + 0]),
          static_cast<int16_t>(Bs3[base + 1]),
          static_cast<int16_t>(Bs3[base + 2]),
          static_cast<int16_t>(Bs3[base + 3])};

      acc0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b0_frag, acc0, 0, 0, 0);
      acc1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b1_frag, acc1, 0, 0, 0);
      acc2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b2_frag, acc2, 0, 0, 0);
      acc3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b3_frag, acc3, 0, 0, 0);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t gm = block_m + out_m[i];
      const int64_t gn0 = block_n + out_n[i];
      if (gm < M) {
        if (gn0 + 0 * TILE_N < N) C[gm * N + (gn0 + 0 * TILE_N)] = float_to_bf16_rn(acc0[i]);
        if (gn0 + 1 * TILE_N < N) C[gm * N + (gn0 + 1 * TILE_N)] = float_to_bf16_rn(acc1[i]);
        if (gn0 + 2 * TILE_N < N) C[gm * N + (gn0 + 2 * TILE_N)] = float_to_bf16_rn(acc2[i]);
        if (gn0 + 3 * TILE_N < N) C[gm * N + (gn0 + 3 * TILE_N)] = float_to_bf16_rn(acc3[i]);
      }
    }
  }

#else
  scalar_fallback_tile(A, B, C, M, N, K, lane, block_m, block_n + 0 * TILE_N);
  scalar_fallback_tile(A, B, C, M, N, K, lane, block_m, block_n + 1 * TILE_N);
  scalar_fallback_tile(A, B, C, M, N, K, lane, block_m, block_n + 2 * TILE_N);
  scalar_fallback_tile(A, B, C, M, N, K, lane, block_m, block_n + 3 * TILE_N);
#endif
}

__global__ __launch_bounds__(BLOCK_THREADS_BALANCED) void gemm_bf16_var_mnk_balanced_kernel(
    const uint16_t* __restrict__ A,
    const uint16_t* __restrict__ B,
    uint16_t* __restrict__ C,
    int64_t M,
    int64_t N,
    int64_t K) {
  const int tid = static_cast<int>(threadIdx.x);
  if (tid >= BLOCK_THREADS_BALANCED) return;

  const int lane = tid & (WAVE_SIZE - 1);
  const int wave = tid / WAVE_SIZE;

  const int row = lane & 15;
  const int k_group = lane >> 4;
  const int base = row * TILE_K + k_group * 4;

  const int64_t block_m = static_cast<int64_t>(blockIdx.y) * TILE_M;
  const int64_t block_n = static_cast<int64_t>(blockIdx.x) * BLOCK_N_BALANCED;
  const int64_t wave_block_n = block_n + static_cast<int64_t>(wave) * (TILES_N_PER_WAVE * TILE_N);

#if KSEARCH_HAS_MFMA_BF16
  __shared__ uint16_t As[TILE_M * TILE_K];
  __shared__ uint16_t Bs[TILES_N_PER_BLOCK_BALANCED * TILE_ELEMS];
  __shared__ int map_valid_block;

  i16x4 a_cal;
  i16x4 b_cal;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int kk = k_group * 4 + i;
    const uint16_t a_bits = (row == kk) ? static_cast<uint16_t>(0x3f80u) : static_cast<uint16_t>(0u);
    const uint16_t b_bits = float_to_bf16_rn(static_cast<float>(kk * 16 + row));
    a_cal[i] = static_cast<int16_t>(a_bits);
    b_cal[i] = static_cast<int16_t>(b_bits);
  }

  fp32x4 map_acc = {0.0f, 0.0f, 0.0f, 0.0f};
  map_acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_cal, b_cal, map_acc, 0, 0, 0);

  int out_m[4];
  int out_n[4];
  bool local_valid = true;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int v = __float2int_rn(map_acc[i]);
    if (static_cast<unsigned>(v) >= 256u) {
      local_valid = false;
    }
    out_m[i] = v >> 4;
    out_n[i] = v & 15;
  }

  if (tid == 0) map_valid_block = 1;
  __syncthreads();
  if (!local_valid) atomicExch(&map_valid_block, 0);
  __syncthreads();

  if (map_valid_block == 0) {
    scalar_fallback_tile(A, B, C, M, N, K, lane, block_m, wave_block_n + 0 * TILE_N);
    scalar_fallback_tile(A, B, C, M, N, K, lane, block_m, wave_block_n + 1 * TILE_N);
    scalar_fallback_tile(A, B, C, M, N, K, lane, block_m, wave_block_n + 2 * TILE_N);
    scalar_fallback_tile(A, B, C, M, N, K, lane, block_m, wave_block_n + 3 * TILE_N);
    return;
  }

  fp32x4 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
  fp32x4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
  fp32x4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
  fp32x4 acc3 = {0.0f, 0.0f, 0.0f, 0.0f};

  const int wave_tile_base = wave * TILES_N_PER_WAVE * TILE_ELEMS;
  const int64_t k_full = (K / TILE_K) * TILE_K;
  const bool full_mn = (block_m + TILE_M <= M) && (block_n + BLOCK_N_BALANCED <= N);

  if (full_mn) {
    for (int64_t k0 = 0; k0 < k_full; k0 += TILE_K) {
      for (int t = tid; t < TILE_M * TILE_K; t += BLOCK_THREADS_BALANCED) {
        const int r = t >> 4;
        const int kk = t & 15;
        const int64_t gk = k0 + kk;
        As[t] = A[(block_m + r) * K + gk];
      }

      for (int t = tid; t < TILES_N_PER_BLOCK_BALANCED * TILE_ELEMS; t += BLOCK_THREADS_BALANCED) {
        const int tile = t / TILE_ELEMS;
        const int rem = t - tile * TILE_ELEMS;
        const int r = rem >> 4;
        const int kk = rem & 15;
        const int64_t gn = block_n + static_cast<int64_t>(tile * TILE_N + r);
        const int64_t gk = k0 + kk;
        Bs[t] = B[gn * K + gk];
      }

      __syncthreads();

      i16x4 a_frag = {
          static_cast<int16_t>(As[base + 0]),
          static_cast<int16_t>(As[base + 1]),
          static_cast<int16_t>(As[base + 2]),
          static_cast<int16_t>(As[base + 3])};

      i16x4 b0_frag = {
          static_cast<int16_t>(Bs[wave_tile_base + 0 * TILE_ELEMS + base + 0]),
          static_cast<int16_t>(Bs[wave_tile_base + 0 * TILE_ELEMS + base + 1]),
          static_cast<int16_t>(Bs[wave_tile_base + 0 * TILE_ELEMS + base + 2]),
          static_cast<int16_t>(Bs[wave_tile_base + 0 * TILE_ELEMS + base + 3])};

      i16x4 b1_frag = {
          static_cast<int16_t>(Bs[wave_tile_base + 1 * TILE_ELEMS + base + 0]),
          static_cast<int16_t>(Bs[wave_tile_base + 1 * TILE_ELEMS + base + 1]),
          static_cast<int16_t>(Bs[wave_tile_base + 1 * TILE_ELEMS + base + 2]),
          static_cast<int16_t>(Bs[wave_tile_base + 1 * TILE_ELEMS + base + 3])};

      i16x4 b2_frag = {
          static_cast<int16_t>(Bs[wave_tile_base + 2 * TILE_ELEMS + base + 0]),
          static_cast<int16_t>(Bs[wave_tile_base + 2 * TILE_ELEMS + base + 1]),
          static_cast<int16_t>(Bs[wave_tile_base + 2 * TILE_ELEMS + base + 2]),
          static_cast<int16_t>(Bs[wave_tile_base + 2 * TILE_ELEMS + base + 3])};

      i16x4 b3_frag = {
          static_cast<int16_t>(Bs[wave_tile_base + 3 * TILE_ELEMS + base + 0]),
          static_cast<int16_t>(Bs[wave_tile_base + 3 * TILE_ELEMS + base + 1]),
          static_cast<int16_t>(Bs[wave_tile_base + 3 * TILE_ELEMS + base + 2]),
          static_cast<int16_t>(Bs[wave_tile_base + 3 * TILE_ELEMS + base + 3])};

      acc0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b0_frag, acc0, 0, 0, 0);
      acc1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b1_frag, acc1, 0, 0, 0);
      acc2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b2_frag, acc2, 0, 0, 0);
      acc3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b3_frag, acc3, 0, 0, 0);

      __syncthreads();
    }

    if (k_full < K) {
      for (int t = tid; t < TILE_M * TILE_K; t += BLOCK_THREADS_BALANCED) {
        const int r = t >> 4;
        const int kk = t & 15;
        const int64_t gk = k_full + kk;
        uint16_t av = 0;
        if (gk < K) {
          av = A[(block_m + r) * K + gk];
        }
        As[t] = av;
      }

      for (int t = tid; t < TILES_N_PER_BLOCK_BALANCED * TILE_ELEMS; t += BLOCK_THREADS_BALANCED) {
        const int tile = t / TILE_ELEMS;
        const int rem = t - tile * TILE_ELEMS;
        const int r = rem >> 4;
        const int kk = rem & 15;
        const int64_t gn = block_n + static_cast<int64_t>(tile * TILE_N + r);
        const int64_t gk = k_full + kk;
        uint16_t bv = 0;
        if (gk < K) {
          bv = B[gn * K + gk];
        }
        Bs[t] = bv;
      }

      __syncthreads();

      i16x4 a_frag = {
          static_cast<int16_t>(As[base + 0]),
          static_cast<int16_t>(As[base + 1]),
          static_cast<int16_t>(As[base + 2]),
          static_cast<int16_t>(As[base + 3])};

      i16x4 b0_frag = {
          static_cast<int16_t>(Bs[wave_tile_base + 0 * TILE_ELEMS + base + 0]),
          static_cast<int16_t>(Bs[wave_tile_base + 0 * TILE_ELEMS + base + 1]),
          static_cast<int16_t>(Bs[wave_tile_base + 0 * TILE_ELEMS + base + 2]),
          static_cast<int16_t>(Bs[wave_tile_base + 0 * TILE_ELEMS + base + 3])};

      i16x4 b1_frag = {
          static_cast<int16_t>(Bs[wave_tile_base + 1 * TILE_ELEMS + base + 0]),
          static_cast<int16_t>(Bs[wave_tile_base + 1 * TILE_ELEMS + base + 1]),
          static_cast<int16_t>(Bs[wave_tile_base + 1 * TILE_ELEMS + base + 2]),
          static_cast<int16_t>(Bs[wave_tile_base + 1 * TILE_ELEMS + base + 3])};

      i16x4 b2_frag = {
          static_cast<int16_t>(Bs[wave_tile_base + 2 * TILE_ELEMS + base + 0]),
          static_cast<int16_t>(Bs[wave_tile_base + 2 * TILE_ELEMS + base + 1]),
          static_cast<int16_t>(Bs[wave_tile_base + 2 * TILE_ELEMS + base + 2]),
          static_cast<int16_t>(Bs[wave_tile_base + 2 * TILE_ELEMS + base + 3])};

      i16x4 b3_frag = {
          static_cast<int16_t>(Bs[wave_tile_base + 3 * TILE_ELEMS + base + 0]),
          static_cast<int16_t>(Bs[wave_tile_base + 3 * TILE_ELEMS + base + 1]),
          static_cast<int16_t>(Bs[wave_tile_base + 3 * TILE_ELEMS + base + 2]),
          static_cast<int16_t>(Bs[wave_tile_base + 3 * TILE_ELEMS + base + 3])};

      acc0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b0_frag, acc0, 0, 0, 0);
      acc1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b1_frag, acc1, 0, 0, 0);
      acc2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b2_frag, acc2, 0, 0, 0);
      acc3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b3_frag, acc3, 0, 0, 0);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t gm = block_m + out_m[i];
      const int64_t gn = wave_block_n + out_n[i];
      C[gm * N + (gn + 0 * TILE_N)] = float_to_bf16_rn(acc0[i]);
      C[gm * N + (gn + 1 * TILE_N)] = float_to_bf16_rn(acc1[i]);
      C[gm * N + (gn + 2 * TILE_N)] = float_to_bf16_rn(acc2[i]);
      C[gm * N + (gn + 3 * TILE_N)] = float_to_bf16_rn(acc3[i]);
    }
  } else {
    for (int64_t k0 = 0; k0 < k_full; k0 += TILE_K) {
      for (int t = tid; t < TILE_M * TILE_K; t += BLOCK_THREADS_BALANCED) {
        const int r = t >> 4;
        const int kk = t & 15;
        const int64_t gm = block_m + r;
        const int64_t gk = k0 + kk;
        uint16_t av = 0;
        if (gm < M) {
          av = A[gm * K + gk];
        }
        As[t] = av;
      }

      for (int t = tid; t < TILES_N_PER_BLOCK_BALANCED * TILE_ELEMS; t += BLOCK_THREADS_BALANCED) {
        const int tile = t / TILE_ELEMS;
        const int rem = t - tile * TILE_ELEMS;
        const int r = rem >> 4;
        const int kk = rem & 15;
        const int64_t gn = block_n + static_cast<int64_t>(tile * TILE_N + r);
        const int64_t gk = k0 + kk;
        uint16_t bv = 0;
        if (gn < N) {
          bv = B[gn * K + gk];
        }
        Bs[t] = bv;
      }

      __syncthreads();

      i16x4 a_frag = {
          static_cast<int16_t>(As[base + 0]),
          static_cast<int16_t>(As[base + 1]),
          static_cast<int16_t>(As[base + 2]),
          static_cast<int16_t>(As[base + 3])};

      i16x4 b0_frag = {
          static_cast<int16_t>(Bs[wave_tile_base + 0 * TILE_ELEMS + base + 0]),
          static_cast<int16_t>(Bs[wave_tile_base + 0 * TILE_ELEMS + base + 1]),
          static_cast<int16_t>(Bs[wave_tile_base + 0 * TILE_ELEMS + base + 2]),
          static_cast<int16_t>(Bs[wave_tile_base + 0 * TILE_ELEMS + base + 3])};

      i16x4 b1_frag = {
          static_cast<int16_t>(Bs[wave_tile_base + 1 * TILE_ELEMS + base + 0]),
          static_cast<int16_t>(Bs[wave_tile_base + 1 * TILE_ELEMS + base + 1]),
          static_cast<int16_t>(Bs[wave_tile_base + 1 * TILE_ELEMS + base + 2]),
          static_cast<int16_t>(Bs[wave_tile_base + 1 * TILE_ELEMS + base + 3])};

      i16x4 b2_frag = {
          static_cast<int16_t>(Bs[wave_tile_base + 2 * TILE_ELEMS + base + 0]),
          static_cast<int16_t>(Bs[wave_tile_base + 2 * TILE_ELEMS + base + 1]),
          static_cast<int16_t>(Bs[wave_tile_base + 2 * TILE_ELEMS + base + 2]),
          static_cast<int16_t>(Bs[wave_tile_base + 2 * TILE_ELEMS + base + 3])};

      i16x4 b3_frag = {
          static_cast<int16_t>(Bs[wave_tile_base + 3 * TILE_ELEMS + base + 0]),
          static_cast<int16_t>(Bs[wave_tile_base + 3 * TILE_ELEMS + base + 1]),
          static_cast<int16_t>(Bs[wave_tile_base + 3 * TILE_ELEMS + base + 2]),
          static_cast<int16_t>(Bs[wave_tile_base + 3 * TILE_ELEMS + base + 3])};

      acc0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b0_frag, acc0, 0, 0, 0);
      acc1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b1_frag, acc1, 0, 0, 0);
      acc2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b2_frag, acc2, 0, 0, 0);
      acc3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b3_frag, acc3, 0, 0, 0);

      __syncthreads();
    }

    if (k_full < K) {
      for (int t = tid; t < TILE_M * TILE_K; t += BLOCK_THREADS_BALANCED) {
        const int r = t >> 4;
        const int kk = t & 15;
        const int64_t gm = block_m + r;
        const int64_t gk = k_full + kk;
        uint16_t av = 0;
        if (gm < M && gk < K) {
          av = A[gm * K + gk];
        }
        As[t] = av;
      }

      for (int t = tid; t < TILES_N_PER_BLOCK_BALANCED * TILE_ELEMS; t += BLOCK_THREADS_BALANCED) {
        const int tile = t / TILE_ELEMS;
        const int rem = t - tile * TILE_ELEMS;
        const int r = rem >> 4;
        const int kk = rem & 15;
        const int64_t gn = block_n + static_cast<int64_t>(tile * TILE_N + r);
        const int64_t gk = k_full + kk;
        uint16_t bv = 0;
        if (gn < N && gk < K) {
          bv = B[gn * K + gk];
        }
        Bs[t] = bv;
      }

      __syncthreads();

      i16x4 a_frag = {
          static_cast<int16_t>(As[base + 0]),
          static_cast<int16_t>(As[base + 1]),
          static_cast<int16_t>(As[base + 2]),
          static_cast<int16_t>(As[base + 3])};

      i16x4 b0_frag = {
          static_cast<int16_t>(Bs[wave_tile_base + 0 * TILE_ELEMS + base + 0]),
          static_cast<int16_t>(Bs[wave_tile_base + 0 * TILE_ELEMS + base + 1]),
          static_cast<int16_t>(Bs[wave_tile_base + 0 * TILE_ELEMS + base + 2]),
          static_cast<int16_t>(Bs[wave_tile_base + 0 * TILE_ELEMS + base + 3])};

      i16x4 b1_frag = {
          static_cast<int16_t>(Bs[wave_tile_base + 1 * TILE_ELEMS + base + 0]),
          static_cast<int16_t>(Bs[wave_tile_base + 1 * TILE_ELEMS + base + 1]),
          static_cast<int16_t>(Bs[wave_tile_base + 1 * TILE_ELEMS + base + 2]),
          static_cast<int16_t>(Bs[wave_tile_base + 1 * TILE_ELEMS + base + 3])};

      i16x4 b2_frag = {
          static_cast<int16_t>(Bs[wave_tile_base + 2 * TILE_ELEMS + base + 0]),
          static_cast<int16_t>(Bs[wave_tile_base + 2 * TILE_ELEMS + base + 1]),
          static_cast<int16_t>(Bs[wave_tile_base + 2 * TILE_ELEMS + base + 2]),
          static_cast<int16_t>(Bs[wave_tile_base + 2 * TILE_ELEMS + base + 3])};

      i16x4 b3_frag = {
          static_cast<int16_t>(Bs[wave_tile_base + 3 * TILE_ELEMS + base + 0]),
          static_cast<int16_t>(Bs[wave_tile_base + 3 * TILE_ELEMS + base + 1]),
          static_cast<int16_t>(Bs[wave_tile_base + 3 * TILE_ELEMS + base + 2]),
          static_cast<int16_t>(Bs[wave_tile_base + 3 * TILE_ELEMS + base + 3])};

      acc0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b0_frag, acc0, 0, 0, 0);
      acc1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b1_frag, acc1, 0, 0, 0);
      acc2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b2_frag, acc2, 0, 0, 0);
      acc3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b3_frag, acc3, 0, 0, 0);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t gm = block_m + out_m[i];
      const int64_t gn = wave_block_n + out_n[i];
      if (gm < M) {
        if (gn + 0 * TILE_N < N) C[gm * N + (gn + 0 * TILE_N)] = float_to_bf16_rn(acc0[i]);
        if (gn + 1 * TILE_N < N) C[gm * N + (gn + 1 * TILE_N)] = float_to_bf16_rn(acc1[i]);
        if (gn + 2 * TILE_N < N) C[gm * N + (gn + 2 * TILE_N)] = float_to_bf16_rn(acc2[i]);
        if (gn + 3 * TILE_N < N) C[gm * N + (gn + 3 * TILE_N)] = float_to_bf16_rn(acc3[i]);
      }
    }
  }

#else
  scalar_fallback_tile(A, B, C, M, N, K, lane, block_m, wave_block_n + 0 * TILE_N);
  scalar_fallback_tile(A, B, C, M, N, K, lane, block_m, wave_block_n + 1 * TILE_N);
  scalar_fallback_tile(A, B, C, M, N, K, lane, block_m, wave_block_n + 2 * TILE_N);
  scalar_fallback_tile(A, B, C, M, N, K, lane, block_m, wave_block_n + 3 * TILE_N);
#endif
}

hipError_t ksearch_launch_gemm_bf16_var_mnk(
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    hipStream_t stream,
    const uint16_t* A,
    const uint16_t* B,
    uint16_t* C,
    int64_t M,
    int64_t N,
    int64_t K) {
  gemm_bf16_var_mnk_kernel<<<grid, block, shared_mem, stream>>>(A, B, C, M, N, K);
  return hipGetLastError();
}

hipError_t ksearch_launch_gemm_bf16_var_mnk_balanced(
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    hipStream_t stream,
    const uint16_t* A,
    const uint16_t* B,
    uint16_t* C,
    int64_t M,
    int64_t N,
    int64_t K) {
  gemm_bf16_var_mnk_balanced_kernel<<<grid, block, shared_mem, stream>>>(A, B, C, M, N, K);
  return hipGetLastError();
}