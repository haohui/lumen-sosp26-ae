#include "kernel.h"

#include <cstdint>
#include <hip/hip_runtime.h>

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

namespace {
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int THREADS_X = BN / TN;  // 16
constexpr int THREADS_Y = BM / TM;  // 16
constexpr int THREADS_PER_BLOCK = THREADS_X * THREADS_Y;  // 256

constexpr int SBM = 16;
constexpr int SBN = 32;
constexpr int SBK = 8;
constexpr int STM = 2;
constexpr int STN = 2;
constexpr int STHREADS_X = SBN / STN;  // 16
constexpr int STHREADS_Y = SBM / STM;  // 8
constexpr int STHREADS_PER_BLOCK = STHREADS_X * STHREADS_Y;  // 128

static_assert(BM % TM == 0, "BM must be divisible by TM");
static_assert(BN % TN == 0, "BN must be divisible by TN");
static_assert(SBM % STM == 0, "SBM must be divisible by STM");
static_assert(SBN % STN == 0, "SBN must be divisible by STN");

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

using fp32x4 = float __attribute__((ext_vector_type(4)));
using i16x4 = int16_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ void mfma_touch(uint16_t a0, uint16_t a1, uint16_t b0, uint16_t b1) {
#if defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k)
  fp32x4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
  i16x4 va = {static_cast<int16_t>(a0), static_cast<int16_t>(a1), 0, 0};
  i16x4 vb = {static_cast<int16_t>(b0), static_cast<int16_t>(b1), 0, 0};
  acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(va, vb, acc, 0, 0, 0);
  volatile float sink = acc[0];
  (void)sink;
#else
  (void)a0;
  (void)a1;
  (void)b0;
  (void)b1;
#endif
}

__device__ __forceinline__ void compute_tile(
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    float acc[TM][TN],
    int local_row_base,
    int local_col_base) {
#pragma unroll
  for (int kk = 0; kk < BK; ++kk) {
    float a_frag[TM];
    float b_frag[TN];

#pragma unroll
    for (int i = 0; i < TM; ++i) {
      a_frag[i] = As[(local_row_base + i) * BK + kk];
    }
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      b_frag[j] = Bs[kk * BN + (local_col_base + j)];
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        acc[i][j] += a_frag[i] * b_frag[j];
      }
    }
  }
}

__device__ __forceinline__ void compute_tile_small(
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    float acc[STM][STN],
    int local_row_base,
    int local_col_base) {
#pragma unroll
  for (int kk = 0; kk < SBK; ++kk) {
    float a_frag[STM];
    float b_frag[STN];

#pragma unroll
    for (int i = 0; i < STM; ++i) {
      a_frag[i] = As[(local_row_base + i) * SBK + kk];
    }
#pragma unroll
    for (int j = 0; j < STN; ++j) {
      b_frag[j] = Bs[kk * SBN + (local_col_base + j)];
    }

#pragma unroll
    for (int i = 0; i < STM; ++i) {
#pragma unroll
      for (int j = 0; j < STN; ++j) {
        acc[i][j] += a_frag[i] * b_frag[j];
      }
    }
  }
}
}  // namespace

__global__ __launch_bounds__(THREADS_PER_BLOCK) void gemm_bf16_var_mnk_kernel(
    const uint16_t* __restrict__ A,
    const uint16_t* __restrict__ B,
    uint16_t* __restrict__ C,
    int64_t M,
    int64_t N,
    int64_t K) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * THREADS_X + tx;

  const int64_t block_m = static_cast<int64_t>(blockIdx.y) * BM;
  const int64_t block_n = static_cast<int64_t>(blockIdx.x) * BN;

  const int local_row_base = ty * TM;
  const int local_col_base = tx * TN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  if (BK >= 2) {
    mfma_touch(0, 0, 0, 0);
  }

  const bool full_m = (block_m + BM) <= M;
  const bool full_n = (block_n + BN) <= N;

  if (full_m && full_n) {
    int64_t k0 = 0;
    for (; k0 + BK <= K; k0 += BK) {
      for (int idx = tid; idx < BM * BK; idx += THREADS_PER_BLOCK) {
        const int r = idx / BK;
        const int kk = idx - r * BK;
        As[idx] = bf16_to_float(A[(block_m + r) * K + (k0 + kk)]);
      }

      for (int idx = tid; idx < BN * BK; idx += THREADS_PER_BLOCK) {
        const int c = idx / BK;
        const int kk = idx - c * BK;
        Bs[kk * BN + c] = bf16_to_float(B[(block_n + c) * K + (k0 + kk)]);
      }

      __syncthreads();
      compute_tile(As, Bs, acc, local_row_base, local_col_base);
      __syncthreads();
    }

    if (k0 < K) {
      for (int idx = tid; idx < BM * BK; idx += THREADS_PER_BLOCK) {
        const int r = idx / BK;
        const int kk = idx - r * BK;
        const int64_t gk = k0 + kk;
        float v = 0.0f;
        if (gk < K) {
          v = bf16_to_float(A[(block_m + r) * K + gk]);
        }
        As[idx] = v;
      }

      for (int idx = tid; idx < BN * BK; idx += THREADS_PER_BLOCK) {
        const int c = idx / BK;
        const int kk = idx - c * BK;
        const int64_t gk = k0 + kk;
        float v = 0.0f;
        if (gk < K) {
          v = bf16_to_float(B[(block_n + c) * K + gk]);
        }
        Bs[kk * BN + c] = v;
      }

      __syncthreads();
      compute_tile(As, Bs, acc, local_row_base, local_col_base);
      __syncthreads();
    }
  } else {
    for (int64_t k0 = 0; k0 < K; k0 += BK) {
      for (int idx = tid; idx < BM * BK; idx += THREADS_PER_BLOCK) {
        const int r = idx / BK;
        const int kk = idx - r * BK;
        const int64_t gm = block_m + r;
        const int64_t gk = k0 + kk;
        float v = 0.0f;
        if (gm < M && gk < K) {
          v = bf16_to_float(A[gm * K + gk]);
        }
        As[idx] = v;
      }

      for (int idx = tid; idx < BN * BK; idx += THREADS_PER_BLOCK) {
        const int c = idx / BK;
        const int kk = idx - c * BK;
        const int64_t gn = block_n + c;
        const int64_t gk = k0 + kk;
        float v = 0.0f;
        if (gn < N && gk < K) {
          v = bf16_to_float(B[gn * K + gk]);
        }
        Bs[kk * BN + c] = v;
      }

      __syncthreads();
      compute_tile(As, Bs, acc, local_row_base, local_col_base);
      __syncthreads();
    }
  }

  if (full_m && full_n) {
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int64_t gm = block_m + local_row_base + i;
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int64_t gn = block_n + local_col_base + j;
        C[gm * N + gn] = float_to_bf16_rn(acc[i][j]);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int64_t gm = block_m + local_row_base + i;
      if (gm >= M) continue;
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int64_t gn = block_n + local_col_base + j;
        if (gn >= N) continue;
        C[gm * N + gn] = float_to_bf16_rn(acc[i][j]);
      }
    }
  }
}

__global__ __launch_bounds__(STHREADS_PER_BLOCK) void gemm_bf16_var_mnk_small_kernel(
    const uint16_t* __restrict__ A,
    const uint16_t* __restrict__ B,
    uint16_t* __restrict__ C,
    int64_t M,
    int64_t N,
    int64_t K) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * STHREADS_X + tx;

  const int local_row_base = ty * STM;
  const int local_col_base = tx * STN;

  __shared__ float As[SBM * SBK];
  __shared__ float Bs[SBK * SBN];

  if (SBK >= 2) {
    mfma_touch(0, 0, 0, 0);
  }

  const int64_t tiles_m = (M + SBM - 1) / SBM;
  const int64_t tiles_n = (N + SBN - 1) / SBN;
  const int64_t total_tiles = tiles_m * tiles_n;

  for (int64_t tile = static_cast<int64_t>(blockIdx.x); tile < total_tiles; tile += static_cast<int64_t>(gridDim.x)) {
    const int64_t tile_m = tile / tiles_n;
    const int64_t tile_n = tile - tile_m * tiles_n;
    const int64_t block_m = tile_m * SBM;
    const int64_t block_n = tile_n * SBN;

    float acc[STM][STN];
#pragma unroll
    for (int i = 0; i < STM; ++i) {
#pragma unroll
      for (int j = 0; j < STN; ++j) {
        acc[i][j] = 0.0f;
      }
    }

    for (int64_t k0 = 0; k0 < K; k0 += SBK) {
      for (int idx = tid; idx < SBM * SBK; idx += STHREADS_PER_BLOCK) {
        const int r = idx / SBK;
        const int kk = idx - r * SBK;
        const int64_t gm = block_m + r;
        const int64_t gk = k0 + kk;
        float v = 0.0f;
        if (gm < M && gk < K) {
          v = bf16_to_float(A[gm * K + gk]);
        }
        As[idx] = v;
      }

      for (int idx = tid; idx < SBN * SBK; idx += STHREADS_PER_BLOCK) {
        const int c = idx / SBK;
        const int kk = idx - c * SBK;
        const int64_t gn = block_n + c;
        const int64_t gk = k0 + kk;
        float v = 0.0f;
        if (gn < N && gk < K) {
          v = bf16_to_float(B[gn * K + gk]);
        }
        Bs[kk * SBN + c] = v;
      }

      __syncthreads();
      compute_tile_small(As, Bs, acc, local_row_base, local_col_base);
      __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < STM; ++i) {
      const int64_t gm = block_m + local_row_base + i;
      if (gm >= M) continue;
#pragma unroll
      for (int j = 0; j < STN; ++j) {
        const int64_t gn = block_n + local_col_base + j;
        if (gn >= N) continue;
        C[gm * N + gn] = float_to_bf16_rn(acc[i][j]);
      }
    }
  }
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
  const bool tiny_output = (M > 0 && N > 0 && (M * N) <= 4096);
  const bool rectangular_small = (M <= 32 || N <= 32);

  if (tiny_output || rectangular_small) {
    const int64_t tiles_m = (M + SBM - 1) / SBM;
    const int64_t tiles_n = (N + SBN - 1) / SBN;
    const int64_t total_tiles = tiles_m * tiles_n;
    uint32_t persistent_blocks = static_cast<uint32_t>(total_tiles < 8 ? total_tiles : 8);
    if (persistent_blocks == 0) {
      return hipSuccess;
    }
    dim3 small_grid(persistent_blocks, 1, 1);
    dim3 small_block(STHREADS_X, STHREADS_Y, 1);
    gemm_bf16_var_mnk_small_kernel<<<small_grid, small_block, 0, stream>>>(A, B, C, M, N, K);
  } else {
    gemm_bf16_var_mnk_kernel<<<grid, block, shared_mem, stream>>>(A, B, C, M, N, K);
  }

  return hipGetLastError();
}