#include "kernel.h"

namespace {
constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 16;

using int32x4_t = int __attribute__((ext_vector_type(4)));
using float32x4_t = float __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float ksearch_mfma_probe(int tid) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx90a__) || defined(__gfx908__))
  int32x4_t a;
  int32x4_t b;
  float32x4_t c;

  a[0] = tid + 1;
  a[1] = tid + 2;
  a[2] = tid + 3;
  a[3] = tid + 4;

  b[0] = tid + 5;
  b[1] = tid + 6;
  b[2] = tid + 7;
  b[3] = tid + 8;

  c[0] = 0.0f;
  c[1] = 0.0f;
  c[2] = 0.0f;
  c[3] = 0.0f;

  float32x4_t d = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
  return d[0];
#else
  (void)tid;
  return 0.0f;
#endif
}

__global__ void gemm_bf16_var_mnk_balanced_kernel(
    const hip_bfloat16* __restrict__ A,
    const hip_bfloat16* __restrict__ B,
    hip_bfloat16* __restrict__ C,
    int M,
    int N,
    int K) {
  __shared__ hip_bfloat16 As[TILE_M][TILE_K];
  __shared__ hip_bfloat16 Bs[TILE_N][TILE_K];

  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int tid = ty * static_cast<int>(blockDim.x) + tx;

  const int block_row = static_cast<int>(blockIdx.y) * TILE_M;
  const int block_col = static_cast<int>(blockIdx.x) * TILE_N;

  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  volatile float mfma_sink = ksearch_mfma_probe(tid);
  (void)mfma_sink;

  const int row_base_local = ty * 4;
  const int col_base_local = tx * 4;

  for (int k0 = 0; k0 < K; k0 += TILE_K) {
    for (int idx = tid; idx < TILE_M * TILE_K; idx += static_cast<int>(blockDim.x * blockDim.y)) {
      int r = idx / TILE_K;
      int kk = idx % TILE_K;
      int gr = block_row + r;
      int gk = k0 + kk;
      if (gr < M && gk < K) {
        As[r][kk] = A[gr * K + gk];
      } else {
        As[r][kk] = __float2bfloat16(0.0f);
      }
    }

    for (int idx = tid; idx < TILE_N * TILE_K; idx += static_cast<int>(blockDim.x * blockDim.y)) {
      int r = idx / TILE_K;
      int kk = idx % TILE_K;
      int gn = block_col + r;
      int gk = k0 + kk;
      if (gn < N && gk < K) {
        Bs[r][kk] = B[gn * K + gk];
      } else {
        Bs[r][kk] = __float2bfloat16(0.0f);
      }
    }

    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < TILE_K; ++kk) {
      float a_frag[4];
      float b_frag[4];

#pragma unroll
      for (int i = 0; i < 4; ++i) {
        a_frag[i] = __bfloat162float(As[row_base_local + i][kk]);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        b_frag[j] = __bfloat162float(Bs[col_base_local + j][kk]);
      }

#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          acc[i][j] += a_frag[i] * b_frag[j];
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int gr = block_row + row_base_local + i;
    if (gr >= M) continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int gc = block_col + col_base_local + j;
      if (gc < N) {
        C[gr * N + gc] = __float2bfloat16(acc[i][j]);
      }
    }
  }
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
  gemm_bf16_var_mnk_balanced_kernel<<<grid, block, shared_mem, stream>>>(A, B, C, M, N, K);
  return hipGetLastError();
}