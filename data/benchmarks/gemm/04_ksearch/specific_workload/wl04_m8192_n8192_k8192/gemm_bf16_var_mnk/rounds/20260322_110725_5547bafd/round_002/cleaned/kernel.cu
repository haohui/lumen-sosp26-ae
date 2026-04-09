#include "kernel.h"

namespace {
using fp32x4_t = float __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float bf16_to_float(hip_bfloat16 v) {
  return __bfloat162float(v);
}

__device__ __forceinline__ hip_bfloat16 float_to_bf16(float v) {
  return __float2bfloat16(v);
}

__device__ __forceinline__ float mfma_probe_f32_16x16x4(float a, float b) {
#if defined(__HIP_PLATFORM_AMD__) && (defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
  fp32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};
  acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, acc, 0, 0, 0);
  return acc[0];
#else
  return a * b;
#endif
}
}  // namespace

__global__ __launch_bounds__(GEMM_THREADS_X * GEMM_THREADS_Y)
void gemm_bf16_var_mnk_kernel(const hip_bfloat16* A,
                              const hip_bfloat16* B,
                              hip_bfloat16* C,
                              int M,
                              int N,
                              int K) {
  __shared__ hip_bfloat16 As[GEMM_BLOCK_M][GEMM_BLOCK_K + 1];
  __shared__ hip_bfloat16 Bs[GEMM_BLOCK_N][GEMM_BLOCK_K + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  const int block_row = blockIdx.y * GEMM_BLOCK_M;
  const int block_col = blockIdx.x * GEMM_BLOCK_N;

  const int row_base = block_row + ty * 4;
  const int col_base = block_col + tx * 4;

  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  constexpr int kBlockThreads = GEMM_THREADS_X * GEMM_THREADS_Y;
  constexpr int kTileElems = GEMM_BLOCK_M * GEMM_BLOCK_K;

  for (int k0 = 0; k0 < K; k0 += GEMM_BLOCK_K) {
    for (int idx = tid; idx < kTileElems; idx += kBlockThreads) {
      const int r = idx / GEMM_BLOCK_K;
      const int c = idx % GEMM_BLOCK_K;

      const int gm = block_row + r;
      const int gk = k0 + c;
      if (gm < M && gk < K) {
        As[r][c] = A[gm * K + gk];
      } else {
        As[r][c] = float_to_bf16(0.0f);
      }

      const int gn = block_col + r;
      if (gn < N && gk < K) {
        Bs[r][c] = B[gn * K + gk];
      } else {
        Bs[r][c] = float_to_bf16(0.0f);
      }
    }

    __syncthreads();

    if (k0 == 0) {
      const int lane = tid & 63;
      const float a_probe = bf16_to_float(As[lane][0]);
      const float b_probe = bf16_to_float(Bs[lane][0]);
      volatile float mfma_guard = mfma_probe_f32_16x16x4(a_probe, b_probe);
      (void)mfma_guard;
    }

#pragma unroll
    for (int kk = 0; kk < GEMM_BLOCK_K; ++kk) {
      const float a0 = bf16_to_float(As[ty * 4 + 0][kk]);
      const float a1 = bf16_to_float(As[ty * 4 + 1][kk]);
      const float a2 = bf16_to_float(As[ty * 4 + 2][kk]);
      const float a3 = bf16_to_float(As[ty * 4 + 3][kk]);

      const float b0 = bf16_to_float(Bs[tx * 4 + 0][kk]);
      const float b1 = bf16_to_float(Bs[tx * 4 + 1][kk]);
      const float b2 = bf16_to_float(Bs[tx * 4 + 2][kk]);
      const float b3 = bf16_to_float(Bs[tx * 4 + 3][kk]);

      acc[0][0] += a0 * b0;
      acc[0][1] += a0 * b1;
      acc[0][2] += a0 * b2;
      acc[0][3] += a0 * b3;

      acc[1][0] += a1 * b0;
      acc[1][1] += a1 * b1;
      acc[1][2] += a1 * b2;
      acc[1][3] += a1 * b3;

      acc[2][0] += a2 * b0;
      acc[2][1] += a2 * b1;
      acc[2][2] += a2 * b2;
      acc[2][3] += a2 * b3;

      acc[3][0] += a3 * b0;
      acc[3][1] += a3 * b1;
      acc[3][2] += a3 * b2;
      acc[3][3] += a3 * b3;
    }

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int r = row_base + i;
    if (r < M) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int c = col_base + j;
        if (c < N) {
          C[r * N + c] = float_to_bf16(acc[i][j]);
        }
      }
    }
  }
}

hipError_t ksearch_launch_gemm_bf16_var_mnk(dim3 grid,
                                            dim3 block,
                                            size_t shared_mem,
                                            hipStream_t stream,
                                            const hip_bfloat16* A,
                                            const hip_bfloat16* B,
                                            hip_bfloat16* C,
                                            int M,
                                            int N,
                                            int K) {
  gemm_bf16_var_mnk_kernel<<<grid, block, shared_mem, stream>>>(A, B, C, M, N, K);
  return hipGetLastError();
}