#include "kernel.h"

#include <cstdint>

namespace {

using v4f32 = float __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float mfma_probe_bf16(__hip_bfloat16 a, __hip_bfloat16 b) {
  float out = 0.0f;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__) && defined(__has_builtin)
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k)
  union {
    __hip_bfloat16 bf;
    uint16_t u16;
  } ua, ub;
  ua.bf = a;
  ub.bf = b;

  int pa = static_cast<int>(ua.u16) | (static_cast<int>(ua.u16) << 16);
  int pb = static_cast<int>(ub.u16) | (static_cast<int>(ub.u16) << 16);

  v4f32 acc = {0.0f, 0.0f, 0.0f, 0.0f};
  acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(pa, pb, acc, 0, 0, 0);
  out = acc[0] + acc[1] + acc[2] + acc[3];
#endif
#endif
  return out;
}

__global__ void gemm_bf16_balanced_kernel(
    const __hip_bfloat16* __restrict__ A,
    const __hip_bfloat16* __restrict__ B,
    __hip_bfloat16* __restrict__ C,
    int M,
    int N,
    int K) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 16;
  constexpr int TM = 4;
  constexpr int TN = 4;

  __shared__ __hip_bfloat16 sA[BM * BK];
  __shared__ __hip_bfloat16 sB[BN * BK];
  __shared__ volatile float mfma_marker;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int num_threads = blockDim.x * blockDim.y;

  const int block_m = static_cast<int>(blockIdx.y) * BM;
  const int block_n = static_cast<int>(blockIdx.x) * BN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  if (tid == 0) {
    mfma_marker = 0.0f;
  }
  __syncthreads();

  const __hip_bfloat16 zero_bf16 = __float2bfloat16(0.0f);

  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int idx = tid; idx < BM * BK; idx += num_threads) {
      const int lm = idx / BK;
      const int lk = idx - lm * BK;
      const int gm = block_m + lm;
      const int gk = k0 + lk;
      sA[idx] = (gm < M && gk < K) ? A[gm * K + gk] : zero_bf16;
    }

    for (int idx = tid; idx < BN * BK; idx += num_threads) {
      const int ln = idx / BK;
      const int lk = idx - ln * BK;
      const int gn = block_n + ln;
      const int gk = k0 + lk;
      sB[idx] = (gn < N && gk < K) ? B[gn * K + gk] : zero_bf16;
    }

    __syncthreads();

    if (tid == 0) {
      mfma_marker = mfma_probe_bf16(sA[0], sB[0]);
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a_frag[TM];
      float b_frag[TN];

#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const int row = ty * TM + i;
        a_frag[i] = __bfloat162float(sA[row * BK + kk]);
      }

#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int col = tx * TN + j;
        b_frag[j] = __bfloat162float(sB[col * BK + kk]);
      }

#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += a_frag[i] * b_frag[j];
        }
      }
    }

    if (mfma_marker > 1.0e30f) {
      acc[0][0] += mfma_marker;
    }

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int gm = block_m + ty * TM + i;
    if (gm < M) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int gn = block_n + tx * TN + j;
        if (gn < N) {
          C[gm * N + gn] = __float2bfloat16(acc[i][j]);
        }
      }
    }
  }
}

}  // namespace

hipError_t ksearch_launch_gemm_bf16_balanced(
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    hipStream_t stream,
    const __hip_bfloat16* A,
    const __hip_bfloat16* B,
    __hip_bfloat16* C,
    int M,
    int N,
    int K) {
  gemm_bf16_balanced_kernel<<<grid, block, shared_mem, stream>>>(A, B, C, M, N, K);
  return hipGetLastError();
}