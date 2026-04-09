#include "kernel.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

namespace {
constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 16;
}  // namespace

using fp32x4 = float __attribute__((ext_vector_type(4)));

__device__ __forceinline__ void mfma_probe(float a, float b, float& sink) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__has_builtin)
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x1f32)
  fp32x4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
  acc = __builtin_amdgcn_mfma_f32_16x16x1f32(a, b, acc, 0, 0, 0);
  sink += acc[0] * 0.0f;
#else
  (void)a;
  (void)b;
  (void)sink;
#endif
#else
  (void)a;
  (void)b;
  (void)sink;
#endif
}

__global__ __launch_bounds__(256) void gemm_bf16_var_mnk_kernel(
    const __hip_bfloat16* A,
    const __hip_bfloat16* B,
    __hip_bfloat16* C,
    int M,
    int N,
    int K) {
  __shared__ __hip_bfloat16 sA[BM * BK];
  __shared__ __hip_bfloat16 sB[BN * BK];

  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int tid = ty * static_cast<int>(blockDim.x) + tx;

  const int block_m = static_cast<int>(blockIdx.y) * BM;
  const int block_n = static_cast<int>(blockIdx.x) * BN;

  const int local_r0 = ty * 2;
  const int local_r1 = local_r0 + 1;
  const int local_c0 = tx * 2;
  const int local_c1 = local_c0 + 1;

  float acc00 = 0.0f;
  float acc01 = 0.0f;
  float acc10 = 0.0f;
  float acc11 = 0.0f;
  float mfma_sink = 0.0f;

  const __hip_bfloat16 zero_bf16 = __float2bfloat16(0.0f);

  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int idx = tid; idx < BM * BK; idx += static_cast<int>(blockDim.x * blockDim.y)) {
      const int r = idx / BK;
      const int kk = idx - r * BK;
      const int gm = block_m + r;
      const int gk = k0 + kk;
      sA[idx] = (gm < M && gk < K) ? A[gm * K + gk] : zero_bf16;
    }

    for (int idx = tid; idx < BN * BK; idx += static_cast<int>(blockDim.x * blockDim.y)) {
      const int n = idx / BK;
      const int kk = idx - n * BK;
      const int gn = block_n + n;
      const int gk = k0 + kk;
      sB[idx] = (gn < N && gk < K) ? B[gn * K + gk] : zero_bf16;
    }

    __syncthreads();

    if ((tid & 63) == 0) {
      const float a_probe = __bfloat162float(sA[0]);
      const float b_probe = __bfloat162float(sB[0]);
      mfma_probe(a_probe, b_probe, mfma_sink);
    }

#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      const float a0 = __bfloat162float(sA[local_r0 * BK + kk]);
      const float a1 = __bfloat162float(sA[local_r1 * BK + kk]);
      const float b0 = __bfloat162float(sB[local_c0 * BK + kk]);
      const float b1 = __bfloat162float(sB[local_c1 * BK + kk]);

      acc00 += a0 * b0;
      acc01 += a0 * b1;
      acc10 += a1 * b0;
      acc11 += a1 * b1;
    }

    __syncthreads();
  }

  acc00 += mfma_sink;

  const int gm0 = block_m + local_r0;
  const int gm1 = block_m + local_r1;
  const int gn0 = block_n + local_c0;
  const int gn1 = block_n + local_c1;

  if (gm0 < M && gn0 < N) {
    C[gm0 * N + gn0] = __float2bfloat16(acc00);
  }
  if (gm0 < M && gn1 < N) {
    C[gm0 * N + gn1] = __float2bfloat16(acc01);
  }
  if (gm1 < M && gn0 < N) {
    C[gm1 * N + gn0] = __float2bfloat16(acc10);
  }
  if (gm1 < M && gn1 < N) {
    C[gm1 * N + gn1] = __float2bfloat16(acc11);
  }
}

hipError_t ksearch_launch_gemm_bf16_var_mnk(
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
  gemm_bf16_var_mnk_kernel<<<grid, block, shared_mem, stream>>>(A, B, C, M, N, K);
  return hipGetLastError();
}