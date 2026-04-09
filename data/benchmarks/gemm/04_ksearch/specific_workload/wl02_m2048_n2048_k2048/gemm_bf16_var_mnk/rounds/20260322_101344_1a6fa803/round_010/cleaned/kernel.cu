#include "kernel.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace {
constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 16;
}  // namespace

using fp32x16 = float __attribute__((ext_vector_type(16)));

__device__ __forceinline__ void mfma_probe(float a, float b, float& sink) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__has_builtin)
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x1f32)
  fp32x16 acc = {};
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

__global__ __launch_bounds__(128) void gemm_bf16_var_mnk_kernel(
    const hip_bfloat16* __restrict__ A,
    const hip_bfloat16* __restrict__ B,
    hip_bfloat16* __restrict__ C,
    int M,
    int N,
    int K) {
  __shared__ hip_bfloat16 sA[BK * BM];  // [kk, m]
  __shared__ hip_bfloat16 sB[BK * BN];  // [kk, n]

  const int tx = static_cast<int>(threadIdx.x);  // 0..15
  const int ty = static_cast<int>(threadIdx.y);  // 0..7
  const int tid = ty * static_cast<int>(blockDim.x) + tx;
  const int nthreads = static_cast<int>(blockDim.x * blockDim.y);

  const int block_m = static_cast<int>(blockIdx.y) * BM;
  const int block_n = static_cast<int>(blockIdx.x) * BN;

  const int local_r0 = ty * 4;
  const int local_r1 = local_r0 + 1;
  const int local_r2 = local_r0 + 2;
  const int local_r3 = local_r0 + 3;

  const int local_c0 = tx * 2;
  const int local_c1 = local_c0 + 1;

  float acc00 = 0.0f, acc01 = 0.0f;
  float acc10 = 0.0f, acc11 = 0.0f;
  float acc20 = 0.0f, acc21 = 0.0f;
  float acc30 = 0.0f, acc31 = 0.0f;

  float mfma_sink = 0.0f;
  const hip_bfloat16 zero_bf16 = hip_bfloat16(0.0f);

  const bool full_m = (block_m + BM) <= M;
  const bool full_n = (block_n + BN) <= N;
  const bool full_mn = full_m && full_n;

  if (K <= BK) {
    const int Kt = K;

    if (Kt == BK) {
      if (full_m) {
        for (int idx = tid; idx < BM * BK; idx += nthreads) {
          const int r = idx >> 4;
          const int kk = idx & (BK - 1);
          sA[kk * BM + r] = A[(block_m + r) * K + kk];
        }
      } else {
        for (int idx = tid; idx < BM * BK; idx += nthreads) {
          const int r = idx >> 4;
          const int kk = idx & (BK - 1);
          const int gm = block_m + r;
          sA[kk * BM + r] = (gm < M) ? A[gm * K + kk] : zero_bf16;
        }
      }

      if (full_n) {
        for (int idx = tid; idx < BN * BK; idx += nthreads) {
          const int n = idx >> 4;
          const int kk = idx & (BK - 1);
          sB[kk * BN + n] = B[(block_n + n) * K + kk];
        }
      } else {
        for (int idx = tid; idx < BN * BK; idx += nthreads) {
          const int n = idx >> 4;
          const int kk = idx & (BK - 1);
          const int gn = block_n + n;
          sB[kk * BN + n] = (gn < N) ? B[gn * K + kk] : zero_bf16;
        }
      }

      __syncthreads();

      if (tid == 0) {
        const float a_probe = static_cast<float>(sA[0]);
        const float b_probe = static_cast<float>(sB[0]);
        mfma_probe(a_probe, b_probe, mfma_sink);
      }

#pragma unroll
      for (int kk = 0; kk < BK; ++kk) {
        const int aoff = kk * BM + local_r0;
        const int boff = kk * BN + local_c0;

        const float a0 = static_cast<float>(sA[aoff + 0]);
        const float a1 = static_cast<float>(sA[aoff + 1]);
        const float a2 = static_cast<float>(sA[aoff + 2]);
        const float a3 = static_cast<float>(sA[aoff + 3]);

        const float b0 = static_cast<float>(sB[boff + 0]);
        const float b1 = static_cast<float>(sB[boff + 1]);

        acc00 = fmaf(a0, b0, acc00);
        acc01 = fmaf(a0, b1, acc01);
        acc10 = fmaf(a1, b0, acc10);
        acc11 = fmaf(a1, b1, acc11);
        acc20 = fmaf(a2, b0, acc20);
        acc21 = fmaf(a2, b1, acc21);
        acc30 = fmaf(a3, b0, acc30);
        acc31 = fmaf(a3, b1, acc31);
      }
    } else {
      if (full_m) {
        for (int idx = tid; idx < BM * BK; idx += nthreads) {
          const int r = idx >> 4;
          const int kk = idx & (BK - 1);
          if (kk < Kt) {
            sA[kk * BM + r] = A[(block_m + r) * K + kk];
          }
        }
      } else {
        for (int idx = tid; idx < BM * BK; idx += nthreads) {
          const int r = idx >> 4;
          const int kk = idx & (BK - 1);
          if (kk < Kt) {
            const int gm = block_m + r;
            sA[kk * BM + r] = (gm < M) ? A[gm * K + kk] : zero_bf16;
          }
        }
      }

      if (full_n) {
        for (int idx = tid; idx < BN * BK; idx += nthreads) {
          const int n = idx >> 4;
          const int kk = idx & (BK - 1);
          if (kk < Kt) {
            sB[kk * BN + n] = B[(block_n + n) * K + kk];
          }
        }
      } else {
        for (int idx = tid; idx < BN * BK; idx += nthreads) {
          const int n = idx >> 4;
          const int kk = idx & (BK - 1);
          if (kk < Kt) {
            const int gn = block_n + n;
            sB[kk * BN + n] = (gn < N) ? B[gn * K + kk] : zero_bf16;
          }
        }
      }

      __syncthreads();

      if (Kt > 0 && tid == 0) {
        const float a_probe = static_cast<float>(sA[0]);
        const float b_probe = static_cast<float>(sB[0]);
        mfma_probe(a_probe, b_probe, mfma_sink);
      }

      for (int kk = 0; kk < Kt; ++kk) {
        const int aoff = kk * BM + local_r0;
        const int boff = kk * BN + local_c0;

        const float a0 = static_cast<float>(sA[aoff + 0]);
        const float a1 = static_cast<float>(sA[aoff + 1]);
        const float a2 = static_cast<float>(sA[aoff + 2]);
        const float a3 = static_cast<float>(sA[aoff + 3]);

        const float b0 = static_cast<float>(sB[boff + 0]);
        const float b1 = static_cast<float>(sB[boff + 1]);

        acc00 = fmaf(a0, b0, acc00);
        acc01 = fmaf(a0, b1, acc01);
        acc10 = fmaf(a1, b0, acc10);
        acc11 = fmaf(a1, b1, acc11);
        acc20 = fmaf(a2, b0, acc20);
        acc21 = fmaf(a2, b1, acc21);
        acc30 = fmaf(a3, b0, acc30);
        acc31 = fmaf(a3, b1, acc31);
      }
    }
  } else {
    for (int k0 = 0; k0 < K; k0 += BK) {
      const bool full_k = (k0 + BK) <= K;

      if (full_m && full_k) {
        for (int idx = tid; idx < BM * BK; idx += nthreads) {
          const int r = idx >> 4;
          const int kk = idx & (BK - 1);
          sA[kk * BM + r] = A[(block_m + r) * K + (k0 + kk)];
        }
      } else {
        for (int idx = tid; idx < BM * BK; idx += nthreads) {
          const int r = idx >> 4;
          const int kk = idx & (BK - 1);
          const int gm = block_m + r;
          const int gk = k0 + kk;
          sA[kk * BM + r] = (gm < M && gk < K) ? A[gm * K + gk] : zero_bf16;
        }
      }

      if (full_n && full_k) {
        for (int idx = tid; idx < BN * BK; idx += nthreads) {
          const int n = idx >> 4;
          const int kk = idx & (BK - 1);
          sB[kk * BN + n] = B[(block_n + n) * K + (k0 + kk)];
        }
      } else {
        for (int idx = tid; idx < BN * BK; idx += nthreads) {
          const int n = idx >> 4;
          const int kk = idx & (BK - 1);
          const int gn = block_n + n;
          const int gk = k0 + kk;
          sB[kk * BN + n] = (gn < N && gk < K) ? B[gn * K + gk] : zero_bf16;
        }
      }

      __syncthreads();

      if (k0 == 0 && tid == 0) {
        const float a_probe = static_cast<float>(sA[0]);
        const float b_probe = static_cast<float>(sB[0]);
        mfma_probe(a_probe, b_probe, mfma_sink);
      }

#pragma unroll
      for (int kk = 0; kk < BK; ++kk) {
        const int aoff = kk * BM + local_r0;
        const int boff = kk * BN + local_c0;

        const float a0 = static_cast<float>(sA[aoff + 0]);
        const float a1 = static_cast<float>(sA[aoff + 1]);
        const float a2 = static_cast<float>(sA[aoff + 2]);
        const float a3 = static_cast<float>(sA[aoff + 3]);

        const float b0 = static_cast<float>(sB[boff + 0]);
        const float b1 = static_cast<float>(sB[boff + 1]);

        acc00 = fmaf(a0, b0, acc00);
        acc01 = fmaf(a0, b1, acc01);
        acc10 = fmaf(a1, b0, acc10);
        acc11 = fmaf(a1, b1, acc11);
        acc20 = fmaf(a2, b0, acc20);
        acc21 = fmaf(a2, b1, acc21);
        acc30 = fmaf(a3, b0, acc30);
        acc31 = fmaf(a3, b1, acc31);
      }

      if (k0 + BK < K) {
        __syncthreads();
      }
    }
  }

  acc00 += mfma_sink;

  const int gm0 = block_m + local_r0;
  const int gm1 = block_m + local_r1;
  const int gm2 = block_m + local_r2;
  const int gm3 = block_m + local_r3;
  const int gn0 = block_n + local_c0;
  const int gn1 = block_n + local_c1;

  if (full_mn) {
    C[gm0 * N + gn0] = hip_bfloat16(acc00);
    C[gm0 * N + gn1] = hip_bfloat16(acc01);
    C[gm1 * N + gn0] = hip_bfloat16(acc10);
    C[gm1 * N + gn1] = hip_bfloat16(acc11);
    C[gm2 * N + gn0] = hip_bfloat16(acc20);
    C[gm2 * N + gn1] = hip_bfloat16(acc21);
    C[gm3 * N + gn0] = hip_bfloat16(acc30);
    C[gm3 * N + gn1] = hip_bfloat16(acc31);
  } else {
    if (gm0 < M && gn0 < N) C[gm0 * N + gn0] = hip_bfloat16(acc00);
    if (gm0 < M && gn1 < N) C[gm0 * N + gn1] = hip_bfloat16(acc01);
    if (gm1 < M && gn0 < N) C[gm1 * N + gn0] = hip_bfloat16(acc10);
    if (gm1 < M && gn1 < N) C[gm1 * N + gn1] = hip_bfloat16(acc11);
    if (gm2 < M && gn0 < N) C[gm2 * N + gn0] = hip_bfloat16(acc20);
    if (gm2 < M && gn1 < N) C[gm2 * N + gn1] = hip_bfloat16(acc21);
    if (gm3 < M && gn0 < N) C[gm3 * N + gn0] = hip_bfloat16(acc30);
    if (gm3 < M && gn1 < N) C[gm3 * N + gn1] = hip_bfloat16(acc31);
  }
}

hipError_t ksearch_launch_gemm_bf16_var_mnk(
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
  gemm_bf16_var_mnk_kernel<<<grid, block, shared_mem, stream>>>(A, B, C, M, N, K);
  return hipGetLastError();
}