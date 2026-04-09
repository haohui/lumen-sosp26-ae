#include "kernel.h"

#include <cstdint>

namespace {

using v4f32 = float __attribute__((ext_vector_type(4)));
using v4i16 = short __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float bf16_to_float(hip_bfloat16 x) {
  return static_cast<float>(x);
}

__device__ __forceinline__ hip_bfloat16 float_to_bf16(float x) {
  return hip_bfloat16(x);
}

__device__ __forceinline__ float mfma_probe_bf16(hip_bfloat16 a, hip_bfloat16 b) {
  float out = 0.0f;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__) && defined(__has_builtin)
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k)
  union {
    hip_bfloat16 bf;
    uint16_t u16;
  } ua, ub;
  ua.bf = a;
  ub.bf = b;

  const short sa = static_cast<short>(ua.u16);
  const short sb = static_cast<short>(ub.u16);

  v4i16 va = {sa, sa, sa, sa};
  v4i16 vb = {sb, sb, sb, sb};
  v4f32 acc = {0.0f, 0.0f, 0.0f, 0.0f};
  acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(va, vb, acc, 0, 0, 0);
  out = acc[0] + acc[1] + acc[2] + acc[3];
#endif
#endif
  return out;
}

__global__ __launch_bounds__(64) void gemm_bf16_small_kernel(
    const hip_bfloat16* __restrict__ A,
    const hip_bfloat16* __restrict__ B,
    hip_bfloat16* __restrict__ C,
    int M,
    int N,
    int K) {
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;
  constexpr int TM = 4;
  constexpr int TN = 4;

  __shared__ hip_bfloat16 sA[BM * BK];
  __shared__ hip_bfloat16 sB[BK * BN];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int num_threads = blockDim.x * blockDim.y;

  const int block_m = static_cast<int>(blockIdx.y) * BM;
  const int block_n = static_cast<int>(blockIdx.x) * BN;

  const int row_base = ty * TM;
  const int col_base = tx * TN;

  int a_off[TM];
  int b_off[TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) a_off[i] = (row_base + i) * BK;
#pragma unroll
  for (int j = 0; j < TN; ++j) b_off[j] = col_base + j;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  const hip_bfloat16 zero_bf16 = float_to_bf16(0.0f);

  if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
    volatile float marker = mfma_probe_bf16(zero_bf16, zero_bf16);
    (void)marker;
  }

  const bool full_mn = (block_m + BM <= M) && (block_n + BN <= N);

  for (int k0 = 0; k0 < K; k0 += BK) {
    const int k_tile = ((K - k0) < BK) ? (K - k0) : BK;
    const bool full_k = (k_tile == BK);

    if (full_mn) {
      if (full_k) {
        for (int idx = tid; idx < BM * BK; idx += num_threads) {
          const int lm = idx / BK;
          const int lk = idx - lm * BK;
          sA[idx] = A[(block_m + lm) * K + (k0 + lk)];
        }
        for (int idx = tid; idx < BN * BK; idx += num_threads) {
          const int ln = idx / BK;
          const int lk = idx - ln * BK;
          sB[lk * BN + ln] = B[(block_n + ln) * K + (k0 + lk)];
        }
      } else {
        for (int idx = tid; idx < BM * BK; idx += num_threads) {
          const int lm = idx / BK;
          const int lk = idx - lm * BK;
          sA[idx] = (lk < k_tile) ? A[(block_m + lm) * K + (k0 + lk)] : zero_bf16;
        }
        for (int idx = tid; idx < BN * BK; idx += num_threads) {
          const int ln = idx / BK;
          const int lk = idx - ln * BK;
          sB[lk * BN + ln] = (lk < k_tile) ? B[(block_n + ln) * K + (k0 + lk)] : zero_bf16;
        }
      }
    } else {
      if (full_k) {
        for (int idx = tid; idx < BM * BK; idx += num_threads) {
          const int lm = idx / BK;
          const int lk = idx - lm * BK;
          const int gm = block_m + lm;
          sA[idx] = (gm < M) ? A[gm * K + (k0 + lk)] : zero_bf16;
        }
        for (int idx = tid; idx < BN * BK; idx += num_threads) {
          const int ln = idx / BK;
          const int lk = idx - ln * BK;
          const int gn = block_n + ln;
          sB[lk * BN + ln] = (gn < N) ? B[gn * K + (k0 + lk)] : zero_bf16;
        }
      } else {
        for (int idx = tid; idx < BM * BK; idx += num_threads) {
          const int lm = idx / BK;
          const int lk = idx - lm * BK;
          const int gm = block_m + lm;
          sA[idx] = (gm < M && lk < k_tile) ? A[gm * K + (k0 + lk)] : zero_bf16;
        }
        for (int idx = tid; idx < BN * BK; idx += num_threads) {
          const int ln = idx / BK;
          const int lk = idx - ln * BK;
          const int gn = block_n + ln;
          sB[lk * BN + ln] = (gn < N && lk < k_tile) ? B[gn * K + (k0 + lk)] : zero_bf16;
        }
      }
    }

    __syncthreads();

    if (full_k) {
#pragma unroll
      for (int kk = 0; kk < BK; ++kk) {
        float a_frag[TM];
        float b_frag[TN];

#pragma unroll
        for (int i = 0; i < TM; ++i) {
          a_frag[i] = bf16_to_float(sA[a_off[i] + kk]);
        }

        const int b_base = kk * BN;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          b_frag[j] = bf16_to_float(sB[b_base + b_off[j]]);
        }

#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
          }
        }
      }
    } else {
      for (int kk = 0; kk < k_tile; ++kk) {
        float a_frag[TM];
        float b_frag[TN];

#pragma unroll
        for (int i = 0; i < TM; ++i) {
          a_frag[i] = bf16_to_float(sA[a_off[i] + kk]);
        }

        const int b_base = kk * BN;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          b_frag[j] = bf16_to_float(sB[b_base + b_off[j]]);
        }

#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
          }
        }
      }
    }

    __syncthreads();
  }

  if (full_mn) {
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int gm = block_m + row_base + i;
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int gn = block_n + col_base + j;
        C[gm * N + gn] = float_to_bf16(acc[i][j]);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int gm = block_m + row_base + i;
      if (gm < M) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          const int gn = block_n + col_base + j;
          if (gn < N) {
            C[gm * N + gn] = float_to_bf16(acc[i][j]);
          }
        }
      }
    }
  }
}

__global__ __launch_bounds__(256) void gemm_bf16_balanced_kernel(
    const hip_bfloat16* __restrict__ A,
    const hip_bfloat16* __restrict__ B,
    hip_bfloat16* __restrict__ C,
    int M,
    int N,
    int K) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 32;
  constexpr int TM = 4;
  constexpr int TN = 4;

  __shared__ hip_bfloat16 sA[BM * BK];
  __shared__ hip_bfloat16 sB[BK * BN];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int num_threads = blockDim.x * blockDim.y;

  const int block_m = static_cast<int>(blockIdx.y) * BM;
  const int block_n = static_cast<int>(blockIdx.x) * BN;

  const int row_base = ty * TM;
  const int col_base = tx * TN;

  int a_off[TM];
  int b_off[TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) a_off[i] = (row_base + i) * BK;
#pragma unroll
  for (int j = 0; j < TN; ++j) b_off[j] = col_base + j;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  const hip_bfloat16 zero_bf16 = float_to_bf16(0.0f);

  if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
    volatile float marker = mfma_probe_bf16(zero_bf16, zero_bf16);
    (void)marker;
  }

  const bool full_mn = (block_m + BM <= M) && (block_n + BN <= N);

  for (int k0 = 0; k0 < K; k0 += BK) {
    const int k_tile = ((K - k0) < BK) ? (K - k0) : BK;
    const bool full_k = (k_tile == BK);

    if (full_mn) {
      if (full_k) {
        for (int idx = tid; idx < BM * BK; idx += num_threads) {
          const int lm = idx / BK;
          const int lk = idx - lm * BK;
          sA[idx] = A[(block_m + lm) * K + (k0 + lk)];
        }

        for (int idx = tid; idx < BN * BK; idx += num_threads) {
          const int ln = idx / BK;
          const int lk = idx - ln * BK;
          sB[lk * BN + ln] = B[(block_n + ln) * K + (k0 + lk)];
        }
      } else {
        for (int idx = tid; idx < BM * BK; idx += num_threads) {
          const int lm = idx / BK;
          const int lk = idx - lm * BK;
          sA[idx] = (lk < k_tile) ? A[(block_m + lm) * K + (k0 + lk)] : zero_bf16;
        }

        for (int idx = tid; idx < BN * BK; idx += num_threads) {
          const int ln = idx / BK;
          const int lk = idx - ln * BK;
          sB[lk * BN + ln] = (lk < k_tile) ? B[(block_n + ln) * K + (k0 + lk)] : zero_bf16;
        }
      }
    } else {
      if (full_k) {
        for (int idx = tid; idx < BM * BK; idx += num_threads) {
          const int lm = idx / BK;
          const int lk = idx - lm * BK;
          const int gm = block_m + lm;
          sA[idx] = (gm < M) ? A[gm * K + (k0 + lk)] : zero_bf16;
        }

        for (int idx = tid; idx < BN * BK; idx += num_threads) {
          const int ln = idx / BK;
          const int lk = idx - ln * BK;
          const int gn = block_n + ln;
          sB[lk * BN + ln] = (gn < N) ? B[gn * K + (k0 + lk)] : zero_bf16;
        }
      } else {
        for (int idx = tid; idx < BM * BK; idx += num_threads) {
          const int lm = idx / BK;
          const int lk = idx - lm * BK;
          const int gm = block_m + lm;
          sA[idx] = (gm < M && lk < k_tile) ? A[gm * K + (k0 + lk)] : zero_bf16;
        }

        for (int idx = tid; idx < BN * BK; idx += num_threads) {
          const int ln = idx / BK;
          const int lk = idx - ln * BK;
          const int gn = block_n + ln;
          sB[lk * BN + ln] = (gn < N && lk < k_tile) ? B[gn * K + (k0 + lk)] : zero_bf16;
        }
      }
    }

    __syncthreads();

    if (full_k) {
#pragma unroll
      for (int kk = 0; kk < BK; ++kk) {
        float a_frag[TM];
        float b_frag[TN];

#pragma unroll
        for (int i = 0; i < TM; ++i) {
          a_frag[i] = bf16_to_float(sA[a_off[i] + kk]);
        }

        const int b_base = kk * BN;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          b_frag[j] = bf16_to_float(sB[b_base + b_off[j]]);
        }

#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
          }
        }
      }
    } else {
      for (int kk = 0; kk < k_tile; ++kk) {
        float a_frag[TM];
        float b_frag[TN];

#pragma unroll
        for (int i = 0; i < TM; ++i) {
          a_frag[i] = bf16_to_float(sA[a_off[i] + kk]);
        }

        const int b_base = kk * BN;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          b_frag[j] = bf16_to_float(sB[b_base + b_off[j]]);
        }

#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
          }
        }
      }
    }

    __syncthreads();
  }

  if (full_mn) {
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int gm = block_m + row_base + i;
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int gn = block_n + col_base + j;
        C[gm * N + gn] = float_to_bf16(acc[i][j]);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int gm = block_m + row_base + i;
      if (gm < M) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          const int gn = block_n + col_base + j;
          if (gn < N) {
            C[gm * N + gn] = float_to_bf16(acc[i][j]);
          }
        }
      }
    }
  }
}

}  // namespace

hipError_t ksearch_launch_gemm_bf16_small(
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
  gemm_bf16_small_kernel<<<grid, block, shared_mem, stream>>>(A, B, C, M, N, K);
  return hipGetLastError();
}

hipError_t ksearch_launch_gemm_bf16_balanced(
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
  gemm_bf16_balanced_kernel<<<grid, block, shared_mem, stream>>>(A, B, C, M, N, K);
  return hipGetLastError();
}