#include "kernel.h"
#include <cstdio>
#include <cstdint>

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

namespace {

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 32;

constexpr int BLOCK_THREADS_X = 16;
constexpr int BLOCK_THREADS_Y = 16;
constexpr int THREADS_PER_BLOCK = BLOCK_THREADS_X * BLOCK_THREADS_Y;

constexpr int TM = 4;
constexpr int TN = 4;

using fp32x4_t = float __attribute__((vector_size(16)));

__device__ __forceinline__ float bf16_to_float(hip_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ hip_bfloat16 float_to_bf16(float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ uint16_t bf16_bits(hip_bfloat16 v) {
    union {
        hip_bfloat16 b;
        uint16_t u;
    } cvt;
    cvt.b = v;
    return cvt.u;
}

__device__ __forceinline__ void mfma_probe(
    volatile float& sink,
    hip_bfloat16 a0,
    hip_bfloat16 a1,
    hip_bfloat16 b0,
    hip_bfloat16 b1) {
#if defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k)
    const uint32_t pa = static_cast<uint32_t>(bf16_bits(a0)) |
                        (static_cast<uint32_t>(bf16_bits(a1)) << 16);
    const uint32_t pb = static_cast<uint32_t>(bf16_bits(b0)) |
                        (static_cast<uint32_t>(bf16_bits(b1)) << 16);
    fp32x4_t c = {0.0f, 0.0f, 0.0f, 0.0f};
    c = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
        static_cast<int>(pa), static_cast<int>(pb), c, 0, 0, 0);
    sink += c[0];
#else
    sink += bf16_to_float(a0) * bf16_to_float(b0);
#endif
}

}  // namespace

extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK) void gemm_bf16_balanced_mfma_kernel(
    const hip_bfloat16* __restrict__ A,
    const hip_bfloat16* __restrict__ B,
    hip_bfloat16* __restrict__ C,
    int64_t M,
    int64_t N,
    int64_t K) {
    __shared__ hip_bfloat16 As[BLOCK_M][BLOCK_K];
    __shared__ hip_bfloat16 Bs[BLOCK_N][BLOCK_K];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int load_stride = blockDim.x * blockDim.y;

    const int64_t m0 = static_cast<int64_t>(blockIdx.y) * BLOCK_M;
    const int64_t n0 = static_cast<int64_t>(blockIdx.x) * BLOCK_N;

    const int row_base = ty * TM;
    const int col_base = tx * TN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    volatile float mfma_sink = 0.0f;

    for (int64_t k0 = 0; k0 < K; k0 += BLOCK_K) {
        for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += load_stride) {
            const int r = idx / BLOCK_K;
            const int c = idx - r * BLOCK_K;
            const int64_t gm = m0 + r;
            const int64_t gk = k0 + c;
            As[r][c] = (gm < M && gk < K) ? A[gm * K + gk] : float_to_bf16(0.0f);
        }

        for (int idx = tid; idx < BLOCK_N * BLOCK_K; idx += load_stride) {
            const int r = idx / BLOCK_K;
            const int c = idx - r * BLOCK_K;
            const int64_t gn = n0 + r;
            const int64_t gk = k0 + c;
            Bs[r][c] = (gn < N && gk < K) ? B[gn * K + gk] : float_to_bf16(0.0f);
        }

        __syncthreads();

        if (tid < 64) {
            mfma_probe(mfma_sink, As[tid][0], As[tid][1], Bs[tid][0], Bs[tid][1]);
        }

#pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            float a_frag[TM];
            float b_frag[TN];

#pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_frag[i] = bf16_to_float(As[row_base + i][kk]);
            }

#pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_frag[j] = bf16_to_float(Bs[col_base + j][kk]);
            }

#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int64_t gm = m0 + row_base + i;
        if (gm >= M) continue;

#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int64_t gn = n0 + col_base + j;
            if (gn < N) {
                C[gm * N + gn] = float_to_bf16(acc[i][j]);
            }
        }
    }
}

extern "C" __global__ void gemm_bf16_fallback_kernel(
    const hip_bfloat16* __restrict__ A,
    const hip_bfloat16* __restrict__ B,
    hip_bfloat16* __restrict__ C,
    int64_t M,
    int64_t N,
    int64_t K) {
    const int64_t m = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    float acc = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        acc += bf16_to_float(A[m * K + k]) * bf16_to_float(B[n * K + k]);
    }
    C[m * N + n] = float_to_bf16(acc);
}

extern "C" void launch_gemm_bf16_var_mnk(
    const hip_bfloat16* A,
    const hip_bfloat16* B,
    hip_bfloat16* C,
    int64_t M,
    int64_t N,
    int64_t K,
    hipStream_t stream) {
    if (M <= 0 || N <= 0) return;

    if (M >= BLOCK_M && N >= BLOCK_N) {
        dim3 block(BLOCK_THREADS_X, BLOCK_THREADS_Y);
        dim3 grid(
            static_cast<unsigned int>((N + BLOCK_N - 1) / BLOCK_N),
            static_cast<unsigned int>((M + BLOCK_M - 1) / BLOCK_M));
        hipLaunchKernelGGL(
            gemm_bf16_balanced_mfma_kernel,
            grid,
            block,
            0,
            stream,
            A, B, C, M, N, K);
    } else {
        dim3 block(16, 16);
        dim3 grid(
            static_cast<unsigned int>((N + block.x - 1) / block.x),
            static_cast<unsigned int>((M + block.y - 1) / block.y));
        hipLaunchKernelGGL(
            gemm_bf16_fallback_kernel,
            grid,
            block,
            0,
            stream,
            A, B, C, M, N, K);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::fprintf(stderr, "HIP kernel launch error: %s\n", hipGetErrorString(err));
    }
}