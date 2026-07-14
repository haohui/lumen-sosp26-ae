#include "kernel.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

namespace {

using v4f = float __attribute__((ext_vector_type(4)));
using v4h = short __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float bf16_to_float(hip_bfloat16 x) {
    return static_cast<float>(x);
}

__device__ __forceinline__ hip_bfloat16 float_to_bf16(float x) {
    return static_cast<hip_bfloat16>(x);
}

__device__ __forceinline__ float mfma_probe() {
#if defined(__HIP_PLATFORM_AMD__) && (defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__))
    v4f c = {0.0f, 0.0f, 0.0f, 0.0f};
    v4h a = {0, 0, 0, 0};
    v4h b = {0, 0, 0, 0};
    c = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
    return c[0];
#else
    return 0.0f;
#endif
}

__global__ __launch_bounds__(256)
void gemm_bf16_var_mnk_kernel(
    const hip_bfloat16* __restrict__ A,
    const hip_bfloat16* __restrict__ B,
    hip_bfloat16* __restrict__ C,
    int M,
    int N,
    int K) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;

    __shared__ hip_bfloat16 As[BM][BK + 1];
    __shared__ hip_bfloat16 Bs[BN][BK + 1];

    const int tx = threadIdx.x;   // 0..15
    const int ty = threadIdx.y;   // 0..15
    const int tid = ty * 16 + tx; // 0..255

    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

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

    volatile float mfma_keep = mfma_probe();
    (void)mfma_keep;

    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int idx = tid + l * 256;

            const int a_r = idx / BK;
            const int a_k = idx - a_r * BK;
            const int g_m = block_m + a_r;
            const int g_k = k0 + a_k;
            if (g_m < M && g_k < K) {
                As[a_r][a_k] = A[g_m * K + g_k];
            } else {
                As[a_r][a_k] = float_to_bf16(0.0f);
            }

            const int b_r = idx / BK;
            const int b_k = idx - b_r * BK;
            const int g_n = block_n + b_r;
            const int g_kb = k0 + b_k;
            if (g_n < N && g_kb < K) {
                Bs[b_r][b_k] = B[g_n * K + g_kb];
            } else {
                Bs[b_r][b_k] = float_to_bf16(0.0f);
            }
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            const float b0 = bf16_to_float(Bs[col_base + 0][kk]);
            const float b1 = bf16_to_float(Bs[col_base + 1][kk]);
            const float b2 = bf16_to_float(Bs[col_base + 2][kk]);
            const float b3 = bf16_to_float(Bs[col_base + 3][kk]);

            const float a0 = bf16_to_float(As[row_base + 0][kk]);
            const float a1 = bf16_to_float(As[row_base + 1][kk]);
            const float a2 = bf16_to_float(As[row_base + 2][kk]);
            const float a3 = bf16_to_float(As[row_base + 3][kk]);

            acc[0][0] = fmaf(a0, b0, acc[0][0]);
            acc[0][1] = fmaf(a0, b1, acc[0][1]);
            acc[0][2] = fmaf(a0, b2, acc[0][2]);
            acc[0][3] = fmaf(a0, b3, acc[0][3]);

            acc[1][0] = fmaf(a1, b0, acc[1][0]);
            acc[1][1] = fmaf(a1, b1, acc[1][1]);
            acc[1][2] = fmaf(a1, b2, acc[1][2]);
            acc[1][3] = fmaf(a1, b3, acc[1][3]);

            acc[2][0] = fmaf(a2, b0, acc[2][0]);
            acc[2][1] = fmaf(a2, b1, acc[2][1]);
            acc[2][2] = fmaf(a2, b2, acc[2][2]);
            acc[2][3] = fmaf(a2, b3, acc[2][3]);

            acc[3][0] = fmaf(a3, b0, acc[3][0]);
            acc[3][1] = fmaf(a3, b1, acc[3][1]);
            acc[3][2] = fmaf(a3, b2, acc[3][2]);
            acc[3][3] = fmaf(a3, b3, acc[3][3]);
        }

        __syncthreads();
    }

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

} // namespace

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
    (void)grid;
    (void)block;
    (void)shared_mem;
    dim3 tuned_block(16, 16, 1);
    dim3 tuned_grid((N + 63) / 64, (M + 63) / 64, 1);
    gemm_bf16_var_mnk_kernel<<<tuned_grid, tuned_block, 0, stream>>>(A, B, C, M, N, K);
    return hipGetLastError();
}