#include "kernel.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <climits>

#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif
#include <rocwmma/rocwmma.hpp>

#define HIP_CHECK(cmd)                                                                 \
    do {                                                                               \
        hipError_t e = (cmd);                                                          \
        if (e != hipSuccess) {                                                         \
            printf("HIP error: %s (%d) at %s:%d\n", hipGetErrorString(e), (int)e,     \
                   __FILE__, __LINE__);                                                \
        }                                                                              \
    } while (0)

using namespace rocwmma;

__global__ __launch_bounds__(512) void gemm_bf16_mfma_balanced_kernel(
    const bfloat16_t* __restrict__ A, // [M, K], row-major
    const bfloat16_t* __restrict__ B, // [N, K], row-major (used as B^T)
    bfloat16_t* __restrict__ C,       // [M, N], row-major
    int M, int N, int K)
{
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 32;
    constexpr int WAVES_PER_BLOCK = 8;
    constexpr int THREADS = WAVES_PER_BLOCK * 64;

    const int tid  = threadIdx.x;
    const int lane = tid & 63;
    const int wave = tid >> 6; // 0..7

    const int wave_m = wave >> 1; // 0..3
    const int wave_n = wave & 1;  // 0..1

    const int block_m0 = static_cast<int>(blockIdx.y) * BLOCK_M;
    const int block_n0 = static_cast<int>(blockIdx.x) * BLOCK_N;

    const int m0 = block_m0 + wave_m * WMMA_M;
    const int n0 = block_n0 + wave_n * WMMA_N;

    extern __shared__ unsigned char smem_raw[];
    bfloat16_t* smemAblk = reinterpret_cast<bfloat16_t*>(smem_raw);                       // [64,16]
    bfloat16_t* smemBblk = smemAblk + (BLOCK_M * WMMA_K);                                 // [32,16]

    constexpr int AB_BF16_ELEMS = (BLOCK_M * WMMA_K) + (BLOCK_N * WMMA_K);
    constexpr int AB_BYTES = AB_BF16_ELEMS * static_cast<int>(sizeof(bfloat16_t));
    constexpr int AB_BYTES_ALIGNED = (AB_BYTES + 3) & ~3;

    float* smemCbase = reinterpret_cast<float*>(smem_raw + AB_BYTES_ALIGNED);             // 8 wave tiles
    float* smemC = smemCbase + wave * (WMMA_M * WMMA_N);

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    fill_fragment(acc_frag, 0.0f);

    const bool full_mn = (block_m0 + BLOCK_M <= M) && (block_n0 + BLOCK_N <= N);

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        const bool full_k = (k0 + WMMA_K <= K);

        if (full_mn && full_k) {
            for (int idx = tid; idx < BLOCK_M * WMMA_K; idx += THREADS) {
                const int r  = idx >> 4;
                const int kk = idx & 15;
                smemAblk[idx] = A[static_cast<int64_t>(block_m0 + r) * K + (k0 + kk)];
            }

            for (int idx = tid; idx < BLOCK_N * WMMA_K; idx += THREADS) {
                const int n  = idx >> 4;
                const int kk = idx & 15;
                smemBblk[idx] = B[static_cast<int64_t>(block_n0 + n) * K + (k0 + kk)];
            }
        } else {
            for (int idx = tid; idx < BLOCK_M * WMMA_K; idx += THREADS) {
                const int r  = idx >> 4;
                const int kk = idx & 15;
                const int gm = block_m0 + r;
                const int gk = k0 + kk;
                smemAblk[idx] = (gm < M && gk < K) ? A[static_cast<int64_t>(gm) * K + gk] : bfloat16_t(0.0f);
            }

            for (int idx = tid; idx < BLOCK_N * WMMA_K; idx += THREADS) {
                const int n  = idx >> 4;
                const int kk = idx & 15;
                const int gn = block_n0 + n;
                const int gk = k0 + kk;
                smemBblk[idx] = (gn < N && gk < K) ? B[static_cast<int64_t>(gn) * K + gk] : bfloat16_t(0.0f);
            }
        }

        __syncthreads();

        const bfloat16_t* aPtr = smemAblk + wave_m * (WMMA_M * WMMA_K);
        const bfloat16_t* bPtr = smemBblk + wave_n * (WMMA_N * WMMA_K);

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, bfloat16_t, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, bfloat16_t, col_major> b_frag;

        load_matrix_sync(a_frag, aPtr, WMMA_K);
        load_matrix_sync(b_frag, bPtr, WMMA_K);
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        __syncthreads();
    }

    store_matrix_sync(smemC, acc_frag, WMMA_N, mem_row_major);

    const bool full_out = (m0 + WMMA_M <= M) && (n0 + WMMA_N <= N);
    if (full_out) {
        for (int idx = lane; idx < WMMA_M * WMMA_N; idx += 64) {
            const int r = idx >> 4;
            const int c = idx & 15;
            C[static_cast<int64_t>(m0 + r) * N + (n0 + c)] = bfloat16_t(smemC[idx]);
        }
    } else {
        for (int idx = lane; idx < WMMA_M * WMMA_N; idx += 64) {
            const int r = idx >> 4;
            const int c = idx & 15;
            const int gm = m0 + r;
            const int gn = n0 + c;
            if (gm < M && gn < N) {
                C[static_cast<int64_t>(gm) * N + gn] = bfloat16_t(smemC[idx]);
            }
        }
    }
}

void launch_gemm_bf16_var_mnk(
    const void* A,
    const void* B,
    void* C,
    int64_t M,
    int64_t N,
    int64_t K,
    hipStream_t stream)
{
    if (M <= 0 || N <= 0 || K <= 0) return;
    if (M > INT32_MAX || N > INT32_MAX || K > INT32_MAX) return;

    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 32;
    constexpr int WMMA_K  = 16;
    constexpr int WAVES_PER_BLOCK = 8;

    dim3 block(WAVES_PER_BLOCK * 64);
    dim3 grid((static_cast<unsigned int>(N) + (BLOCK_N - 1)) / BLOCK_N,
              (static_cast<unsigned int>(M) + (BLOCK_M - 1)) / BLOCK_M);

    constexpr int AB_BF16_ELEMS = (BLOCK_M * WMMA_K) + (BLOCK_N * WMMA_K);
    constexpr int AB_BYTES = AB_BF16_ELEMS * static_cast<int>(sizeof(bfloat16_t));
    constexpr int AB_BYTES_ALIGNED = (AB_BYTES + 3) & ~3;
    constexpr int C_F32_ELEMS = WAVES_PER_BLOCK * (16 * 16);

    const size_t shmem_bytes = AB_BYTES_ALIGNED + C_F32_ELEMS * sizeof(float);

    hipLaunchKernelGGL(
        gemm_bf16_mfma_balanced_kernel,
        grid,
        block,
        shmem_bytes,
        stream,
        reinterpret_cast<const bfloat16_t*>(A),
        reinterpret_cast<const bfloat16_t*>(B),
        reinterpret_cast<bfloat16_t*>(C),
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K));

    HIP_CHECK(hipGetLastError());
}
