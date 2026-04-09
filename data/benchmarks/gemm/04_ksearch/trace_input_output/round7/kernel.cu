#include "kernel.h"

#include <cstdint>
#include <cstddef>

#if defined(__HIP_PLATFORM_AMD__) && __has_include(<rocwmma/rocwmma.hpp>)
#include <rocwmma/rocwmma.hpp>
#define GEMM_USE_ROCWMMA 1
#else
#define GEMM_USE_ROCWMMA 0
#endif

__device__ __forceinline__ float bf16_to_float(hip_bfloat16 x) {
    return static_cast<float>(x);
}

__global__ void gemm_scalar_direct_kernel(
    const hip_bfloat16* __restrict__ A,
    const hip_bfloat16* __restrict__ B,
    hip_bfloat16* __restrict__ C,
    int M, int N, int K) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= M || n >= N) return;

    float acc = 0.0f;
    const int a_base = m * K;
    const int b_base = n * K;
    for (int k = 0; k < K; ++k) {
        acc += bf16_to_float(A[a_base + k]) * bf16_to_float(B[b_base + k]);
    }

    C[static_cast<int64_t>(m) * static_cast<int64_t>(N) + n] = __float2bfloat16(acc);
}

__global__ void splitk_gemm_scalar_kernel(
    const hip_bfloat16* __restrict__ A,
    const hip_bfloat16* __restrict__ B,
    float* __restrict__ partial,
    int M, int N, int K,
    int part_size) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int p = blockIdx.z;

    if (m >= M || n >= N) return;

    int k0 = p * part_size;
    int k1 = k0 + part_size;
    if (k1 > K) k1 = K;

    float acc = 0.0f;
    const int a_base = m * K;
    const int b_base = n * K;

    for (int k = k0; k < k1; ++k) {
        acc += bf16_to_float(A[a_base + k]) * bf16_to_float(B[b_base + k]);
    }

    const int64_t elem_idx = static_cast<int64_t>(m) * static_cast<int64_t>(N) + n;
    const int64_t out_idx =
        static_cast<int64_t>(p) * static_cast<int64_t>(M) * static_cast<int64_t>(N) + elem_idx;
    partial[out_idx] = acc;
}

#if GEMM_USE_ROCWMMA
using namespace rocwmma;

__global__ __launch_bounds__(64) void splitk_gemm_mfma_kernel(
    const hip_bfloat16* __restrict__ A,
    const hip_bfloat16* __restrict__ B,
    float* __restrict__ partial,
    int M, int N, int K,
    int part_size) {

    constexpr int BlockM = 16;
    constexpr int BlockN = 16;
    constexpr int BlockK = 16;

    const int m0 = blockIdx.y * BlockM;
    const int n0 = blockIdx.x * BlockN;
    const int p  = blockIdx.z;

    if (m0 + BlockM > M || n0 + BlockN > N) return;

    const int k0 = p * part_size;
    int k1 = k0 + part_size;
    if (k1 > K) k1 = K;

    fragment<matrix_a, BlockM, BlockN, BlockK, bfloat16_t, row_major> fragA;
    fragment<matrix_b, BlockM, BlockN, BlockK, bfloat16_t, col_major> fragB;
    fragment<accumulator, BlockM, BlockN, BlockK, float> fragAcc;

    fill_fragment(fragAcc, 0.0f);

    const bfloat16_t* A_bf16 = reinterpret_cast<const bfloat16_t*>(A);
    const bfloat16_t* B_bf16 = reinterpret_cast<const bfloat16_t*>(B);

    for (int k = k0; k < k1; k += BlockK) {
        const bfloat16_t* aTile = A_bf16 + static_cast<int64_t>(m0) * K + k;
        const bfloat16_t* bTile = B_bf16 + static_cast<int64_t>(n0) * K + k;

        load_matrix_sync(fragA, aTile, K);
        load_matrix_sync(fragB, bTile, K);
        mma_sync(fragAcc, fragA, fragB, fragAcc);
    }

    float* outTile = partial + (static_cast<int64_t>(p) * M + m0) * N + n0;
    store_matrix_sync(outTile, fragAcc, N, mem_row_major);
}
#endif

__global__ void splitk_merge_kernel(
    const float* __restrict__ partial,
    hip_bfloat16* __restrict__ C,
    int64_t elements,
    int partitions) {

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elements) return;

    float acc = 0.0f;
    for (int p = 0; p < partitions; ++p) {
        acc += partial[static_cast<int64_t>(p) * elements + idx];
    }
    C[idx] = __float2bfloat16(acc);
}

hipError_t launch_gemm_bf16_var_mnk(
    const hip_bfloat16* A,
    const hip_bfloat16* B,
    hip_bfloat16* C,
    int M,
    int N,
    int K,
    hipStream_t stream) {

    if (M <= 0 || N <= 0) {
        return hipSuccess;
    }

    const int64_t elements = static_cast<int64_t>(M) * static_cast<int64_t>(N);

    if (K <= 0) {
        return hipMemsetAsync(C, 0, static_cast<size_t>(elements) * sizeof(hip_bfloat16), stream);
    }

    const bool k_dominant = (K >= 2048) && (elements <= 262144);
    const int part_size = k_dominant ? 1024 : K;
    const int partitions = (K + part_size - 1) / part_size;

#if GEMM_USE_ROCWMMA
    const bool use_mfma =
        (M % 16 == 0) &&
        (N % 16 == 0) &&
        (K % 16 == 0) &&
        (part_size % 16 == 0);
#else
    const bool use_mfma = false;
#endif

    if (partitions == 1 && !use_mfma) {
        dim3 block(16, 16, 1);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y, 1);
        hipLaunchKernelGGL(
            gemm_scalar_direct_kernel,
            grid,
            block,
            0,
            stream,
            A, B, C, M, N, K
        );
        return hipGetLastError();
    }

    float* partial = nullptr;
    const size_t partial_bytes =
        static_cast<size_t>(elements) * static_cast<size_t>(partitions) * sizeof(float);

    hipError_t err = hipMalloc(reinterpret_cast<void**>(&partial), partial_bytes);
    if (err != hipSuccess) return err;

    if (use_mfma) {
#if GEMM_USE_ROCWMMA
        dim3 block(64, 1, 1);
        dim3 grid((N + 15) / 16, (M + 15) / 16, partitions);
        hipLaunchKernelGGL(
            splitk_gemm_mfma_kernel,
            grid,
            block,
            0,
            stream,
            A, B, partial, M, N, K, part_size
        );
        err = hipGetLastError();
        if (err != hipSuccess) {
            hipFree(partial);
            return err;
        }
#endif
    } else {
        dim3 block(16, 16, 1);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y, partitions);
        hipLaunchKernelGGL(
            splitk_gemm_scalar_kernel,
            grid,
            block,
            0,
            stream,
            A, B, partial, M, N, K, part_size
        );
        err = hipGetLastError();
        if (err != hipSuccess) {
            hipFree(partial);
            return err;
        }
    }

    {
        constexpr int threads = 256;
        const int blocks = static_cast<int>((elements + threads - 1) / threads);
        hipLaunchKernelGGL(
            splitk_merge_kernel,
            dim3(blocks),
            dim3(threads),
            0,
            stream,
            partial, C, elements, partitions
        );
        err = hipGetLastError();
        if (err != hipSuccess) {
            hipFree(partial);
            return err;
        }
    }

    err = hipFree(partial);
    return err;
}