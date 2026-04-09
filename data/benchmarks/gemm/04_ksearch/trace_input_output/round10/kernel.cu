#include "kernel.h"

#include <cstddef>
#include <cstdint>

namespace {

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;

__device__ __forceinline__ float bf16_to_float(hip_bfloat16 x) {
    return static_cast<float>(x);
}

__global__ void gemm_tiled_direct_kernel(
    const hip_bfloat16* __restrict__ A,
    const hip_bfloat16* __restrict__ B,
    hip_bfloat16* __restrict__ C,
    int M, int N, int K) {

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_N][TILE_K];

    const int tx = threadIdx.x; // [0, TILE_N)
    const int ty = threadIdx.y; // [0, TILE_M)

    const int m = blockIdx.y * TILE_M + ty;
    const int n = blockIdx.x * TILE_N + tx;

    float acc = 0.0f;

    const int n_tile_base = blockIdx.x * TILE_N;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        const int kA = k0 + tx;
        if (m < M && kA < K) {
            As[ty][tx] = bf16_to_float(A[static_cast<int64_t>(m) * K + kA]);
        } else {
            As[ty][tx] = 0.0f;
        }

        const int b_row = n_tile_base + ty;
        const int kB = k0 + tx;
        if (b_row < N && kB < K) {
            Bs[ty][tx] = bf16_to_float(B[static_cast<int64_t>(b_row) * K + kB]);
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            acc += As[ty][kk] * Bs[tx][kk];
        }

        __syncthreads();
    }

    if (m < M && n < N) {
        C[static_cast<int64_t>(m) * N + n] = __float2bfloat16(acc);
    }
}

__global__ void splitk_gemm_tiled_kernel(
    const hip_bfloat16* __restrict__ A,
    const hip_bfloat16* __restrict__ B,
    float* __restrict__ partial,
    int M, int N, int K,
    int part_size) {

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_N][TILE_K];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int m = blockIdx.y * TILE_M + ty;
    const int n = blockIdx.x * TILE_N + tx;
    const int p = blockIdx.z;

    if (m >= M || n >= N) return;

    const int k_begin = p * part_size;
    const int k_end = (k_begin + part_size < K) ? (k_begin + part_size) : K;
    if (k_begin >= k_end) return;

    float acc = 0.0f;
    const int n_tile_base = blockIdx.x * TILE_N;

    for (int k0 = k_begin; k0 < k_end; k0 += TILE_K) {
        const int kA = k0 + tx;
        if (kA < k_end) {
            As[ty][tx] = bf16_to_float(A[static_cast<int64_t>(m) * K + kA]);
        } else {
            As[ty][tx] = 0.0f;
        }

        const int b_row = n_tile_base + ty;
        const int kB = k0 + tx;
        if (b_row < N && kB < k_end) {
            Bs[ty][tx] = bf16_to_float(B[static_cast<int64_t>(b_row) * K + kB]);
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            acc += As[ty][kk] * Bs[tx][kk];
        }

        __syncthreads();
    }

    const int64_t elem_idx = static_cast<int64_t>(m) * N + n;
    partial[static_cast<int64_t>(p) * static_cast<int64_t>(M) * N + elem_idx] = acc;
}

__global__ void splitk_merge_kernel(
    const float* __restrict__ partial,
    hip_bfloat16* __restrict__ C,
    int64_t elements,
    int partitions) {

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elements) return;

    float acc = 0.0f;
    for (int p = 0; p < partitions; ++p) {
        acc += partial[static_cast<int64_t>(p) * elements + idx];
    }
    C[idx] = __float2bfloat16(acc);
}

} // namespace

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
        return hipMemsetAsync(
            C,
            0,
            static_cast<size_t>(elements) * sizeof(hip_bfloat16),
            stream);
    }

    const bool k_dominant = (K >= 2048) && (elements <= 262144);
    const int part_size = k_dominant ? 1024 : K;
    const int partitions = (K + part_size - 1) / part_size;

    if (partitions == 1) {
        dim3 block(TILE_N, TILE_M, 1);
        dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, 1);

        hipLaunchKernelGGL(
            gemm_tiled_direct_kernel,
            grid,
            block,
            0,
            stream,
            A, B, C, M, N, K);

        return hipGetLastError();
    }

    float* partial = nullptr;
    const size_t partial_bytes =
        static_cast<size_t>(elements) * static_cast<size_t>(partitions) * sizeof(float);

    hipError_t err = hipMalloc(reinterpret_cast<void**>(&partial), partial_bytes);
    if (err != hipSuccess) return err;

    {
        dim3 block(TILE_N, TILE_M, 1);
        dim3 grid(
            (N + TILE_N - 1) / TILE_N,
            (M + TILE_M - 1) / TILE_M,
            partitions);

        hipLaunchKernelGGL(
            splitk_gemm_tiled_kernel,
            grid,
            block,
            0,
            stream,
            A, B, partial, M, N, K, part_size);

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
            partial, C, elements, partitions);

        err = hipGetLastError();
        if (err != hipSuccess) {
            hipFree(partial);
            return err;
        }
    }

    err = hipFree(partial);
    return err;
}