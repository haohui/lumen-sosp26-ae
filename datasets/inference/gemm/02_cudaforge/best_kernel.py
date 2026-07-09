import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <climits>

#define BLOCK_X 16
#define BLOCK_Y 16
#define TILE_M 64
#define TILE_N 64
#define TILE_K 32
#define THREADS_PER_BLOCK (BLOCK_X * BLOCK_Y)

__device__ __forceinline__ float bf16_to_float(__hip_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __hip_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

__global__ __launch_bounds__(THREADS_PER_BLOCK)
void bf16_gemm_bt_kernel_64x64x32(
    const __hip_bfloat16* __restrict__ A,   // [M, K]
    const __hip_bfloat16* __restrict__ B,   // [N, K]
    __hip_bfloat16* __restrict__ C,         // [M, N]
    int M, int N, int K
) {
    __shared__ __hip_bfloat16 As[TILE_M][TILE_K + 1];
    __shared__ __hip_bfloat16 Bs[TILE_K][TILE_N + 1]; // stored as [k][n]

    const int tx = threadIdx.x;                 // 0..15
    const int ty = threadIdx.y;                 // 0..15
    const int tid = ty * BLOCK_X + tx;          // 0..255

    const int row_base = blockIdx.y * TILE_M;
    const int col_base = blockIdx.x * TILE_N;

    const int row0 = row_base + ty;
    const int row1 = row0 + BLOCK_Y;
    const int row2 = row1 + BLOCK_Y;
    const int row3 = row2 + BLOCK_Y;

    const int col0 = col_base + tx;
    const int col1 = col0 + BLOCK_X;
    const int col2 = col1 + BLOCK_X;
    const int col3 = col2 + BLOCK_X;

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f, acc23 = 0.0f;
    float acc30 = 0.0f, acc31 = 0.0f, acc32 = 0.0f, acc33 = 0.0f;

    for (int t = 0; t < K; t += TILE_K) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int idx = tid + i * THREADS_PER_BLOCK; // 0..2047

            // A tile: [64 x 32], row-major
            const int a_r = idx >> 5;       // /32
            const int a_k = idx & 31;       // %32
            const int g_ar = row_base + a_r;
            const int g_ak = t + a_k;

            As[a_r][a_k] =
                (g_ar < M && g_ak < K) ? A[g_ar * K + g_ak] : float_to_bf16(0.0f);

            // B tile: load B[n][k] and store as Bs[k][n]
            const int b_n = idx >> 5;       // /32
            const int b_k = idx & 31;       // %32
            const int g_bn = col_base + b_n;
            const int g_bk = t + b_k;

            Bs[b_k][b_n] =
                (g_bn < N && g_bk < K) ? B[g_bn * K + g_bk] : float_to_bf16(0.0f);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            const float a0 = bf16_to_float(As[ty][k]);
            const float a1 = bf16_to_float(As[ty + BLOCK_Y][k]);
            const float a2 = bf16_to_float(As[ty + 2 * BLOCK_Y][k]);
            const float a3 = bf16_to_float(As[ty + 3 * BLOCK_Y][k]);

            const float b0 = bf16_to_float(Bs[k][tx]);
            const float b1 = bf16_to_float(Bs[k][tx + BLOCK_X]);
            const float b2 = bf16_to_float(Bs[k][tx + 2 * BLOCK_X]);
            const float b3 = bf16_to_float(Bs[k][tx + 3 * BLOCK_X]);

            acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
            acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
            acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
            acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
        }

        __syncthreads();
    }

    if (row0 < M) {
        if (col0 < N) C[row0 * N + col0] = float_to_bf16(acc00);
        if (col1 < N) C[row0 * N + col1] = float_to_bf16(acc01);
        if (col2 < N) C[row0 * N + col2] = float_to_bf16(acc02);
        if (col3 < N) C[row0 * N + col3] = float_to_bf16(acc03);
    }
    if (row1 < M) {
        if (col0 < N) C[row1 * N + col0] = float_to_bf16(acc10);
        if (col1 < N) C[row1 * N + col1] = float_to_bf16(acc11);
        if (col2 < N) C[row1 * N + col2] = float_to_bf16(acc12);
        if (col3 < N) C[row1 * N + col3] = float_to_bf16(acc13);
    }
    if (row2 < M) {
        if (col0 < N) C[row2 * N + col0] = float_to_bf16(acc20);
        if (col1 < N) C[row2 * N + col1] = float_to_bf16(acc21);
        if (col2 < N) C[row2 * N + col2] = float_to_bf16(acc22);
        if (col3 < N) C[row2 * N + col3] = float_to_bf16(acc23);
    }
    if (row3 < M) {
        if (col0 < N) C[row3 * N + col0] = float_to_bf16(acc30);
        if (col1 < N) C[row3 * N + col1] = float_to_bf16(acc31);
        if (col2 < N) C[row3 * N + col2] = float_to_bf16(acc32);
        if (col3 < N) C[row3 * N + col3] = float_to_bf16(acc33);
    }
}

torch::Tensor bf16_gemm_bt(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a HIP tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a HIP tensor");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be torch.bfloat16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be torch.bfloat16");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A.shape[1] must equal B.shape[1]");
    TORCH_CHECK(A.get_device() == B.get_device(), "A and B must be on the same device");

    auto A_c = A.contiguous();
    auto B_c = B.contiguous();

    const int64_t M64 = A_c.size(0);
    const int64_t K64 = A_c.size(1);
    const int64_t N64 = B_c.size(0);

    TORCH_CHECK(M64 <= INT_MAX && N64 <= INT_MAX && K64 <= INT_MAX, "Dimensions too large");

    const int M = static_cast<int>(M64);
    const int N = static_cast<int>(N64);
    const int K = static_cast<int>(K64);

    auto C = torch::empty({M64, N64}, A_c.options().dtype(torch::kBFloat16));

    const __hip_bfloat16* A_ptr =
        reinterpret_cast<const __hip_bfloat16*>(A_c.data_ptr<at::BFloat16>());
    const __hip_bfloat16* B_ptr =
        reinterpret_cast<const __hip_bfloat16*>(B_c.data_ptr<at::BFloat16>());
    __hip_bfloat16* C_ptr =
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr<at::BFloat16>());

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    hipStream_t stream = c10::hip::getCurrentHIPStream();

    hipLaunchKernelGGL(
        bf16_gemm_bt_kernel_64x64x32,
        grid,
        block,
        0,
        stream,
        A_ptr,
        B_ptr,
        C_ptr,
        M,
        N,
        K
    );

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "bf16_gemm_bt_kernel_64x64x32 launch failed: ", hipGetErrorString(err));

    return C;
}
"""

cpp_src = r"""
torch::Tensor bf16_gemm_bt(torch::Tensor A, torch::Tensor B);
"""

_bf16_gemm_ext = load_inline(
    name="bf16_gemm_bt_ext_v6_tile64x64x32",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["bf16_gemm_bt"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._ext = _bf16_gemm_ext

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A_bf16 = A if A.dtype == torch.bfloat16 else A.to(dtype=torch.bfloat16)
        B_bf16 = B if B.dtype == torch.bfloat16 else B.to(dtype=torch.bfloat16)
        return self._ext.bf16_gemm_bt(A_bf16, B_bf16)
