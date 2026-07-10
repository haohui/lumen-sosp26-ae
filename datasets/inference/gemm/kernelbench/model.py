import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

_cpp_source = r'''
#include <torch/extension.h>

torch::Tensor gemm_abt_hip(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_abt_hip", &gemm_abt_hip, "GEMM ABT HIP (C = A @ B^T)");
}
'''

_hip_source = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <c10/util/BFloat16.h>
#include <cstdint>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

__device__ __forceinline__ float bf16_to_float_u16(uint16_t x) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = static_cast<uint32_t>(x) << 16;
    return v.f;
}

__device__ __forceinline__ uint16_t float_to_bf16_rn(float x) {
    union {
        uint32_t u;
        float f;
    } v;
    v.f = x;
    uint32_t u = v.u;
    uint32_t lsb = (u >> 16) & 1U;
    uint32_t rounding_bias = 0x7FFFU + lsb;
    return static_cast<uint16_t>((u + rounding_bias) >> 16);
}

__global__ void gemm_abt_f32_kernel(
    const float* __restrict__ A,   // [M, K]
    const float* __restrict__ B,   // [N, K]
    float* __restrict__ C,         // [M, N]
    int M, int N, int K
) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_N][TILE_K];

    const int tx = threadIdx.x; // n dimension inside tile
    const int ty = threadIdx.y; // m dimension inside tile

    const int row = blockIdx.y * TILE_M + ty; // m
    const int col = blockIdx.x * TILE_N + tx; // n

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        const int a_k = k0 + tx;
        As[ty][tx] = (row < M && a_k < K) ? A[(int64_t)row * K + a_k] : 0.0f;

        const int b_k = k0 + ty;
        // B is indexed as B[n, k]
        Bs[tx][ty] = (col < N && b_k < K) ? B[(int64_t)col * K + b_k] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            acc += As[ty][kk] * Bs[tx][kk];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[(int64_t)row * N + col] = acc;
    }
}

__global__ void gemm_abt_bf16_kernel(
    const uint16_t* __restrict__ A, // [M, K] bf16 raw
    const uint16_t* __restrict__ B, // [N, K] bf16 raw
    uint16_t* __restrict__ C,       // [M, N] bf16 raw
    int M, int N, int K
) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_N][TILE_K];

    const int tx = threadIdx.x; // n dimension inside tile
    const int ty = threadIdx.y; // m dimension inside tile

    const int row = blockIdx.y * TILE_M + ty; // m
    const int col = blockIdx.x * TILE_N + tx; // n

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        const int a_k = k0 + tx;
        As[ty][tx] = (row < M && a_k < K) ? bf16_to_float_u16(A[(int64_t)row * K + a_k]) : 0.0f;

        const int b_k = k0 + ty;
        // B is indexed as B[n, k]
        Bs[tx][ty] = (col < N && b_k < K) ? bf16_to_float_u16(B[(int64_t)col * K + b_k]) : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            acc += As[ty][kk] * Bs[tx][kk];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[(int64_t)row * N + col] = float_to_bf16_rn(acc);
    }
}

torch::Tensor gemm_abt_hip(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA/HIP tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA/HIP tensor");
    TORCH_CHECK(A.dim() == 2, "A must be 2D with shape (M, K)");
    TORCH_CHECK(B.dim() == 2, "B must be 2D with shape (N, K)");
    TORCH_CHECK(A.size(1) == B.size(1), "A.shape[1] must equal B.shape[1] (K)");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "A and B dtypes must match");

    auto A_c = A.contiguous();
    auto B_c = B.contiguous();

    const int M = static_cast<int>(A_c.size(0));
    const int K = static_cast<int>(A_c.size(1));
    const int N = static_cast<int>(B_c.size(0));

    auto C = torch::empty({A_c.size(0), B_c.size(0)}, A_c.options());

    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    hipStream_t stream = c10::hip::getCurrentHIPStream();

    if (A_c.scalar_type() == at::ScalarType::Float) {
        gemm_abt_f32_kernel<<<grid, block, 0, stream>>>(
            A_c.data_ptr<float>(),
            B_c.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N, K
        );
    } else if (A_c.scalar_type() == at::ScalarType::BFloat16) {
        const uint16_t* A_ptr = reinterpret_cast<const uint16_t*>(A_c.data_ptr<at::BFloat16>());
        const uint16_t* B_ptr = reinterpret_cast<const uint16_t*>(B_c.data_ptr<at::BFloat16>());
        uint16_t* C_ptr = reinterpret_cast<uint16_t*>(C.data_ptr<at::BFloat16>());

        gemm_abt_bf16_kernel<<<grid, block, 0, stream>>>(
            A_ptr, B_ptr, C_ptr, M, N, K
        );
    } else {
        TORCH_CHECK(false, "Only float32 and bfloat16 are supported");
    }

    auto err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return C;
}
'''

_gemm_abt_ext = load_inline(
    name="gemm_abt_ext",
    cpp_sources=_cpp_source,
    cuda_sources=_hip_source,
    functions=None,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model with custom HIP kernel for C = A @ B^T
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self._ext = _gemm_abt_ext

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self._ext.gemm_abt_hip(A, B)

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.rand(N, K, device="cuda", dtype=torch.bfloat16)
    return [A, B]

def get_init_inputs():
    return []

# ANTI_HACK_MANIFEST
# target_semantics: A@B^T
# forbidden_api_used: []
# fallback_path: false
