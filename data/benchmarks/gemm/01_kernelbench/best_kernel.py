import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

_bf16_matmul_ext = None
if torch.cuda.is_available():
    bf16_matmul_cpp_source = r"""
    #include <torch/extension.h>
    #include <ATen/hip/HIPContext.h>
    #include <hip/hip_runtime.h>
    #include <cstdint>

    #define TILE_M 16
    #define TILE_N 16
    #define TILE_K 16

    __device__ __forceinline__ float bf16_to_float(uint16_t x) {
        uint32_t tmp = static_cast<uint32_t>(x) << 16;
        return __uint_as_float(tmp);
    }

    __device__ __forceinline__ uint16_t float_to_bf16_rn(float x) {
        uint32_t v = __float_as_uint(x);
        uint32_t lsb = (v >> 16) & 1u;
        uint32_t rounding_bias = 0x7FFFu + lsb;
        v += rounding_bias;
        return static_cast<uint16_t>(v >> 16);
    }

    __global__ void bf16_gemm_tiled_kernel(
        const uint16_t* __restrict__ A,
        const uint16_t* __restrict__ B,
        uint16_t* __restrict__ C,
        int M, int K, int N
    ) {
        __shared__ uint16_t As[TILE_M][TILE_K];
        __shared__ uint16_t Bs[TILE_K][TILE_N];

        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int row = blockIdx.y * TILE_M + ty;
        const int col = blockIdx.x * TILE_N + tx;

        float acc = 0.0f;

        for (int k0 = 0; k0 < K; k0 += TILE_K) {
            const int a_col = k0 + tx;
            const int b_row = k0 + ty;

            As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : static_cast<uint16_t>(0);
            Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : static_cast<uint16_t>(0);

            __syncthreads();

            #pragma unroll
            for (int k = 0; k < TILE_K; ++k) {
                acc = fmaf(bf16_to_float(As[ty][k]), bf16_to_float(Bs[k][tx]), acc);
            }

            __syncthreads();
        }

        if (row < M && col < N) {
            C[row * N + col] = float_to_bf16_rn(acc);
        }
    }

    torch::Tensor bf16_matmul_hip(torch::Tensor A, torch::Tensor B) {
        TORCH_CHECK(A.is_cuda(), "A must be on HIP device");
        TORCH_CHECK(B.is_cuda(), "B must be on HIP device");
        TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
        TORCH_CHECK(A.scalar_type() == at::kBFloat16, "A must be bfloat16");
        TORCH_CHECK(B.scalar_type() == at::kBFloat16, "B must be bfloat16");
        TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
        TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
        TORCH_CHECK(A.size(1) == B.size(0), "Incompatible matrix shapes");

        const int M = static_cast<int>(A.size(0));
        const int K = static_cast<int>(A.size(1));
        const int N = static_cast<int>(B.size(1));

        auto C = torch::empty({M, N}, A.options().dtype(at::kBFloat16));

        const uint16_t* A_ptr = reinterpret_cast<const uint16_t*>(A.data_ptr<at::BFloat16>());
        const uint16_t* B_ptr = reinterpret_cast<const uint16_t*>(B.data_ptr<at::BFloat16>());
        uint16_t* C_ptr = reinterpret_cast<uint16_t*>(C.data_ptr<at::BFloat16>());

        dim3 block(TILE_N, TILE_M);
        dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
        auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

        hipLaunchKernelGGL(
            bf16_gemm_tiled_kernel,
            grid,
            block,
            0,
            stream.stream(),
            A_ptr, B_ptr, C_ptr, M, K, N
        );

        hipError_t err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "bf16_gemm_tiled_kernel launch failed: ", hipGetErrorString(err));

        return C;
    }
    """

    _bf16_matmul_ext = load_inline(
        name="bf16_matmul_hip_ext_v1",
        cpp_sources=bf16_matmul_cpp_source,
        functions=["bf16_matmul_hip"],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


class ModelNew(nn.Module):
    """
    BF16-optimized matmul model using a custom HIP kernel on AMD GPUs.
    Falls back to torch.matmul for non-BF16 or non-GPU inputs.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self._ext = _bf16_matmul_ext

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if (
            self._ext is not None
            and A.is_cuda
            and B.is_cuda
            and A.dtype == torch.bfloat16
            and B.dtype == torch.bfloat16
        ):
            if not A.is_contiguous():
                A = A.contiguous()
            if not B.is_contiguous():
                B = B.contiguous()
            return self._ext.bf16_matmul_hip(A, B)
        return torch.matmul(A, B)


M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    if torch.cuda.is_available():
        A = torch.rand(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.rand(K, N, device="cuda", dtype=torch.bfloat16)
    else:
        A = torch.rand(M, K, dtype=torch.float32)
        B = torch.rand(K, N, dtype=torch.float32)
    return [A, B]

def get_init_inputs():
    return []
