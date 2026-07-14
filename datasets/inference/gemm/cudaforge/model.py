import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#define BLOCK_SIZE 16

__global__ void gemm_bf16_kernel(const hip_bfloat16* A, const hip_bfloat16* B, hip_bfloat16* C, int N) {
    __shared__ hip_bfloat16 As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ hip_bfloat16 Bs[BLOCK_SIZE * BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    float acc = 0.0f;
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int k = 0; k < num_blocks; ++k) {
        int a_row = row;
        int a_col = k * BLOCK_SIZE + tx;
        if (a_row < N && a_col < N) {
            As[ty * BLOCK_SIZE + tx] = A[a_row * N + a_col];
        } else {
            As[ty * BLOCK_SIZE + tx] = hip_bfloat16(0.0f);
        }

        int b_row = k * BLOCK_SIZE + ty;
        int b_col = col;
        if (b_row < N && b_col < N) {
            Bs[ty * BLOCK_SIZE + tx] = B[b_row * N + b_col];
        } else {
            Bs[ty * BLOCK_SIZE + tx] = hip_bfloat16(0.0f);
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            float a_val = static_cast<float>(As[ty * BLOCK_SIZE + i]);
            float b_val = static_cast<float>(Bs[i * BLOCK_SIZE + tx]);
            acc += a_val * b_val;
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = hip_bfloat16(acc);
    }
}

torch::Tensor gemm_bf16(torch::Tensor A, torch::Tensor B, unsigned long long stream_ptr) {
    TORCH_CHECK(A.is_cuda(), "A must be a HIP/ROCm tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a HIP/ROCm tensor");
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16, "A must be BFloat16");
    TORCH_CHECK(B.scalar_type() == torch::kBFloat16, "B must be BFloat16");
    TORCH_CHECK(A.size(0) == B.size(1), "Incompatible dimensions");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible dimensions");

    int N = A.size(0);
    auto C = torch::empty({N, N}, torch::dtype(torch::kBFloat16).device(A.device()));

    const int block_size = BLOCK_SIZE;
    dim3 blocks((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);
    dim3 threads(block_size, block_size);

    const hip_bfloat16* A_ptr = reinterpret_cast<const hip_bfloat16*>(A.data_ptr());
    const hip_bfloat16* B_ptr = reinterpret_cast<const hip_bfloat16*>(B.data_ptr());
    hip_bfloat16* C_ptr = reinterpret_cast<hip_bfloat16*>(C.data_ptr());

    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
    gemm_bf16_kernel<<<blocks, threads, 0, stream>>>(A_ptr, B_ptr, C_ptr, N);

    auto err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error("HIP kernel launch failed");
    }

    return C;
}
"""

cpp_src = """
#include <torch/extension.h>
torch::Tensor gemm_bf16(torch::Tensor A, torch::Tensor B, unsigned long long stream_ptr);
"""

gemm_op = load_inline(
    name="gemm_bf16_op",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["gemm_bf16"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized Model using custom HIP/ROCm BF16 GEMM kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gemm_op = gemm_op

    def _forward_b_kn(self, A: torch.Tensor, B_kn: torch.Tensor) -> torch.Tensor:
        # Store original device to return output on same device
        original_device = A.device

        # Move inputs to HIP/ROCm device if not already there (FIX for RuntimeError)
        if A.device.type != 'cuda':
            A = A.to('cuda')
        if B_kn.device.type != 'cuda':
            B_kn = B_kn.to('cuda')

        # Convert to bfloat16 if needed
        if A.dtype != torch.bfloat16:
            A = A.to(torch.bfloat16)
        if B_kn.dtype != torch.bfloat16:
            B_kn = B_kn.to(torch.bfloat16)

        # Call the HIP kernel
        stream_ptr = int(torch.cuda.current_stream(device=A.device).cuda_stream)
        C = self.gemm_op.gemm_bf16(A, B_kn, stream_ptr)

        # Move output back to original device if needed
        if original_device.type != 'cuda':
            C = C.to(original_device)

        return C

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs C = A @ B.T using the generated square GEMM kernel.
        """
        return self._forward_b_kn(A, B.t().contiguous())

    def build_call(self, *, a_mk: torch.Tensor, b_nk: torch.Tensor):
        b_kn = b_nk.t().contiguous()
        return lambda: self._forward_b_kn(a_mk, b_kn)
