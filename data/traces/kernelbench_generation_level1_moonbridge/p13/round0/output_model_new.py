import torch
import torch.nn as nn
from avelang_kernels.amdgpu_gemm import gemm_pipeline_transposed_b

N = 4096


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_bf16_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B using AveLang BF16 pipeline GEMM kernel.

    A and B are both (n, n) square matrices.
    B is transposed before launch since the kernel expects B in (N, K) layout.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    a_bf16 = _prepare_bf16_cuda_contiguous(a)
    b_bf16 = _prepare_bf16_cuda_contiguous(b)

    n = a_bf16.shape[0]

    # Transpose B: kernel expects B in (N, K) row-major layout for C = A @ B.
    b_t = b_bf16.T.contiguous()

    return gemm_pipeline_transposed_b(a_bf16, b_t)


class ModelNew(nn.Module):
    """BF16 matmul via AveLang pipelined MFMA GEMM kernel."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return avelang_bf16_gemm(A, B)


def get_inputs():
    A = torch.rand(N, N)
    A = (A + A.T) / 2
    B = torch.rand(N, N)
    B = (B + B.T) / 2
    return [A, B]


def get_init_inputs():
    return []
