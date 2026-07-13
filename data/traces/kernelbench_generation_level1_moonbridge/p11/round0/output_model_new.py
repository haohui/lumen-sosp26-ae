import torch
import torch.nn as nn

from avelang_kernels.amdgpu_gemm import gemm_pipeline_transposed_b


def avelang_4d_tensor_matmul(
    A: torch.Tensor, B: torch.Tensor
) -> torch.Tensor:
    """
    Performs 4D tensor-matrix multiplication:
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Uses the amdgpu_gemm pipeline kernel which expects:
        A: (M, K) bf16
        B: (N, K) bf16  (transposed B)
        Output: (M, N) bf16
    """
    b, i, j, l = A.shape
    lb, k = B.shape
    if l != lb:
        raise ValueError(f"Contraction dimension mismatch: A has l={l}, B has l={lb}")

    m = b * i * j

    # Flatten A from (b, i, j, l) to (M, l)
    A_flat = A.reshape(m, l)
    if not (A_flat.is_cuda and A_flat.dtype == torch.bfloat16 and A_flat.is_contiguous()):
        A_flat = A_flat.contiguous().cuda().to(dtype=torch.bfloat16)

    # Transpose B from (l, k) to (k, l) = (N, K) for the gemm kernel
    B_T = B.T.contiguous()
    if not (B_T.is_cuda and B_T.dtype == torch.bfloat16 and B_T.is_contiguous()):
        B_T = B_T.contiguous().cuda().to(dtype=torch.bfloat16)

    out = gemm_pipeline_transposed_b(A_flat, B_T)

    # Reshape output from (M, k) back to (b, i, j, k)
    return out.reshape(b, i, j, k)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return avelang_4d_tensor_matmul(A, B)
