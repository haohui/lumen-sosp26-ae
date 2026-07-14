import torch
import torch.nn as nn

import avelang
import avelang.language as al


N = 4096


@avelang.jit
def gemm_kernel(
    A: al.Tensor((N, N), al.bf16),
    B: al.Tensor((N, N), al.bf16),
    C: al.Tensor((N, N), al.bf16),
):
    for i in al.range(N):
        for j in al.range(N):
            acc = al.convert(0.0, al.f32)
            for k in al.range(N):
                acc += al.convert(A[i, k], al.f32) * al.convert(B[k, j], al.f32)
            C[i, j] = al.convert(acc, al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if (
            tuple(A.shape) != (N, N)
            or tuple(B.shape) != (N, N)
            or A.dtype != torch.bfloat16
            or B.dtype != torch.bfloat16
            or A.device != B.device
        ):
            raise RuntimeError('Shape/dtype fallback is intentionally disabled for the AveLang naive seed.')

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((N, N), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: ((1, 1, 1), (1, 1, 1))](A, B, C)
        return C
