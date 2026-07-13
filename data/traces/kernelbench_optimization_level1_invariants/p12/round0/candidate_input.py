import torch
import torch.nn as nn

import avelang
import avelang.language as al


M = 4096
N = 4096


@avelang.jit
def diag_left_kernel(
    A: al.Tensor((4096,), al.bf16),
    B: al.Tensor((4096, 4096), al.bf16),
    C: al.Tensor((4096, 4096), al.bf16),
):
    for i in al.range(M):
        for j in al.range(N):
            C[i, j] = A[i] * B[i, j]


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (4096,) or tuple(B.shape) != (4096, 4096) or A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise RuntimeError('Shape/dtype fallback is intentionally disabled for the AveLang naive seed.')
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((4096, 4096), device=B.device, dtype=B.dtype)
        diag_left_kernel[lambda: ((1, 1, 1), (1, 1, 1))](A, B, C)
        return C
