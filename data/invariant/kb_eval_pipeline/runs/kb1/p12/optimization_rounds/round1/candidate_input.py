import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 4096
N = 4096


@substrate.jit
def diag_left_kernel(
    A: S.Tensor((4096,), S.bf16),
    B: S.Tensor((4096, 4096), S.bf16),
    C: S.Tensor((4096, 4096), S.bf16),
):
    for i in S.range(M):
        for j in S.range(N):
            C[i, j] = A[i] * B[i, j]


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (4096,) or tuple(B.shape) != (4096, 4096):
            return torch.diag(A) @ B
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((4096, 4096), device=B.device, dtype=B.dtype)
        diag_left_kernel[lambda: ((1, 1, 1), (1, 1, 1))](A, B, C)
        return C
