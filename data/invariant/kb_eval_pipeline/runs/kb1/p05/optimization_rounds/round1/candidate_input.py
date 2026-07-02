import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 65536
N = 16384


@substrate.jit
def scale_kernel(
    A: S.Tensor((65536, 16384), S.bf16),
    C: S.Tensor((65536, 16384), S.bf16),
    scalar: S.bf16,
):
    for i in S.range(M):
        for j in S.range(N):
            C[i, j] = A[i, j] * scalar


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (65536, 16384):
            return A * B
        A = A.contiguous()
        scalar = torch.tensor(B, device=A.device, dtype=A.dtype).item()
        C = torch.empty((65536, 16384), device=A.device, dtype=A.dtype)
        scale_kernel[lambda: ((1, 1, 1), (1, 1, 1))](A, C, scalar)
        return C
