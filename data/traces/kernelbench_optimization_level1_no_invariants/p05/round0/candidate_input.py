import torch
import torch.nn as nn

import avelang
import avelang.language as al


M = 65536
N = 16384


@avelang.jit
def scale_kernel(
    A: al.Tensor((65536, 16384), al.bf16),
    C: al.Tensor((65536, 16384), al.bf16),
    scalar: al.bf16,
):
    for i in al.range(M):
        for j in al.range(N):
            C[i, j] = A[i, j] * scalar


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (65536, 16384) or A.dtype != torch.bfloat16:
            raise RuntimeError('Shape/dtype fallback is intentionally disabled for the AveLang naive seed.')
        A = A.contiguous()
        scalar = B
        C = torch.empty((65536, 16384), device=A.device, dtype=A.dtype)
        scale_kernel[lambda: ((1, 1, 1), (1, 1, 1))](A, C, scalar)
        return C
