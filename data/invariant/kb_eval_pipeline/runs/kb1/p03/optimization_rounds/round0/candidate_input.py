import torch
import torch.nn as nn

import substrate
import substrate.language as S


BATCH = 128
M = 512
K = 1024
N = 2048


@substrate.jit
def bmm_kernel(
    A: S.Tensor((128, 512, 1024), S.bf16),
    B: S.Tensor((128, 1024, 2048), S.bf16),
    C: S.Tensor((128, 512, 2048), S.bf16),
):
    for b in S.range(BATCH):
        for i in S.range(M):
            for j in S.range(N):
                acc = S.convert(0.0, S.f32)
                for kk in S.range(K):
                    acc += S.convert(A[b, i, kk], S.f32) * S.convert(B[b, kk, j], S.f32)
                C[b, i, j] = S.convert(acc, S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (128, 512, 1024) or tuple(B.shape) != (128, 1024, 2048):
            return torch.bmm(A, B)
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((128, 512, 2048), device=A.device, dtype=A.dtype)
        bmm_kernel[lambda: ((1, 1, 1), (1, 1, 1))](A, B, C)
        return C
