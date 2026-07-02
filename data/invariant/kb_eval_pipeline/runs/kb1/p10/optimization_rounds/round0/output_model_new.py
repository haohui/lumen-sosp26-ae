import torch
import torch.nn as nn

import substrate
import substrate.language as S


BATCH = 16
M = 1024
K = 2048
N = 768


@substrate.jit
def matmul3d_kernel(
    A: S.Tensor((16, 1024, 2048), S.bf16),
    B: S.Tensor((2048, 768), S.bf16),
    C: S.Tensor((16, 1024, 768), S.bf16),
):
    for b in S.range(BATCH):
        for i in S.range(M):
            for j in S.range(N):
                acc = S.convert(0.0, S.f32)
                for kk in S.range(K):
                    acc += S.convert(A[b, i, kk], S.f32) * S.convert(B[kk, j], S.f32)
                C[b, i, j] = S.convert(acc, S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (16, 1024, 2048) or tuple(B.shape) != (2048, 768):
            return torch.matmul(A, B)
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((16, 1024, 768), device=A.device, dtype=A.dtype)
        matmul3d_kernel[lambda: ((1, 1, 1), (1, 1, 1))](A, B, C)
        return C
