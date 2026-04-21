import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 32768
K = 64
N = 32768


@substrate.jit
def gemm_kernel(
    A: S.Tensor((32768, 64), S.bf16),
    B: S.Tensor((64, 32768), S.bf16),
    C: S.Tensor((32768, 32768), S.bf16),
):
    for i in S.range(M):
        for j in S.range(N):
            acc = S.convert(0.0, S.f32)
            for kk in S.range(K):
                acc += S.convert(A[i, kk], S.f32) * S.convert(B[kk, j], S.f32)
            C[i, j] = S.convert(acc, S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (32768, 64) or tuple(B.shape) != (64, 32768):
            return torch.matmul(A, B)
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((32768, 32768), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: ((1, 1, 1), (1, 1, 1))](A, B, C)
        return C
