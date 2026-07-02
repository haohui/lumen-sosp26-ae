import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 2048
K = 1048576
N = 1


@substrate.jit
def gemm_kernel(
    A: S.Tensor((2048, 1048576), S.bf16),
    B: S.Tensor((1048576, 1), S.bf16),
    C: S.Tensor((2048, 1), S.bf16),
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
        if tuple(A.shape) != (2048, 1048576) or tuple(B.shape) != (1048576, 1):
            return torch.matmul(A, B)
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((2048, 1), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: ((1, 1, 1), (1, 1, 1))](A, B, C)
        return C
