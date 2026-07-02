import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 2048
K = 8192
N = 4096


@substrate.jit
def gemm_kernel(
    A: S.Tensor((2048, 8192), S.bf16),
    B: S.Tensor((8192, 4096), S.bf16),
    C: S.Tensor((2048, 4096), S.bf16),
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
        if tuple(A.shape) != (2048, 8192) or tuple(B.shape) != (4096, 8192):
            return torch.matmul(A, B.T)
        A2 = A.contiguous()
        B2 = B.transpose(-2, -1).contiguous()
        C = torch.empty((2048, 4096), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: ((1, 1, 1), (1, 1, 1))](A2, B2, C)
        return C
