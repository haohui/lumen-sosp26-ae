import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 4096
K = 4096
N = 4096


@substrate.jit
def tri_gemm_kernel(
    A: S.Tensor((4096, 4096), S.bf16),
    B: S.Tensor((4096, 4096), S.bf16),
    C: S.Tensor((4096, 4096), S.bf16),
):
    for i in S.range(M):
        for j in S.range(N):
            acc = S.convert(0.0, S.f32)
            for kk in S.range(K):
                acc += S.convert(A[i, kk], S.f32) * S.convert(B[kk, j], S.f32)
            if j >= i:
                C[i, j] = S.convert(acc, S.bf16)
            else:
                C[i, j] = S.convert(0.0, S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (4096, 4096) or tuple(B.shape) != (4096, 4096):
            return torch.triu(torch.matmul(A, B))
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((4096, 4096), device=A.device, dtype=A.dtype)
        tri_gemm_kernel[lambda: ((1, 1, 1), (1, 1, 1))](A, B, C)
        return C
