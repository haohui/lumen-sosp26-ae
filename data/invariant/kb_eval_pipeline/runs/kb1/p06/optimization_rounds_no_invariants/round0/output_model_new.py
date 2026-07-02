import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 256
K = 524288
N = 256


@substrate.jit
def gemm_kernel(
    A: S.Tensor((256, 524288), S.bf16),
    B: S.Tensor((524288, 256), S.bf16),
    C: S.Tensor((256, 256), S.bf16),
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
        if tuple(A.shape) != (256, 524288) or tuple(B.shape) != (524288, 256):
            return torch.matmul(A, B)
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((256, 256), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: ((1, 1, 1), (1, 1, 1))](A, B, C)
        return C
