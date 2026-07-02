import torch
import torch.nn as nn

import substrate
import substrate.language as S


BATCH = 8
I = 256
J = 512
L = 256
K = 768


@substrate.jit
def einsum4d_kernel(
    A: S.Tensor((8, 256, 512, 256), S.bf16),
    B: S.Tensor((256, 768), S.bf16),
    C: S.Tensor((8, 256, 512, 768), S.bf16),
):
    for b in S.range(BATCH):
        for i in S.range(I):
            for j in S.range(J):
                for k in S.range(K):
                    acc = S.convert(0.0, S.f32)
                    for l in S.range(L):
                        acc += S.convert(A[b, i, j, l], S.f32) * S.convert(B[l, k], S.f32)
                    C[b, i, j, k] = S.convert(acc, S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (8, 256, 512, 256) or tuple(B.shape) != (256, 768):
            return torch.einsum("bijl,lk->bijk", A, B)
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((8, 256, 512, 768), device=A.device, dtype=A.dtype)
        einsum4d_kernel[lambda: ((1, 1, 1), (1, 1, 1))](A, B, C)
        return C
