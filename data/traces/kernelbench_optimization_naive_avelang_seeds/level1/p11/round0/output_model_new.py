import torch
import torch.nn as nn

import avelang
import avelang.language as al


BATCH = 8
I = 256
J = 512
L = 256
K = 768


@avelang.jit
def einsum4d_kernel(
    A: al.Tensor((8, 256, 512, 256), al.bf16),
    B: al.Tensor((256, 768), al.bf16),
    C: al.Tensor((8, 256, 512, 768), al.bf16),
):
    for b in al.range(BATCH):
        for i in al.range(I):
            for j in al.range(J):
                for k in al.range(K):
                    acc = al.convert(0.0, al.f32)
                    for l in al.range(L):
                        acc += al.convert(A[b, i, j, l], al.f32) * al.convert(B[l, k], al.f32)
                    C[b, i, j, k] = al.convert(acc, al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (8, 256, 512, 256) or tuple(B.shape) != (256, 768):
            raise RuntimeError('Shape/dtype fallback is intentionally disabled for the AveLang naive seed.')
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((8, 256, 512, 768), device=A.device, dtype=A.dtype)
        einsum4d_kernel[lambda: ((1, 1, 1), (1, 1, 1))](A, B, C)
        return C
