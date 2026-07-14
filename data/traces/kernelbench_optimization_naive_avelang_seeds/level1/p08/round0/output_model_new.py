import torch
import torch.nn as nn

import avelang
import avelang.language as al


M = 8205
K = 2949
N = 5921


@avelang.jit
def gemm_kernel(
    A: al.Tensor((8205, 2949), al.bf16),
    B: al.Tensor((2949, 5921), al.bf16),
    C: al.Tensor((8205, 5921), al.bf16),
):
    for i in al.range(M):
        for j in al.range(N):
            acc = al.convert(0.0, al.f32)
            for kk in al.range(K):
                acc += al.convert(A[i, kk], al.f32) * al.convert(B[kk, j], al.f32)
            C[i, j] = al.convert(acc, al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (8205, 2949) or tuple(B.shape) != (2949, 5921):
            raise RuntimeError('Shape/dtype fallback is intentionally disabled for the AveLang naive seed.')
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((8205, 5921), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: ((1, 1, 1), (1, 1, 1))](A, B, C)
        return C
