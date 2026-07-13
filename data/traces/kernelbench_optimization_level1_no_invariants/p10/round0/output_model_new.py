import torch
import torch.nn as nn

import avelang
import avelang.language as al


@avelang.jit
def matmul3d_kernel(
    A: al.Tensor((16, 1024, 2048), al.bf16),
    B: al.Tensor((2048, 768), al.bf16),
    C: al.Tensor((16, 1024, 768), al.bf16),
):
    for b in al.range(16):
        for i in al.range(1024):
            for j in al.range(768):
                acc = al.convert(0.0, al.f32)
                for kk in al.range(2048):
                    acc += al.convert(A[b, i, kk], al.f32) * al.convert(B[kk, j], al.f32)
                C[b, i, j] = al.convert(acc, al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (16, 1024, 2048) or tuple(B.shape) != (2048, 768):
            raise RuntimeError(
                "Shape/dtype fallback is intentionally disabled."
            )
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((16, 1024, 768), device=A.device, dtype=A.dtype)
        matmul3d_kernel[lambda: ((1, 1, 1), (1, 1, 1))](A, B, C)
        return C
