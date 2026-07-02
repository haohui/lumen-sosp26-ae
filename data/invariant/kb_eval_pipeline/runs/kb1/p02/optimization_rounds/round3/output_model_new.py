import torch
import torch.nn as nn


M = 2048
K = 8192
N = 4096


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise ValueError(
                f"ModelNew only supports A={(M, K)} and B={(K, N)}, "
                f"got {tuple(A.shape)} and {tuple(B.shape)}"
            )
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError(
                f"ModelNew requires bfloat16 inputs, got {A.dtype} and {B.dtype}"
            )
        return torch.matmul(A.contiguous(), B.contiguous())
