import torch
import torch.nn as nn
import torch.nn.functional as F
import substrate
import substrate.language as S


BATCH_SIZE = 16384
IN_FEATURES = 4096
OUT_FEATURES = 4096


class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int, scale_shape: tuple[int, ...], eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.linear(x, self.gemm.weight, self.gemm.bias)
        x = x * self.scale
        x = self.bn(x)
        return x


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_FEATURES)]


def get_init_inputs():
    return [IN_FEATURES, OUT_FEATURES, (OUT_FEATURES,)]
