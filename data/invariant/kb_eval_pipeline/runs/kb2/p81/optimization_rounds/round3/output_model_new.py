import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        x = self.gemm(x)
        x = x * torch.sigmoid(x)
        x = x / 2.0
        x = torch.clamp(x, min=-1.0, max=1.0)
        x = torch.tanh(x)
        x = torch.clamp(x, min=-1.0, max=1.0)
        return x
