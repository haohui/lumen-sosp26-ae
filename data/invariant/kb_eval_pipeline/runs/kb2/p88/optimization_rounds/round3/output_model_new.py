import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

    def forward(self, x):
        x = self.gemm(x)
        x = self.group_norm(x)
        x = x * torch.sigmoid(x)
        x = x * self.multiply_weight
        x = x * torch.sigmoid(x)
        return x
