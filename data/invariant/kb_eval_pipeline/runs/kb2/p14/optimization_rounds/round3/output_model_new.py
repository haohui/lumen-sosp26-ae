import torch
import torch.nn as nn
import torch.nn.functional as F


BATCH_SIZE = 1024
INPUT_SIZE = 8192
HIDDEN_SIZE = 8192
SCALING_FACTOR = 1.5


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.output_scale = scaling_factor / 2.0

    def forward(self, x):
        y = F.linear(x, self.weight)
        y = y.sum(dim=1, keepdim=True)
        return y * self.output_scale
