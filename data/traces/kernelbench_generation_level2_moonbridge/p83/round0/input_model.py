import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, minimum, clamp, and dropout.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.register_buffer('_min_tensor', torch.tensor(min_value), persistent=False)
        self.register_buffer('_max_tensor', torch.tensor(max_value), persistent=False)
        self._min_value = min_value
        self._max_value = max_value

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = torch.min(x, self._min_tensor)
        x = torch.clamp(x, min=self._min_value, max=self._max_value)
        x = self.dropout(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 64, 64
kernel_size = 3
groups = 8
min_value = 0.0
max_value = 1.0
dropout_p = 0.2

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p]
