import torch
import torch.nn as nn
import substrate
import substrate.language as S
import math

WARP_SIZE = 64


@substrate.jit
def test_tanh_kernel(
    a_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.f32),
    n: S.u32,
):
    """Test tanh kernel."""
    tid = S.thread_id(0)
    if tid >= n:
        return

    a_val = a_ptr[tid]
    a_f32 = S.convert(a_val, S.f32)
    result = S.tanh(a_f32)
    out_ptr[tid] = result


class ModelNew(nn.Module):
    """
    Test implementation with tanh.
    """

    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()
        self.eps = eps
        self.num_groups = groups

    def forward(self, x):
        # Step 1: Convolution
        x_conv = self.conv(x)

        # Step 2: Group Normalization
        x_norm = self.group_norm(x_conv)

        # Step 3: Tanh
        x_tanh = self.tanh(x_norm)

        # Step 4: HardSwish
        x_hard_swish = self.hard_swish(x_tanh)

        # Step 5: Residual Addition
        x_res = x_conv + x_hard_swish

        # Step 6: LogSumExp
        x_logsumexp = torch.logsumexp(x_res, dim=1, keepdim=True)

        return x_logsumexp


# Required for evaluation
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
groups = 16


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups]
