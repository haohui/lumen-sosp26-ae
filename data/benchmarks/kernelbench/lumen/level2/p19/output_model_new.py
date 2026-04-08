import torch
import torch.nn as nn
import torch.nn.functional as F
import substrate
import substrate.language as S
import math


@substrate.jit
def gelu_kernel(
    x: S.Tensor((1,), S.bf16),
    out: S.Tensor((1,), S.bf16),
    n: S.i32,
):
    """Element-wise GELU activation using exact formula with erf."""
    tid = S.thread_id(0)
    bid = S.block_id(0)
    block_size = S.block_dim(0)
    idx = bid * block_size + tid

    if idx < n:
        val = x[idx]
        val_f32 = val
        # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        sqrt2 = 1.4142135623730951
        scaled = val_f32 / sqrt2
        erf_val = S.erf(scaled)
        one_plus_erf = 1.0 + erf_val
        half_factor = 0.5 * one_plus_erf
        gelu_val = val_f32 * half_factor
        out[idx] = S.convert(gelu_val, S.bf16)


def substrate_gelu(x: torch.Tensor) -> torch.Tensor:
    """Apply GELU using substrate kernel."""
    n = x.numel()
    out = torch.empty_like(x)
    block_size = 256
    num_blocks = (n + block_size - 1) // block_size

    x_flat = x.view(-1)
    out_flat = out.view(-1)

    gelu_kernel[lambda: ((num_blocks, 1, 1), (block_size, 1, 1))](x_flat, out_flat, n)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using substrate kernel for GELU.
    ConvTranspose2d and GroupNorm use PyTorch implementation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transposed convolution
        x = self.conv_transpose(x)

        # GELU activation using substrate kernel
        x = substrate_gelu(x)

        # GroupNorm using PyTorch
        x = self.group_norm(x)

        return x


batch_size   = 128
in_channels  = 64
out_channels = 64
height = width = 256
kernel_size  = 3
stride       = 1
groups = 8
num_groups = 8


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, dtype=torch.bfloat16, device='cuda')]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]
