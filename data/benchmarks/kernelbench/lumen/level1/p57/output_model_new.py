"""Transposed 2D convolution kernel using Substrate DSL.

This implements nn.ConvTranspose2d with stride=1, padding=0.
Input: (N, C_in, H, W)
Output: (N, C_out, H + K - 1, W + K - 1) where K is kernel_size
"""

import math

import substrate
import substrate.language as S
import torch
import torch.nn as nn


@substrate.jit
def conv_transpose2d_kernel(
    input_ptr: S.Pointer(S.bf16),
    weight_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    in_channels: S.i32,
    out_channels: S.i32,
    in_h: S.i32,
    in_w: S.i32,
    out_h: S.i32,
    out_w: S.i32,
    k: S.i32,
):
    """Transposed 2D convolution kernel.

    Each thread computes one output pixel.
    Weight layout: (in_channels, out_channels, k, k)
    """
    n = S.block_id(0)
    oc = S.block_id(1)
    spatial_id = S.block_id(2)

    out_w_tiles = (out_w + 15) // 16
    oh = (spatial_id // out_w_tiles) * 16 + S.thread_id(1)
    ow = (spatial_id % out_w_tiles) * 16 + S.thread_id(0)

    if oh >= out_h or ow >= out_w:
        return

    # Create tensor views
    input_tensor = S.make_tensor(
        input_ptr, S.bf16,
        S.make_layout(
            (batch_size, in_channels, in_h, in_w),
            (in_channels * in_h * in_w, in_h * in_w, in_w, 1)
        )
    )
    weight_tensor = S.make_tensor(
        weight_ptr, S.bf16,
        S.make_layout(
            (in_channels, out_channels, k, k),
            (out_channels * k * k, k * k, k, 1)
        )
    )
    output_tensor = S.make_tensor(
        output_ptr, S.bf16,
        S.make_layout(
            (batch_size, out_channels, out_h, out_w),
            (out_channels * out_h * out_w, out_h * out_w, out_w, 1)
        )
    )

    acc = S.convert(0.0, S.f32)

    # Iterate over input channels
    for ic in S.range(in_channels):
        # Iterate over kernel positions
        for kh in S.range(k):
            ih = oh - kh
            if ih >= 0:
                if ih < in_h:
                    for kw in S.range(k):
                        iw = ow - kw
                        if iw >= 0:
                            if iw < in_w:
                                in_val = S.convert(input_tensor[n, ic, ih, iw], S.f32)
                                w_val = S.convert(weight_tensor[ic, oc, kh, kw], S.f32)
                                acc = acc + in_val * w_val

    output_tensor[n, oc, oh, ow] = S.convert(acc, S.bf16)


class ModelNew(nn.Module):
    """Optimized transposed 2D convolution using Substrate DSL."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Weight shape for ConvTranspose2d: (in_channels, out_channels // groups, kH, kW)
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias_param", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        batch_size, _, in_h, in_w = x.shape
        k = self.kernel_size

        # Output dimensions for transposed convolution
        out_h = (in_h - 1) * self.stride - 2 * self.padding + k + self.output_padding
        out_w = (in_w - 1) * self.stride - 2 * self.padding + k + self.output_padding

        # Convert to BF16 for kernel
        original_dtype = x.dtype
        x_bf16 = x.to(torch.bfloat16)
        weight_bf16 = self.weight.detach().to(x.device).to(torch.bfloat16).contiguous()

        output = torch.empty(
            (batch_size, self.out_channels, out_h, out_w),
            dtype=torch.bfloat16,
            device=x.device,
        )

        # Launch configuration: one block per (batch, out_channel) pair
        # Each block has 16x16 threads for output tile
        out_w_tiles = (out_w + 15) // 16
        out_h_tiles = (out_h + 15) // 16

        grid = (batch_size, self.out_channels, out_h_tiles * out_w_tiles)
        block = (16, 16, 1)

        conv_transpose2d_kernel[lambda: (grid, block)](
            x_bf16,
            weight_bf16,
            output,
            batch_size,
            self.in_channels,
            self.out_channels,
            in_h,
            in_w,
            out_h,
            out_w,
            k,
        )

        # Add bias if present
        if self.bias_param is not None:
            bias_bf16 = self.bias_param.detach().to(x.device).to(torch.bfloat16)
            output = output + bias_bf16.view(1, -1, 1, 1)

        # Restore original dtype if needed
        if original_dtype != torch.bfloat16:
            output = output.to(original_dtype)

        return output
