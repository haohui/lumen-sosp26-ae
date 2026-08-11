"""Optimized 1D Convolution using Substrate DSL."""

import math

import substrate
import substrate.language as S
import torch
import torch.nn as nn


# Tile size for output length per block
TILE_SIZE = 256


@substrate.jit
def conv1d_kernel(
    input_ptr: S.Pointer(S.bf16),
    weight_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    in_channels: S.i32,
    out_channels: S.i32,
    length: S.i32,
    out_length: S.i32,
    kernel_size: S.i32,
    stride: S.i32,
    padding: S.i32,
    dilation: S.i32,
):
    """
    1D convolution kernel.

    Each block computes one (batch, out_channel) tile of output positions.
    Each thread computes one output element.
    """
    # block_id(0) = batch * out_channels + out_channel (flattened)
    # block_id(1) = length tile
    bc_linear = S.block_id(0)
    n = bc_linear // out_channels
    oc = bc_linear % out_channels

    tile_l = S.block_id(1)
    tid = S.thread_id(0)

    ol = tile_l * TILE_SIZE + tid

    if n >= batch_size or oc >= out_channels or ol >= out_length:
        return

    # Input layout: (batch, in_channels, length)
    input_stride_l = S.convert(1, S.i32)
    input_stride_c = input_stride_l * length
    input_stride_b = input_stride_c * in_channels

    input_layout = S.make_layout(
        (batch_size, in_channels, length),
        (input_stride_b, input_stride_c, input_stride_l)
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, input_layout)

    # Weight layout: (out_channels, in_channels, kernel_size)
    weight_stride_k = S.convert(1, S.i32)
    weight_stride_c = weight_stride_k * kernel_size
    weight_stride_o = weight_stride_c * in_channels

    weight_layout = S.make_layout(
        (out_channels, in_channels, kernel_size),
        (weight_stride_o, weight_stride_c, weight_stride_k)
    )
    weight_tensor = S.make_tensor(weight_ptr, S.bf16, weight_layout)

    # Output layout: (batch, out_channels, out_length)
    output_stride_l = S.convert(1, S.i32)
    output_stride_c = output_stride_l * out_length
    output_stride_b = output_stride_c * out_channels

    output_layout = S.make_layout(
        (batch_size, out_channels, out_length),
        (output_stride_b, output_stride_c, output_stride_l)
    )
    output_tensor = S.make_tensor(output_ptr, S.bf16, output_layout)

    # Compute convolution - accumulate in FP32 for precision
    acc = S.convert(0.0, S.f32)

    for ic in S.range(in_channels):
        for k in S.range(kernel_size):
            input_l = ol * stride + k * dilation - padding
            if input_l < 0 or input_l >= length:
                continue
            in_val = S.convert(input_tensor[n, ic, input_l], S.f32)
            w_val = S.convert(weight_tensor[oc, ic, k], S.f32)
            acc = acc + in_val * w_val

    output_tensor[n, oc, ol] = S.convert(acc, S.bf16)


class ModelNew(nn.Module):
    """
    Optimized 1D convolution using Substrate DSL.
    Supports standard convolution (groups=1) with BF16 optimization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()

        assert groups == 1, "Only groups=1 is supported"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Weight shape for Conv1d with groups=1: (out_channels, in_channels, kernel_size)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))

        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias_param", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.in_channels * self.kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()

        batch_size, in_channels, length = x.shape
        orig_dtype = x.dtype

        # Convert to BF16 for optimized kernel execution
        x_bf16 = x.to(torch.bfloat16)
        weight_bf16 = self.weight.to(torch.bfloat16).contiguous()

        # Compute output length
        out_length = (
            length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
        ) // self.stride + 1

        # Allocate output tensor in BF16
        output_bf16 = torch.empty(
            (batch_size, self.out_channels, out_length),
            dtype=torch.bfloat16,
            device=x.device,
        )

        # Calculate grid and block dimensions
        tiles_length = (out_length + TILE_SIZE - 1) // TILE_SIZE

        grid = (batch_size * self.out_channels, tiles_length, 1)
        block = (TILE_SIZE, 1, 1)

        conv1d_kernel[lambda: (grid, block)](
            x_bf16,
            weight_bf16,
            output_bf16,
            batch_size,
            in_channels,
            self.out_channels,
            length,
            out_length,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )

        # Convert back to original dtype
        output = output_bf16.to(orig_dtype)

        # Add bias if present
        if self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1)

        return output
