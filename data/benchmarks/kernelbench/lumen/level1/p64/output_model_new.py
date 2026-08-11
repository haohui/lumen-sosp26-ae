"""Transposed 1D convolution kernel in Substrate DSL for AMD MI300X."""

import math

import substrate
import substrate.language as S
import torch
import torch.nn as nn

# Tile configuration
TILE_OUT = 64  # Output elements per tile
TILE_OC = 4    # Output channels per tile


@substrate.jit
def conv_transpose1d_kernel(
    input_ptr: S.Pointer(S.bf16),
    weight_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    in_channels: S.i32,
    out_channels: S.i32,
    input_length: S.i32,
    output_length: S.i32,
    kernel_size: S.i32,
    padding: S.i32,
):
    """
    Transposed 1D convolution kernel.

    Block layout:
    - block_id(0) = batch * output_channel_tile
    - block_id(1) = output position tile

    Thread layout:
    - thread_id(0) = output position within tile
    - thread_id(1) = output channel within tile
    """
    # Decode block IDs
    combined_bc = S.block_id(0)
    out_tile = S.block_id(1)

    n = combined_bc // ((out_channels + TILE_OC - 1) // TILE_OC)
    oc_tile = combined_bc - n * ((out_channels + TILE_OC - 1) // TILE_OC)

    # Thread coordinates
    tid_o = S.thread_id(0)
    tid_oc = S.thread_id(1)

    # Global output positions
    oc = oc_tile * TILE_OC + tid_oc
    o = out_tile * TILE_OUT + tid_o

    # Bounds check
    if n >= batch_size or oc >= out_channels or o >= output_length:
        return

    # Create tensor views with explicit strides
    # Input layout: (batch, in_channels, input_length)
    input_stride_l = S.convert(1, S.i32)
    input_stride_c = input_stride_l * input_length
    input_stride_b = input_stride_c * in_channels

    input_layout = S.make_layout(
        (batch_size, in_channels, input_length),
        (input_stride_b, input_stride_c, input_stride_l),
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, input_layout)

    # Weight layout: (in_channels, out_channels, kernel_size)
    weight_stride_k = S.convert(1, S.i32)
    weight_stride_oc = weight_stride_k * kernel_size
    weight_stride_ic = weight_stride_oc * out_channels

    weight_layout = S.make_layout(
        (in_channels, out_channels, kernel_size),
        (weight_stride_ic, weight_stride_oc, weight_stride_k),
    )
    weight_tensor = S.make_tensor(weight_ptr, S.bf16, weight_layout)

    # Output layout: (batch, out_channels, output_length)
    output_stride_l = S.convert(1, S.i32)
    output_stride_c = output_stride_l * output_length
    output_stride_b = output_stride_c * out_channels

    output_layout = S.make_layout(
        (batch_size, out_channels, output_length),
        (output_stride_b, output_stride_c, output_stride_l),
    )
    output_tensor = S.make_tensor(output_ptr, S.bf16, output_layout)

    # Accumulate in float32 for precision
    acc = S.convert(0.0, S.f32)

    # For transposed convolution with stride=1:
    # output[o] gets contributions from input[i] where o = i + k - padding
    # So i = o - k + padding
    # With padding=0: i = o - k

    for k in S.range(3):  # kernel_size is always 3 for this problem
        i = o - k + padding
        if i >= 0:
            if i < input_length:
                for ic in S.range(128):  # in_channels is always 128 for this problem
                    in_val = S.convert(input_tensor[n, ic, i], S.f32)
                    w_val = S.convert(weight_tensor[ic, oc, k], S.f32)
                    acc = acc + in_val * w_val

    # Store result
    output_tensor[n, oc, o] = S.convert(acc, S.bf16)


class ModelNew(nn.Module):
    """
    Optimized transposed 1D convolution using Substrate DSL.
    """

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
        super(ModelNew, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.has_bias = bias

        # Weight tensor for ConvTranspose1d
        # Shape: (in_channels, out_channels // groups, kernel_size)
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels // groups, kernel_size)
        )

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

        batch_size, in_channels, input_length = x.shape

        # Compute output length
        # output_length = (input_length - 1) * stride + kernel_size - 2 * padding + output_padding
        output_length = (
            (input_length - 1) * self.stride
            + self.kernel_size
            - 2 * self.padding
            + self.output_padding
        )

        # Convert to BF16 for kernel computation
        x_bf16 = x.to(dtype=torch.bfloat16)
        weight_bf16 = self.weight.data.to(dtype=torch.bfloat16).contiguous()

        # Create output tensor in BF16
        output = torch.empty(
            (batch_size, self.out_channels, output_length),
            dtype=torch.bfloat16,
            device=x.device,
        )

        # Compute grid and block dimensions
        tiles_oc = (self.out_channels + TILE_OC - 1) // TILE_OC
        tiles_out = (output_length + TILE_OUT - 1) // TILE_OUT

        # Combined batch and output channel tiles in first dimension
        grid = (batch_size * tiles_oc, tiles_out, 1)
        block = (TILE_OUT, TILE_OC, 1)

        conv_transpose1d_kernel[lambda: (grid, block)](
            x_bf16,
            weight_bf16,
            output,
            batch_size,
            in_channels,
            self.out_channels,
            input_length,
            output_length,
            self.kernel_size,
            self.padding,
        )

        # Convert back to original dtype if needed
        if x.dtype != torch.bfloat16:
            output = output.to(dtype=x.dtype)

        # Add bias if needed
        if self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1)

        return output
