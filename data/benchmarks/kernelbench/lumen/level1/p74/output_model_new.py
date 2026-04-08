import torch
import torch.nn as nn
import math
import substrate
import substrate.language as S


# Block tile size for output spatial dimension
TILE_L = 128


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
    stride: S.i32,
    dilation: S.i32,
):
    """
    Transposed 1D convolution kernel.
    Each thread handles one output element and accumulates across all in_channels and kernel positions.
    """
    # Block IDs: (batch, out_channel, tile_idx)
    n = S.block_id(0)
    oc = S.block_id(1)
    tile_idx = S.block_id(2)

    tile_ol = tile_idx * TILE_L
    tid = S.thread_id(0)
    ol = tile_ol + tid

    # Create tensor views with proper layouts
    # Input layout: (batch, in_channels, length)
    input_stride_l = S.convert(1, S.i32)
    input_stride_c = input_stride_l * input_length
    input_stride_b = input_stride_c * in_channels

    input_layout = S.make_layout(
        (batch_size, in_channels, input_length),
        (input_stride_b, input_stride_c, input_stride_l)
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, input_layout)

    # Weight layout: (in_channels, out_channels, kernel_size)
    weight_stride_k = S.convert(1, S.i32)
    weight_stride_oc = weight_stride_k * kernel_size
    weight_stride_ic = weight_stride_oc * out_channels

    weight_layout = S.make_layout(
        (in_channels, out_channels, kernel_size),
        (weight_stride_ic, weight_stride_oc, weight_stride_k)
    )
    weight_tensor = S.make_tensor(weight_ptr, S.bf16, weight_layout)

    # Output layout: (batch, out_channels, output_length)
    output_stride_l = S.convert(1, S.i32)
    output_stride_c = output_stride_l * output_length
    output_stride_b = output_stride_c * out_channels

    output_layout = S.make_layout(
        (batch_size, out_channels, output_length),
        (output_stride_b, output_stride_c, output_stride_l)
    )
    output_tensor = S.make_tensor(output_ptr, S.bf16, output_layout)

    # Check bounds
    if n >= batch_size or oc >= out_channels or ol >= output_length:
        return

    # Accumulate over all in_channels and kernel positions
    acc = S.convert(0.0, S.f32)

    for ic in S.range(in_channels):
        for k in S.range(kernel_size):
            # For transposed conv: input_pos = output_pos - k * dilation
            input_pos = ol - k * dilation

            # Check if this input position is valid
            if input_pos >= 0 and input_pos < input_length:
                in_val = S.convert(input_tensor[n, ic, input_pos], S.f32)
                w_val = S.convert(weight_tensor[ic, oc, k], S.f32)
                acc = acc + in_val * w_val

    # Store result
    output_tensor[n, oc, ol] = S.convert(acc, S.bf16)


class ModelNew(torch.nn.Module):
    """
    Optimized transposed 1D convolution using Substrate DSL.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        # Weight tensor for transposed conv
        # ConvTranspose1d weight shape: (in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))

        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)

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
        # L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
        output_length = (input_length - 1) * self.stride - 2 * self.padding + \
                        self.dilation * (self.kernel_size - 1) + 1

        # Create output tensor
        output = torch.empty((batch_size, self.out_channels, output_length),
                            dtype=x.dtype, device=x.device)

        weight = self.weight.data.contiguous()

        tiles_l = (output_length + TILE_L - 1) // TILE_L

        grid = (batch_size, self.out_channels, tiles_l)
        block = (TILE_L, 1, 1)

        conv_transpose1d_kernel[lambda: (grid, block)](
            x, weight, output,
            batch_size, self.in_channels, self.out_channels,
            input_length, output_length,
            self.kernel_size, self.stride, self.dilation
        )

        # Add bias if needed
        if self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1)

        return output
