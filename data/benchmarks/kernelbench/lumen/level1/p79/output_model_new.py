import torch
import torch.nn as nn
import math
import substrate
import substrate.language as S


# Block tile sizes
TILE_O = 256  # Output spatial positions per block
BLOCK_THREADS = 256


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
    padding: S.i32,
    dilation: S.i32,
):
    """
    Transposed 1D convolution kernel.
    Grid: (batch, out_channels, num_spatial_tiles)
    Block: (TILE_O,) threads, each handling one spatial position
    """
    n = S.block_id(0)
    oc = S.block_id(1)
    tile_o = S.block_id(2)
    tid = S.thread_id(0)

    # Compute output position
    o = tile_o * TILE_O + tid

    if n >= batch_size or oc >= out_channels or o >= output_length:
        return

    # Create layout for input: (batch, in_channels, length)
    input_layout = S.make_layout(
        (batch_size, in_channels, input_length),
        (in_channels * input_length, input_length, 1)
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, input_layout)

    # Create layout for weight: (in_channels, out_channels, kernel_size)
    weight_layout = S.make_layout(
        (in_channels, out_channels, kernel_size),
        (out_channels * kernel_size, kernel_size, 1)
    )
    weight_tensor = S.make_tensor(weight_ptr, S.bf16, weight_layout)

    # Create layout for output: (batch, out_channels, output_length)
    output_layout = S.make_layout(
        (batch_size, out_channels, output_length),
        (out_channels * output_length, output_length, 1)
    )
    output_tensor = S.make_tensor(output_ptr, S.bf16, output_layout)

    # Accumulate contributions
    acc = S.convert(0.0, S.f32)

    # Loop over kernel elements
    for k in S.range(kernel_size):
        # For transposed conv: output_pos = input_pos * stride - padding + k * dilation
        # Rearranged: input_pos = (output_pos + padding - k * dilation) / stride
        kd = k * dilation
        num = o + padding - kd

        if num >= 0:
            i_candidate = num // stride
            check = i_candidate * stride

            if check == num:
                i = i_candidate
                if i < input_length:
                    # Accumulate over all input channels
                    for ic in S.range(in_channels):
                        in_val = S.convert(input_tensor[n, ic, i], S.f32)
                        w_val = S.convert(weight_tensor[ic, oc, k], S.f32)
                        acc = acc + in_val * w_val

    # Store result
    output_tensor[n, oc, o] = S.convert(acc, S.bf16)


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

        # Weight for ConvTranspose1d has shape (in_channels, out_channels, kernel_size)
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
        # Ensure contiguous input
        x = x.contiguous()

        batch_size = int(x.shape[0])
        in_channels = int(x.shape[1])
        input_length = int(x.shape[2])
        input_dtype = x.dtype

        # Move to GPU if needed
        if not x.is_cuda:
            x = x.cuda()

        # Compute output length using PyTorch formula
        output_length = (input_length - 1) * self.stride - 2 * self.padding + \
                        self.dilation * (self.kernel_size - 1) + 1

        # Create output tensor
        output = torch.zeros(
            (batch_size, self.out_channels, output_length),
            dtype=x.dtype, device=x.device
        )

        # Ensure weight is contiguous
        weight = self.weight.data.contiguous()

        # Launch configuration
        num_tiles = (output_length + TILE_O - 1) // TILE_O

        grid = (batch_size, self.out_channels, num_tiles)
        block = (BLOCK_THREADS, 1, 1)

        conv_transpose1d_kernel[lambda: (grid, block)](
            x, weight, output,
            batch_size, in_channels, self.out_channels,
            input_length, output_length,
            self.kernel_size, self.stride, self.padding, self.dilation
        )

        # Add bias if needed
        if self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1)

        return output
