import torch
import torch.nn as nn
import math
import substrate
import substrate.language as S


# Block tile size for output spatial dimensions
TILE_D = 4
TILE_H = 4
TILE_W = 4


@substrate.jit
def conv_transpose3d_kernel(
    input_ptr: S.Pointer(S.bf16),
    weight_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    in_channels: S.i32,
    out_channels: S.i32,
    in_depth: S.i32,
    in_height: S.i32,
    in_width: S.i32,
    out_depth: S.i32,
    out_height: S.i32,
    out_width: S.i32,
    kernel_size: S.i32,
    padding: S.i32,
    tiles_d: S.i32,
    tiles_h: S.i32,
    tiles_w: S.i32,
):
    # Decode block IDs
    # block_id(0) = batch index
    # block_id(1) = output channel
    # block_id(2) = linearized spatial tile
    n = S.block_id(0)
    oc = S.block_id(1)

    linear_id = S.block_id(2)
    tiles_per_slice = tiles_h * tiles_w
    tile_od = (linear_id // tiles_per_slice) * TILE_D
    rem = linear_id - tile_od // TILE_D * tiles_per_slice
    tile_oh = (rem // tiles_w) * TILE_H
    tile_ow = (rem - tile_oh // TILE_H * tiles_w) * TILE_W

    # Thread ID within block
    tid_x = S.thread_id(0)
    tid_y = S.thread_id(1)
    tid_z = S.thread_id(2)

    od = tile_od + tid_z
    oh = tile_oh + tid_y
    ow = tile_ow + tid_x

    # Check bounds early
    if n >= batch_size or oc >= out_channels or od >= out_depth or oh >= out_height or ow >= out_width:
        return

    # Create tensor views using S.make_layout
    # Input layout: (batch, in_channels, in_depth, in_height, in_width)
    input_layout = S.make_layout(
        (batch_size, in_channels, in_depth, in_height, in_width),
        (in_channels * in_depth * in_height * in_width, in_depth * in_height * in_width, in_height * in_width, in_width, 1)
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, input_layout)

    # Weight layout: (in_channels, out_channels, kernel_d, kernel_h, kernel_w)
    weight_layout = S.make_layout(
        (in_channels, out_channels, kernel_size, kernel_size, kernel_size),
        (out_channels * kernel_size * kernel_size * kernel_size, kernel_size * kernel_size * kernel_size, kernel_size * kernel_size, kernel_size, 1)
    )
    weight_tensor = S.make_tensor(weight_ptr, S.bf16, weight_layout)

    # Output layout: (batch, out_channels, out_depth, out_height, out_width)
    output_layout = S.make_layout(
        (batch_size, out_channels, out_depth, out_height, out_width),
        (out_channels * out_depth * out_height * out_width, out_depth * out_height * out_width, out_height * out_width, out_width, 1)
    )
    output_tensor = S.make_tensor(output_ptr, S.bf16, output_layout)

    # Accumulator
    acc = S.convert(0.0, S.f32)

    # For transposed convolution with stride=1:
    # output[od, oh, ow] = sum over ic, kd, kh, kw of input[od - kd + padding, oh - kh + padding, ow - kw + padding] * weight[ic, oc, kd, kh, kw]

    for ic in S.range(in_channels):
        for kd in S.range(kernel_size):
            input_d = od - kd + padding

            if input_d >= 0:
                if input_d < in_depth:
                    for kh in S.range(kernel_size):
                        input_h = oh - kh + padding

                        if input_h >= 0:
                            if input_h < in_height:
                                for kw in S.range(kernel_size):
                                    input_w = ow - kw + padding

                                    if input_w >= 0:
                                        if input_w < in_width:
                                            in_val = S.convert(input_tensor[n, ic, input_d, input_h, input_w], S.f32)
                                            w_val = S.convert(weight_tensor[ic, oc, kd, kh, kw], S.f32)
                                            acc = acc + in_val * w_val

    # Store result
    output_tensor[n, oc, od, oh, ow] = S.convert(acc, S.bf16)


class ModelNew(torch.nn.Module):
    """
    Optimized transposed 3D convolution using Substrate DSL.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, output_padding: int = 0,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.has_bias = bias

        # Initialize weight tensor for transposed convolution
        # Weight shape: (in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size))

        # Initialize bias if needed
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.in_channels * self.kernel_size ** 3
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous and on correct device
        x = x.contiguous()

        batch_size, in_channels, in_depth, in_height, in_width = x.shape
        dtype = x.dtype

        # Compute output dimensions for transposed convolution
        # out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding
        out_depth = (in_depth - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_height = (in_height - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_width = (in_width - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        # Create output tensor (same dtype as input to match reference model behavior)
        output = torch.empty((batch_size, self.out_channels, out_depth, out_height, out_width),
                            dtype=dtype, device=x.device)

        # Ensure weight is contiguous
        weight = self.weight.data.contiguous()

        # Launch kernel
        tiles_d = (out_depth + TILE_D - 1) // TILE_D
        tiles_h = (out_height + TILE_H - 1) // TILE_H
        tiles_w = (out_width + TILE_W - 1) // TILE_W

        grid = (batch_size, self.out_channels, tiles_d * tiles_h * tiles_w)
        block = (TILE_W, TILE_H, TILE_D)

        conv_transpose3d_kernel[lambda: (grid, block)](
            x, weight, output,
            batch_size, in_channels, self.out_channels,
            in_depth, in_height, in_width,
            out_depth, out_height, out_width,
            self.kernel_size, self.padding,
            tiles_d, tiles_h, tiles_w
        )

        # Add bias if needed
        if self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1, 1, 1)

        return output
