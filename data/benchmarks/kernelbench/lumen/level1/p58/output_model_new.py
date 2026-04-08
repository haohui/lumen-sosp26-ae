import torch
import torch.nn as nn
import math
import substrate
import substrate.language as S


# Tile sizes for output spatial dimensions
TILE_D = 4
TILE_H = 8
TILE_W = 8


@substrate.jit
def conv_transpose3d_kernel(
    input_ptr: S.Pointer(S.bf16),
    weight_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    in_channels: S.i32,
    out_channels: S.i32,
    depth_in: S.i32,
    height_in: S.i32,
    width_in: S.i32,
    depth_out: S.i32,
    height_out: S.i32,
    width_out: S.i32,
    kernel_d: S.i32,
    kernel_h: S.i32,
    kernel_w: S.i32,
):
    """ConvTranspose3d kernel for asymmetric kernel."""
    # Block mapping: (batch, out_channel, tile_index)
    # tile_index encodes position in output spatial grid
    n = S.block_id(0)
    oc = S.block_id(1)
    tile_idx = S.block_id(2)

    # Compute tile position in output spatial dimensions
    tiles_d = (depth_out + TILE_D - 1) // TILE_D
    tiles_h = (height_out + TILE_H - 1) // TILE_H
    tiles_w = (width_out + TILE_W - 1) // TILE_W
    tiles_per_d = tiles_h * tiles_w

    tile_d = tile_idx // tiles_per_d
    remaining = tile_idx - tile_d * tiles_per_d
    tile_h = remaining // tiles_w
    tile_w = remaining - tile_h * tiles_w

    tile_od = tile_d * TILE_D
    tile_oh = tile_h * TILE_H
    tile_ow = tile_w * TILE_W

    # Thread mapping within the tile
    tid_x = S.thread_id(0)
    tid_y = S.thread_id(1)
    tid_z = S.thread_id(2)

    od = tile_od + tid_z
    oh = tile_oh + tid_y
    ow = tile_ow + tid_x

    # Bounds check
    if n >= batch_size or oc >= out_channels or od >= depth_out or oh >= height_out or ow >= width_out:
        return

    # Create input tensor layout: (batch, in_channels, depth, height, width)
    in_stride_w = S.convert(1, S.i32)
    in_stride_h = in_stride_w * width_in
    in_stride_d = in_stride_h * height_in
    in_stride_c = in_stride_d * depth_in
    in_stride_b = in_stride_c * in_channels

    in_layout = S.make_layout(
        (batch_size, in_channels, depth_in, height_in, width_in),
        (in_stride_b, in_stride_c, in_stride_d, in_stride_h, in_stride_w)
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, in_layout)

    # Create weight tensor layout: (in_channels, out_channels, kernel_d, kernel_h, kernel_w)
    w_stride_w = S.convert(1, S.i32)
    w_stride_h = w_stride_w * kernel_w
    w_stride_d = w_stride_h * kernel_h
    w_stride_oc = w_stride_d * kernel_d
    w_stride_ic = w_stride_oc * out_channels

    w_layout = S.make_layout(
        (in_channels, out_channels, kernel_d, kernel_h, kernel_w),
        (w_stride_ic, w_stride_oc, w_stride_d, w_stride_h, w_stride_w)
    )
    weight_tensor = S.make_tensor(weight_ptr, S.bf16, w_layout)

    # Create output tensor layout: (batch, out_channels, depth_out, height_out, width_out)
    out_stride_w = S.convert(1, S.i32)
    out_stride_h = out_stride_w * width_out
    out_stride_d = out_stride_h * height_out
    out_stride_c = out_stride_d * depth_out
    out_stride_b = out_stride_c * out_channels

    out_layout = S.make_layout(
        (batch_size, out_channels, depth_out, height_out, width_out),
        (out_stride_b, out_stride_c, out_stride_d, out_stride_h, out_stride_w)
    )
    output_tensor = S.make_tensor(output_ptr, S.bf16, out_layout)

    # Compute the transposed convolution
    # For each output position (od, oh, ow), accumulate contributions from all
    # input positions that would have contributed to this output in a forward conv
    acc = S.convert(0.0, S.f32)

    for ic in S.range(in_channels):
        for kd in S.range(kernel_d):
            for kh in S.range(kernel_h):
                for kw in S.range(kernel_w):
                    # In transposed conv, output[i] receives from input[i - k]
                    id = od - kd
                    ih = oh - kh
                    iw = ow - kw

                    # Check input bounds
                    if id >= 0 and id < depth_in and ih >= 0 and ih < height_in and iw >= 0 and iw < width_in:
                        in_val = S.convert(input_tensor[n, ic, id, ih, iw], S.f32)
                        w_val = S.convert(weight_tensor[ic, oc, kd, kh, kw], S.f32)
                        acc = acc + in_val * w_val

    output_tensor[n, oc, od, oh, ow] = S.convert(acc, S.bf16)


class ModelNew(torch.nn.Module):
    """
    Optimized transposed 3D convolution using Substrate DSL.
    Supports asymmetric kernel sizes.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0),
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.has_bias = bias

        # Validate groups (currently only supporting groups=1)
        if groups != 1:
            raise NotImplementedError("Only groups=1 is currently supported")

        # Initialize weight tensor
        # ConvTranspose3d weight shape: (in_channels, out_channels // groups, kernel_d, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.empty(
            in_channels, out_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]
        ))

        # Initialize bias if needed
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous and on GPU
        x = x.contiguous()

        batch_size = x.shape[0]
        depth_in = x.shape[2]
        height_in = x.shape[3]
        width_in = x.shape[4]

        # Compute output dimensions for ConvTranspose3d
        # output = (input - 1) * stride - 2 * padding + kernel_size + output_padding
        depth_out = (depth_in - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0] + self.output_padding[0]
        height_out = (height_in - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1] + self.output_padding[1]
        width_out = (width_in - 1) * self.stride[2] - 2 * self.padding[2] + self.kernel_size[2] + self.output_padding[2]

        # Create output tensor
        output = torch.empty(
            (batch_size, self.out_channels, depth_out, height_out, width_out),
            dtype=x.dtype, device=x.device
        )

        # Ensure weight is contiguous
        weight = self.weight.data.contiguous()

        # Convert to bf16 for kernel execution
        x_bf16 = x.to(torch.bfloat16)
        weight_bf16 = weight.to(torch.bfloat16)
        output_bf16 = torch.empty(
            (batch_size, self.out_channels, depth_out, height_out, width_out),
            dtype=torch.bfloat16, device=x.device
        )

        # Compute grid and block dimensions
        tiles_d = (depth_out + TILE_D - 1) // TILE_D
        tiles_h = (height_out + TILE_H - 1) // TILE_H
        tiles_w = (width_out + TILE_W - 1) // TILE_W
        total_tiles = tiles_d * tiles_h * tiles_w

        grid = (batch_size, self.out_channels, total_tiles)
        block = (TILE_W, TILE_H, TILE_D)

        # Launch kernel
        conv_transpose3d_kernel[lambda: (grid, block)](
            x_bf16, weight_bf16, output_bf16,
            batch_size, self.in_channels, self.out_channels,
            depth_in, height_in, width_in,
            depth_out, height_out, width_out,
            self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]
        )

        # Convert back to original dtype if needed
        if x.dtype != torch.bfloat16:
            output = output_bf16.to(x.dtype)
        else:
            output = output_bf16

        # Add bias if needed
        if self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1, 1, 1)

        return output
