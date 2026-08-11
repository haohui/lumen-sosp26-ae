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
def conv3d_asymmetric_kernel(
    input_ptr: S.Pointer(S.bf16),
    weight_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    in_channels: S.i32,
    out_channels: S.i32,
    depth: S.i32,
    height: S.i32,
    width: S.i32,
    out_depth: S.i32,
    out_height: S.i32,
    out_width: S.i32,
    kernel_d: S.i32,
    kernel_h: S.i32,
    kernel_w: S.i32,
    stride_d: S.i32,
    stride_h: S.i32,
    stride_w: S.i32,
    pad_d: S.i32,
    pad_h: S.i32,
    pad_w: S.i32,
    dil_d: S.i32,
    dil_h: S.i32,
    dil_w: S.i32,
    tiles_dhw: S.i32,
    tiles_w: S.i32,
    tiles_hw: S.i32,
):
    """
    3D convolution kernel for asymmetric kernel sizes.
    Grid layout: (batch, out_channel_block, spatial_tile)
    Block layout: (TILE_W, TILE_H, TILE_D) threads per block
    """
    n = S.block_id(0)
    oc_block = S.block_id(1)
    spatial_tile = S.block_id(2)

    # Decode spatial tile to (od, oh, ow) base
    tile_od = (spatial_tile // tiles_hw) * TILE_D
    tile_oh = ((spatial_tile % tiles_hw) // tiles_w) * TILE_H
    tile_ow = (spatial_tile % tiles_w) * TILE_W

    # Thread indices
    tid_x = S.thread_id(0)
    tid_y = S.thread_id(1)
    tid_z = S.thread_id(2)

    # Output coordinates
    od = tile_od + tid_z
    oh = tile_oh + tid_y
    ow = tile_ow + tid_x

    # Compute output channel from block
    oc = oc_block

    # Create tensor views with proper layouts
    # Input layout: (batch, in_channels, depth, height, width)
    input_stride_w = S.convert(1, S.i32)
    input_stride_h = input_stride_w * width
    input_stride_d = input_stride_h * height
    input_stride_c = input_stride_d * depth
    input_stride_b = input_stride_c * in_channels

    input_layout = S.make_layout(
        (batch_size, in_channels, depth, height, width),
        (input_stride_b, input_stride_c, input_stride_d, input_stride_h, input_stride_w)
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, input_layout)

    # Weight layout: (out_channels, in_channels, kernel_d, kernel_h, kernel_w)
    weight_stride_kw = S.convert(1, S.i32)
    weight_stride_kh = weight_stride_kw * kernel_w
    weight_stride_kd = weight_stride_kh * kernel_h
    weight_stride_ic = weight_stride_kd * kernel_d
    weight_stride_oc = weight_stride_ic * in_channels

    weight_layout = S.make_layout(
        (out_channels, in_channels, kernel_d, kernel_h, kernel_w),
        (weight_stride_oc, weight_stride_ic, weight_stride_kd, weight_stride_kh, weight_stride_kw)
    )
    weight_tensor = S.make_tensor(weight_ptr, S.bf16, weight_layout)

    # Output layout: (batch, out_channels, out_depth, out_height, out_width)
    output_stride_w = S.convert(1, S.i32)
    output_stride_h = output_stride_w * out_width
    output_stride_d = output_stride_h * out_height
    output_stride_c = output_stride_d * out_depth
    output_stride_b = output_stride_c * out_channels

    output_layout = S.make_layout(
        (batch_size, out_channels, out_depth, out_height, out_width),
        (output_stride_b, output_stride_c, output_stride_d, output_stride_h, output_stride_w)
    )
    output_tensor = S.make_tensor(output_ptr, S.bf16, output_layout)

    # Bounds check
    if n >= batch_size or oc >= out_channels or od >= out_depth or oh >= out_height or ow >= out_width:
        return

    # Compute convolution - accumulate in f32 for precision
    acc = S.convert(0.0, S.f32)

    # Loop over input channels
    for ic in S.range(in_channels):
        # Loop over kernel depth
        for kd in S.range(kernel_d):
            input_d = od * stride_d + kd * dil_d - pad_d
            if input_d < 0 or input_d >= depth:
                continue

            # Loop over kernel height
            for kh in S.range(kernel_h):
                input_h = oh * stride_h + kh * dil_h - pad_h
                if input_h < 0 or input_h >= height:
                    continue

                # Loop over kernel width
                for kw in S.range(kernel_w):
                    input_w = ow * stride_w + kw * dil_w - pad_w
                    if input_w < 0 or input_w >= width:
                        continue

                    in_val = S.convert(input_tensor[n, ic, input_d, input_h, input_w], S.f32)
                    w_val = S.convert(weight_tensor[oc, ic, kd, kh, kw], S.f32)
                    acc = acc + in_val * w_val

    # Convert back to bf16 and store
    output_tensor[n, oc, od, oh, ow] = S.convert(acc, S.bf16)


class ModelNew(torch.nn.Module):
    """
    Optimized 3D convolution using Substrate DSL for asymmetric kernel sizes.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0),
                 dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias_enabled = bias

        # For groups=1, weight shape is (out_channels, in_channels, kD, kH, kW)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups,
                                               kernel_size[0], kernel_size[1], kernel_size[2]))

        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3] * self.weight.shape[4]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous and on correct device
        x = x.contiguous()

        batch_size, in_channels, depth, height, width = x.shape
        dtype = x.dtype

        # Compute output dimensions
        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        pad_d, pad_h, pad_w = self.padding
        dil_d, dil_h, dil_w = self.dilation

        out_depth = (depth + 2 * pad_d - dil_d * (kernel_d - 1) - 1) // stride_d + 1
        out_height = (height + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
        out_width = (width + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1

        # Create output tensor
        output = torch.empty((batch_size, self.out_channels, out_depth, out_height, out_width),
                             dtype=dtype, device=x.device)

        # Ensure weight is contiguous
        weight = self.weight.data.contiguous()

        # Compute tile counts
        tiles_d = (out_depth + TILE_D - 1) // TILE_D
        tiles_h = (out_height + TILE_H - 1) // TILE_H
        tiles_w = (out_width + TILE_W - 1) // TILE_W
        tiles_hw = tiles_h * tiles_w
        tiles_dhw = tiles_d * tiles_hw

        # Launch kernel
        # Grid: (batch, out_channels, spatial_tiles)
        grid = (batch_size, self.out_channels, tiles_dhw)
        block = (TILE_W, TILE_H, TILE_D)

        conv3d_asymmetric_kernel[lambda: (grid, block)](
            x, weight, output,
            batch_size, in_channels, self.out_channels,
            depth, height, width,
            out_depth, out_height, out_width,
            kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            dil_d, dil_h, dil_w,
            tiles_dhw, tiles_w, tiles_hw
        )

        # Add bias if needed
        if self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1, 1, 1)

        return output
