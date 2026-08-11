import torch
import torch.nn as nn
import math
import substrate
import substrate.language as S


# Block tile sizes for output spatial dimensions
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
    in_depth: S.i32,
    in_height: S.i32,
    in_width: S.i32,
    out_depth: S.i32,
    out_height: S.i32,
    out_width: S.i32,
    kernel_d: S.i32,
    kernel_h: S.i32,
    kernel_w: S.i32,
    stride_d: S.i32,
    stride_h: S.i32,
    stride_w: S.i32,
    padding_d: S.i32,
    padding_h: S.i32,
    padding_w: S.i32,
    groups: S.i32,
):
    """
    ConvTranspose3d kernel.
    Each thread computes one output element.

    For transposed convolution:
    - output[n, c_out, d_out, h_out, w_out] = sum over input positions that contribute
    - d_out = d_in * stride_d + kd - padding_d
    - So: d_in = (d_out + padding_d - kd) / stride_d
    """
    # Decode block IDs
    # block_id(0) = batch * out_channels + out_channel
    # block_id(1) = tile_d index
    # block_id(2) = tile_h * tiles_w + tile_w
    bc_block = S.block_id(0)
    n = bc_block // out_channels
    c_out = bc_block % out_channels

    tile_d_idx = S.block_id(1)
    tile_hw_linear = S.block_id(2)

    tiles_h = (out_height + TILE_H - 1) // TILE_H
    tiles_w = (out_width + TILE_W - 1) // TILE_W

    tile_h_idx = tile_hw_linear // tiles_w
    tile_w_idx = tile_hw_linear % tiles_w

    # Thread IDs within block
    tid_d = S.thread_id(0)
    tid_h = S.thread_id(1)
    tid_w = S.thread_id(2)

    # Compute output coordinates
    d_out = tile_d_idx * TILE_D + tid_d
    h_out = tile_h_idx * TILE_H + tid_h
    w_out = tile_w_idx * TILE_W + tid_w

    # Bounds check
    if n >= batch_size or c_out >= out_channels or d_out >= out_depth or h_out >= out_height or w_out >= out_width:
        return

    # Compute group for this output channel
    out_channels_per_group = out_channels // groups
    in_channels_per_group = in_channels // groups
    g = c_out // out_channels_per_group

    # Create tensor views with proper layouts
    # Input layout: (batch, in_channels, depth, height, width)
    input_stride_w = S.convert(1, S.i32)
    input_stride_h = input_stride_w * in_width
    input_stride_d = input_stride_h * in_height
    input_stride_c = input_stride_d * in_depth
    input_stride_b = input_stride_c * in_channels

    input_layout = S.make_layout(
        (batch_size, in_channels, in_depth, in_height, in_width),
        (input_stride_b, input_stride_c, input_stride_d, input_stride_h, input_stride_w)
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, input_layout)

    # Weight layout: (in_channels, out_channels_per_group, kernel_d, kernel_h, kernel_w)
    weight_stride_w = S.convert(1, S.i32)
    weight_stride_h = weight_stride_w * kernel_w
    weight_stride_d = weight_stride_h * kernel_h
    weight_stride_oc = weight_stride_d * kernel_d
    weight_stride_ic = weight_stride_oc * out_channels_per_group

    weight_layout = S.make_layout(
        (in_channels, out_channels_per_group, kernel_d, kernel_h, kernel_w),
        (weight_stride_ic, weight_stride_oc, weight_stride_d, weight_stride_h, weight_stride_w)
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

    # Local output channel within the group
    c_out_local = c_out % out_channels_per_group

    # Accumulate in FP32
    acc = S.convert(0.0, S.f32)

    # Iterate over kernel positions
    for kd in S.range(kernel_d):
        # Compute input depth that contributes to this output position
        # d_out = d_in * stride_d + kd - padding_d
        # d_in = (d_out + padding_d - kd) / stride_d
        d_in_scaled = d_out + padding_d - kd
        # Check if this maps to a valid input position
        # We need d_in_scaled to be divisible by stride_d and in bounds
        if d_in_scaled >= 0:
            d_in = d_in_scaled // stride_d
            d_in_rem = d_in_scaled % stride_d
            if d_in_rem == 0 and d_in < in_depth:
                for kh in S.range(kernel_h):
                    h_in_scaled = h_out + padding_h - kh
                    if h_in_scaled >= 0:
                        h_in = h_in_scaled // stride_h
                        h_in_rem = h_in_scaled % stride_h
                        if h_in_rem == 0 and h_in < in_height:
                            for kw in S.range(kernel_w):
                                w_in_scaled = w_out + padding_w - kw
                                if w_in_scaled >= 0:
                                    w_in = w_in_scaled // stride_w
                                    w_in_rem = w_in_scaled % stride_w
                                    if w_in_rem == 0 and w_in < in_width:
                                        # Iterate over input channels in this group
                                        for c_in_local in S.range(in_channels_per_group):
                                            c_in = g * in_channels_per_group + c_in_local
                                            # Get input value
                                            in_val = S.convert(input_tensor[n, c_in, d_in, h_in, w_in], S.f32)
                                            # Get weight value
                                            w_val = S.convert(weight_tensor[c_in, c_out_local, kd, kh, kw], S.f32)
                                            acc = acc + in_val * w_val

    # Store result
    output_tensor[n, c_out, d_out, h_out, w_out] = S.convert(acc, S.bf16)


class ModelNew(torch.nn.Module):
    """
    Optimized 3D transposed convolution using Substrate DSL.
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
        self.bias_enabled = bias

        # For ConvTranspose3d, weight shape is:
        # (in_channels, out_channels // groups, *kernel_size)
        out_channels_per_group = out_channels // groups
        self.weight = nn.Parameter(torch.empty(
            in_channels, out_channels_per_group, kernel_size[0], kernel_size[1], kernel_size[2]
        ))

        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()

        batch_size, in_channels, in_depth, in_height, in_width = x.shape

        # Compute output dimensions
        # H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
        out_depth = (in_depth - 1) * self.stride[0] - 2 * self.padding[0] + (self.kernel_size[0] - 1) + self.output_padding[0] + 1
        out_height = (in_height - 1) * self.stride[1] - 2 * self.padding[1] + (self.kernel_size[1] - 1) + self.output_padding[1] + 1
        out_width = (in_width - 1) * self.stride[2] - 2 * self.padding[2] + (self.kernel_size[2] - 1) + self.output_padding[2] + 1

        # Create output tensor
        output = torch.zeros((batch_size, self.out_channels, out_depth, out_height, out_width),
                            dtype=x.dtype, device=x.device)

        # Ensure weight is contiguous
        weight = self.weight.data.contiguous()

        # Compute grid and block dimensions
        tiles_d = (out_depth + TILE_D - 1) // TILE_D
        tiles_h = (out_height + TILE_H - 1) // TILE_H
        tiles_w = (out_width + TILE_W - 1) // TILE_W

        grid = (batch_size * self.out_channels, tiles_d, tiles_h * tiles_w)
        block = (TILE_D, TILE_H, TILE_W)

        conv_transpose3d_kernel[lambda: (grid, block)](
            x, weight, output,
            batch_size, self.in_channels, self.out_channels,
            in_depth, in_height, in_width,
            out_depth, out_height, out_width,
            self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.groups
        )

        # Add bias if needed
        if self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1, 1, 1)

        return output
