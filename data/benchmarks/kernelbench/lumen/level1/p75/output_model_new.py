import torch
import torch.nn as nn
import math
import substrate
import substrate.language as S


# Block tile size for output spatial dimensions
TILE_H = 16
TILE_W = 16


@substrate.jit
def conv_transpose2d_kernel(
    input_ptr: S.Pointer(S.bf16),
    weight_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    in_channels: S.i32,
    out_channels: S.i32,
    height: S.i32,
    width: S.i32,
    out_height: S.i32,
    out_width: S.i32,
    kernel_h: S.i32,
    kernel_w: S.i32,
    stride_h: S.i32,
    stride_w: S.i32,
    padding_h: S.i32,
    padding_w: S.i32,
    dilation_h: S.i32,
    dilation_w: S.i32,
    groups: S.i32,
    in_channels_per_group: S.i32,
    out_channels_per_group: S.i32,
):
    # Block layout: block_id(0) = batch * out_channels + output_channel
    #                block_id(1) = tile index
    # Thread layout: thread_id(0) = x within tile, thread_id(1) = y within tile

    block_id = S.block_id(0)
    n = block_id // out_channels
    oc = block_id - n * out_channels

    tile_idx = S.block_id(1)
    tiles_w = (out_width + TILE_W - 1) // TILE_W
    tile_oh = (tile_idx // tiles_w) * TILE_H
    tile_ow = (tile_idx - (tile_idx // tiles_w) * tiles_w) * TILE_W

    # Thread ID within block
    tid_x = S.thread_id(0)
    tid_y = S.thread_id(1)

    oh = tile_oh + tid_y
    ow = tile_ow + tid_x

    # Determine which group this output channel belongs to
    g = oc // out_channels_per_group
    oc_in_group = oc - g * out_channels_per_group

    # Bounds check for output
    if oh >= out_height or ow >= out_width:
        return

    # Create tensor views with proper layouts
    # Input layout: (batch, in_channels, height, width)
    input_stride_w = S.convert(1, S.i32)
    input_stride_h = input_stride_w * width
    input_stride_c = input_stride_h * height
    input_stride_b = input_stride_c * in_channels

    input_layout = S.make_layout(
        (batch_size, in_channels, height, width),
        (input_stride_b, input_stride_c, input_stride_h, input_stride_w)
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, input_layout)

    # Weight layout for ConvTranspose2d: (in_channels, out_channels // groups, kernel_h, kernel_w)
    weight_stride_kw = S.convert(1, S.i32)
    weight_stride_kh = weight_stride_kw * kernel_w
    weight_stride_oc = weight_stride_kh * kernel_h
    weight_stride_ic = weight_stride_oc * out_channels_per_group

    weight_layout = S.make_layout(
        (in_channels, out_channels_per_group, kernel_h, kernel_w),
        (weight_stride_ic, weight_stride_oc, weight_stride_kh, weight_stride_kw)
    )
    weight_tensor = S.make_tensor(weight_ptr, S.bf16, weight_layout)

    # Output layout: (batch, out_channels, out_height, out_width)
    output_stride_w = S.convert(1, S.i32)
    output_stride_h = output_stride_w * out_width
    output_stride_c = output_stride_h * out_height
    output_stride_b = output_stride_c * out_channels

    output_layout = S.make_layout(
        (batch_size, out_channels, out_height, out_width),
        (output_stride_b, output_stride_c, output_stride_h, output_stride_w)
    )
    output_tensor = S.make_tensor(output_ptr, S.bf16, output_layout)

    # Compute convolution transpose for this output element
    # For transposed conv:
    # oh = h_in * stride_h + kh * dilation_h - padding_h
    # => h_in = (oh + padding_h - kh * dilation_h) / stride_h
    #
    # We iterate over kernel positions and compute the potential input position.

    acc = S.convert(0.0, S.f32)

    # Iterate over kernel positions
    for kh in S.range(kernel_h):
        # Compute the input position that would contribute through this kernel position
        # h_in = (oh + padding_h - kh * dilation_h) / stride_h
        num_h = oh + padding_h - kh * dilation_h

        # Check bounds: num_h must be >= 0
        if num_h >= 0:
            # Compute h_in using integer division
            h_in = num_h // stride_h

            # Check if the division is exact and h_in is in bounds
            if h_in < height:
                # Verify: h_in * stride_h == num_h
                if h_in * stride_h == num_h:
                    for kw in S.range(kernel_w):
                        num_w = ow + padding_w - kw * dilation_w

                        if num_w >= 0:
                            w_in = num_w // stride_w

                            if w_in < width:
                                if w_in * stride_w == num_w:
                                    # Accumulate contribution
                                    for ic_idx in S.range(in_channels_per_group):
                                        ic = g * in_channels_per_group + ic_idx
                                        in_val = S.convert(input_tensor[n, ic, h_in, w_in], S.f32)
                                        w_val = S.convert(weight_tensor[ic, oc_in_group, kh, kw], S.f32)
                                        acc = acc + in_val * w_val

    # Convert back to bf16 and store
    output_tensor[n, oc, oh, ow] = S.convert(acc, S.bf16)


class ModelNew(torch.nn.Module):
    """
    Optimized 2D transposed convolution using Substrate DSL.
    Supports asymmetric input, asymmetric kernel, grouped, padded, and dilated configurations.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1), padding: tuple = (0, 0),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias_enabled = bias

        # For ConvTranspose2d: weight shape is (in_channels, out_channels // groups, kernel_h, kernel_w)
        in_channels_per_group = in_channels // groups
        out_channels_per_group = out_channels // groups

        self.weight = nn.Parameter(torch.empty(
            in_channels, out_channels_per_group, kernel_size[0], kernel_size[1]
        ))

        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.weight.shape[1] * self.weight.shape[2] * self.kernel_size[3]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous and on correct device
        x = x.contiguous()

        batch_size, in_channels, height, width = x.shape
        dtype = x.dtype

        # Compute output dimensions for transposed convolution
        # H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + 1
        # W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + 1
        out_height = (height - 1) * self.stride[0] - 2 * self.padding[0] + \
                     self.dilation[0] * (self.kernel_size[0] - 1) + 1
        out_width = (width - 1) * self.stride[1] - 2 * self.padding[1] + \
                    self.dilation[1] * (self.kernel_size[1] - 1) + 1

        # Create output tensor
        output = torch.empty((batch_size, self.out_channels, out_height, out_width),
                            dtype=dtype, device=x.device)

        # Ensure weight is contiguous
        weight = self.weight.data.contiguous()

        # Launch kernel
        tiles_h = (out_height + TILE_H - 1) // TILE_H
        tiles_w = (out_width + TILE_W - 1) // TILE_W
        total_tiles = tiles_h * tiles_w

        in_channels_per_group = self.in_channels // self.groups
        out_channels_per_group = self.out_channels // self.groups

        grid = (batch_size * self.out_channels, total_tiles, 1)
        block = (TILE_W, TILE_H, 1)

        conv_transpose2d_kernel[lambda: (grid, block)](
            x, weight, output,
            batch_size, self.in_channels, self.out_channels,
            height, width, out_height, out_width,
            self.kernel_size[0], self.kernel_size[1],
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups, in_channels_per_group, out_channels_per_group
        )

        # Add bias if needed
        if self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1, 1)

        return output
