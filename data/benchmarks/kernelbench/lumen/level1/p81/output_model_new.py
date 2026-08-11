import torch
import torch.nn as nn
import math
import substrate
import substrate.language as S


# Block tile sizes for output spatial dimensions
TILE_H = 8
TILE_W = 16
BLOCK_THREADS_H = TILE_H
BLOCK_THREADS_W = TILE_W


@substrate.jit
def conv_transpose2d_kernel(
    input_ptr: S.Pointer(S.bf16),
    weight_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    in_channels: S.i32,
    out_channels: S.i32,
    height_in: S.i32,
    width_in: S.i32,
    height_out: S.i32,
    width_out: S.i32,
    kernel_size: S.i32,
    stride: S.i32,
    padding: S.i32,
    dilation: S.i32,
    tiles_w: S.i32,
):
    """
    Transposed 2D convolution kernel.

    Grid: (batch, out_channels, num_spatial_tiles)
    Block: (TILE_W, TILE_H) threads, each handling one output spatial position
    """
    n = S.block_id(0)
    oc = S.block_id(1)
    linear_id = S.block_id(2)

    # Decode spatial tile
    tiles_h = (height_out + TILE_H - 1) // TILE_H
    tile_oh = (linear_id // tiles_w) * TILE_H
    tile_ow = (linear_id % tiles_w) * TILE_W

    # Thread ID within block
    tid_x = S.thread_id(0)
    tid_y = S.thread_id(1)

    # Compute output position
    oh = tile_oh + tid_y
    ow = tile_ow + tid_x

    # Bounds check
    if n >= batch_size or oc >= out_channels or oh >= height_out or ow >= width_out:
        return

    # Create layout for input: (batch, in_channels, height_in, width_in)
    input_stride_w = S.convert(1, S.i32)
    input_stride_h = input_stride_w * width_in
    input_stride_c = input_stride_h * height_in
    input_stride_b = input_stride_c * in_channels

    input_layout = S.make_layout(
        (batch_size, in_channels, height_in, width_in),
        (input_stride_b, input_stride_c, input_stride_h, input_stride_w)
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, input_layout)

    # Create layout for weight: (in_channels, out_channels, kernel_size, kernel_size)
    weight_stride_kw = S.convert(1, S.i32)
    weight_stride_kh = weight_stride_kw * kernel_size
    weight_stride_oc = weight_stride_kh * kernel_size
    weight_stride_ic = weight_stride_oc * out_channels

    weight_layout = S.make_layout(
        (in_channels, out_channels, kernel_size, kernel_size),
        (weight_stride_ic, weight_stride_oc, weight_stride_kh, weight_stride_kw)
    )
    weight_tensor = S.make_tensor(weight_ptr, S.bf16, weight_layout)

    # Create layout for output: (batch, out_channels, height_out, width_out)
    output_stride_w = S.convert(1, S.i32)
    output_stride_h = output_stride_w * width_out
    output_stride_c = output_stride_h * height_out
    output_stride_b = output_stride_c * out_channels

    output_layout = S.make_layout(
        (batch_size, out_channels, height_out, width_out),
        (output_stride_b, output_stride_c, output_stride_h, output_stride_w)
    )
    output_tensor = S.make_tensor(output_ptr, S.bf16, output_layout)

    # Accumulate contributions
    acc = S.convert(0.0, S.f32)

    # Loop over kernel elements
    for kh in S.range(kernel_size):
        for kw in S.range(kernel_size):
            # For transposed conv:
            # output_pos = input_pos * stride - padding + kernel_idx * dilation
            # Rearranged to find input from output:
            # input_pos = (output_pos + padding - kernel_idx * dilation) / stride

            khd = kh * dilation
            kwd = kw * dilation

            num_h = oh + padding - khd
            num_w = ow + padding - kwd

            # Check if this (kh, kw) contributes to output position (oh, ow)
            # Condition: (oh + padding - khd) % stride == 0
            #           (ow + padding - kwd) % stride == 0
            if num_h >= 0:
                ih_candidate = num_h // stride
                check_h = ih_candidate * stride

                if check_h == num_h:
                    ih = ih_candidate
                    if ih < height_in:
                        if num_w >= 0:
                            iw_candidate = num_w // stride
                            check_w = iw_candidate * stride

                            if check_w == num_w:
                                iw = iw_candidate
                                if iw < width_in:
                                    # Accumulate over all input channels
                                    for ic in S.range(in_channels):
                                        in_val = S.convert(input_tensor[n, ic, ih, iw], S.f32)
                                        w_val = S.convert(weight_tensor[ic, oc, kh, kw], S.f32)
                                        acc = acc + in_val * w_val

    # Store result
    output_tensor[n, oc, oh, ow] = S.convert(acc, S.bf16)


class ModelNew(torch.nn.Module):
    """
    Optimized transposed 2D convolution using Substrate DSL.
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

        # Weight for ConvTranspose2d has shape (in_channels, out_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))

        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous input
        x = x.contiguous()

        batch_size = int(x.shape[0])
        in_channels = int(x.shape[1])
        height_in = int(x.shape[2])
        width_in = int(x.shape[3])
        input_dtype = x.dtype

        # Move to GPU if needed
        if not x.is_cuda:
            x = x.cuda()

        # Compute output dimensions using PyTorch formula for ConvTranspose2d
        # output = (input - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1
        height_out = (height_in - 1) * self.stride - 2 * self.padding + \
                     self.dilation * (self.kernel_size - 1) + 1
        width_out = (width_in - 1) * self.stride - 2 * self.padding + \
                    self.dilation * (self.kernel_size - 1) + 1

        # Create output tensor
        output = torch.zeros(
            (batch_size, self.out_channels, height_out, width_out),
            dtype=x.dtype, device=x.device
        )

        # Ensure weight is contiguous
        weight = self.weight.data.contiguous()

        # Launch configuration
        tiles_h = (height_out + TILE_H - 1) // TILE_H
        tiles_w = (width_out + TILE_W - 1) // TILE_W

        grid = (batch_size, self.out_channels, tiles_h * tiles_w)
        block = (BLOCK_THREADS_W, BLOCK_THREADS_H, 1)

        conv_transpose2d_kernel[lambda: (grid, block)](
            x, weight, output,
            batch_size, in_channels, self.out_channels,
            height_in, width_in, height_out, width_out,
            self.kernel_size, self.stride, self.padding, self.dilation,
            tiles_w
        )

        # Add bias if needed
        if self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1, 1)

        return output
