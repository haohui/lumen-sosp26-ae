import torch
import torch.nn as nn
import math
import substrate
import substrate.language as S


# Tile sizes for output spatial dimensions
TILE_H = 8
TILE_W = 8


@substrate.jit
def conv2d_kernel(
    input_ptr: S.Pointer(S.bf16),
    weight_ptr: S.Pointer(S.bf16),
    bias_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    in_channels: S.i32,
    in_height: S.i32,
    in_width: S.i32,
    out_channels: S.i32,
    out_height: S.i32,
    out_width: S.i32,
    kernel_size: S.i32,
    stride: S.i32,
    padding: S.i32,
    has_bias: S.i32,
    out_width_tiles: S.i32,
):
    # Block mapping:
    # block_id(0) = batch
    # block_id(1) = output channel
    # block_id(2) = spatial tile (oh_tile * out_width_tiles + ow_tile)

    n = S.block_id(0)
    oc = S.block_id(1)
    spatial_id = S.block_id(2)

    tile_oh = (spatial_id // out_width_tiles) * TILE_H
    tile_ow = (spatial_id % out_width_tiles) * TILE_W

    # Thread indices
    tid_x = S.thread_id(0)
    tid_y = S.thread_id(1)

    # Output coordinates for this thread
    oh = tile_oh + tid_y
    ow = tile_ow + tid_x

    # Bounds check
    if n >= batch_size or oc >= out_channels:
        return
    if oh >= out_height or ow >= out_width:
        return

    # Create input tensor view
    # Layout: (batch, in_channels, height, width)
    in_stride_w = S.convert(1, S.i32)
    in_stride_h = in_stride_w * in_width
    in_stride_c = in_stride_h * in_height
    in_stride_b = in_stride_c * in_channels

    in_layout = S.make_layout(
        (batch_size, in_channels, in_height, in_width),
        (in_stride_b, in_stride_c, in_stride_h, in_stride_w)
    )
    in_tensor = S.make_tensor(input_ptr, S.bf16, in_layout)

    # Create weight tensor view
    # Layout: (out_channels, in_channels, kernel_h, kernel_w)
    w_stride_w = S.convert(1, S.i32)
    w_stride_h = w_stride_w * kernel_size
    w_stride_c = w_stride_h * kernel_size
    w_stride_oc = w_stride_c * in_channels

    w_layout = S.make_layout(
        (out_channels, in_channels, kernel_size, kernel_size),
        (w_stride_oc, w_stride_c, w_stride_h, w_stride_w)
    )
    w_tensor = S.make_tensor(weight_ptr, S.bf16, w_layout)

    # Create output tensor view
    # Layout: (batch, out_channels, out_height, out_width)
    out_stride_w = S.convert(1, S.i32)
    out_stride_h = out_stride_w * out_width
    out_stride_c = out_stride_h * out_height
    out_stride_b = out_stride_c * out_channels

    out_layout = S.make_layout(
        (batch_size, out_channels, out_height, out_width),
        (out_stride_b, out_stride_c, out_stride_h, out_stride_w)
    )
    out_tensor = S.make_tensor(output_ptr, S.bf16, out_layout)

    # Compute convolution
    acc = S.convert(0.0, S.f32)

    for ic in S.range(in_channels):
        for kh in S.range(kernel_size):
            for kw in S.range(kernel_size):
                # Input position
                ih = oh * stride + kh - padding
                iw = ow * stride + kw - padding

                # Check input bounds (padding)
                if ih >= 0 and ih < in_height and iw >= 0 and iw < in_width:
                    in_val = S.convert(in_tensor[n, ic, ih, iw], S.f32)
                    w_val = S.convert(w_tensor[oc, ic, kh, kw], S.f32)
                    acc = acc + in_val * w_val

    # Add bias if present
    if has_bias != 0:
        bias_layout = S.make_layout((out_channels,), (S.convert(1, S.i32),))
        bias_tensor = S.make_tensor(bias_ptr, S.bf16, bias_layout)
        bias_val = S.convert(bias_tensor[oc], S.f32)
        acc = acc + bias_val

    # Store output
    out_tensor[n, oc, oh, ow] = S.convert(acc, S.bf16)


class ModelNew(nn.Module):
    """
    Optimized Conv2d using Substrate DSL for AMD GPU.
    Targets BF16 precision with FP32 accumulation.
    """
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        # Conv2d parameters
        self.in_channels = 3
        self.out_channels = 96
        self.kernel_size = 11
        self.stride = 4
        self.padding = 2

        # Initialize weights (out_channels, in_channels, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.empty(
            self.out_channels, self.in_channels,
            self.kernel_size, self.kernel_size
        ))
        self.bias = nn.Parameter(torch.empty(self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous and BF16
        x = x.contiguous()
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        batch_size, in_channels, in_height, in_width = x.shape

        # Compute output dimensions
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Prepare weights
        weight = self.weight.data.to(torch.bfloat16).contiguous()
        bias = self.bias.data.to(torch.bfloat16).contiguous()

        # Create output tensor
        output = torch.empty(
            (batch_size, self.out_channels, out_height, out_width),
            dtype=torch.bfloat16, device=x.device
        )

        # Calculate grid and block dimensions
        out_width_tiles = (out_width + TILE_W - 1) // TILE_W
        out_height_tiles = (out_height + TILE_H - 1) // TILE_H

        grid = (batch_size, self.out_channels, out_height_tiles * out_width_tiles)
        block = (TILE_W, TILE_H, 1)

        # Launch kernel
        conv2d_kernel[lambda: (grid, block)](
            x, weight, bias, output,
            batch_size, in_channels, in_height, in_width,
            self.out_channels, out_height, out_width,
            self.kernel_size, self.stride, self.padding,
            1, out_width_tiles  # has_bias, out_width_tiles
        )

        return output
