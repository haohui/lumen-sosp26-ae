import torch
import torch.nn as nn
import math
import substrate
import substrate.language as S


# Block tile size for output spatial dimensions
TILE_H = 16
TILE_W = 16


@substrate.jit
def depthwise_conv2d_full_kernel(
    input_ptr: S.Pointer(S.bf16),
    weight_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    channels: S.i32,
    height: S.i32,
    width: S.i32,
    out_height: S.i32,
    out_width: S.i32,
    kernel_size_h: S.i32,
    kernel_size_w: S.i32,
    stride_h: S.i32,
    stride_w: S.i32,
    padding_h: S.i32,
    padding_w: S.i32,
    dilation_h: S.i32,
    dilation_w: S.i32,
    tiles_w: S.i32,
):
    # Decode block IDs
    # block_id(0) = batch
    # block_id(1) = channel
    # block_id(2) = tile_h * tiles_w + tile_w

    n = S.block_id(0)
    c = S.block_id(1)

    linear_id = S.block_id(2)
    tiles_h = (out_height + TILE_H - 1) // TILE_H

    tile_oh = (linear_id // tiles_w) * TILE_H
    tile_ow = (linear_id % tiles_w) * TILE_W

    # Thread ID within block
    tid_x = S.thread_id(0)
    tid_y = S.thread_id(1)

    oh = tile_oh + tid_y
    ow = tile_ow + tid_x

    # Create tensor views with proper layouts
    # Input layout: (batch, channels, height, width)
    input_stride_w = S.convert(1, S.i32)
    input_stride_h = input_stride_w * width
    input_stride_c = input_stride_h * height
    input_stride_b = input_stride_c * channels

    input_layout = S.make_layout(
        (batch_size, channels, height, width),
        (input_stride_b, input_stride_c, input_stride_h, input_stride_w)
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, input_layout)

    # Weight layout: (channels, 1, kernel_h, kernel_w)
    # For depthwise conv with asymmetric kernel (kernel_size_h, kernel_size_w)
    # PyTorch Conv2d weight is contiguous with shape (out_channels, in_channels/groups, kH, kW)
    # For depthwise: (in_channels, 1, kH, kW)
    # Strides for contiguous layout: (kH*kW, kH*kW, kW, 1)
    weight_stride_kw = S.convert(1, S.i32)
    weight_stride_kh = weight_stride_kw * kernel_size_w
    weight_stride_c = weight_stride_kh * kernel_size_h

    weight_layout = S.make_layout(
        (channels, 1, kernel_size_h, kernel_size_w),
        (weight_stride_c, weight_stride_c, weight_stride_kh, weight_stride_kw)
    )
    weight_tensor = S.make_tensor(weight_ptr, S.bf16, weight_layout)

    # Output layout: (batch, channels, out_height, out_width)
    output_stride_w = S.convert(1, S.i32)
    output_stride_h = output_stride_w * out_width
    output_stride_c = output_stride_h * out_height
    output_stride_b = output_stride_c * channels

    output_layout = S.make_layout(
        (batch_size, channels, out_height, out_width),
        (output_stride_b, output_stride_c, output_stride_h, output_stride_w)
    )
    output_tensor = S.make_tensor(output_ptr, S.bf16, output_layout)

    # Check bounds
    if n >= batch_size or c >= channels or oh >= out_height or ow >= out_width:
        return

    # Compute convolution for this output element
    # Kernel is (kernel_size_h, kernel_size_w) - convolves along both dimensions
    # For depthwise: each channel uses its own kernel
    acc = S.convert(0.0, S.f32)

    for kh in S.range(kernel_size_h):
        input_h = oh * stride_h + kh * dilation_h - padding_h
        if input_h < 0 or input_h >= height:
            continue

        for kw in S.range(kernel_size_w):
            input_w = ow * stride_w + kw * dilation_w - padding_w
            if input_w < 0 or input_w >= width:
                continue

            in_val = S.convert(input_tensor[n, c, input_h, input_w], S.f32)
            w_val = S.convert(weight_tensor[c, 0, kh, kw], S.f32)
            acc = acc + in_val * w_val

    # Convert back to bf16 and store
    output_tensor[n, c, oh, ow] = S.convert(acc, S.bf16)


class ModelNew(torch.nn.Module):
    """
    Optimized depthwise 2D convolution using Substrate DSL.
    Uses an asymmetric kernel (kernel_size_h, kernel_size_w) that convolves along both dimensions.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int,
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0,
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()

        self.in_channels = in_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.bias = bias

        # Initialize weight tensor for depthwise convolution
        # For Conv2d with groups=in_channels: weight shape is (in_channels, 1, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size_h, kernel_size_w))

        # Initialize bias if needed
        if bias:
            self.bias_param = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter('bias_param', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous and on correct device
        x = x.contiguous()

        batch_size, in_channels, height, width = x.shape
        dtype = x.dtype

        # Compute output dimensions
        # out_height = (height + 2*padding_h - dilation_h*(kernel_size_h-1) - 1) // stride_h + 1
        # out_width = (width + 2*padding_w - dilation_w*(kernel_size_w-1) - 1) // stride_w + 1
        out_height = (height + 2 * self.padding_h - self.dilation_h * (self.kernel_size_h - 1) - 1) // self.stride_h + 1
        out_width = (width + 2 * self.padding_w - self.dilation_w * (self.kernel_size_w - 1) - 1) // self.stride_w + 1

        # Create output tensor (same dtype as input to match reference model behavior)
        output = torch.empty((batch_size, self.in_channels, out_height, out_width),
                            dtype=dtype, device=x.device)

        # Ensure weight is contiguous
        weight = self.weight.data.contiguous()

        # Launch kernel
        tiles_h = (out_height + TILE_H - 1) // TILE_H
        tiles_w = (out_width + TILE_W - 1) // TILE_W

        grid = (batch_size, self.in_channels, tiles_h * tiles_w)
        block = (TILE_W, TILE_H, 1)

        depthwise_conv2d_full_kernel[lambda: (grid, block)](
            x, weight, output,
            batch_size, in_channels, height, width,
            out_height, out_width,
            self.kernel_size_h, self.kernel_size_w,
            self.stride_h, self.stride_w,
            self.padding_h, self.padding_w,
            self.dilation_h, self.dilation_w,
            tiles_w
        )

        # Add bias if needed
        if self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1, 1)

        return output
