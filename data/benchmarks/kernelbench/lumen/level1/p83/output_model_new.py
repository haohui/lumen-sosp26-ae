import torch
import torch.nn as nn
import math
import substrate
import substrate.language as S


# Block tile size for output spatial dimensions
TILE_H = 16
TILE_W = 16


@substrate.jit
def depthwise_conv2d_asymmetric_kernel(
    input_ptr: S.Pointer(S.bf16),
    weight_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    channels: S.i32,
    height: S.i32,
    width: S.i32,
    out_height: S.i32,
    out_width: S.i32,
    kernel_size: S.i32,
    stride: S.i32,
    padding: S.i32,
    dilation: S.i32,
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
    # For depthwise conv: (in_channels, 1, kernel_size, 1)
    weight_stride_kw = S.convert(1, S.i32)
    weight_stride_kh = weight_stride_kw * 1  # kernel_w = 1
    weight_stride_c = weight_stride_kh * kernel_size

    weight_layout = S.make_layout(
        (channels, 1, kernel_size, 1),
        (weight_stride_c, weight_stride_kh, weight_stride_kw, weight_stride_kw)
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
    # Kernel is (kernel_size, 1) - only convolves along height dimension
    # For depthwise: each channel uses its own kernel
    acc = S.convert(0.0, S.f32)

    for kh in S.range(kernel_size):
        input_h = oh * stride + kh * dilation - padding
        if input_h < 0 or input_h >= height:
            continue

        # For width: kernel_size = 1, so input_w = ow * stride - padding
        input_w = ow * stride - padding
        if input_w < 0 or input_w >= width:
            continue

        in_val = S.convert(input_tensor[n, c, input_h, input_w], S.f32)
        w_val = S.convert(weight_tensor[c, 0, kh, 0], S.f32)
        acc = acc + in_val * w_val

    # Convert back to bf16 and store
    output_tensor[n, c, oh, ow] = S.convert(acc, S.bf16)


class ModelNew(torch.nn.Module):
    """
    Optimized depthwise 2D convolution using Substrate DSL.
    Uses an asymmetric kernel (kernel_size, 1) that only convolves along the height dimension.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        # Initialize weight tensor for depthwise convolution
        # For Conv2d with groups=in_channels: weight shape is (in_channels, 1, kernel_h, kernel_w)
        # Our kernel is (kernel_size, 1) in (H, W)
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, 1))

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
        # For asymmetric kernel (kernel_size, 1):
        # out_height = (height + 2*padding - dilation*(kernel_size-1) - 1) // stride + 1
        # out_width = (width + 2*padding - dilation*(1-1) - 1) // stride + 1
        out_height = (height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_width = (width + 2 * self.padding - self.dilation * (1 - 1) - 1) // self.stride + 1

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

        depthwise_conv2d_asymmetric_kernel[lambda: (grid, block)](
            x, weight, output,
            batch_size, in_channels, height, width,
            out_height, out_width,
            self.kernel_size, self.stride, self.padding, self.dilation,
            tiles_w
        )

        # Add bias if needed
        if self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1, 1)

        return output
