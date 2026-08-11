import torch
import torch.nn as nn
import math
import substrate
import substrate.language as S

# Block tile size for output spatial dimensions
TILE_H = 16
TILE_W = 16


@substrate.jit
def conv2d_standard_kernel(
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
    pad_h: S.i32,
    pad_w: S.i32,
    dilation_h: S.i32,
    dilation_w: S.i32,
    tiles_w: S.i32,
):
    """Standard 2D convolution kernel with asymmetric kernel size."""
    # Decode block IDs
    # block_id(0) = batch
    # block_id(1) = output channel
    # block_id(2) = spatial tile (linearized)

    n = S.block_id(0)
    oc = S.block_id(1)

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

    # Weight layout: (out_channels, in_channels, kernel_h, kernel_w)
    weight_stride_kw = S.convert(1, S.i32)
    weight_stride_kh = weight_stride_kw * kernel_w
    weight_stride_ic = weight_stride_kh * kernel_h
    weight_stride_oc = weight_stride_ic * in_channels

    weight_layout = S.make_layout(
        (out_channels, in_channels, kernel_h, kernel_w),
        (weight_stride_oc, weight_stride_ic, weight_stride_kh, weight_stride_kw)
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

    # Bounds check
    if n >= batch_size or oc >= out_channels or oh >= out_height or ow >= out_width:
        return

    # Compute convolution for this output element
    # Standard convolution: sum over all input channels and kernel positions
    acc = S.convert(0.0, S.f32)

    for ic in S.range(in_channels):
        for kh in S.range(kernel_h):
            for kw in S.range(kernel_w):
                # Input coordinates with stride, padding, dilation
                ih = oh * stride_h + kh * dilation_h - pad_h
                iw = ow * stride_w + kw * dilation_w - pad_w

                # Check input bounds (handle padding)
                if ih < 0 or ih >= height or iw < 0 or iw >= width:
                    continue

                # Load input and weight values
                in_val = S.convert(input_tensor[n, ic, ih, iw], S.f32)
                w_val = S.convert(weight_tensor[oc, ic, kh, kw], S.f32)

                acc = acc + in_val * w_val

    # Store output (convert back to bf16)
    output_tensor[n, oc, oh, ow] = S.convert(acc, S.bf16)


class ModelNew(torch.nn.Module):
    """
    Optimized standard 2D convolution using Substrate DSL.
    Supports asymmetric kernel sizes with configurable stride, padding, and dilation.
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

        # Initialize weight tensor for standard convolution
        # Shape: (out_channels, in_channels // groups, kernel_h, kernel_w)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size[0], kernel_size[1])
        )

        # Initialize bias if needed
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.in_channels // self.groups * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()

        batch_size, in_channels, height, width = x.shape

        # Convert to BF16 for computation
        input_bf16 = x.to(torch.bfloat16)

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dilation_h, dilation_w = self.dilation

        # Compute output dimensions
        out_height = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        out_width = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

        # Create output tensor in BF16
        output_bf16 = torch.empty(
            (batch_size, self.out_channels, out_height, out_width),
            dtype=torch.bfloat16, device=x.device
        )

        # Ensure weight is contiguous and in BF16
        weight_bf16 = self.weight.data.to(torch.bfloat16).contiguous()

        # Compute grid and block dimensions
        tiles_h = (out_height + TILE_H - 1) // TILE_H
        tiles_w = (out_width + TILE_W - 1) // TILE_W

        grid = (batch_size, self.out_channels, tiles_h * tiles_w)
        block = (TILE_W, TILE_H, 1)

        # Launch kernel
        conv2d_standard_kernel[lambda: (grid, block)](
            input_bf16, weight_bf16, output_bf16,
            batch_size, in_channels, self.out_channels,
            height, width, out_height, out_width,
            kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w,
            dilation_h, dilation_w,
            tiles_w
        )

        # Convert output back to original dtype
        output = output_bf16.to(x.dtype)

        # Add bias if needed
        if self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1, 1)

        return output
