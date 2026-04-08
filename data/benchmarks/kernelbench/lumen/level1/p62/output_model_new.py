import torch
import torch.nn as nn
import math
import substrate
import substrate.language as S


# Block tile sizes for output spatial dimensions
TILE_H = 8
TILE_W = 8
TILE_OC = 4  # Number of output channels per block


@substrate.jit
def conv2d_asymmetric_kernel(
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
    stride: S.i32,
    padding: S.i32,
    dilation: S.i32,
    tiles_spatial: S.i32,
):
    # Block layout:
    # block_id(0) = batch
    # block_id(1) = output channel tile
    # block_id(2) = spatial tile (h_tile * tiles_spatial + w_tile)

    n = S.block_id(0)
    oc_tile = S.block_id(1)
    spatial_tile = S.block_id(2)

    tile_oh = (spatial_tile // tiles_spatial) * TILE_H
    tile_ow = (spatial_tile % tiles_spatial) * TILE_W

    # Thread IDs
    tid_x = S.thread_id(0)
    tid_y = S.thread_id(1)
    tid_z = S.thread_id(2)

    # Compute output position
    oc_local = tid_z
    oc = oc_tile * TILE_OC + oc_local
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
    # Accumulate in FP32 for precision
    acc = S.convert(0.0, S.f32)

    # Loop over all input channels
    for ic in S.range(in_channels):
        # Loop over kernel positions (asymmetric kernel)
        for kh in S.range(kernel_h):
            input_h = oh * stride + kh * dilation - padding

            # Check height bounds
            if input_h < 0 or input_h >= height:
                continue

            for kw in S.range(kernel_w):
                input_w = ow * stride + kw * dilation - padding

                # Check width bounds
                if input_w < 0 or input_w >= width:
                    continue

                # Load input and weight values
                in_val = S.convert(input_tensor[n, ic, input_h, input_w], S.f32)
                w_val = S.convert(weight_tensor[oc, ic, kh, kw], S.f32)
                acc = acc + in_val * w_val

    # Store result
    output_tensor[n, oc, oh, ow] = S.convert(acc, S.bf16)


class ModelNew(torch.nn.Module):
    """
    Optimized 2D convolution with asymmetric kernel using Substrate DSL.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias

        # Weight tensor for Conv2d: (out_channels, in_channels // groups, kernel_h, kernel_w)
        kernel_h, kernel_w = self.kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_h, kernel_w))

        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous and in BF16
        x = x.contiguous()
        original_dtype = x.dtype
        if x.dtype != torch.bfloat16:
            x = x.to(dtype=torch.bfloat16)

        batch_size, in_channels, height, width = x.shape
        kernel_h, kernel_w = self.kernel_size

        # Compute output dimensions
        out_height = (height + 2 * self.padding - self.dilation * (kernel_h - 1) - 1) // self.stride + 1
        out_width = (width + 2 * self.padding - self.dilation * (kernel_w - 1) - 1) // self.stride + 1

        # Create output tensor
        output = torch.empty((batch_size, self.out_channels, out_height, out_width),
                            dtype=torch.bfloat16, device=x.device)

        # Ensure weight is contiguous and in BF16
        weight = self.weight.data.contiguous()
        if weight.dtype != torch.bfloat16:
            weight = weight.to(dtype=torch.bfloat16)

        # Calculate grid dimensions
        tiles_h = (out_height + TILE_H - 1) // TILE_H
        tiles_w = (out_width + TILE_W - 1) // TILE_W
        tiles_spatial = tiles_w
        tiles_oc = (self.out_channels + TILE_OC - 1) // TILE_OC

        grid = (batch_size, tiles_oc, tiles_h * tiles_w)
        block = (TILE_W, TILE_H, TILE_OC)

        # Launch kernel
        conv2d_asymmetric_kernel[lambda: (grid, block)](
            x, weight, output,
            batch_size, in_channels, self.out_channels,
            height, width, out_height, out_width,
            kernel_h, kernel_w, self.stride, self.padding, self.dilation,
            tiles_spatial
        )

        # Add bias if needed
        if self.bias_param is not None:
            bias = self.bias_param if self.bias_param.dtype == torch.bfloat16 else self.bias_param.to(torch.bfloat16)
            output = output + bias.view(1, -1, 1, 1)

        # Convert back to original dtype if needed
        if original_dtype != torch.bfloat16:
            output = output.to(original_dtype)

        return output
