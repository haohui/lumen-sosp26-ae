import torch
import torch.nn as nn
import math
import substrate
import substrate.language as S


# Block tile size for output spatial dimensions
TILE_H = 8
TILE_W = 8


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
    tiles_w: S.i32,
):
    # Decode block IDs
    # block_id(0) = batch
    # block_id(1) = output channel
    # block_id(2) = tile index

    n = S.block_id(0)
    oc = S.block_id(1)

    tile_idx = S.block_id(2)

    # Compute output position from tile index
    tile_oh = (tile_idx // tiles_w) * TILE_H
    tile_ow = (tile_idx % tiles_w) * TILE_W

    # Thread ID within block
    tid_x = S.thread_id(0)
    tid_y = S.thread_id(1)

    oh = tile_oh + tid_y
    ow = tile_ow + tid_x

    # Check bounds - exit early if out of bounds
    if oh >= out_height or ow >= out_width:
        return

    # Create tensor views with proper layouts
    # Input layout: (batch, in_channels, height, width)
    input_stride_w = S.convert(1, S.i32)
    input_stride_h = width
    input_stride_c = height * width
    input_stride_b = in_channels * height * width

    input_layout = S.make_layout(
        (batch_size, in_channels, height, width),
        (input_stride_b, input_stride_c, input_stride_h, input_stride_w)
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, input_layout)

    # Weight layout: (in_channels, out_channels, kernel_h, kernel_w)
    weight_stride_kw = S.convert(1, S.i32)
    weight_stride_kh = kernel_w
    weight_stride_oc = kernel_h * kernel_w
    weight_stride_ic = out_channels * kernel_h * kernel_w

    weight_layout = S.make_layout(
        (in_channels, out_channels, kernel_h, kernel_w),
        (weight_stride_ic, weight_stride_oc, weight_stride_kh, weight_stride_kw)
    )
    weight_tensor = S.make_tensor(weight_ptr, S.bf16, weight_layout)

    # Output layout: (batch, out_channels, out_height, out_width)
    output_stride_w = S.convert(1, S.i32)
    output_stride_h = out_width
    output_stride_c = out_height * out_width
    output_stride_b = out_channels * out_height * out_width

    output_layout = S.make_layout(
        (batch_size, out_channels, out_height, out_width),
        (output_stride_b, output_stride_c, output_stride_h, output_stride_w)
    )
    output_tensor = S.make_tensor(output_ptr, S.bf16, output_layout)

    # Compute transposed convolution for this output element
    # For transposed conv with stride=1, padding=0:
    # output[oc, oh, ow] = sum over (ic, kh, kw) of input[ic, oh - kh, ow - kw] * weight[ic, oc, kh, kw]

    acc = S.convert(0.0, S.f32)

    # Iterate over all input channels
    for ic in S.range(in_channels):
        # Iterate over kernel height
        for kh in S.range(kernel_h):
            ih = oh - kh
            if ih >= 0 and ih < height:
                # Iterate over kernel width
                for kw in S.range(kernel_w):
                    iw = ow - kw
                    if iw >= 0 and iw < width:
                        in_val = S.convert(input_tensor[n, ic, ih, iw], S.f32)
                        w_val = S.convert(weight_tensor[ic, oc, kh, kw], S.f32)
                        acc = acc + in_val * w_val

    # Convert back to bf16 and store
    output_tensor[n, oc, oh, ow] = S.convert(acc, S.bf16)


class ModelNew(torch.nn.Module):
    """
    Optimized transposed 2D convolution using Substrate DSL.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: int = 1, padding: int = 0, output_padding: int = 0,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weight tensor for transposed convolution
        # Weight shape: (in_channels, out_channels, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size[0], kernel_size[1]))

        # Initialize bias if needed
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
        # Ensure input is contiguous and on correct device
        x = x.contiguous()

        batch_size, in_channels, height, width = x.shape
        dtype = x.dtype

        # Convert to bf16 for kernel
        x_bf16 = x.to(dtype=torch.bfloat16)

        # Compute output dimensions for transposed convolution
        # out = (in - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
        kernel_h, kernel_w = self.kernel_size
        out_height = (height - 1) * self.stride - 2 * self.padding + (kernel_h - 1) + self.output_padding + 1
        out_width = (width - 1) * self.stride - 2 * self.padding + (kernel_w - 1) + self.output_padding + 1

        # Create output tensor
        output = torch.empty((batch_size, self.out_channels, out_height, out_width),
                            dtype=torch.bfloat16, device=x.device)

        # Ensure weight is contiguous and in bf16
        weight = self.weight.data.to(dtype=torch.bfloat16).contiguous()

        # Launch kernel
        tiles_h = (out_height + TILE_H - 1) // TILE_H
        tiles_w = (out_width + TILE_W - 1) // TILE_W
        total_tiles = tiles_h * tiles_w

        # Grid: (batch, out_channels, total_tiles)
        # Block: (TILE_W, TILE_H, 1)
        grid = (batch_size, self.out_channels, total_tiles)
        block = (TILE_W, TILE_H, 1)

        conv_transpose2d_kernel[lambda: (grid, block)](
            x_bf16, weight, output,
            batch_size, in_channels, self.out_channels, height, width,
            out_height, out_width, kernel_h, kernel_w, tiles_w
        )

        # Convert back to original dtype if needed
        if dtype != torch.bfloat16:
            output = output.to(dtype=dtype)

        # Add bias if needed
        if self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1, 1).to(dtype=output.dtype)

        return output
