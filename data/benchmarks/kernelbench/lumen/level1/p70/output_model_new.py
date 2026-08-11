import torch
import torch.nn as nn
import math
import substrate
import substrate.language as S


# Block tile size for output spatial dimensions
TILE_D = 4
TILE_H = 4
TILE_W = 4


@substrate.jit
def conv_transpose3d_kernel(
    input_ptr: S.Pointer(S.bf16),
    weight_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    in_channels: S.i32,
    out_channels: S.i32,
    depth_in: S.i32,
    height_in: S.i32,
    width_in: S.i32,
    depth_out: S.i32,
    height_out: S.i32,
    width_out: S.i32,
    kernel_size: S.i32,
    padding: S.i32,
):
    """ConvTranspose3d kernel for stride=1, dilation=1, padding=0.

    Each thread computes one output element.
    For transposed conv with stride=1, padding=0:
      output[od, oh, ow] = sum over ic, kd, kh, kw of
        input[od - kd, oh - kh, ow - kw] * weight[ic, oc, kd, kh, kw]
    where input indices must be in valid range.
    """
    # Decode block IDs
    n = S.block_id(0)
    oc = S.block_id(1)

    # Spatial tile from block_id(2)
    linear_id = S.block_id(2)
    tiles_d = (depth_out + TILE_D - 1) // TILE_D
    tiles_h = (height_out + TILE_H - 1) // TILE_H
    tiles_w = (width_out + TILE_W - 1) // TILE_W
    tiles_hw = tiles_h * tiles_w

    tile_od = (linear_id // tiles_hw) * TILE_D
    remaining = linear_id % tiles_hw
    tile_oh = (remaining // tiles_w) * TILE_H
    tile_ow = (remaining % tiles_w) * TILE_W

    # Thread ID within block
    tid_x = S.thread_id(0)
    tid_y = S.thread_id(1)
    tid_z = S.thread_id(2)

    od = tile_od + tid_z
    oh = tile_oh + tid_y
    ow = tile_ow + tid_x

    # Check bounds early
    if n >= batch_size or oc >= out_channels or od >= depth_out or oh >= height_out or ow >= width_out:
        return

    # Create tensor views using S.make_layout
    # Input layout: (batch, in_channels, depth, height, width)
    input_layout = S.make_layout(
        (batch_size, in_channels, depth_in, height_in, width_in),
        (in_channels * depth_in * height_in * width_in,
         depth_in * height_in * width_in,
         height_in * width_in,
         width_in,
         1)
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, input_layout)

    # Weight layout: (in_channels, out_channels, kernel_d, kernel_h, kernel_w)
    weight_layout = S.make_layout(
        (in_channels, out_channels, kernel_size, kernel_size, kernel_size),
        (out_channels * kernel_size * kernel_size * kernel_size,
         kernel_size * kernel_size * kernel_size,
         kernel_size * kernel_size,
         kernel_size,
         1)
    )
    weight_tensor = S.make_tensor(weight_ptr, S.bf16, weight_layout)

    # Output layout: (batch, out_channels, depth_out, height_out, width_out)
    output_layout = S.make_layout(
        (batch_size, out_channels, depth_out, height_out, width_out),
        (out_channels * depth_out * height_out * width_out,
         depth_out * height_out * width_out,
         height_out * width_out,
         width_out,
         1)
    )
    output_tensor = S.make_tensor(output_ptr, S.bf16, output_layout)

    # Accumulator for this output element
    acc = S.convert(0.0, S.f32)

    # For transposed convolution with stride=1, padding=0:
    # output[od, oh, ow] accumulates from input positions
    # id = od - kd, ih = oh - kh, iw = ow - kw for valid indices
    # Kernel is 3x3x3, so kd, kh, kw each range [0, 3)

    # Unrolled loops for kernel_size=3
    for kd in S.range(3):
        id_in = od - kd + padding
        if id_in >= 0:
            if id_in < depth_in:
                for kh in S.range(3):
                    ih = oh - kh + padding
                    if ih >= 0:
                        if ih < height_in:
                            for kw in S.range(3):
                                iw = ow - kw + padding
                                if iw >= 0:
                                    if iw < width_in:
                                        # Inner loop over input channels
                                        for ic in S.range(in_channels):
                                            in_val = S.convert(input_tensor[n, ic, id_in, ih, iw], S.f32)
                                            w_val = S.convert(weight_tensor[ic, oc, kd, kh, kw], S.f32)
                                            acc = acc + in_val * w_val

    # Store result
    output_tensor[n, oc, od, oh, ow] = S.convert(acc, S.bf16)


class ModelNew(torch.nn.Module):
    """
    Optimized 3D transposed convolution using Substrate DSL.
    Supports cubic kernel, stride=1, padding=0.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, output_padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias

        # Initialize weight tensor for transposed convolution
        # Weight shape: (in_channels, out_channels // groups, kernel_d, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.empty(
            in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size
        ))

        # Initialize bias if needed
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.in_channels * self.kernel_size ** 3
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous and convert to bf16
        x = x.contiguous()
        original_dtype = x.dtype
        x_bf16 = x.to(dtype=torch.bfloat16)

        batch_size, in_channels, depth_in, height_in, width_in = x.shape

        # Compute output dimensions for ConvTranspose3d
        # D_out = (D_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
        depth_out = (depth_in - 1) * self.stride - 2 * self.padding + \
                    self.dilation * (self.kernel_size - 1) + self.output_padding + 1
        height_out = (height_in - 1) * self.stride - 2 * self.padding + \
                     self.dilation * (self.kernel_size - 1) + self.output_padding + 1
        width_out = (width_in - 1) * self.stride - 2 * self.padding + \
                    self.dilation * (self.kernel_size - 1) + self.output_padding + 1

        # Create output tensor
        output = torch.zeros((batch_size, self.out_channels, depth_out, height_out, width_out),
                            dtype=torch.bfloat16, device=x.device)

        # Ensure weight is contiguous and bf16
        weight = self.weight.data.to(dtype=torch.bfloat16).contiguous()

        # Compute grid dimensions
        tiles_d = (depth_out + TILE_D - 1) // TILE_D
        tiles_h = (height_out + TILE_H - 1) // TILE_H
        tiles_w = (width_out + TILE_W - 1) // TILE_W

        # Launch kernel
        grid = (batch_size, self.out_channels, tiles_d * tiles_h * tiles_w)
        block = (TILE_W, TILE_H, TILE_D)

        conv_transpose3d_kernel[lambda: (grid, block)](
            x_bf16, weight, output,
            batch_size, self.in_channels, self.out_channels,
            depth_in, height_in, width_in,
            depth_out, height_out, width_out,
            self.kernel_size, self.padding
        )

        # Add bias if needed
        if self.bias_param is not None:
            output = output + self.bias_param.to(torch.bfloat16).view(1, -1, 1, 1, 1)

        # Convert back to original dtype if needed
        if original_dtype != torch.bfloat16:
            output = output.to(dtype=original_dtype)

        return output
