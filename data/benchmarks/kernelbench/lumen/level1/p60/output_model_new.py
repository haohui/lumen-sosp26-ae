import torch
import torch.nn as nn
import math
import substrate
import substrate.language as S


# Block size - each thread computes one output element
BLOCK_SIZE = 256


@substrate.jit
def conv3d_standard_kernel(
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
    stride: S.i32,
    padding: S.i32,
    dilation: S.i32,
    total_elements: S.i32,
):
    # Linear thread index
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    # Check bounds
    if tid >= total_elements:
        return

    # Decode linear index to (batch, out_channel, out_depth, out_height, out_width)
    # Layout: (batch, out_channel, out_depth, out_height, out_width)
    ow = tid % out_width
    remainder = tid // out_width
    oh = remainder % out_height
    remainder = remainder // out_height
    od = remainder % out_depth
    remainder = remainder // out_depth
    oc = remainder % out_channels
    n = remainder // out_channels

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

    # Weight layout: (out_channels, in_channels, kernel_d, kernel_h, kernel_w)
    weight_stride_kw = S.convert(1, S.i32)
    weight_stride_kh = weight_stride_kw * kernel_w
    weight_stride_kd = weight_stride_kh * kernel_h
    weight_stride_ic = weight_stride_kd * kernel_d
    weight_stride_oc = weight_stride_ic * in_channels

    weight_layout = S.make_layout(
        (out_channels, in_channels, kernel_d, kernel_h, kernel_w),
        (weight_stride_oc, weight_stride_ic, weight_stride_kd, weight_stride_kh, weight_stride_kw)
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

    # Compute convolution for this output element
    # Standard convolution: sum over (in_channels, kernel_d, kernel_h, kernel_w)
    acc = S.convert(0.0, S.f32)

    for ic in S.range(in_channels):
        for kd in S.range(kernel_d):
            # Compute input depth index
            input_d = od * stride + kd * dilation - padding
            if input_d < 0 or input_d >= in_depth:
                continue

            for kh in S.range(kernel_h):
                # Compute input height index
                input_h = oh * stride + kh * dilation - padding
                if input_h < 0 or input_h >= in_height:
                    continue

                for kw in S.range(kernel_w):
                    # Compute input width index
                    input_w = ow * stride + kw * dilation - padding
                    if input_w < 0 or input_w >= in_width:
                        continue

                    in_val = S.convert(input_tensor[n, ic, input_d, input_h, input_w], S.f32)
                    w_val = S.convert(weight_tensor[oc, ic, kd, kh, kw], S.f32)
                    acc = acc + in_val * w_val

    # Convert back to bf16 and store
    output_tensor[n, oc, od, oh, ow] = S.convert(acc, S.bf16)


class ModelNew(torch.nn.Module):
    """
    Optimized 3D convolution using Substrate DSL.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Weight tensor for standard convolution
        # Shape: (out_channels, in_channels // groups, kernel_d, kernel_h, kernel_w)
        kernel_d, kernel_h, kernel_w = self.kernel_size
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, kernel_d, kernel_h, kernel_w
        ))

        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3] * self.weight.shape[4]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()

        batch_size, in_channels, in_depth, in_height, in_width = x.shape
        dtype = x.dtype

        # Compute output dimensions
        kernel_d, kernel_h, kernel_w = self.kernel_size
        out_depth = (in_depth + 2 * self.padding - self.dilation * (kernel_d - 1) - 1) // self.stride + 1
        out_height = (in_height + 2 * self.padding - self.dilation * (kernel_h - 1) - 1) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.dilation * (kernel_w - 1) - 1) // self.stride + 1

        # Create output tensor
        output = torch.empty(
            (batch_size, self.out_channels, out_depth, out_height, out_width),
            dtype=dtype, device=x.device
        )

        # Ensure weight is contiguous
        weight = self.weight.data.contiguous()

        # Total output elements
        total_elements = batch_size * self.out_channels * out_depth * out_height * out_width

        # Grid and block configuration
        grid_x = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = (grid_x, 1, 1)
        block = (BLOCK_SIZE, 1, 1)

        conv3d_standard_kernel[lambda: (grid, block)](
            x, weight, output,
            batch_size, in_channels, self.out_channels,
            in_depth, in_height, in_width,
            out_depth, out_height, out_width,
            kernel_d, kernel_h, kernel_w,
            self.stride, self.padding, self.dilation,
            total_elements
        )

        # Add bias if needed
        if self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1, 1, 1)

        return output
