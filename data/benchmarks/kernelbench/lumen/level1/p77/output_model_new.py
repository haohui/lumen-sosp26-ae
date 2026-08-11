import torch
import torch.nn as nn
import math
import substrate
import substrate.language as S


# Use simple 1D tile approach
BLOCK_SIZE: S.constexpr = 256


@substrate.jit
def conv_transpose3d_kernel(
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
    kernel_size: S.i32,
    stride: S.i32,
    padding: S.i32,
    dilation: S.i32,
    total_output_elements: S.i32,
):
    # Simple 1D grid: each thread computes one output element
    linear_tid = S.block_id(0) * BLOCK_SIZE + S.thread_id(0)

    if linear_tid >= total_output_elements:
        return

    # Decode linear index to (n, oc, od, oh, ow)
    out_spatial_size = out_depth * out_height * out_width
    out_channel_size = out_channels * out_spatial_size

    n = linear_tid // out_channel_size
    remaining = linear_tid % out_channel_size
    oc = remaining // out_spatial_size
    spatial_idx = remaining % out_spatial_size

    od = spatial_idx // (out_height * out_width)
    spatial_rem = spatial_idx % (out_height * out_width)
    oh = spatial_rem // out_width
    ow = spatial_rem % out_width

    # Create tensor views with layouts
    # Input layout: (batch, in_channels, depth, height, width)
    input_layout = S.make_layout(
        (batch_size, in_channels, in_depth, in_height, in_width),
        (in_channels * in_depth * in_height * in_width, in_depth * in_height * in_width, in_height * in_width, in_width, 1)
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, input_layout)

    # Weight layout: (in_channels, out_channels, kernel_d, kernel_h, kernel_w)
    weight_layout = S.make_layout(
        (in_channels, out_channels, kernel_size, kernel_size, kernel_size),
        (out_channels * kernel_size * kernel_size * kernel_size, kernel_size * kernel_size * kernel_size, kernel_size * kernel_size, kernel_size, 1)
    )
    weight_tensor = S.make_tensor(weight_ptr, S.bf16, weight_layout)

    # Output layout: (batch, out_channels, out_depth, out_height, out_width)
    output_layout = S.make_layout(
        (batch_size, out_channels, out_depth, out_height, out_width),
        (out_channels * out_depth * out_height * out_width, out_depth * out_height * out_width, out_height * out_width, out_width, 1)
    )
    output_tensor = S.make_tensor(output_ptr, S.bf16, output_layout)

    # Compute transposed convolution
    acc = S.convert(0.0, S.f32)

    # For each input channel and kernel position, check if it contributes
    for ic in S.range(in_channels):
        for kd in S.range(kernel_size):
            # Compute input depth: id = (od + padding - kd * dilation) / stride
            temp_d = od + padding - kd * dilation
            if temp_d >= 0:
                id = temp_d // stride
                if id < in_depth:
                    # Check exact divisibility
                    if temp_d == id * stride:
                        for kh in S.range(kernel_size):
                            temp_h = oh + padding - kh * dilation
                            if temp_h >= 0:
                                ih = temp_h // stride
                                if ih < in_height:
                                    if temp_h == ih * stride:
                                        for kw in S.range(kernel_size):
                                            temp_w = ow + padding - kw * dilation
                                            if temp_w >= 0:
                                                iw = temp_w // stride
                                                if iw < in_width:
                                                    if temp_w == iw * stride:
                                                        # Access using tensor indexing
                                                        in_val = S.convert(input_tensor[n, ic, id, ih, iw], S.f32)
                                                        w_val = S.convert(weight_tensor[ic, oc, kd, kh, kw], S.f32)
                                                        acc = acc + in_val * w_val

    # Store result using tensor indexing
    output_tensor[n, oc, od, oh, ow] = S.convert(acc, S.bf16)


class ModelNew(torch.nn.Module):
    """
    Optimized 3D transposed convolution using Substrate DSL.
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
        self.has_bias = bias

        # Weight tensor for ConvTranspose3d: (in_channels, out_channels, kernel_d, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size))

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
        # Convert to BF16 and ensure contiguous
        x_bf16 = x.to(dtype=torch.bfloat16, device=x.device).contiguous()

        batch_size, in_channels, in_depth, in_height, in_width = x_bf16.shape

        # Compute output dimensions for ConvTranspose3d
        out_depth = (in_depth - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1
        out_height = (in_height - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1
        out_width = (in_width - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1

        # Create output tensor
        output = torch.empty((batch_size, self.out_channels, out_depth, out_height, out_width),
                            dtype=torch.bfloat16, device=x.device)

        # Ensure weight is contiguous
        weight = self.weight.data.to(dtype=torch.bfloat16, device=x.device).contiguous()

        # Total output elements
        total_output_elements = batch_size * self.out_channels * out_depth * out_height * out_width

        # Grid calculation
        grid_size = (total_output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Launch kernel
        conv_transpose3d_kernel[lambda: ((grid_size, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_bf16, weight, output,
            batch_size, in_channels, self.out_channels,
            in_depth, in_height, in_width,
            out_depth, out_height, out_width,
            self.kernel_size, self.stride, self.padding, self.dilation,
            total_output_elements
        )

        # Add bias if needed
        if self.bias_param is not None:
            output = output + self.bias_param.to(dtype=torch.bfloat16, device=x.device).view(1, -1, 1, 1, 1)

        # Convert back to original dtype if needed
        if x.dtype != torch.bfloat16:
            output = output.to(dtype=x.dtype)

        return output
