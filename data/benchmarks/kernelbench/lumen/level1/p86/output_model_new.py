"""Depthwise-separable 2D convolution using Substrate DSL with optimized kernels."""

import math

import substrate
import substrate.language as S
import torch
import torch.nn as nn

BLOCK_SIZE: S.constexpr = 256


@substrate.jit
def depthwise_conv2d_kernel(
    input_ptr: S.Pointer(S.bf16),
    weight_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.u32,
    channels: S.u32,
    in_h: S.u32,
    in_w: S.u32,
    out_h: S.u32,
    out_w: S.u32,
    kernel_size: S.u32,
    padding: S.u32,
    stride: S.u32,
    dilation: S.u32,
):
    """Depthwise convolution kernel where each channel has its own filter."""
    tid = S.thread_id(0)
    bid = S.block_id(0)

    idx = bid * BLOCK_SIZE + tid
    total = batch_size * channels * out_h * out_w

    if idx >= total:
        return

    # Decode linear index to (b, c, h_out, w_out)
    w_out = idx % out_w
    tmp = idx // out_w
    h_out = tmp % out_h
    tmp = tmp // out_h
    c = tmp % channels
    b = tmp // channels

    # Create tensor views
    input_layout = S.make_layout(
        (batch_size, channels, in_h, in_w),
        (channels * in_h * in_w, in_h * in_w, in_w, 1),
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, input_layout)

    # Weight shape: (channels, kernel_size, kernel_size) for depthwise conv
    weight_layout = S.make_layout(
        (channels, kernel_size, kernel_size),
        (kernel_size * kernel_size, kernel_size, 1),
    )
    weight_tensor = S.make_tensor(weight_ptr, S.bf16, weight_layout)

    output_layout = S.make_layout(
        (batch_size, channels, out_h, out_w),
        (channels * out_h * out_w, out_h * out_w, out_w, 1),
    )
    output_tensor = S.make_tensor(output_ptr, S.bf16, output_layout)

    # Compute convolution with FP32 accumulation
    acc = S.convert(0.0, S.f32)

    for kh in S.range(kernel_size):
        for kw in S.range(kernel_size):
            h_in = h_out * stride + kh * dilation
            w_in = w_out * stride + kw * dilation

            # Handle padding by adjusting coordinates
            h_in_padded = h_in - padding
            w_in_padded = w_in - padding

            # Bounds check (using unsigned comparison - negative values wrap to large positive)
            if h_in_padded < in_h and w_in_padded < in_w:
                input_val = input_tensor[b, c, h_in_padded, w_in_padded]
                weight_val = weight_tensor[c, kh, kw]
                acc = acc + S.convert(input_val, S.f32) * S.convert(weight_val, S.f32)

    output_tensor[b, c, h_out, w_out] = S.convert(acc, S.bf16)


@substrate.jit
def pointwise_conv2d_kernel(
    input_ptr: S.Pointer(S.bf16),
    weight_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.u32,
    in_channels: S.u32,
    out_channels: S.u32,
    height: S.u32,
    width: S.u32,
):
    """Pointwise (1x1) convolution kernel - essentially a matrix multiply per spatial location."""
    tid = S.thread_id(0)
    bid = S.block_id(0)

    idx = bid * BLOCK_SIZE + tid
    total = batch_size * out_channels * height * width

    if idx >= total:
        return

    # Decode linear index to (b, c_out, h, w)
    w_idx = idx % width
    tmp = idx // width
    h_idx = tmp % height
    tmp = tmp // height
    c_out = tmp % out_channels
    b = tmp // out_channels

    # Create tensor views
    input_layout = S.make_layout(
        (batch_size, in_channels, height, width),
        (in_channels * height * width, height * width, width, 1),
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, input_layout)

    # Weight shape: (out_channels, in_channels) for pointwise conv
    weight_layout = S.make_layout(
        (out_channels, in_channels),
        (in_channels, 1),
    )
    weight_tensor = S.make_tensor(weight_ptr, S.bf16, weight_layout)

    output_layout = S.make_layout(
        (batch_size, out_channels, height, width),
        (out_channels * height * width, height * width, width, 1),
    )
    output_tensor = S.make_tensor(output_ptr, S.bf16, output_layout)

    # Compute 1x1 convolution with FP32 accumulation
    acc = S.convert(0.0, S.f32)

    for c_in in S.range(in_channels):
        input_val = input_tensor[b, c_in, h_idx, w_idx]
        weight_val = weight_tensor[c_out, c_in]
        acc = acc + S.convert(input_val, S.f32) * S.convert(weight_val, S.f32)

    output_tensor[b, c_out, h_idx, w_idx] = S.convert(acc, S.bf16)


class ModelNew(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation using Substrate DSL kernels.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Depthwise conv weight: (in_channels, 1, kernel_size, kernel_size)
        self.depthwise_weight = nn.Parameter(
            torch.empty(in_channels, 1, kernel_size, kernel_size)
        )

        # Pointwise conv weight: (out_channels, in_channels, 1, 1)
        self.pointwise_weight = nn.Parameter(
            torch.empty(out_channels, in_channels, 1, 1)
        )

        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias_param", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.depthwise_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pointwise_weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        x = x.contiguous()
        original_dtype = x.dtype

        batch_size, in_channels, in_h, in_w = x.shape

        # Compute output dimensions
        out_h = (
            in_h + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
        ) // self.stride + 1
        out_w = (
            in_w + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
        ) // self.stride + 1

        # Convert to BF16
        x_bf16 = x.to(dtype=torch.bfloat16)
        depthwise_weight_bf16 = (
            self.depthwise_weight.squeeze(1).to(dtype=torch.bfloat16).contiguous()
        )
        pointwise_weight_bf16 = (
            self.pointwise_weight.squeeze(2)
            .squeeze(2)
            .to(dtype=torch.bfloat16)
            .contiguous()
        )

        # Allocate intermediate tensor for depthwise conv output
        depthwise_out = torch.empty(
            (batch_size, in_channels, out_h, out_w),
            dtype=torch.bfloat16,
            device=x.device,
        )

        # Launch depthwise conv kernel
        total_elements = batch_size * in_channels * out_h * out_w
        grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

        depthwise_conv2d_kernel[lambda: ((grid_size, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_bf16,
            depthwise_weight_bf16,
            depthwise_out,
            batch_size,
            in_channels,
            in_h,
            in_w,
            out_h,
            out_w,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        )

        # Allocate output tensor for pointwise conv
        pointwise_out = torch.empty(
            (batch_size, self.out_channels, out_h, out_w),
            dtype=torch.bfloat16,
            device=x.device,
        )

        # Launch pointwise conv kernel
        total_elements = batch_size * self.out_channels * out_h * out_w
        grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

        pointwise_conv2d_kernel[lambda: ((grid_size, 1, 1), (BLOCK_SIZE, 1, 1))](
            depthwise_out,
            pointwise_weight_bf16,
            pointwise_out,
            batch_size,
            in_channels,
            self.out_channels,
            out_h,
            out_w,
        )

        # Add bias if present
        if self.bias_param is not None:
            bias_bf16 = self.bias_param.to(dtype=torch.bfloat16)
            pointwise_out = pointwise_out + bias_bf16.view(1, -1, 1, 1)

        # Convert back to original dtype if needed
        if original_dtype != torch.bfloat16:
            pointwise_out = pointwise_out.to(original_dtype)

        return pointwise_out
