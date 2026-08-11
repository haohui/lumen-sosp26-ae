"""AMDGPU Conv2D kernels for asymmetric kernel, padding, and dilation."""

import math

import substrate
import substrate.language as S
import torch
import torch.nn as nn

# Thread block configuration
BLOCK_SIZE = 256


@substrate.jit
def _conv2d_kernel(
    input_ptr: S.Pointer(S.bf16),
    weight_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.u32,
    in_channels: S.u32,
    in_h: S.u32,
    in_w: S.u32,
    out_channels: S.u32,
    out_h: S.u32,
    out_w: S.u32,
    kernel_h: S.u32,
    kernel_w: S.u32,
    pad_h: S.u32,
    pad_w: S.u32,
    stride_h: S.u32,
    stride_w: S.u32,
    dilation_h: S.u32,
    dilation_w: S.u32,
):
    """
    Conv2D kernel using direct global memory access with FP32 accumulation.
    Each thread computes one output pixel for one output channel.
    """
    # Flatten output: (batch, out_channels, out_h, out_w) -> linear index
    total_outputs = batch_size * out_channels * out_h * out_w
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx >= total_outputs:
        return

    # Decode linear index to (batch, out_c, out_h_idx, out_w_idx)
    # Layout: (batch, out_channels, out_h, out_w)
    hw_out = out_h * out_w
    chw_out = out_channels * hw_out

    batch = idx // chw_out
    rest = idx % chw_out
    out_c = rest // hw_out
    hw_idx = rest % hw_out
    out_h_idx = hw_idx // out_w
    out_w_idx = hw_idx % out_w

    # Input base position
    h_in_base = out_h_idx * stride_h - pad_h
    w_in_base = out_w_idx * stride_w - pad_w

    # Input tensor layout: (batch, in_channels, in_h, in_w)
    input_layout = S.make_layout(
        (batch_size, in_channels, in_h, in_w),
        (in_channels * in_h * in_w, in_h * in_w, in_w, 1),
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, input_layout)

    # Weight tensor layout: (out_channels, in_channels, kernel_h, kernel_w)
    weight_layout = S.make_layout(
        (out_channels, in_channels, kernel_h, kernel_w),
        (in_channels * kernel_h * kernel_w, kernel_h * kernel_w, kernel_w, 1),
    )
    weight_tensor = S.make_tensor(weight_ptr, S.bf16, weight_layout)

    # Output tensor layout: (batch, out_channels, out_h, out_w)
    output_layout = S.make_layout(
        (batch_size, out_channels, out_h, out_w),
        (out_channels * out_h * out_w, out_h * out_w, out_w, 1),
    )
    output_tensor = S.make_tensor(output_ptr, S.bf16, output_layout)

    # Accumulate in FP32
    acc = S.convert(0.0, S.f32)

    # Convolution loop
    for ic in S.range(in_channels):
        for kh in S.range(kernel_h):
            for kw in S.range(kernel_w):
                h_in = h_in_base + kh * dilation_h
                w_in = w_in_base + kw * dilation_w

                # Bounds check
                if h_in >= 0 and h_in < in_h and w_in >= 0 and w_in < in_w:
                    in_val = input_tensor[batch, ic, h_in, w_in]
                    w_val = weight_tensor[out_c, ic, kh, kw]
                    acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

    # Store output
    output_tensor[batch, out_c, out_h_idx, out_w_idx] = S.convert(acc, S.bf16)


def conv2d_asym(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride_h: int = 1,
    stride_w: int = 1,
    dilation_h: int = 1,
    dilation_w: int = 1,
    pad_h: int = 0,
    pad_w: int = 0,
) -> torch.Tensor:
    """
    Conv2D with asymmetric kernel, padding, and dilation support.

    Args:
        input: Input tensor of shape (batch_size, in_channels, in_h, in_w)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_h, kernel_w)
        stride_h: Stride in height dimension.
        stride_w: Stride in width dimension.
        dilation_h: Dilation in height dimension.
        dilation_w: Dilation in width dimension.
        pad_h: Padding in height dimension (top and bottom).
        pad_w: Padding in width dimension (left and right).

    Returns:
        Output tensor of shape (batch_size, out_channels, out_h, out_w)
    """
    if not isinstance(input, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("input and weight must be torch.Tensor")
    if input.ndim != 4 or weight.ndim != 4:
        raise ValueError(
            f"input must be rank-4 and weight must be rank-4 "
            f"(got input.ndim={input.ndim}, weight.ndim={weight.ndim})"
        )

    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    # Compute output dimensions
    out_h = (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    # Allocate output tensor
    out = torch.empty(
        (batch_size, out_channels, out_h, out_w),
        dtype=torch.bfloat16,
        device=input.device,
    )

    # Grid dimensions
    total_outputs = batch_size * out_channels * out_h * out_w
    num_blocks = (total_outputs + BLOCK_SIZE - 1) // BLOCK_SIZE

    grid = (num_blocks, 1, 1)
    block = (BLOCK_SIZE, 1, 1)

    _conv2d_kernel[lambda: (grid, block)](
        input,
        weight,
        out,
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
    )
    return out


class ModelNew(nn.Module):
    """
    KernelBench-style Conv2d wrapper for asymmetric kernel, padding, and dilation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        bias: bool = False,
    ):
        super().__init__()

        if not isinstance(stride, int):
            raise NotImplementedError(
                "This implementation currently supports integer stride only."
            )
        if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
            raise TypeError(f"kernel_size must be a 2-tuple (got {kernel_size!r})")
        if not isinstance(padding, tuple) or len(padding) != 2:
            raise TypeError(f"padding must be a 2-tuple (got {padding!r})")
        if not isinstance(dilation, tuple) or len(dilation) != 2:
            raise TypeError(f"dilation must be a 2-tuple (got {dilation!r})")

        kernel_h, kernel_w = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_h, kernel_w)
        )
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias_param", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        original_dtype = x.dtype

        x_bf16 = x.to(dtype=torch.bfloat16)
        weight_bf16 = self.weight.detach().to(device=x.device, dtype=torch.bfloat16).contiguous()

        out = conv2d_asym(
            x_bf16,
            weight_bf16,
            stride_h=self.stride,
            stride_w=self.stride,
            dilation_h=self.dilation[0],
            dilation_w=self.dilation[1],
            pad_h=self.padding[0],
            pad_w=self.padding[1],
        )

        if self.bias_param is not None:
            bias = self.bias_param.detach().to(device=x.device, dtype=torch.bfloat16)
            out = out + bias.view(1, -1, 1, 1)

        if original_dtype != torch.bfloat16:
            out = out.to(original_dtype)
        return out
