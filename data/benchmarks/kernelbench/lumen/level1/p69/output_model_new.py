"""ConvTranspose2D using im2col + optimized GEMM approach with Substrate."""

import math
import sys

import substrate
import substrate.language as S
import torch
import torch.nn as nn

# Add path for substrate_kernels
sys.path.insert(0, "/root/kernel-benchmark/substrate/python")
from substrate_kernels.amdgpu_gemm import (
    gemm_1stage_pipeline_row_major,
    gemm_1stage_validate_shape,
)

THREADS = 256


@substrate.jit
def _transpose_input_nchw_to_nhwc_kernel(
    src: S.Pointer(S.bf16),
    dst: S.Pointer(S.bf16),
    batch_size: S.u32,
    in_channels: S.u32,
    in_h: S.u32,
    in_w: S.u32,
):
    """Transpose input from NCHW to NHWC."""
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = batch_size * in_channels * in_h * in_w
    if idx >= total:
        return

    c = idx % in_channels
    tmp = idx // in_channels
    w = tmp % in_w
    tmp = tmp // in_w
    h = tmp % in_h
    n = tmp // in_h

    src_tensor = S.make_tensor(
        src,
        S.bf16,
        S.make_layout(
            (batch_size, in_channels, in_h, in_w),
            (in_channels * in_h * in_w, in_h * in_w, in_w, 1),
        ),
    )
    dst_tensor = S.make_tensor(
        dst,
        S.bf16,
        S.make_layout(
            (batch_size, in_h, in_w, in_channels),
            (in_h * in_w * in_channels, in_w * in_channels, in_channels, 1),
        ),
    )
    dst_tensor[n, h, w, c] = src_tensor[n, c, h, w]


@substrate.jit
def _im2col_conv_transpose_kernel(
    input_nhwc: S.Pointer(S.bf16),
    col: S.Pointer(S.bf16),
    batch_size: S.u32,
    in_h: S.u32,
    in_w: S.u32,
    in_channels: S.u32,
    out_h: S.u32,
    out_w: S.u32,
    kernel_h: S.u32,
    kernel_w: S.u32,
    stride_h: S.u32,
    stride_w: S.u32,
    implicit_pad_h: S.u32,
    implicit_pad_w: S.u32,
    dilation_h: S.u32,
    dilation_w: S.u32,
):
    """Im2col for ConvTranspose2d."""
    gemm_m = batch_size * out_h * out_w
    gemm_k = in_channels * kernel_h * kernel_w

    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = gemm_m * gemm_k

    input_tensor = S.make_tensor(
        input_nhwc,
        S.bf16,
        S.make_layout(
            (batch_size, in_h, in_w, in_channels),
            (in_h * in_w * in_channels, in_w * in_channels, in_channels, 1),
        ),
    )
    col_tensor = S.make_tensor(
        col,
        S.bf16,
        S.make_layout((gemm_m, gemm_k), (gemm_k, 1)),
    )

    while idx < total:
        m_idx = idx // gemm_k
        k_idx = idx % gemm_k

        # Decode m_idx to (batch, oh, ow)
        hw_idx = m_idx % (out_h * out_w)
        batch = m_idx // (out_h * out_w)
        oh = hw_idx // out_w
        ow = hw_idx % out_w

        # Decode k_idx to (kh, kw, ic)
        kw = k_idx % kernel_w
        tmp = k_idx // kernel_w
        kh = tmp % kernel_h
        ic = tmp // kernel_h

        # For ConvTranspose2d: ih = oh + kh - implicit_pad
        ih = oh * stride_h + kh * dilation_h - implicit_pad_h
        iw = ow * stride_w + kw * dilation_w - implicit_pad_w

        # Check bounds and load
        if ih >= 0 and ih < in_h and iw >= 0 and iw < in_w:
            col_tensor[m_idx, k_idx] = input_tensor[batch, ih, iw, ic]
        else:
            col_tensor[m_idx, k_idx] = S.convert(0.0, S.bf16)

        idx = idx + (S.block_dim(0) * S.grid_dim(0))


@substrate.jit
def _gemm_simple_kernel(
    A: S.Pointer(S.bf16),
    B: S.Pointer(S.bf16),
    C: S.Pointer(S.bf16),
    M: S.u32,
    N: S.u32,
    K: S.u32,
):
    """Simple element-wise GEMM for correctness."""
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = M * N

    A_tensor = S.make_tensor(A, S.bf16, S.make_layout((M, K), (K, 1)))
    B_tensor = S.make_tensor(B, S.bf16, S.make_layout((K, N), (N, 1)))
    C_tensor = S.make_tensor(C, S.bf16, S.make_layout((M, N), (N, 1)))

    while idx < total:
        m = idx // N
        n = idx % N

        sum_val = S.convert(0.0, S.f32)
        for k in S.range(K):
            a_val = S.convert(A_tensor[m, k], S.f32)
            b_val = S.convert(B_tensor[k, n], S.f32)
            sum_val = sum_val + a_val * b_val

        C_tensor[m, n] = S.convert(sum_val, S.bf16)
        idx = idx + (S.block_dim(0) * S.grid_dim(0))


@substrate.jit
def _transpose_output_nhwc_to_nchw_kernel(
    src: S.Pointer(S.bf16),
    dst: S.Pointer(S.bf16),
    batch_size: S.u32,
    out_channels: S.u32,
    out_h: S.u32,
    out_w: S.u32,
):
    """Transpose output from NHWC to NCHW."""
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = batch_size * out_channels * out_h * out_w
    if idx >= total:
        return

    c = idx % out_channels
    tmp = idx // out_channels
    w = tmp % out_w
    tmp = tmp // out_w
    h = tmp % out_h
    n = tmp // out_h

    src_tensor = S.make_tensor(
        src,
        S.bf16,
        S.make_layout(
            (batch_size, out_h, out_w, out_channels),
            (out_h * out_w * out_channels, out_w * out_channels, out_channels, 1),
        ),
    )
    dst_tensor = S.make_tensor(
        dst,
        S.bf16,
        S.make_layout(
            (batch_size, out_channels, out_h, out_w),
            (out_channels * out_h * out_w, out_h * out_w, out_w, 1),
        ),
    )
    dst_tensor[n, c, h, w] = src_tensor[n, h, w, c]


def conv_transpose2d_im2col(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple = (1, 1),
    dilation: tuple = (1, 1),
    padding: tuple = (0, 0),
    groups: int = 1,
) -> torch.Tensor:
    """ConvTranspose2D using im2col + GEMM approach."""
    batch_size, in_channels, in_h, in_w = input.shape
    in_channels_w, out_channels, kernel_h, kernel_w = weight.shape

    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    padding_h, padding_w = padding

    # Compute output dimensions for ConvTranspose2d
    out_h = (in_h - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1
    out_w = (in_w - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1

    gemm_m = batch_size * out_h * out_w
    gemm_k = in_channels * kernel_h * kernel_w
    gemm_n = out_channels

    # Implicit padding for ConvTranspose2d
    implicit_pad_h = dilation_h * (kernel_h - 1)
    implicit_pad_w = dilation_w * (kernel_w - 1)

    # Allocate intermediate buffers
    input_nhwc = torch.empty(
        (batch_size, in_h, in_w, in_channels),
        dtype=torch.bfloat16,
        device=input.device,
    )
    col = torch.empty(
        (gemm_m, gemm_k),
        dtype=torch.bfloat16,
        device=input.device,
    )
    output_nhwc = torch.empty(
        (gemm_m, gemm_n),
        dtype=torch.bfloat16,
        device=input.device,
    )
    out = torch.empty(
        (batch_size, out_channels, out_h, out_w),
        dtype=torch.bfloat16,
        device=input.device,
    )

    # Transpose input NCHW -> NHWC
    input_elems = batch_size * in_channels * in_h * in_w
    grid_trans = ((input_elems + THREADS - 1) // THREADS, 1, 1)
    block = (THREADS, 1, 1)
    _transpose_input_nchw_to_nhwc_kernel[lambda: (grid_trans, block)](
        input, input_nhwc, batch_size, in_channels, in_h, in_w
    )

    # Prepare weight: reshape from (in_channels, out_channels, kernel_h, kernel_w)
    # For ConvTranspose2d, we need to FLIP the kernel spatially first!
    flipped_weight = weight.flip(2, 3)
    weight_flat = flipped_weight.permute(0, 2, 3, 1).reshape(gemm_k, gemm_n).contiguous()

    # im2col: convert input to column matrix
    col_elems = gemm_m * gemm_k
    grid_im2col = ((col_elems + THREADS - 1) // THREADS, 1, 1)
    _im2col_conv_transpose_kernel[lambda: (grid_im2col, block)](
        input_nhwc,
        col,
        batch_size,
        in_h,
        in_w,
        in_channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        implicit_pad_h,
        implicit_pad_w,
        dilation_h,
        dilation_w,
    )

    # GEMM: col @ weight_flat -> output matrix
    # Use optimized GEMM if dimensions are aligned, otherwise fall back to simple
    try:
        gemm_1stage_validate_shape(gemm_m, gemm_n, gemm_k)
        output_nhwc = gemm_1stage_pipeline_row_major(col, weight_flat, out=output_nhwc)
    except ValueError:
        # Fall back to simple kernel for non-aligned shapes
        gemm_elems = gemm_m * gemm_n
        grid_gemm = ((gemm_elems + THREADS - 1) // THREADS, 1, 1)
        _gemm_simple_kernel[lambda: (grid_gemm, block)](
            col, weight_flat, output_nhwc, gemm_m, gemm_n, gemm_k
        )

    # Transpose output (batch*out_h*out_w, out_channels) -> NCHW
    out_nhwc = output_nhwc.view(batch_size, out_h, out_w, out_channels)
    out_elems = batch_size * out_channels * out_h * out_w
    grid_outtrans = ((out_elems + THREADS - 1) // THREADS, 1, 1)
    _transpose_output_nhwc_to_nchw_kernel[lambda: (grid_outtrans, block)](
        out_nhwc, out, batch_size, out_channels, out_h, out_w
    )

    return out


class ModelNew(nn.Module):
    """ConvTranspose2d wrapper using the im2col + GEMM kernel."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        output_padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()

        if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
            raise TypeError(f"kernel_size must be a 2-tuple (got {kernel_size!r})")

        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = (
            output_padding
            if isinstance(output_padding, tuple)
            else (output_padding, output_padding)
        )
        self.dilation = (
            dilation if isinstance(dilation, tuple) else (dilation, dilation)
        )
        self.groups = groups
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels // groups, kernel_size[0], kernel_size[1])
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
        weight_bf16 = (
            self.weight.detach().to(device=x.device, dtype=torch.bfloat16).contiguous()
        )

        out = conv_transpose2d_im2col(
            x_bf16,
            weight_bf16,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            groups=self.groups,
        )

        if self.bias_param is not None:
            bias = self.bias_param.detach().to(device=x.device, dtype=torch.bfloat16)
            out = out + bias.view(1, -1, 1, 1)

        if original_dtype != torch.bfloat16:
            out = out.to(original_dtype)
        return out
