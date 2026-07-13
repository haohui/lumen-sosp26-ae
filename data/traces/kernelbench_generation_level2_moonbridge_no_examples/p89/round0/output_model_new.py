import torch
import torch.nn as nn
import torch.nn.functional as F
import avelang
import avelang.language as al


@avelang.jit
def channel_softmax_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
):
    """Softmax along channel dimension (dim=1) for each (n,d,h,w) position."""
    stride_c = D * H * W
    stride_n = C * stride_c
    total_elements = N * C * D * H * W

    layout = al.make_layout((total_elements,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    total_positions = N * D * H * W
    idx = al.block_id(0) * al.block_dim(0) + al.thread_id(0)

    if idx < total_positions:
        w_idx = idx % W
        t = idx // W
        h_idx = t % H
        t = t // H
        d_idx = t % D
        n_idx = t // D

        base = n_idx * stride_n + d_idx * H * W + h_idx * W + w_idx

        # Find max across channels for numerical stability
        max_val = al.convert(x[base], al.f32)
        ci = al.convert(1, al.i32)
        for _c in al.range(1, C):
            offset = base + ci * stride_c
            val = al.convert(x[offset], al.f32)
            if val > max_val:
                max_val = val
            ci = ci + al.convert(1, al.i32)

        # Compute sum of exp(x - max)
        val0 = al.convert(x[base], al.f32)
        sum_exp = al.exp(val0 - max_val)
        ci = al.convert(1, al.i32)
        for _c in al.range(1, C):
            offset = base + ci * stride_c
            val = al.convert(x[offset], al.f32)
            sum_exp = sum_exp + al.exp(val - max_val)
            ci = ci + al.convert(1, al.i32)

        # Write normalized output channel 0
        val0 = al.convert(x[base], al.f32)
        out[base] = al.convert(al.exp(val0 - max_val) / sum_exp, al.bf16)

        # Write normalized output channels 1..C-1
        ci = al.convert(1, al.i32)
        for _c in al.range(1, C):
            offset = base + ci * stride_c
            val = al.convert(x[offset], al.f32)
            out[offset] = al.convert(al.exp(val - max_val) / sum_exp, al.bf16)
            ci = ci + al.convert(1, al.i32)


@avelang.jit
def subtract_swish_max_kernel(
    x_ptr: al.Pointer(al.bf16),
    subtract_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
):
    """Fused subtract channel-wise, swish activation, and max reduction over channels."""
    stride_c = D * H * W
    stride_n = C * stride_c

    total_x = N * C * D * H * W
    x_layout = al.make_layout((total_x,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    sub_layout = al.make_layout((C,), (1,))
    subtract = al.make_tensor(subtract_ptr, al.bf16, sub_layout)

    total_out = N * D * H * W
    out_layout = al.make_layout((total_out,), (1,))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    total_positions = N * D * H * W
    idx = al.block_id(0) * al.block_dim(0) + al.thread_id(0)

    if idx < total_positions:
        w_idx = idx % W
        t = idx // W
        h_idx = t % H
        t = t // H
        d_idx = t % D
        n_idx = t // D

        base = n_idx * stride_n + d_idx * H * W + h_idx * W + w_idx

        one_f32 = al.convert(1.0, al.f32)
        zero_f32 = al.convert(0.0, al.f32)

        # Channel 0: subtract, swish, init max
        val0 = al.convert(x[base], al.f32)
        sub0 = al.convert(subtract[0], al.f32)
        v0 = val0 - sub0
        # Swish: v0 * sigmoid(v0) = v0 / (1 + exp(-v0))
        swish0 = v0 / (one_f32 + al.exp(zero_f32 - v0))
        max_val = swish0

        # Channels 1..C-1
        ci = al.convert(1, al.i32)
        for _c in al.range(1, C):
            offset = base + ci * stride_c
            val = al.convert(x[offset], al.f32)
            sub_val = al.convert(subtract[ci], al.f32)
            v = val - sub_val

            swish_v = v / (one_f32 + al.exp(zero_f32 - v))

            if swish_v > max_val:
                max_val = swish_v

            ci = ci + al.convert(1, al.i32)

        out_idx = n_idx * D * H * W + d_idx * H * W + h_idx * W + w_idx
        out[out_idx] = al.convert(max_val, al.bf16)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        pool_kernel_size,
        pool_stride,
        pool_padding,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        self.max_pool = nn.MaxPool3d(
            kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding,
        )
        self.subtract = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        # Convert to BF16
        x = x.to(torch.bfloat16)
        w = self.conv_transpose.weight.to(torch.bfloat16)
        b = self.conv_transpose.bias.to(torch.bfloat16) if self.conv_transpose.bias is not None else None
        sub = self.subtract.to(torch.bfloat16)

        # ConvTranspose3d
        x = F.conv_transpose3d(
            x, w, b,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding,
        )

        # MaxPool3d
        x = F.max_pool3d(
            x,
            kernel_size=self.max_pool.kernel_size,
            stride=self.max_pool.stride,
            padding=self.max_pool.padding,
        )

        # x shape: (N, C, D, H, W)
        N, C, D, H, W = x.shape
        total_positions = N * D * H * W
        grid = (total_positions + 255) // 256

        # Kernel 1: channel softmax
        softmax_out = torch.empty_like(x)
        channel_softmax_kernel[lambda: ((grid, 1, 1), (256, 1, 1))](
            x.contiguous(), softmax_out,
            N, C, D, H, W,
        )

        # Kernel 2: subtract + swish + channel max
        max_out = torch.empty(N, D, H, W, dtype=torch.bfloat16, device=x.device)
        subtract_swish_max_kernel[lambda: ((grid, 1, 1), (256, 1, 1))](
            softmax_out.contiguous(), sub.contiguous(), max_out,
            N, C, D, H, W,
        )

        return max_out
