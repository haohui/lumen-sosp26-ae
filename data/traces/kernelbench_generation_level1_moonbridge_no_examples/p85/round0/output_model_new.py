import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def depthwise_conv2d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    KH: al.constexpr,
    KW: al.constexpr,
    H_out: al.i32,
    W_out: al.i32,
    BLOCK_H: al.constexpr,
    BLOCK_W: al.constexpr,
):
    one = al.convert(1, al.i32)

    # Input: shape (N, C, H, W), row-major strides
    inp_layout = al.make_layout(
        (N, C, H, W),
        (C * H * W, H * W, W, one),
    )
    inp = al.make_tensor(input_ptr, al.bf16, inp_layout)

    # Weight: raw shape (C, 1, KH, KW); view as (C, KH, KW) dropping the size-1 dim
    wgt_layout = al.make_layout(
        (C, KH, KW),
        (KH * KW, KW, one),
    )
    wgt = al.make_tensor(weight_ptr, al.bf16, wgt_layout)

    # Output: shape (N, C, H_out, W_out), row-major strides
    out_layout = al.make_layout(
        (N, C, H_out, W_out),
        (C * H_out * W_out, H_out * W_out, W_out, one),
    )
    out = al.make_tensor(output_ptr, al.bf16, out_layout)

    bid_w = al.block_id(0)
    bid_h = al.block_id(1)
    bid_nc = al.block_id(2)

    tid_w = al.thread_id(0)
    tid_h = al.thread_id(1)

    w_out = bid_w * BLOCK_W + tid_w
    h_out = bid_h * BLOCK_H + tid_h

    n = bid_nc // C
    c = bid_nc % C

    if h_out < H_out and w_out < W_out:
        acc = al.convert(0, al.f32)
        for kh in al.range(KH):
            h_in = h_out + kh
            for kw in al.range(KW):
                w_in = w_out + kw
                inp_val = al.convert(inp[n, c, h_in, w_in], al.f32)
                w_val = al.convert(wgt[c, kh, kw], al.f32)
                acc = acc + inp_val * w_val
        out[n, c, h_out, w_out] = al.convert(acc, al.bf16)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_h: int,
        kernel_size_w: int,
        stride_h: int = 1,
        stride_w: int = 1,
        padding_h: int = 0,
        padding_w: int = 0,
        dilation_h: int = 1,
        dilation_w: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            in_channels,
            (kernel_size_h, kernel_size_w),
            stride=(stride_h, stride_w),
            padding=(padding_h, padding_w),
            dilation=(dilation_h, dilation_w),
            groups=in_channels,
            bias=bias,
        )
        self.kh = kernel_size_h
        self.kw = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv2d.weight
        N, C, H, W = x.shape
        KH = self.kh
        KW = self.kw

        H_out = (
            H + 2 * self.padding_h - self.dilation_h * (KH - 1) - 1
        ) // self.stride_h + 1
        W_out = (
            W + 2 * self.padding_w - self.dilation_w * (KW - 1) - 1
        ) // self.stride_w + 1

        input_dtype = x.dtype
        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = weight.to(torch.bfloat16).contiguous()
        out = torch.empty(N, C, H_out, W_out, dtype=torch.bfloat16, device=x.device)

        BLOCK_H = 16
        BLOCK_W = 16

        grid_w = (W_out + BLOCK_W - 1) // BLOCK_W
        grid_h = (H_out + BLOCK_H - 1) // BLOCK_H
        grid_nc = N * C

        depthwise_conv2d_kernel[lambda: ((grid_w, grid_h, grid_nc), (BLOCK_W, BLOCK_H, 1))](
            x_bf16.data_ptr(), w_bf16.data_ptr(), out.data_ptr(),
            N, C, H, W, KH, KW, H_out, W_out, BLOCK_H, BLOCK_W,
        )

        return out.to(input_dtype)
