import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_transpose_min_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.i32,
    stride: al.i32,
    pad: al.i32,
    BLOCK_H: al.constexpr,
    BLOCK_W: al.constexpr,
):
    bx = al.block_id(0)
    by = al.block_id(1)
    bz = al.block_id(2)
    tx = al.thread_id(0)
    ty = al.thread_id(1)

    oh = by * BLOCK_H + ty
    ow = bx * BLOCK_W + tx
    n = bz

    if oh < H_out and ow < W_out:
        x_layout = al.make_layout(
            (N, C_in, H_in, W_in),
            (C_in * H_in * W_in, H_in * W_in, W_in, 1),
        )
        x = al.make_tensor(x_ptr, al.bf16, x_layout)

        w_layout = al.make_layout(
            (C_in, C_out, K, K),
            (C_out * K * K, K * K, K, 1),
        )
        w = al.make_tensor(w_ptr, al.bf16, w_layout)

        b_layout = al.make_layout((C_out,), (1,))
        b_tensor = al.make_tensor(b_ptr, al.f32, b_layout)

        out_layout = al.make_layout(
            (N, 1, H_out, W_out),
            (H_out * W_out, H_out * W_out, W_out, 1),
        )
        out = al.make_tensor(out_ptr, al.bf16, out_layout)

        first = 1
        min_val = al.convert(0.0, al.f32)

        for oc in al.range(C_out):
            sum_val = b_tensor[oc]
            for ic in al.range(C_in):
                for kh in al.range(K):
                    for kw in al.range(K):
                        rem_h = (oh + pad - kh) % stride
                        rem_w = (ow + pad - kw) % stride
                        if rem_h == 0 and rem_w == 0:
                            ih = (oh + pad - kh) // stride
                            iw = (ow + pad - kw) // stride
                            if ih >= 0 and ih < H_in and iw >= 0 and iw < W_in:
                                x_val = al.convert(x[n, ic, ih, iw], al.f32)
                                w_val = al.convert(w[ic, oc, kh, kw], al.f32)
                                sum_val = sum_val + x_val * w_val
            if first == 1:
                min_val = sum_val
                first = 0
            else:
                if sum_val < min_val:
                    min_val = sum_val

        out[n, 0, oh, ow] = al.convert(min_val, al.bf16)


@avelang.jit
def sum_height_kernel(
    in_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    H: al.i32,
    W: al.i32,
    BLOCK_W: al.constexpr,
):
    bx = al.block_id(0)
    bz = al.block_id(2)
    tx = al.thread_id(0)

    w = bx * BLOCK_W + tx
    n = bz

    if w < W:
        in_layout = al.make_layout(
            (N, 1, H, W),
            (H * W, H * W, W, 1),
        )
        in_tensor = al.make_tensor(in_ptr, al.bf16, in_layout)

        out_layout = al.make_layout(
            (N, 1, 1, W),
            (W, W, W, 1),
        )
        out_tensor = al.make_tensor(out_ptr, al.bf16, out_layout)

        sum_val = al.convert(0.0, al.f32)
        for h in al.range(H):
            sum_val = sum_val + al.convert(in_tensor[n, 0, h, w], al.f32)

        out_tensor[n, 0, 0, w] = al.convert(sum_val, al.bf16)


@avelang.jit
def gelu_kernel(
    in_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    W: al.i32,
    BLOCK_W: al.constexpr,
):
    bx = al.block_id(0)
    bz = al.block_id(2)
    tx = al.thread_id(0)

    w = bx * BLOCK_W + tx
    n = bz

    if w < W:
        in_layout = al.make_layout(
            (N, 1, 1, W),
            (W, W, W, 1),
        )
        in_tensor = al.make_tensor(in_ptr, al.bf16, in_layout)

        out_layout = al.make_layout(
            (N, 1, 1, W),
            (W, W, W, 1),
        )
        out_tensor = al.make_tensor(out_ptr, al.bf16, out_layout)

        val = al.convert(in_tensor[n, 0, 0, w], al.f32)

        sqrt_2_pi = al.convert(0.7978845608028654, al.f32)
        coeff = al.convert(0.044715, al.f32)
        half = al.convert(0.5, al.f32)
        one = al.convert(1.0, al.f32)

        x3 = val * val * val
        inner = sqrt_2_pi * (val + coeff * x3)
        tanh_val = al.tanh(inner)
        gelu_val = val * half * (one + tanh_val)

        out_tensor[n, 0, 0, w] = al.convert(gelu_val, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        w = self.conv_transpose.weight
        b_conv = self.conv_transpose.bias
        final_bias = self.bias

        N, C_in, H_in, W_in = x.shape
        C_out = w.shape[1]
        K = w.shape[2]
        stride_val = self.conv_transpose.stride[0]
        pad_val = self.conv_transpose.padding[0]
        out_pad_val = self.conv_transpose.output_padding[0]

        H_out = (H_in - 1) * stride_val - 2 * pad_val + K + out_pad_val
        W_out = (W_in - 1) * stride_val - 2 * pad_val + K + out_pad_val

        x_bf16 = x.contiguous().to(torch.bfloat16)
        w_bf16 = w.contiguous().to(torch.bfloat16)
        b_conv_f32 = b_conv.contiguous().to(torch.float32)

        mid1 = torch.empty(N, 1, H_out, W_out, dtype=torch.bfloat16, device=x.device)

        BLOCK_H = 8
        BLOCK_W = 8
        grid_x = (W_out + BLOCK_W - 1) // BLOCK_W
        grid_y = (H_out + BLOCK_H - 1) // BLOCK_H
        conv_transpose_min_kernel[lambda: ((grid_x, grid_y, N), (BLOCK_W, BLOCK_H, 1))](
            x_bf16,
            w_bf16,
            b_conv_f32,
            mid1,
            N,
            C_in,
            C_out,
            H_in,
            W_in,
            H_out,
            W_out,
            K,
            stride_val,
            pad_val,
            BLOCK_H,
            BLOCK_W,
        )

        mid2 = torch.empty(N, 1, 1, W_out, dtype=torch.bfloat16, device=x.device)

        BLOCK_W2 = 256
        grid_w2 = (W_out + BLOCK_W2 - 1) // BLOCK_W2
        sum_height_kernel[lambda: ((grid_w2, 1, N), (BLOCK_W2, 1, 1))](
            mid1, mid2, N, H_out, W_out, BLOCK_W2,
        )

        mid3 = torch.empty(N, 1, 1, W_out, dtype=torch.bfloat16, device=x.device)

        gelu_kernel[lambda: ((grid_w2, 1, N), (BLOCK_W2, 1, 1))](
            mid2, mid3, N, W_out, BLOCK_W2,
        )

        out_bf16 = mid3 + final_bias.to(torch.bfloat16)
        return out_bf16
