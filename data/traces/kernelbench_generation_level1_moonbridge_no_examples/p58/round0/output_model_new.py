import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_transpose3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K_D: al.i32,
    K_H: al.i32,
    K_W: al.i32,
    BLOCK_W: al.i32,
):
    w_block = al.block_id(0)
    dh_bid = al.block_id(1)
    noc_bid = al.block_id(2)
    tid = al.thread_id(0)

    w = w_block * BLOCK_W + tid
    d = dh_bid // H_out
    h = dh_bid % H_out
    n = noc_bid // C_out
    oc = noc_bid % C_out

    one = al.convert(1, al.i32)
    zero = al.convert(0, al.i32)

    if w < W_out:
        inp_s0 = C_in * D_in * H_in * W_in
        inp_s1 = D_in * H_in * W_in
        inp_s2 = H_in * W_in
        inp_s3 = W_in
        inp_layout = al.make_layout(
            (N, C_in, D_in, H_in, W_in),
            (inp_s0, inp_s1, inp_s2, inp_s3, one),
        )
        inp = al.make_tensor(input_ptr, al.bf16, inp_layout)

        w_s0 = C_out * K_D * K_H * K_W
        w_s1 = K_D * K_H * K_W
        w_s2 = K_H * K_W
        w_s3 = K_W
        w_layout = al.make_layout(
            (C_in, C_out, K_D, K_H, K_W),
            (w_s0, w_s1, w_s2, w_s3, one),
        )
        weight = al.make_tensor(weight_ptr, al.bf16, w_layout)

        out_s0 = C_out * D_out * H_out * W_out
        out_s1 = D_out * H_out * W_out
        out_s2 = H_out * W_out
        out_s3 = W_out
        out_layout = al.make_layout(
            (N, C_out, D_out, H_out, W_out),
            (out_s0, out_s1, out_s2, out_s3, one),
        )
        out = al.make_tensor(output_ptr, al.bf16, out_layout)

        acc = al.convert(0.0, al.f32)

        for kd in al.range(K_D):
            d_in = d - kd
            if d_in >= zero and d_in < D_in:
                for kh in al.range(K_H):
                    h_in = h - kh
                    if h_in >= zero and h_in < H_in:
                        for kw in al.range(K_W):
                            w_in = w - kw
                            if w_in >= zero and w_in < W_in:
                                for ic in al.range(C_in):
                                    inp_val = al.convert(
                                        inp[n, ic, d_in, h_in, w_in], al.f32
                                    )
                                    w_val = al.convert(
                                        weight[ic, oc, kd, kh, kw], al.f32
                                    )
                                    acc = acc + inp_val * w_val

        out[n, oc, d, h, w] = al.convert(acc, al.bf16)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.has_bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_transpose3d.weight.data
        bias_tensor = self.conv_transpose3d.bias

        N, C_in, D_in, H_in, W_in = x.shape
        K_D, K_H, K_W = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        pad_d, pad_h, pad_w = self.padding
        out_pad_d, out_pad_h, out_pad_w = self.output_padding

        D_out = (D_in - 1) * stride_d - 2 * pad_d + K_D + out_pad_d
        H_out = (H_in - 1) * stride_h - 2 * pad_h + K_H + out_pad_h
        W_out = (W_in - 1) * stride_w - 2 * pad_w + K_W + out_pad_w
        C_out = self.out_channels

        x_bf16 = x.contiguous().to(torch.bfloat16)
        w_bf16 = weight.contiguous().to(torch.bfloat16)
        out_bf16 = torch.empty(
            N, C_out, D_out, H_out, W_out, dtype=torch.bfloat16, device=x.device
        )

        BLOCK_W = 64
        grid_w = (W_out + BLOCK_W - 1) // BLOCK_W
        grid_dh = D_out * H_out
        grid_noc = N * C_out

        conv_transpose3d_kernel[
            lambda: ((grid_w, grid_dh, grid_noc), (BLOCK_W, 1, 1))
        ](
            x_bf16,
            w_bf16,
            out_bf16,
            N,
            C_in,
            C_out,
            D_in,
            H_in,
            W_in,
            D_out,
            H_out,
            W_out,
            K_D,
            K_H,
            K_W,
            BLOCK_W,
        )

        if self.has_bias and bias_tensor is not None:
            bias_bf16 = bias_tensor.to(torch.bfloat16).view(1, C_out, 1, 1, 1)
            out_bf16 = out_bf16 + bias_bf16

        return out_bf16
