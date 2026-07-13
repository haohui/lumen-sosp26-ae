import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv3d_fused_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    conv_bias_ptr: al.Pointer(al.bf16),
    final_bias_ptr: al.Pointer(al.bf16),
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
    K: al.i32,
):
    one = al.convert(1, al.i32)

    inp = al.make_tensor(
        input_ptr, al.bf16,
        al.make_layout((N * C_in * D_in * H_in * W_in,), (one,)),
    )
    wgt = al.make_tensor(
        weight_ptr, al.bf16,
        al.make_layout((C_out * C_in * K * K * K,), (one,)),
    )
    cb = al.make_tensor(
        conv_bias_ptr, al.bf16,
        al.make_layout((C_out,), (one,)),
    )
    fb = al.make_tensor(
        final_bias_ptr, al.bf16,
        al.make_layout((C_out,), (one,)),
    )
    out = al.make_tensor(
        output_ptr, al.bf16,
        al.make_layout((N * C_out * D_out * H_out * W_out,), (one,)),
    )

    sN_in = C_in * D_in * H_in * W_in
    sC_in = D_in * H_in * W_in
    sD_in = H_in * W_in
    sH_in = W_in

    sCout_w = C_in * K * K * K
    sCin_w = K * K * K
    sKd_w = K * K
    sKh_w = K

    sN_out = C_out * D_out * H_out * W_out
    sC_out = D_out * H_out * W_out
    sD_out = H_out * W_out
    sH_out = W_out

    tid_h = al.thread_id(0)
    tid_w = al.thread_id(1)
    bdim_h = al.block_dim(0)
    bdim_w = al.block_dim(1)

    bid_w = al.block_id(0)
    bid_h = al.block_id(1)
    bid_z = al.block_id(2)

    CD = C_out * D_out
    n = bid_z // CD
    rem = bid_z - n * CD
    oc = rem // D_out
    d = rem - oc * D_out

    h = bid_h * bdim_h + tid_h
    w = bid_w * bdim_w + tid_w

    if n < N:
        if oc < C_out:
            if d < D_out:
                if h < H_out:
                    if w < W_out:
                        zf = al.convert(0.0, al.f32)
                        acc = zf

                        off_out = (
                            n * sN_out + oc * sC_out
                            + d * sD_out + h * sH_out + w
                        )

                        for ic in al.range(C_in):
                            wgt_base = oc * sCout_w + ic * sCin_w
                            in_base_ic = n * sN_in + ic * sC_in
                            for kd in al.range(K):
                                off_d = in_base_ic + (d + kd) * sD_in
                                for kh in al.range(K):
                                    off_h = off_d + (h + kh) * sH_in
                                    for kw in al.range(K):
                                        off_wgt = (
                                            wgt_base
                                            + kd * sKd_w
                                            + kh * sKh_w
                                            + kw
                                        )
                                        ival = al.convert(inp[off_h + (w + kw)], al.f32)
                                        wval = al.convert(wgt[off_wgt], al.f32)
                                        acc = acc + ival * wval

                        acc = acc + al.convert(cb[oc], al.f32)

                        if acc < zf:
                            acc = zf

                        neg_slope = al.convert(0.01, al.f32)
                        if acc < zf:
                            acc = acc * neg_slope

                        half = al.convert(0.5, al.f32)
                        fone = al.convert(1.0, al.f32)
                        sqrt_2_pi = al.convert(0.7978845608028654, al.f32)
                        coeff = al.convert(0.044715, al.f32)
                        x3 = acc * acc * acc
                        inner = sqrt_2_pi * (acc + coeff * x3)
                        acc = half * acc * (fone + al.tanh(inner))

                        exp_neg = al.exp(zf - acc)
                        acc = fone / (fone + exp_neg)

                        acc = acc + al.convert(fb[oc], al.f32)

                        out[off_out] = al.convert(acc, al.bf16)


def _launch(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_bias: torch.Tensor,
    final_bias: torch.Tensor,
) -> torch.Tensor:
    N, C_in, D_in, H_in, W_in = x.shape
    C_out = weight.shape[0]
    K = weight.shape[2]
    D_out = D_in - K + 1
    H_out = H_in - K + 1
    W_out = W_in - K + 1

    if not x.is_cuda:
        x = x.cuda()
    device = x.device

    x_flat = x.contiguous().reshape(-1)
    w_flat = weight.to(device).contiguous().reshape(-1)
    cb_flat = conv_bias.to(device).contiguous().reshape(-1)
    fb_flat = final_bias.to(device).contiguous().reshape(-1)

    out = torch.empty(
        N, C_out, D_out, H_out, W_out,
        device=device, dtype=x.dtype,
    )
    out_flat = out.reshape(-1)

    TILE = 16
    grid_x = (W_out + TILE - 1) // TILE
    grid_y = (H_out + TILE - 1) // TILE
    grid_z = N * C_out * D_out

    conv3d_fused_kernel[lambda: ((grid_x, grid_y, grid_z), (TILE, TILE, 1))](
        x_flat,
        w_flat,
        cb_flat,
        fb_flat,
        out_flat,
        N,
        C_in,
        C_out,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return _launch(x, self.conv.weight, self.conv.bias, self.bias)
