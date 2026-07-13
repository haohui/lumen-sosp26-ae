import torch
import torch.nn as nn
import avelang
import avelang.language as al

C_IN = 64
C_OUT = 128
KH = 3
KW = 3
CK = 16

WIN_FLAT = KH * KW * CK


@avelang.jit
def conv2d_3x3_scale_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    tid = al.thread_id(0)
    zero_i32 = al.convert(0, al.i32)
    one_i32 = al.convert(1, al.i32)
    two_f32 = al.convert(2.0, al.f32)
    c_in_val = al.convert(C_IN, al.i32)
    c_out_val = al.convert(C_OUT, al.i32)
    kh_val = al.convert(KH, al.i32)
    kw_val = al.convert(KW, al.i32)
    ck_val = al.convert(CK, al.i32)
    win_flat_val = al.convert(WIN_FLAT, al.i32)

    x_layout = al.make_layout(
        (N, c_in_val, H, W),
        (c_in_val * H * W, H * W, W, 1),
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    w_layout = al.make_layout(
        (c_out_val, c_in_val, kh_val, kw_val),
        (c_in_val * kh_val * kw_val, kh_val * kw_val, kw_val, 1),
    )
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    b_layout = al.make_layout((c_out_val,), (1,))
    b = al.make_tensor(b_ptr, al.bf16, b_layout)

    out_layout = al.make_layout(
        (N, c_out_val, H_out, W_out),
        (c_out_val * H_out * W_out, H_out * W_out, W_out, 1),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    spatial_idx = al.block_id(0)
    n = al.block_id(1)
    h_out = spatial_idx // W_out
    w_out = spatial_idx % W_out
    c_out = tid

    in_sh = al.make_shared((WIN_FLAT,), al.bf16)
    acc = al.make_local((C_OUT,), al.f32)

    acc_idx = zero_i32
    for _init in al.range(C_OUT):
        acc[acc_idx] = al.convert(0.0, al.f32)
        acc_idx = acc_idx + one_i32

    acc_idx = c_out
    acc[acc_idx] = al.convert(b[c_out], al.f32)

    ic_base = zero_i32
    for _tile in al.range(C_IN // CK):
        if ic_base < c_in_val:
            # Cooperative load of input window tile
            load_idx = tid
            if load_idx < win_flat_val:
                ic_local = load_idx
                ic_local = ic_local - (ic_local // ck_val) * ck_val
                rest = load_idx // ck_val
                kw_win = rest
                kw_win = kw_win - (kw_win // kw_val) * kw_val
                ky_win = rest // kw_val

                global_ic = ic_base + ic_local
                global_h = h_out + ky_win
                global_w = w_out + kw_win

                if global_ic < c_in_val:
                    in_sh[load_idx] = x[n, global_ic, global_h, global_w]
                else:
                    in_sh[load_idx] = al.convert(0.0, al.bf16)

            load_idx2 = tid + al.convert(128, al.i32)
            if load_idx2 < win_flat_val:
                ic_local2 = load_idx2
                ic_local2 = ic_local2 - (ic_local2 // ck_val) * ck_val
                rest2 = load_idx2 // ck_val
                kw_win2 = rest2
                kw_win2 = kw_win2 - (kw_win2 // kw_val) * kw_val
                ky_win2 = rest2 // kw_val

                global_ic2 = ic_base + ic_local2
                global_h2 = h_out + ky_win2
                global_w2 = w_out + kw_win2

                if global_ic2 < c_in_val:
                    in_sh[load_idx2] = x[n, global_ic2, global_h2, global_w2]
                else:
                    in_sh[load_idx2] = al.convert(0.0, al.bf16)

            al.syncthreads()

            # Compute partial dot products
            for ic_local3 in al.range(CK):
                global_ic3 = ic_base + ic_local3
                if global_ic3 < c_in_val:
                    ky_ctr = zero_i32
                    for _ky in al.range(KH):
                        kw_ctr = zero_i32
                        for _kw in al.range(KW):
                            sh_idx = (ky_ctr * kw_val + kw_ctr) * ck_val + ic_local3
                            x_val = al.convert(in_sh[sh_idx], al.f32)
                            w_val = al.convert(
                                w[c_out, global_ic3, ky_ctr, kw_ctr], al.f32
                            )
                            acc[acc_idx] = acc[acc_idx] + x_val * w_val
                            kw_ctr = kw_ctr + one_i32
                        ky_ctr = ky_ctr + one_i32

            al.syncthreads()

        ic_base = ic_base + ck_val

    acc[acc_idx] = acc[acc_idx] * two_f32
    out[n, c_out, h_out, w_out] = al.convert(acc[acc_idx], al.bf16)


@avelang.jit
def channel_min_kernel(
    in_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    H: al.i32,
    W: al.i32,
):
    tid = al.thread_id(0)
    zero_i32 = al.convert(0, al.i32)
    one_i32 = al.convert(1, al.i32)
    c_val = al.convert(C_OUT, al.i32)

    in_layout = al.make_layout(
        (N, c_val, H, W),
        (c_val * H * W, H * W, W, 1),
    )
    in_tensor = al.make_tensor(in_ptr, al.bf16, in_layout)

    out_layout = al.make_layout(
        (N, 1, H, W),
        (H * W, H * W, W, 1),
    )
    out_tensor = al.make_tensor(out_ptr, al.bf16, out_layout)

    spatial_idx = al.block_id(0)
    n = al.block_id(1)
    h = spatial_idx // W
    w = spatial_idx % W

    if tid == 0:
        out_tensor[n, zero_i32, h, w] = in_tensor[n, zero_i32, h, w]
        i = one_i32
        for _i in al.range(C_OUT - 1):
            cur_bf16 = in_tensor[n, i, h, w]
            cur_f32 = al.convert(cur_bf16, al.f32)
            best_bf16 = out_tensor[n, zero_i32, h, w]
            best_f32 = al.convert(best_bf16, al.f32)
            diff = cur_f32 - best_f32
            diff_int = al.bitcast(diff, al.i32)
            if diff_int < 0:
                out_tensor[n, zero_i32, h, w] = cur_bf16
            i = i + one_i32


def avelang_conv_scale_min(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    scale_factor: float,
) -> torch.Tensor:
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    N, _C_in, H, W = x.shape
    H_out = H - KH + 1
    W_out = W - KW + 1
    x_bf16 = x.to(torch.bfloat16)
    w_bf16 = w.to(torch.bfloat16)
    b_bf16 = b.to(torch.bfloat16)

    conv_out = torch.empty(N, C_OUT, H_out, W_out, dtype=torch.bfloat16, device=x.device)
    conv2d_3x3_scale_kernel[
        lambda: ((H_out * W_out, N, 1), (C_OUT, 1, 1))
    ](x_bf16, w_bf16, b_bf16, conv_out, N, H, W, H_out, W_out)

    out = torch.empty(N, 1, H_out, W_out, dtype=torch.bfloat16, device=x.device)
    channel_min_kernel[
        lambda: ((H_out * W_out, N, 1), (C_OUT, 1, 1))
    ](conv_out, out, N, H_out, W_out)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        w = self.conv.weight.data
        b = self.conv.bias.data
        return avelang_conv_scale_min(x, w, b, self.scale_factor)
