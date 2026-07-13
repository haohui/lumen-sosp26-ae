import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_H = 16
TILE_W = 16
CK = 16
MISH_BLOCK = 256
C_OUT = 128

WIN_H = TILE_H + 2
WIN_W = TILE_W + 2
WIN_FLAT = WIN_H * WIN_W * CK
BLOCK_THREADS = TILE_H * TILE_W
CHUNK = (WIN_FLAT + BLOCK_THREADS - 1) // BLOCK_THREADS


@avelang.jit
def conv2d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_in: al.i32,
    H: al.i32,
    W: al.i32,
    C_out: al.i32,
    OH: al.i32,
    OW: al.i32,
    KH: al.i32,
    KW: al.i32,
    has_bias: al.i32,
):
    batch = al.block_id(2)
    ow_tile = al.block_id(0)
    oh_tile = al.block_id(1)

    tid_x = al.thread_id(0)
    tid_y = al.thread_id(1)

    block_w = al.block_dim(0)
    block_h = al.block_dim(1)

    ow = ow_tile * block_w + tid_x
    oh = oh_tile * block_h + tid_y

    zero_i32 = al.convert(0, al.i32)
    one_i32 = al.convert(1, al.i32)

    in_s0 = C_in * H * W
    in_s1 = H * W
    in_s2 = W
    in_layout = al.make_layout((B, C_in, H, W), (in_s0, in_s1, in_s2, 1))
    inp = al.make_tensor(input_ptr, al.bf16, in_layout)

    w_s0 = C_in * KH * KW
    w_s1 = KH * KW
    w_s2 = KW
    w_layout = al.make_layout((C_out, C_in, KH, KW), (w_s0, w_s1, w_s2, 1))
    wgt = al.make_tensor(weight_ptr, al.bf16, w_layout)

    out_s0 = C_out * OH * OW
    out_s1 = OH * OW
    out_s2 = OW
    out_layout = al.make_layout((B, C_out, OH, OW), (out_s0, out_s1, out_s2, 1))
    out = al.make_tensor(output_ptr, al.bf16, out_layout)

    bias_layout = al.make_layout((C_out,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    in_sh = al.make_shared((WIN_FLAT,), al.bf16)
    acc = al.make_local((C_OUT,), al.f32)

    for a_idx in al.range(C_OUT):
        acc[a_idx] = al.convert(0.0, al.f32)

    win_flat = al.convert(WIN_FLAT, al.i32)
    win_w_val = al.convert(WIN_W, al.i32)
    ck_val = al.convert(CK, al.i32)
    chunk = al.convert(CHUNK, al.i32)

    valid_thread = zero_i32
    if ow < OW and oh < OH:
        valid_thread = one_i32

    linear_tid = tid_x + tid_y * block_w

    ic_base = zero_i32
    for _ic in al.range(C_in):
        if ic_base < C_in:
            in_start = linear_tid * chunk
            for offset in al.range(chunk):
                idx = in_start + offset
                if idx < win_flat:
                    ic_local = idx - (idx // ck_val) * ck_val
                    rest = idx // ck_val
                    kw_win = rest - (rest // win_w_val) * win_w_val
                    kh_win = rest // win_w_val

                    global_ic = ic_base + ic_local
                    global_kh = oh_tile * block_h + kh_win
                    global_kw = ow_tile * block_w + kw_win

                    if global_ic < C_in and global_kh < H and global_kw < W:
                        in_sh[idx] = inp[batch, global_ic, global_kh, global_kw]
                    else:
                        in_sh[idx] = al.convert(0.0, al.bf16)

            al.syncthreads()

            if valid_thread != zero_i32:
                oc = zero_i32
                oc_idx = zero_i32
                for _oc in al.range(C_out):
                    if oc < C_out:
                        for ic_local in al.range(CK):
                            global_ic = ic_base + ic_local
                            if global_ic < C_in:
                                for kh in al.range(KH):
                                    for kw in al.range(KW):
                                        kw_win = tid_x + kw
                                        kh_win = tid_y + kh
                                        sh_idx = (
                                            kh_win * win_w_val * ck_val
                                            + kw_win * ck_val
                                            + ic_local
                                        )
                                        inp_val = al.convert(in_sh[sh_idx], al.f32)
                                        w_val = al.convert(
                                            wgt[oc, global_ic, kh, kw], al.f32
                                        )
                                        acc[oc_idx] = acc[oc_idx] + inp_val * w_val
                    oc = oc + one_i32
                    oc_idx = oc_idx + one_i32

            al.syncthreads()

        ic_base = ic_base + CK

    if valid_thread != zero_i32:
        oc = zero_i32
        oc_idx = zero_i32
        for _oc in al.range(C_out):
            if oc < C_out:
                if has_bias != zero_i32:
                    acc[oc_idx] = acc[oc_idx] + al.convert(bias[oc], al.f32)
                out[batch, oc, oh, ow] = al.convert(acc[oc_idx], al.bf16)
            oc = oc + one_i32
            oc_idx = oc_idx + one_i32


@avelang.jit
def mish_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
):
    tid = al.thread_id(0)
    gid = al.block_id(0) * al.block_dim(0) + tid

    if gid < N:
        layout = al.make_layout((N,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout)
        out = al.make_tensor(out_ptr, al.bf16, layout)

        val = al.convert(x[gid], al.f32)
        exp_val = al.exp(val)
        one = al.convert(1.0, al.f32)
        softplus = al.log(one + exp_val)
        tanh_sp = al.tanh(softplus)
        result = val * tanh_sp
        out[gid] = al.convert(result, al.bf16)


def _launch_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    B, C_in, H, W = x.shape
    C_out = weight.shape[0]
    KH = weight.shape[2]
    KW = weight.shape[3]
    OH = H - KH + 1
    OW = W - KW + 1

    out = torch.empty(B, C_out, OH, OW, dtype=x.dtype, device=x.device)

    weight_bf16 = weight.to(x.dtype).contiguous()
    if bias is not None:
        bias_bf16 = bias.to(x.dtype).contiguous()
        has_bias = 1
    else:
        bias_bf16 = torch.empty(1, dtype=x.dtype, device=x.device)
        has_bias = 0

    grid_ow = (OW + TILE_W - 1) // TILE_W
    grid_oh = (OH + TILE_H - 1) // TILE_H
    conv2d_kernel[lambda: ((grid_ow, grid_oh, B), (TILE_W, TILE_H, 1))](
        x, weight_bf16, bias_bf16, out,
        B, C_in, H, W, C_out, OH, OW, KH, KW, has_bias,
    )
    return out


def _launch_mish(x: torch.Tensor) -> torch.Tensor:
    N = x.numel()
    out = torch.empty_like(x)
    grid = (N + MISH_BLOCK - 1) // MISH_BLOCK
    mish_kernel[lambda: ((grid, 1, 1), (MISH_BLOCK, 1, 1))](x, out, N)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        weight = self.conv.weight.data.contiguous()
        bias = self.conv.bias.data.contiguous() if self.conv.bias is not None else None

        x = _launch_conv2d(x, weight, bias)
        x = _launch_mish(x)
        x = _launch_mish(x)
        return x
