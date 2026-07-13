import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_H = 8
TILE_W = 8
STRIDE = 4
PAD = 2
KH = 11
KW = 11
C_IN = 3
OC_TOTAL = 96
OC_PER_BLOCK = 4

# Input window for 8x8 output tile with stride 4
IN_WIN_H = TILE_H * STRIDE + KH - 1  # 42
IN_WIN_W = TILE_W * STRIDE + KW - 1  # 42


@avelang.jit
def conv2d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    H: al.i32,
    W: al.i32,
    OC: al.i32,
    OH: al.i32,
    OW: al.i32,
):
    one = al.convert(1, al.i32)

    in_total = B * C_IN * H * W
    in_layout = al.make_layout((in_total,), (one,))
    inp = al.make_tensor(input_ptr, al.bf16, in_layout)

    wt_total = OC * C_IN * KH * KW
    wt_layout = al.make_layout((wt_total,), (one,))
    weight = al.make_tensor(weight_ptr, al.bf16, wt_layout)

    bias_layout = al.make_layout((OC,), (one,))
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    out_total = B * OC * OH * OW
    out_layout = al.make_layout((out_total,), (one,))
    out = al.make_tensor(output_ptr, al.bf16, out_layout)

    tx = al.thread_id(0)
    ty = al.thread_id(1)
    bx = al.block_id(0)
    by = al.block_id(1)
    b = al.block_id(2)

    ow = bx * TILE_W + tx
    oh = by * TILE_H + ty

    # Shared memory: 5292 BF16 input window + 1452 BF16 weight tile
    smem_in = al.make_shared((5292,), al.bf16)
    smem_wt = al.make_shared((1452,), al.bf16)

    in_win_oh0 = by * TILE_H * STRIDE - PAD
    in_win_ow0 = bx * TILE_W * STRIDE - PAD

    tid_flat = ty * TILE_W + tx  # 0..63

    # Cooperative load: input window (5292 elements, 64 threads)
    for load_idx in al.range(tid_flat, 5292, 64):
        ic = load_idx // (IN_WIN_H * IN_WIN_W)
        rest = load_idx - ic * (IN_WIN_H * IN_WIN_W)
        ih = rest // IN_WIN_W
        iw = rest - ih * IN_WIN_W
        g_h = in_win_oh0 + ih
        g_w = in_win_ow0 + iw
        g_off = b * (C_IN * H * W) + ic * (H * W) + g_h * W + g_w
        if g_h >= 0:
            if g_h < H:
                if g_w >= 0:
                    if g_w < W:
                        smem_in[load_idx] = inp[g_off]

    # Thread's window-relative offsets
    ih_win_off = ty * STRIDE
    iw_win_off = tx * STRIDE

    out_base_b = b * OC * OH * OW
    out_spatial = oh * OW + ow

    # Loop over output-channel groups
    for g in al.range(0, 24):
        oc_start = g * 4

        # Cooperative load: weight tile (1452 elements, 64 threads)
        for load_idx in al.range(tid_flat, 1452, 64):
            oc_l = load_idx // (C_IN * KH * KW)
            rest_w = load_idx - oc_l * (C_IN * KH * KW)
            ic_w = rest_w // (KH * KW)
            rest_w2 = rest_w - ic_w * (KH * KW)
            kh_w = rest_w2 // KW
            kw_w = rest_w2 - kh_w * KW
            wt_off = (oc_start + oc_l) * (C_IN * KH * KW) + ic_w * (KH * KW) + kh_w * KW + kw_w
            if (oc_start + oc_l) < OC:
                smem_wt[load_idx] = weight[wt_off]

        al.syncthreads()

        if ow < OW:
            if oh < OH:
                # Accumulate 4 output channels
                oc = oc_start
                if oc < OC:
                    acc = al.convert(0.0, al.f32)
                    oc_wt_base = 0 * C_IN * KH * KW
                    for ic in al.range(0, 3):
                        ic_in_base = ic * IN_WIN_H * IN_WIN_W
                        ic_wt_base = ic * KH * KW
                        for kh_l in al.range(0, 11):
                            ih_win = ih_win_off + kh_l
                            ih_abs = in_win_oh0 + ih_win
                            in_row_base = ic_in_base + ih_win * IN_WIN_W
                            wt_row_base = ic_wt_base + kh_l * KW
                            for kw_l in al.range(0, 11):
                                iw_win = iw_win_off + kw_l
                                iw_abs = in_win_ow0 + iw_win
                                if ih_abs >= 0:
                                    if ih_abs < H:
                                        if iw_abs >= 0:
                                            if iw_abs < W:
                                                in_val = smem_in[in_row_base + iw_win]
                                                w_val = smem_wt[oc_wt_base + wt_row_base + kw_l]
                                                in_f32 = al.convert(in_val, al.f32)
                                                w_f32 = al.convert(w_val, al.f32)
                                                acc = acc + in_f32 * w_f32
                    acc = acc + al.convert(bias[oc], al.f32)
                    out[out_base_b + oc * OH * OW + out_spatial] = al.convert(acc, al.bf16)

                oc = oc_start + 1
                if oc < OC:
                    acc = al.convert(0.0, al.f32)
                    oc_wt_base = 1 * C_IN * KH * KW
                    for ic in al.range(0, 3):
                        ic_in_base = ic * IN_WIN_H * IN_WIN_W
                        ic_wt_base = ic * KH * KW
                        for kh_l in al.range(0, 11):
                            ih_win = ih_win_off + kh_l
                            ih_abs = in_win_oh0 + ih_win
                            in_row_base = ic_in_base + ih_win * IN_WIN_W
                            wt_row_base = ic_wt_base + kh_l * KW
                            for kw_l in al.range(0, 11):
                                iw_win = iw_win_off + kw_l
                                iw_abs = in_win_ow0 + iw_win
                                if ih_abs >= 0:
                                    if ih_abs < H:
                                        if iw_abs >= 0:
                                            if iw_abs < W:
                                                in_val = smem_in[in_row_base + iw_win]
                                                w_val = smem_wt[oc_wt_base + wt_row_base + kw_l]
                                                acc = acc + al.convert(in_val, al.f32) * al.convert(w_val, al.f32)
                    acc = acc + al.convert(bias[oc], al.f32)
                    out[out_base_b + oc * OH * OW + out_spatial] = al.convert(acc, al.bf16)

                oc = oc_start + 2
                if oc < OC:
                    acc = al.convert(0.0, al.f32)
                    oc_wt_base = 2 * C_IN * KH * KW
                    for ic in al.range(0, 3):
                        ic_in_base = ic * IN_WIN_H * IN_WIN_W
                        ic_wt_base = ic * KH * KW
                        for kh_l in al.range(0, 11):
                            ih_win = ih_win_off + kh_l
                            ih_abs = in_win_oh0 + ih_win
                            in_row_base = ic_in_base + ih_win * IN_WIN_W
                            wt_row_base = ic_wt_base + kh_l * KW
                            for kw_l in al.range(0, 11):
                                iw_win = iw_win_off + kw_l
                                iw_abs = in_win_ow0 + iw_win
                                if ih_abs >= 0:
                                    if ih_abs < H:
                                        if iw_abs >= 0:
                                            if iw_abs < W:
                                                in_val = smem_in[in_row_base + iw_win]
                                                w_val = smem_wt[oc_wt_base + wt_row_base + kw_l]
                                                acc = acc + al.convert(in_val, al.f32) * al.convert(w_val, al.f32)
                    acc = acc + al.convert(bias[oc], al.f32)
                    out[out_base_b + oc * OH * OW + out_spatial] = al.convert(acc, al.bf16)

                oc = oc_start + 3
                if oc < OC:
                    acc = al.convert(0.0, al.f32)
                    oc_wt_base = 3 * C_IN * KH * KW
                    for ic in al.range(0, 3):
                        ic_in_base = ic * IN_WIN_H * IN_WIN_W
                        ic_wt_base = ic * KH * KW
                        for kh_l in al.range(0, 11):
                            ih_win = ih_win_off + kh_l
                            ih_abs = in_win_oh0 + ih_win
                            in_row_base = ic_in_base + ih_win * IN_WIN_W
                            wt_row_base = ic_wt_base + kh_l * KW
                            for kw_l in al.range(0, 11):
                                iw_win = iw_win_off + kw_l
                                iw_abs = in_win_ow0 + iw_win
                                if ih_abs >= 0:
                                    if ih_abs < H:
                                        if iw_abs >= 0:
                                            if iw_abs < W:
                                                in_val = smem_in[in_row_base + iw_win]
                                                w_val = smem_wt[oc_wt_base + wt_row_base + kw_l]
                                                acc = acc + al.convert(in_val, al.f32) * al.convert(w_val, al.f32)
                    acc = acc + al.convert(bias[oc], al.f32)
                    out[out_base_b + oc * OH * OW + out_spatial] = al.convert(acc, al.bf16)

        al.syncthreads()


def avelang_conv2d(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    bias_tensor: torch.Tensor,
) -> torch.Tensor:
    B, C, H, W = input_tensor.shape
    OC, C2, KH_val, KW_val = weight_tensor.shape
    assert C == C2, "Input and weight channel counts must match"

    stride = 4
    pad = 2
    OH = (H + 2 * pad - KH_val) // stride + 1
    OW = (W + 2 * pad - KW_val) // stride + 1

    inp_bf16 = input_tensor.to(torch.bfloat16).contiguous()
    w_bf16 = weight_tensor.to(torch.bfloat16).contiguous()
    bias_bf16 = bias_tensor.to(torch.bfloat16).contiguous()

    out_bf16 = torch.empty(
        (B, OC, OH, OW),
        dtype=torch.bfloat16,
        device=input_tensor.device,
    )

    grid_x = (OW + TILE_W - 1) // TILE_W
    grid_y = (OH + TILE_H - 1) // TILE_H
    grid_z = B

    conv2d_kernel[lambda: ((grid_x, grid_y, grid_z), (TILE_W, TILE_H, 1))](
        inp_bf16,
        w_bf16,
        bias_bf16,
        out_bf16,
        B,
        H,
        W,
        OC,
        OH,
        OW,
    )

    return out_bf16


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=96,
            kernel_size=11,
            stride=4,
            padding=2,
        )

    def forward(self, x):
        return avelang_conv2d(x, self.conv1.weight, self.conv1.bias)
