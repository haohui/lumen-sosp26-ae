import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_H = 16
TILE_W = 16
TILE_IC = 16
THREADS = TILE_H * TILE_W


@avelang.jit
def conv2d_tiled_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
    OH: al.i32,
    OW: al.i32,
    KH: al.constexpr,
    KW: al.constexpr,
    stride_h: al.i32,
    stride_w: al.i32,
    pad_h: al.i32,
    pad_w: al.i32,
    dilation_h: al.i32,
    dilation_w: al.i32,
):
    bx = al.block_id(0)
    by = al.block_id(1)
    bz = al.block_id(2)
    tid = al.thread_id(0)

    th = tid // TILE_W
    tw = tid % TILE_W

    oh = by * TILE_H + th
    ow = bx * TILE_W + tw
    oc = bz % OC
    b = bz // OC

    R_H = TILE_H + KH - 1
    R_W = TILE_W + KW - 1

    ih_start = by * TILE_H * stride_h - pad_h
    iw_start = bx * TILE_W * stride_w - pad_w

    inp_shared = al.make_shared((TILE_IC, R_H, R_W), al.bf16)
    wt_shared = al.make_shared((TILE_IC, KH, KW), al.bf16)

    inp_layout = al.make_layout((B, IC, H, W), (IC * H * W, H * W, W, 1))
    inp = al.make_tensor(input_ptr, al.bf16, inp_layout)

    wt_layout = al.make_layout((OC, IC, KH, KW), (IC * KH * KW, KH * KW, KW, 1))
    wt = al.make_tensor(weight_ptr, al.bf16, wt_layout)

    acc = al.convert(0.0, al.f32)

    inp_total = TILE_IC * R_H * R_W
    wt_total = TILE_IC * KH * KW
    num_ic_tiles = (IC + TILE_IC - 1) // TILE_IC

    for ic_tile in al.range(num_ic_tiles):
        ic_start = ic_tile * TILE_IC

        for idx in al.range(tid, inp_total, THREADS):
            rw = idx % R_W
            tmp = idx // R_W
            rh = tmp % R_H
            ic_l = tmp // R_H

            ic_g = ic_start + ic_l
            ih_g = ih_start + rh
            iw_g = iw_start + rw

            val = al.convert(0.0, al.bf16)
            if ic_g < IC:
                if ih_g >= 0:
                    if ih_g < H:
                        if iw_g >= 0:
                            if iw_g < W:
                                val = inp[b, ic_g, ih_g, iw_g]
            inp_shared[ic_l, rh, rw] = val

        for idx in al.range(tid, wt_total, THREADS):
            kw_l = idx % KW
            tmp = idx // KW
            kh_l = tmp % KH
            ic_l = tmp // KH

            ic_g = ic_start + ic_l

            val = al.convert(0.0, al.bf16)
            if ic_g < IC:
                if oc < OC:
                    val = wt[oc, ic_g, kh_l, kw_l]
            wt_shared[ic_l, kh_l, kw_l] = val

        al.syncthreads()

        if oh < OH:
            if ow < OW:
                if oc < OC:
                    if b < B:
                        for ic_l in al.range(TILE_IC):
                            ic_g = ic_start + ic_l
                            if ic_g < IC:
                                for kh_l in al.range(KH):
                                    ih_off = th * stride_h + kh_l * dilation_h
                                    for kw_l in al.range(KW):
                                        iw_off = tw * stride_w + kw_l * dilation_w
                                        inp_val = inp_shared[ic_l, ih_off, iw_off]
                                        wt_val = wt_shared[ic_l, kh_l, kw_l]
                                        acc = acc + al.convert(inp_val, al.f32) * al.convert(wt_val, al.f32)

        al.syncthreads()

    if oh < OH:
        if ow < OW:
            if oc < OC:
                if b < B:
                    out_layout = al.make_layout((B, OC, OH, OW), (OC * OH * OW, OH * OW, OW, 1))
                    out = al.make_tensor(output_ptr, al.bf16, out_layout)
                    out[b, oc, oh, ow] = al.convert(acc, al.bf16)


def avelang_conv2d(x, weight, bias, stride, padding, dilation, groups):
    if groups != 1:
        raise ValueError("Only groups=1 is currently supported.")

    B, IC, H, W = x.shape
    OC, IC_G, KH, KW = weight.shape

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    OH = (H + 2 * pad_h - dilation_h * (KH - 1) - 1) // stride_h + 1
    OW = (W + 2 * pad_w - dilation_w * (KW - 1) - 1) // stride_w + 1

    grid_x = (OW + TILE_W - 1) // TILE_W
    grid_y = (OH + TILE_H - 1) // TILE_H
    grid_z = B * OC

    out = torch.empty(B, OC, OH, OW, dtype=x.dtype, device=x.device)

    conv2d_tiled_kernel[lambda: ((grid_x, grid_y, grid_z), (THREADS, 1, 1))](
        x,
        weight,
        out,
        B,
        IC,
        OC,
        H,
        W,
        OH,
        OW,
        KH,
        KW,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
    )

    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)

    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        bias=False,
    ):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups

    def forward(self, x):
        weight = self.conv2d.weight
        bias = self.conv2d.bias
        return avelang_conv2d(
            x, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )
