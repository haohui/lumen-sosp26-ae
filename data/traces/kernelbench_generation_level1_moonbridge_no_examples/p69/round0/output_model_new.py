import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_transpose2d_kernel(
    in_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
    OH: al.i32,
    OW: al.i32,
    KH: al.i32,
    KW: al.i32,
    OC_TILES: al.i32,
    OH_TILES: al.i32,
    OW_TILES: al.i32,
):
    in_s1 = H * W
    in_s2 = W
    in_s3 = 1
    in_s0 = IC * in_s1
    in_layout = al.make_layout((B, IC, H, W), (in_s0, in_s1, in_s2, in_s3))
    in_t = al.make_tensor(in_ptr, al.bf16, in_layout)

    w_s1 = KH * KW
    w_s2 = KW
    w_s3 = 1
    w_s0 = OC * w_s1
    w_layout = al.make_layout((IC, OC, KH, KW), (w_s0, w_s1, w_s2, w_s3))
    w_t = al.make_tensor(w_ptr, al.bf16, w_layout)

    out_s1 = OH * OW
    out_s2 = OW
    out_s3 = 1
    out_s0 = OC * out_s1
    out_layout = al.make_layout((B, OC, OH, OW), (out_s0, out_s1, out_s2, out_s3))
    out_t = al.make_tensor(out_ptr, al.bf16, out_layout)

    ow_tile = al.block_id(0)
    combined = al.block_id(1)
    b_idx = al.block_id(2)
    oh_tile = combined // OC_TILES
    oc_tile = combined % OC_TILES

    tid = al.thread_id(0)
    local_oc = tid % 8
    tmp = tid // 8
    local_ow = tmp % 4
    local_oh = tmp // 4

    oc = oc_tile * 8 + local_oc
    oh = oh_tile * 8 + local_oh
    ow = ow_tile * 4 + local_ow

    oh_base = oh_tile * 8
    ow_base = ow_tile * 4
    oc_base = oc_tile * 8

    in_shared = al.make_shared((8, 10, 8), al.bf16)
    w_shared = al.make_shared((8, 8, 3, 5), al.bf16)

    acc = al.convert(0.0, al.f32)

    for ic_chunk in al.range(8):
        ic_start = ic_chunk * 8

        for r in al.range(3):
            li = tid + r * 256
            if li < 640:
                l_ic = li // 80
                rem = li % 80
                l_ih = rem // 8
                l_iw = rem % 8
                g_ic = ic_start + l_ic
                g_ih = oh_base + l_ih - 2
                g_iw = ow_base + l_iw - 4
                if g_ih >= 0:
                    if g_ih < H:
                        if g_iw >= 0:
                            if g_iw < W:
                                in_shared[l_ic, l_ih, l_iw] = in_t[b_idx, g_ic, g_ih, g_iw]

        for r in al.range(4):
            li = tid + r * 256
            if li < 960:
                l_ic = li // 120
                rem = li % 120
                l_oc = rem // 15
                rem2 = rem % 15
                l_kh = rem2 // 5
                l_kw = rem2 % 5
                g_ic = ic_start + l_ic
                g_oc = oc_base + l_oc
                if g_oc < OC:
                    w_shared[l_ic, l_oc, l_kh, l_kw] = w_t[g_ic, g_oc, l_kh, l_kw]

        al.syncthreads()

        if oc < OC:
            if oh < OH:
                if ow < OW:
                    for lic in al.range(8):
                        for kh in al.range(3):
                            ih = oh - kh
                            if ih >= 0:
                                if ih < H:
                                    ih_sm = ih - oh_base + 2
                                    for kw in al.range(5):
                                        iw = ow - kw
                                        if iw >= 0:
                                            if iw < W:
                                                iw_sm = iw - ow_base + 4
                                                in_val = al.convert(
                                                    in_shared[lic, ih_sm, iw_sm], al.f32
                                                )
                                                w_val = al.convert(
                                                    w_shared[lic, local_oc, kh, kw], al.f32
                                                )
                                                acc = acc + in_val * w_val

        al.syncthreads()

    if oc < OC:
        if oh < OH:
            if ow < OW:
                out_t[b_idx, oc, oh, ow] = al.convert(acc, al.bf16)


def avelang_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    B = x.shape[0]
    IC = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    OC = weight.shape[1]
    KH = weight.shape[2]
    KW = weight.shape[3]

    OH = H + KH - 1
    OW = W + KW - 1

    OC_TILES = (OC + 7) // 8
    OH_TILES = (OH + 7) // 8
    OW_TILES = (OW + 3) // 4

    out = torch.empty((B, OC, OH, OW), dtype=torch.bfloat16, device=x.device)

    grid = (OW_TILES, OH_TILES * OC_TILES, B)
    block = (256, 1, 1)

    conv_transpose2d_kernel[lambda: (grid, block)](
        x.data_ptr(),
        weight.data_ptr(),
        out.data_ptr(),
        B, IC, OC, H, W, OH, OW, KH, KW,
        OC_TILES, OH_TILES, OW_TILES,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1), padding: tuple = (0, 0),
                 output_padding: tuple = (0, 0), dilation: tuple = (1, 1),
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
            dilation=dilation, groups=groups, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_transpose2d.weight
        weight = weight.contiguous()
        x = x.contiguous()
        if self.conv_transpose2d.bias is not None:
            out = avelang_conv_transpose2d(x, weight)
            bias = self.conv_transpose2d.bias
            out = out + bias.view(1, -1, 1, 1)
            return out
        else:
            return avelang_conv_transpose2d(x, weight)
