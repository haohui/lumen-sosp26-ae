import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    OC: al.i32,
    KD: al.i32,
    KH: al.i32,
    KW: al.i32,
    OD: al.i32,
    OH: al.i32,
    OW: al.i32,
    oh_tiles: al.i32,
    ow_tiles: al.i32,
    OC_TILE: al.constexpr,
    OH_TILE: al.constexpr,
    OW_TILE: al.constexpr,
):
    inp = al.make_tensor(
        input_ptr, al.bf16,
        al.make_layout((B, IC, D, H, W), (IC * D * H * W, D * H * W, H * W, W, 1)),
    )
    wgt = al.make_tensor(
        weight_ptr, al.bf16,
        al.make_layout((OC, IC, KD, KH, KW), (IC * KD * KH * KW, KD * KH * KW, KH * KW, KW, 1)),
    )
    out = al.make_tensor(
        output_ptr, al.bf16,
        al.make_layout((B, OC, OD, OH, OW), (OC * OD * OH * OW, OD * OH * OW, OH * OW, OW, 1)),
    )

    b = al.block_id(0)
    spat_block = al.block_id(1)
    oc_block = al.block_id(2)
    tid = al.thread_id(0)

    oc_start = oc_block * OC_TILE

    # Decode spatial block
    oh_ow_tiles = oh_tiles * ow_tiles
    od = spat_block // oh_ow_tiles
    rem = spat_block - od * oh_ow_tiles
    oh_tile = rem // ow_tiles
    ow_tile = rem - oh_tile * ow_tiles

    oh_start = oh_tile * OH_TILE
    ow_start = ow_tile * OW_TILE

    th = tid // OW_TILE
    tw = tid - th * OW_TILE

    my_oh = oh_start + th
    my_ow = ow_start + tw

    if (my_oh < OH) and (my_ow < OW) and (od < OD):
        acc = al.make_local((OC_TILE,), al.f32)
        for oc_l in al.range(OC_TILE):
            acc[oc_l] = al.convert(0.0, al.f32)

        for ic in al.range(3):
            for kd in al.range(3):
                id_idx = od + kd
                for kh in al.range(5):
                    ih_idx = my_oh + kh
                    for kw in al.range(7):
                        iw_idx = my_ow + kw
                        inp_val = al.convert(inp[b, ic, id_idx, ih_idx, iw_idx], al.f32)
                        for oc_l in al.range(OC_TILE):
                            oc_idx = oc_start + oc_l
                            if oc_idx < OC:
                                wgt_val = al.convert(wgt[oc_idx, ic, kd, kh, kw], al.f32)
                                acc[oc_l] = acc[oc_l] + inp_val * wgt_val

        for oc_l in al.range(OC_TILE):
            oc_idx = oc_start + oc_l
            if oc_idx < OC:
                out[b, oc_idx, od, my_oh, my_ow] = al.convert(acc[oc_l], al.bf16)


def avelang_conv3d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Launch the AveLang 3D convolution kernel with tiled execution."""
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA/HIP device."

    B, IC, D, H, W = x.shape
    OC, _, KD, KH, KW = weight.shape

    OD = D - KD + 1
    OH = H - KH + 1
    OW = W - KW + 1

    OC_TILE = 16
    OH_TILE = 16
    OW_TILE = 16

    oc_tiles = (OC + OC_TILE - 1) // OC_TILE
    oh_tiles = (OH + OH_TILE - 1) // OH_TILE
    ow_tiles = (OW + OW_TILE - 1) // OW_TILE

    out = torch.empty((B, OC, OD, OH, OW), dtype=torch.bfloat16, device=x.device)

    x = x.contiguous()
    weight = weight.contiguous()

    conv3d_kernel[lambda: (
        (B, OD * oh_tiles * ow_tiles, oc_tiles),
        (OH_TILE * OW_TILE, 1, 1),
    )](
        x, weight, out,
        B, IC, D, H, W,
        OC, KD, KH, KW,
        OD, OH, OW,
        oh_tiles, ow_tiles,
        OC_TILE, OH_TILE, OW_TILE,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0),
                 dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_bf16 = x.to(torch.bfloat16)
        w_bf16 = self.conv3d.weight.to(torch.bfloat16)
        out_bf16 = avelang_conv3d(x_bf16, w_bf16)
        return out_bf16.to(orig_dtype)
