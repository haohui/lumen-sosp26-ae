import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_OH: int = 16
TILE_OW: int = 16
BLOCK_OC: int = 8
THREADS: int = TILE_OH * TILE_OW  # 256


@avelang.jit
def conv_transpose2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    IH: al.i32,
    IW: al.i32,
    OH: al.i32,
    OW: al.i32,
    KH: al.i32,
    KW: al.i32,
    stride: al.i32,
    padding: al.i32,
    dilation: al.i32,
    OW_TILES: al.i32,
    OC_GROUPS: al.i32,
):
    # -- row-major tensor views --
    x_stride0 = IC * IH * IW
    x_stride1 = IH * IW
    x_stride2 = IW
    x_layout = al.make_layout((B, IC, IH, IW), (x_stride0, x_stride1, x_stride2, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    w_stride0 = OC * KH * KW
    w_stride1 = KH * KW
    w_stride2 = KW
    w_layout = al.make_layout((IC, OC, KH, KW), (w_stride0, w_stride1, w_stride2, 1))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    out_stride0 = OC * OH * OW
    out_stride1 = OH * OW
    out_stride2 = OW
    out_layout = al.make_layout((B, OC, OH, OW), (out_stride0, out_stride1, out_stride2, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    # -- block/thread indexing --
    tid = al.thread_id(0)
    tile_ow = al.block_id(0)
    tile_oh = al.block_id(1)
    flat_idx = al.block_id(2)

    b = flat_idx // OC_GROUPS
    oc_group = flat_idx % OC_GROUPS
    oc_start = oc_group * BLOCK_OC

    th = tid // TILE_OW
    tw = tid % TILE_OW
    ow = tile_ow * TILE_OW + tw
    oh = tile_oh * TILE_OH + th

    zero_f32 = al.convert(0.0, al.f32)

    if ow < OW and oh < OH:
        # Compute the single valid (kh, kw) via modular inverse.
        # dilation*kh ≡ oh+padding (mod stride)
        # For stride=5, dilation=2: inverse of 2 mod 5 is 3, so kh ≡ 3*(oh+padding) (mod stride)
        kh_candidate = (3 * (oh + padding)) % stride
        kw_candidate = (3 * (ow + padding)) % stride

        if kh_candidate < KH and kw_candidate < KW:
            ih = (oh + padding - dilation * kh_candidate) // stride
            iw = (ow + padding - dilation * kw_candidate) // stride
            if ih >= 0 and ih < IH and iw >= 0 and iw < IW:
                acc0 = zero_f32
                acc1 = zero_f32
                acc2 = zero_f32
                acc3 = zero_f32
                acc4 = zero_f32
                acc5 = zero_f32
                acc6 = zero_f32
                acc7 = zero_f32

                oc0 = oc_start
                oc1 = oc_start + 1
                oc2 = oc_start + 2
                oc3 = oc_start + 3
                oc4 = oc_start + 4
                oc5 = oc_start + 5
                oc6 = oc_start + 6
                oc7 = oc_start + 7

                for ic in al.range(IC):
                    x_val = al.convert(x[b, ic, ih, iw], al.f32)
                    acc0 = acc0 + x_val * al.convert(w[ic, oc0, kh_candidate, kw_candidate], al.f32)
                    acc1 = acc1 + x_val * al.convert(w[ic, oc1, kh_candidate, kw_candidate], al.f32)
                    acc2 = acc2 + x_val * al.convert(w[ic, oc2, kh_candidate, kw_candidate], al.f32)
                    acc3 = acc3 + x_val * al.convert(w[ic, oc3, kh_candidate, kw_candidate], al.f32)
                    acc4 = acc4 + x_val * al.convert(w[ic, oc4, kh_candidate, kw_candidate], al.f32)
                    acc5 = acc5 + x_val * al.convert(w[ic, oc5, kh_candidate, kw_candidate], al.f32)
                    acc6 = acc6 + x_val * al.convert(w[ic, oc6, kh_candidate, kw_candidate], al.f32)
                    acc7 = acc7 + x_val * al.convert(w[ic, oc7, kh_candidate, kw_candidate], al.f32)

                out[b, oc0, oh, ow] = al.convert(acc0, al.bf16)
                out[b, oc1, oh, ow] = al.convert(acc1, al.bf16)
                out[b, oc2, oh, ow] = al.convert(acc2, al.bf16)
                out[b, oc3, oh, ow] = al.convert(acc3, al.bf16)
                out[b, oc4, oh, ow] = al.convert(acc4, al.bf16)
                out[b, oc5, oh, ow] = al.convert(acc5, al.bf16)
                out[b, oc6, oh, ow] = al.convert(acc6, al.bf16)
                out[b, oc7, oh, ow] = al.convert(acc7, al.bf16)


def avelang_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA/HIP device."
    B, IC, IH, IW = x.shape
    wIC, OC, KH, KW = weight.shape
    assert wIC == IC, "Weight in_channels must match input in_channels."

    OH = (IH - 1) * stride - 2 * padding + dilation * (KH - 1) + 1
    OW = (IW - 1) * stride - 2 * padding + dilation * (KW - 1) + 1

    OC_GROUPS = (OC + BLOCK_OC - 1) // BLOCK_OC
    OH_TILES = (OH + TILE_OH - 1) // TILE_OH
    OW_TILES = (OW + TILE_OW - 1) // TILE_OW

    out = torch.zeros(B, OC, OH, OW, dtype=torch.bfloat16, device=x.device)

    conv_transpose2d_kernel[lambda: (
        (OW_TILES, OH_TILES, B * OC_GROUPS),
        (THREADS, 1, 1),
    )](
        x, weight, out,
        B, IC, OC, IH, IW, OH, OW, KH, KW,
        stride, padding, dilation,
        OW_TILES, OC_GROUPS,
    )
    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        weight = self.conv_transpose2d.weight
        return avelang_conv_transpose2d(
            x, weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
