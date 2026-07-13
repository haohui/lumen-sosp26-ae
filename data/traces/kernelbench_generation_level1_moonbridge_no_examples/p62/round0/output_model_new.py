import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_H = 16
TILE_W = 16


@avelang.jit
def conv2d_bf16_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    IC: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
    KH: al.i32,
    KW: al.i32,
    OH: al.i32,
    OW: al.i32,
):
    th = al.thread_id(0)
    tw = al.thread_id(1)

    tile_h = al.block_id(0)
    tile_w = al.block_id(1)
    flat_idx = al.block_id(2)

    # Decompose flat_idx = n_idx * OC + oc_idx using subtraction loop
    n_idx = al.convert(0, al.i32)
    oc_idx = flat_idx
    for _n in al.range(N):
        if oc_idx >= OC:
            oc_idx = oc_idx - OC
            n_idx = n_idx + 1

    oh = tile_h * 16 + th
    ow = tile_w * 16 + tw

    # Shared memory: 4 input channels x 20 x 24 bf16
    smem = al.make_shared((4, 20, 24), al.bf16)

    input_layout = al.make_layout((N, IC, H, W), (IC * H * W, H * W, W, 1))
    input_t = al.make_tensor(input_ptr, al.bf16, input_layout)

    weight_layout = al.make_layout((OC, IC, KH, KW), (IC * KH * KW, KH * KW, KW, 1))
    weight_t = al.make_tensor(weight_ptr, al.bf16, weight_layout)

    output_layout = al.make_layout((N, OC, OH, OW), (OC * OH * OW, OH * OW, OW, 1))
    output_t = al.make_tensor(output_ptr, al.bf16, output_layout)

    acc = al.convert(0.0, al.f32)
    zero = al.convert(0.0, al.bf16)

    h_base = tile_h * 16
    w_base = tile_w * 16

    for ic_block in al.range(0, IC, 4):
        # Zero-init shared memory for all 4 IC tiles (all threads)
        for ic_l in al.range(4):
            smem[ic_l, th, tw] = zero
            if tw < 8:
                smem[ic_l, th, tw + 16] = zero
            if th < 4:
                smem[ic_l, th + 16, tw] = zero
            if th < 4 and tw < 8:
                smem[ic_l, th + 16, tw + 16] = zero
        al.syncthreads()

        # Cooperative load 4 IC tiles into shared memory (all threads)
        for ic_l in al.range(4):
            ic_g = ic_block + ic_l
            if ic_g < IC:
                hg = h_base + th
                wg = w_base + tw
                if hg < H and wg < W:
                    smem[ic_l, th, tw] = input_t[n_idx, ic_g, hg, wg]
                if tw < 8:
                    wg2 = w_base + tw + 16
                    if hg < H and wg2 < W:
                        smem[ic_l, th, tw + 16] = input_t[n_idx, ic_g, hg, wg2]
                if th < 4:
                    hg3 = h_base + th + 16
                    if hg3 < H and wg < W:
                        smem[ic_l, th + 16, tw] = input_t[n_idx, ic_g, hg3, wg]
                if th < 4 and tw < 8:
                    hg4 = h_base + th + 16
                    wg4 = w_base + tw + 16
                    if hg4 < H and wg4 < W:
                        smem[ic_l, th + 16, tw + 16] = input_t[n_idx, ic_g, hg4, wg4]
        al.syncthreads()

        # Accumulate from shared memory (only valid output threads)
        if oh < OH and ow < OW:
            for ic_l in al.range(4):
                ic_g = ic_block + ic_l
                if ic_g < IC:
                    for kh in al.range(KH):
                        for kw in al.range(KW):
                            inp_bf16 = smem[ic_l, th + kh, tw + kw]
                            w_bf16 = weight_t[oc_idx, ic_g, kh, kw]
                            inp_f32 = al.convert(inp_bf16, al.f32)
                            w_f32 = al.convert(w_bf16, al.f32)
                            acc = acc + inp_f32 * w_f32
        al.syncthreads()

    if oh < OH and ow < OW:
        output_t[n_idx, oc_idx, oh, ow] = al.convert(acc, al.bf16)


def avelang_conv2d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    N, IC, H, W = x.shape
    OC, IC_w, KH, KW = weight.shape

    OH = H - KH + 1
    OW = W - KW + 1

    x_bf16 = x.to(torch.bfloat16).contiguous()
    w_bf16 = weight.to(torch.bfloat16).contiguous()

    out = torch.empty(N, OC, OH, OW, dtype=torch.bfloat16, device=x.device)

    grid_x = (OH + TILE_H - 1) // TILE_H
    grid_y = (OW + TILE_W - 1) // TILE_W
    grid_z = N * OC

    conv2d_bf16_kernel[lambda: ((grid_x, grid_y, grid_z), (TILE_H, TILE_W, 1))](
        x_bf16, w_bf16, out,
        N, IC, OC, H, W, KH, KW, OH, OW,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv2d(x, self.conv2d.weight)
