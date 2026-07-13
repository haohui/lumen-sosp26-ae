import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv3d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    IN_D: al.i32,
    IN_H: al.i32,
    IN_W: al.i32,
    KD: al.i32,
    KH: al.i32,
    KW: al.i32,
    OD: al.i32,
    OH: al.i32,
    OW: al.i32,
    total_in: al.i32,
    total_w: al.i32,
    total_out: al.i32,
    in_stride_b: al.i32,
    in_stride_ic: al.i32,
    in_stride_d: al.i32,
    in_stride_h: al.i32,
    w_stride_oc: al.i32,
    w_stride_ic: al.i32,
    w_stride_kd: al.i32,
    w_stride_kh: al.i32,
    out_stride_b: al.i32,
    out_stride_oc: al.i32,
    out_stride_od: al.i32,
    out_stride_oh: al.i32,
    BLOCK_W: al.constexpr,
    BLOCK_H: al.constexpr,
    BLOCK_D: al.constexpr,
    BLOCK_OC: al.constexpr,
):
    one = al.convert(1, al.i32)

    # Flat 1D global views
    x_flat = al.make_tensor(x_ptr, al.bf16,
        al.make_layout((total_in,), (one,)))
    w_flat = al.make_tensor(w_ptr, al.bf16,
        al.make_layout((total_w,), (one,)))
    out_flat = al.make_tensor(out_ptr, al.bf16,
        al.make_layout((total_out,), (one,)))

    # Thread and block indexing
    num_spatial = BLOCK_W * BLOCK_H * BLOCK_D
    tid = al.thread_id(0)
    oc_off = tid // num_spatial
    spat_off = tid % num_spatial
    w_off = spat_off % BLOCK_W
    h_off = (spat_off // BLOCK_W) % BLOCK_H
    d_off = spat_off // (BLOCK_W * BLOCK_H)

    tiles_per_w = (OW + BLOCK_W - 1) // BLOCK_W
    tiles_per_h = (OH + BLOCK_H - 1) // BLOCK_H
    bid_spatial = al.block_id(0)
    bid_oc_tile = al.block_id(1)
    bid_batch = al.block_id(2)

    tile_w = bid_spatial % tiles_per_w
    tile_h = (bid_spatial // tiles_per_w) % tiles_per_h
    tile_d = bid_spatial // (tiles_per_w * tiles_per_h)

    global_ow = tile_w * BLOCK_W + w_off
    global_oh = tile_h * BLOCK_H + h_off
    global_od = tile_d * BLOCK_D + d_off
    global_oc = bid_oc_tile * BLOCK_OC + oc_off

    if (global_ow < OW) and (global_oh < OH) and (global_od < OD) and (global_oc < OC):
        acc = al.convert(0, al.f32)

        in_b_base = bid_batch * in_stride_b
        w_oc_base = global_oc * w_stride_oc

        for ic in al.range(IC):
            in_ic_base = in_b_base + ic * in_stride_ic
            w_ic_base = w_oc_base + ic * w_stride_ic
            for kd in al.range(KD):
                in_d_base = in_ic_base + (global_od + kd) * in_stride_d
                w_kd_base = w_ic_base + kd * w_stride_kd
                for kh in al.range(KH):
                    in_h_base = in_d_base + (global_oh + kh) * in_stride_h
                    w_kh_base = w_kd_base + kh * w_stride_kh
                    for kw in al.range(KW):
                        in_idx = in_h_base + (global_ow + kw)
                        w_idx = w_kh_base + kw
                        x_val = al.convert(x_flat[in_idx], al.f32)
                        w_val = al.convert(w_flat[w_idx], al.f32)
                        acc = acc + x_val * w_val

        out_b_base = bid_batch * out_stride_b
        out_idx = (out_b_base + global_oc * out_stride_oc +
                   global_od * out_stride_od + global_oh * out_stride_oh + global_ow)
        out_flat[out_idx] = al.convert(acc, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )

    def forward(self, x):
        w = self.conv.weight.data
        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = w.to(torch.bfloat16).contiguous()

        B = x_bf16.shape[0]
        IC = x_bf16.shape[1]
        IN_D = x_bf16.shape[2]
        IN_H = x_bf16.shape[3]
        IN_W = x_bf16.shape[4]

        OC = w_bf16.shape[0]
        KD = w_bf16.shape[2]
        KH = w_bf16.shape[3]
        KW = w_bf16.shape[4]

        OD = IN_D - KD + 1
        OH = IN_H - KH + 1
        OW = IN_W - KW + 1

        out = torch.empty(B, OC, OD, OH, OW, device=x.device, dtype=torch.bfloat16)

        in_stride_h = IN_W
        in_stride_d = IN_H * IN_W
        in_stride_ic = IN_D * IN_H * IN_W
        in_stride_b = IC * IN_D * IN_H * IN_W
        total_in = B * in_stride_b

        w_stride_kh = KW
        w_stride_kd = KH * KW
        w_stride_ic = KD * KH * KW
        w_stride_oc = IC * KD * KH * KW
        total_w = OC * w_stride_oc

        out_stride_oh = OW
        out_stride_od = OH * OW
        out_stride_oc = OD * OH * OW
        out_stride_b = OC * OD * OH * OW
        total_out = B * out_stride_b

        BLOCK_W = 8
        BLOCK_H = 4
        BLOCK_D = 4
        BLOCK_OC = 2

        tiles_w = (OW + BLOCK_W - 1) // BLOCK_W
        tiles_h = (OH + BLOCK_H - 1) // BLOCK_H
        tiles_d = (OD + BLOCK_D - 1) // BLOCK_D
        tiles_oc = (OC + BLOCK_OC - 1) // BLOCK_OC

        grid = (tiles_w * tiles_h * tiles_d, tiles_oc, B)
        block = (BLOCK_W * BLOCK_H * BLOCK_D * BLOCK_OC, 1, 1)

        conv3d_kernel[lambda: (grid, block)](
            x_bf16, w_bf16, out,
            B, IC, OC, IN_D, IN_H, IN_W,
            KD, KH, KW, OD, OH, OW,
            total_in, total_w, total_out,
            in_stride_b, in_stride_ic, in_stride_d, in_stride_h,
            w_stride_oc, w_stride_ic, w_stride_kd, w_stride_kh,
            out_stride_b, out_stride_oc, out_stride_od, out_stride_oh,
            BLOCK_W, BLOCK_H, BLOCK_D, BLOCK_OC,
        )
        return out.to(x.dtype)
