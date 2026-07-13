import torch
import torch.nn as nn
import avelang
import avelang.language as al


BLOCK_SIZE: al.constexpr = 256
OC_C: al.constexpr = 16
IC_C: al.constexpr = 3
KD_C: al.constexpr = 3
KH_C: al.constexpr = 3
KW_C: al.constexpr = 3
KERNEL_VOL: al.constexpr = IC_C * KD_C * KH_C * KW_C  # 81
WEIGHT_SIZE: al.constexpr = OC_C * KERNEL_VOL  # 1296


@avelang.jit
def _conv3d_fused_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    conv_bias_ptr: al.Pointer(al.bf16),
    scaling_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    OC: al.i32,
    ID: al.i32,
    IH: al.i32,
    IW: al.i32,
    OD: al.i32,
    OH: al.i32,
    OW: al.i32,
    total_spatial: al.i32,
):
    tid = al.thread_id(0)
    gid = al.block_id(0) * BLOCK_SIZE + tid

    if gid < total_spatial:
        # Decode flat spatial index into (b, od, oh, ow)
        ow = gid % OW
        tmp1 = gid // OW
        oh = tmp1 % OH
        tmp2 = tmp1 // OH
        od = tmp2 % OD
        b = tmp2 // OD

        # Input tensor view: (B, IC, ID, IH, IW)
        x_layout = al.make_layout(
            (B, al.convert(IC_C, al.i32), ID, IH, IW),
            (IC_C * ID * IH * IW, ID * IH * IW, IH * IW, IW, 1),
        )
        x = al.make_tensor(x_ptr, al.bf16, x_layout)

        # Weight tensor view: (OC, IC, KD, KH, KW) — 5D
        w_layout = al.make_layout(
            (OC, al.convert(IC_C, al.i32), al.convert(KD_C, al.i32),
             al.convert(KH_C, al.i32), al.convert(KW_C, al.i32)),
            (KERNEL_VOL, KD_C * KH_C * KW_C, KH_C * KW_C, KW_C, 1),
        )
        w = al.make_tensor(w_ptr, al.bf16, w_layout)

        # Bias/scaling tensors: (OC,) 1D
        bias1d_layout = al.make_layout((OC,), (1,))
        cb = al.make_tensor(conv_bias_ptr, al.bf16, bias1d_layout)
        sf = al.make_tensor(scaling_ptr, al.bf16, bias1d_layout)
        bt = al.make_tensor(bias_ptr, al.bf16, bias1d_layout)

        # Accumulators for all 16 output channels (f32 registers)
        acc0 = al.convert(0.0, al.f32)
        acc1 = al.convert(0.0, al.f32)
        acc2 = al.convert(0.0, al.f32)
        acc3 = al.convert(0.0, al.f32)
        acc4 = al.convert(0.0, al.f32)
        acc5 = al.convert(0.0, al.f32)
        acc6 = al.convert(0.0, al.f32)
        acc7 = al.convert(0.0, al.f32)
        acc8 = al.convert(0.0, al.f32)
        acc9 = al.convert(0.0, al.f32)
        acc10 = al.convert(0.0, al.f32)
        acc11 = al.convert(0.0, al.f32)
        acc12 = al.convert(0.0, al.f32)
        acc13 = al.convert(0.0, al.f32)
        acc14 = al.convert(0.0, al.f32)
        acc15 = al.convert(0.0, al.f32)

        # Load input window into registers: 3 * 3 * 3 * 3 = 81 values
        # x_in[ic][kd][kh][kw] for ic=0..2, kd=0..2, kh=0..2, kw=0..2
        # We'll compute inline

        for ic in al.range(IC_C):
            id0 = od
            id1 = od + 1
            id2 = od + 2
            ih0 = oh
            ih1 = oh + 1
            ih2 = oh + 2
            iw0 = ow
            iw1 = ow + 1
            iw2 = ow + 2

            # Load 27 input values for this input channel
            x000 = al.convert(x[b, ic, id0, ih0, iw0], al.f32)
            x001 = al.convert(x[b, ic, id0, ih0, iw1], al.f32)
            x002 = al.convert(x[b, ic, id0, ih0, iw2], al.f32)
            x010 = al.convert(x[b, ic, id0, ih1, iw0], al.f32)
            x011 = al.convert(x[b, ic, id0, ih1, iw1], al.f32)
            x012 = al.convert(x[b, ic, id0, ih1, iw2], al.f32)
            x020 = al.convert(x[b, ic, id0, ih2, iw0], al.f32)
            x021 = al.convert(x[b, ic, id0, ih2, iw1], al.f32)
            x022 = al.convert(x[b, ic, id0, ih2, iw2], al.f32)

            x100 = al.convert(x[b, ic, id1, ih0, iw0], al.f32)
            x101 = al.convert(x[b, ic, id1, ih0, iw1], al.f32)
            x102 = al.convert(x[b, ic, id1, ih0, iw2], al.f32)
            x110 = al.convert(x[b, ic, id1, ih1, iw0], al.f32)
            x111 = al.convert(x[b, ic, id1, ih1, iw1], al.f32)
            x112 = al.convert(x[b, ic, id1, ih1, iw2], al.f32)
            x120 = al.convert(x[b, ic, id1, ih2, iw0], al.f32)
            x121 = al.convert(x[b, ic, id1, ih2, iw1], al.f32)
            x122 = al.convert(x[b, ic, id1, ih2, iw2], al.f32)

            x200 = al.convert(x[b, ic, id2, ih0, iw0], al.f32)
            x201 = al.convert(x[b, ic, id2, ih0, iw1], al.f32)
            x202 = al.convert(x[b, ic, id2, ih0, iw2], al.f32)
            x210 = al.convert(x[b, ic, id2, ih1, iw0], al.f32)
            x211 = al.convert(x[b, ic, id2, ih1, iw1], al.f32)
            x212 = al.convert(x[b, ic, id2, ih1, iw2], al.f32)
            x220 = al.convert(x[b, ic, id2, ih2, iw0], al.f32)
            x221 = al.convert(x[b, ic, id2, ih2, iw1], al.f32)
            x222 = al.convert(x[b, ic, id2, ih2, iw2], al.f32)

            # Accumulate over all 16 output channels
            # Channel 0
            w000 = al.convert(w[0, ic, 0, 0, 0], al.f32)
            w001 = al.convert(w[0, ic, 0, 0, 1], al.f32)
            w002 = al.convert(w[0, ic, 0, 0, 2], al.f32)
            w010 = al.convert(w[0, ic, 0, 1, 0], al.f32)
            w011 = al.convert(w[0, ic, 0, 1, 1], al.f32)
            w012 = al.convert(w[0, ic, 0, 1, 2], al.f32)
            w020 = al.convert(w[0, ic, 0, 2, 0], al.f32)
            w021 = al.convert(w[0, ic, 0, 2, 1], al.f32)
            w022 = al.convert(w[0, ic, 0, 2, 2], al.f32)
            w100 = al.convert(w[0, ic, 1, 0, 0], al.f32)
            w101 = al.convert(w[0, ic, 1, 0, 1], al.f32)
            w102 = al.convert(w[0, ic, 1, 0, 2], al.f32)
            w110 = al.convert(w[0, ic, 1, 1, 0], al.f32)
            w111 = al.convert(w[0, ic, 1, 1, 1], al.f32)
            w112 = al.convert(w[0, ic, 1, 1, 2], al.f32)
            w120 = al.convert(w[0, ic, 1, 2, 0], al.f32)
            w121 = al.convert(w[0, ic, 1, 2, 1], al.f32)
            w122 = al.convert(w[0, ic, 1, 2, 2], al.f32)
            w200 = al.convert(w[0, ic, 2, 0, 0], al.f32)
            w201 = al.convert(w[0, ic, 2, 0, 1], al.f32)
            w202 = al.convert(w[0, ic, 2, 0, 2], al.f32)
            w210 = al.convert(w[0, ic, 2, 1, 0], al.f32)
            w211 = al.convert(w[0, ic, 2, 1, 1], al.f32)
            w212 = al.convert(w[0, ic, 2, 1, 2], al.f32)
            w220 = al.convert(w[0, ic, 2, 2, 0], al.f32)
            w221 = al.convert(w[0, ic, 2, 2, 1], al.f32)
            w222 = al.convert(w[0, ic, 2, 2, 2], al.f32)

            acc0 = acc0 + x000 * w000 + x001 * w001 + x002 * w002
            acc0 = acc0 + x010 * w010 + x011 * w011 + x012 * w012
            acc0 = acc0 + x020 * w020 + x021 * w021 + x022 * w022
            acc0 = acc0 + x100 * w100 + x101 * w101 + x102 * w102
            acc0 = acc0 + x110 * w110 + x111 * w111 + x112 * w112
            acc0 = acc0 + x120 * w120 + x121 * w121 + x122 * w122
            acc0 = acc0 + x200 * w200 + x201 * w201 + x202 * w202
            acc0 = acc0 + x210 * w210 + x211 * w211 + x212 * w212
            acc0 = acc0 + x220 * w220 + x221 * w221 + x222 * w222

            # Channel 1
            w000 = al.convert(w[1, ic, 0, 0, 0], al.f32)
            w001 = al.convert(w[1, ic, 0, 0, 1], al.f32)
            w002 = al.convert(w[1, ic, 0, 0, 2], al.f32)
            w010 = al.convert(w[1, ic, 0, 1, 0], al.f32)
            w011 = al.convert(w[1, ic, 0, 1, 1], al.f32)
            w012 = al.convert(w[1, ic, 0, 1, 2], al.f32)
            w020 = al.convert(w[1, ic, 0, 2, 0], al.f32)
            w021 = al.convert(w[1, ic, 0, 2, 1], al.f32)
            w022 = al.convert(w[1, ic, 0, 2, 2], al.f32)
            w100 = al.convert(w[1, ic, 1, 0, 0], al.f32)
            w101 = al.convert(w[1, ic, 1, 0, 1], al.f32)
            w102 = al.convert(w[1, ic, 1, 0, 2], al.f32)
            w110 = al.convert(w[1, ic, 1, 1, 0], al.f32)
            w111 = al.convert(w[1, ic, 1, 1, 1], al.f32)
            w112 = al.convert(w[1, ic, 1, 1, 2], al.f32)
            w120 = al.convert(w[1, ic, 1, 2, 0], al.f32)
            w121 = al.convert(w[1, ic, 1, 2, 1], al.f32)
            w122 = al.convert(w[1, ic, 1, 2, 2], al.f32)
            w200 = al.convert(w[1, ic, 2, 0, 0], al.f32)
            w201 = al.convert(w[1, ic, 2, 0, 1], al.f32)
            w202 = al.convert(w[1, ic, 2, 0, 2], al.f32)
            w210 = al.convert(w[1, ic, 2, 1, 0], al.f32)
            w211 = al.convert(w[1, ic, 2, 1, 1], al.f32)
            w212 = al.convert(w[1, ic, 2, 1, 2], al.f32)
            w220 = al.convert(w[1, ic, 2, 2, 0], al.f32)
            w221 = al.convert(w[1, ic, 2, 2, 1], al.f32)
            w222 = al.convert(w[1, ic, 2, 2, 2], al.f32)

            acc1 = acc1 + x000 * w000 + x001 * w001 + x002 * w002
            acc1 = acc1 + x010 * w010 + x011 * w011 + x012 * w012
            acc1 = acc1 + x020 * w020 + x021 * w021 + x022 * w022
            acc1 = acc1 + x100 * w100 + x101 * w101 + x102 * w102
            acc1 = acc1 + x110 * w110 + x111 * w111 + x112 * w112
            acc1 = acc1 + x120 * w120 + x121 * w121 + x122 * w122
            acc1 = acc1 + x200 * w200 + x201 * w201 + x202 * w202
            acc1 = acc1 + x210 * w210 + x211 * w211 + x212 * w212
            acc1 = acc1 + x220 * w220 + x221 * w221 + x222 * w222

            # Channel 2
            w000 = al.convert(w[2, ic, 0, 0, 0], al.f32)
            w001 = al.convert(w[2, ic, 0, 0, 1], al.f32)
            w002 = al.convert(w[2, ic, 0, 0, 2], al.f32)
            w010 = al.convert(w[2, ic, 0, 1, 0], al.f32)
            w011 = al.convert(w[2, ic, 0, 1, 1], al.f32)
            w012 = al.convert(w[2, ic, 0, 1, 2], al.f32)
            w020 = al.convert(w[2, ic, 0, 2, 0], al.f32)
            w021 = al.convert(w[2, ic, 0, 2, 1], al.f32)
            w022 = al.convert(w[2, ic, 0, 2, 2], al.f32)
            w100 = al.convert(w[2, ic, 1, 0, 0], al.f32)
            w101 = al.convert(w[2, ic, 1, 0, 1], al.f32)
            w102 = al.convert(w[2, ic, 1, 0, 2], al.f32)
            w110 = al.convert(w[2, ic, 1, 1, 0], al.f32)
            w111 = al.convert(w[2, ic, 1, 1, 1], al.f32)
            w112 = al.convert(w[2, ic, 1, 1, 2], al.f32)
            w120 = al.convert(w[2, ic, 1, 2, 0], al.f32)
            w121 = al.convert(w[2, ic, 1, 2, 1], al.f32)
            w122 = al.convert(w[2, ic, 1, 2, 2], al.f32)
            w200 = al.convert(w[2, ic, 2, 0, 0], al.f32)
            w201 = al.convert(w[2, ic, 2, 0, 1], al.f32)
            w202 = al.convert(w[2, ic, 2, 0, 2], al.f32)
            w210 = al.convert(w[2, ic, 2, 1, 0], al.f32)
            w211 = al.convert(w[2, ic, 2, 1, 1], al.f32)
            w212 = al.convert(w[2, ic, 2, 1, 2], al.f32)
            w220 = al.convert(w[2, ic, 2, 2, 0], al.f32)
            w221 = al.convert(w[2, ic, 2, 2, 1], al.f32)
            w222 = al.convert(w[2, ic, 2, 2, 2], al.f32)

            acc2 = acc2 + x000 * w000 + x001 * w001 + x002 * w002
            acc2 = acc2 + x010 * w010 + x011 * w011 + x012 * w012
            acc2 = acc2 + x020 * w020 + x021 * w021 + x022 * w022
            acc2 = acc2 + x100 * w100 + x101 * w101 + x102 * w102
            acc2 = acc2 + x110 * w110 + x111 * w111 + x112 * w112
            acc2 = acc2 + x120 * w120 + x121 * w121 + x122 * w122
            acc2 = acc2 + x200 * w200 + x201 * w201 + x202 * w202
            acc2 = acc2 + x210 * w210 + x211 * w211 + x212 * w212
            acc2 = acc2 + x220 * w220 + x221 * w221 + x222 * w222

            # Channel 3
            w000 = al.convert(w[3, ic, 0, 0, 0], al.f32)
            w001 = al.convert(w[3, ic, 0, 0, 1], al.f32)
            w002 = al.convert(w[3, ic, 0, 0, 2], al.f32)
            w010 = al.convert(w[3, ic, 0, 1, 0], al.f32)
            w011 = al.convert(w[3, ic, 0, 1, 1], al.f32)
            w012 = al.convert(w[3, ic, 0, 1, 2], al.f32)
            w020 = al.convert(w[3, ic, 0, 2, 0], al.f32)
            w021 = al.convert(w[3, ic, 0, 2, 1], al.f32)
            w022 = al.convert(w[3, ic, 0, 2, 2], al.f32)
            w100 = al.convert(w[3, ic, 1, 0, 0], al.f32)
            w101 = al.convert(w[3, ic, 1, 0, 1], al.f32)
            w102 = al.convert(w[3, ic, 1, 0, 2], al.f32)
            w110 = al.convert(w[3, ic, 1, 1, 0], al.f32)
            w111 = al.convert(w[3, ic, 1, 1, 1], al.f32)
            w112 = al.convert(w[3, ic, 1, 1, 2], al.f32)
            w120 = al.convert(w[3, ic, 1, 2, 0], al.f32)
            w121 = al.convert(w[3, ic, 1, 2, 1], al.f32)
            w122 = al.convert(w[3, ic, 1, 2, 2], al.f32)
            w200 = al.convert(w[3, ic, 2, 0, 0], al.f32)
            w201 = al.convert(w[3, ic, 2, 0, 1], al.f32)
            w202 = al.convert(w[3, ic, 2, 0, 2], al.f32)
            w210 = al.convert(w[3, ic, 2, 1, 0], al.f32)
            w211 = al.convert(w[3, ic, 2, 1, 1], al.f32)
            w212 = al.convert(w[3, ic, 2, 1, 2], al.f32)
            w220 = al.convert(w[3, ic, 2, 2, 0], al.f32)
            w221 = al.convert(w[3, ic, 2, 2, 1], al.f32)
            w222 = al.convert(w[3, ic, 2, 2, 2], al.f32)

            acc3 = acc3 + x000 * w000 + x001 * w001 + x002 * w002
            acc3 = acc3 + x010 * w010 + x011 * w011 + x012 * w012
            acc3 = acc3 + x020 * w020 + x021 * w021 + x022 * w022
            acc3 = acc3 + x100 * w100 + x101 * w101 + x102 * w102
            acc3 = acc3 + x110 * w110 + x111 * w111 + x112 * w112
            acc3 = acc3 + x120 * w120 + x121 * w121 + x122 * w122
            acc3 = acc3 + x200 * w200 + x201 * w201 + x202 * w202
            acc3 = acc3 + x210 * w210 + x211 * w211 + x212 * w212
            acc3 = acc3 + x220 * w220 + x221 * w221 + x222 * w222

            # Channel 4
            w000 = al.convert(w[4, ic, 0, 0, 0], al.f32)
            w001 = al.convert(w[4, ic, 0, 0, 1], al.f32)
            w002 = al.convert(w[4, ic, 0, 0, 2], al.f32)
            w010 = al.convert(w[4, ic, 0, 1, 0], al.f32)
            w011 = al.convert(w[4, ic, 0, 1, 1], al.f32)
            w012 = al.convert(w[4, ic, 0, 1, 2], al.f32)
            w020 = al.convert(w[4, ic, 0, 2, 0], al.f32)
            w021 = al.convert(w[4, ic, 0, 2, 1], al.f32)
            w022 = al.convert(w[4, ic, 0, 2, 2], al.f32)
            w100 = al.convert(w[4, ic, 1, 0, 0], al.f32)
            w101 = al.convert(w[4, ic, 1, 0, 1], al.f32)
            w102 = al.convert(w[4, ic, 1, 0, 2], al.f32)
            w110 = al.convert(w[4, ic, 1, 1, 0], al.f32)
            w111 = al.convert(w[4, ic, 1, 1, 1], al.f32)
            w112 = al.convert(w[4, ic, 1, 1, 2], al.f32)
            w120 = al.convert(w[4, ic, 1, 2, 0], al.f32)
            w121 = al.convert(w[4, ic, 1, 2, 1], al.f32)
            w122 = al.convert(w[4, ic, 1, 2, 2], al.f32)
            w200 = al.convert(w[4, ic, 2, 0, 0], al.f32)
            w201 = al.convert(w[4, ic, 2, 0, 1], al.f32)
            w202 = al.convert(w[4, ic, 2, 0, 2], al.f32)
            w210 = al.convert(w[4, ic, 2, 1, 0], al.f32)
            w211 = al.convert(w[4, ic, 2, 1, 1], al.f32)
            w212 = al.convert(w[4, ic, 2, 1, 2], al.f32)
            w220 = al.convert(w[4, ic, 2, 2, 0], al.f32)
            w221 = al.convert(w[4, ic, 2, 2, 1], al.f32)
            w222 = al.convert(w[4, ic, 2, 2, 2], al.f32)

            acc4 = acc4 + x000 * w000 + x001 * w001 + x002 * w002
            acc4 = acc4 + x010 * w010 + x011 * w011 + x012 * w012
            acc4 = acc4 + x020 * w020 + x021 * w021 + x022 * w022
            acc4 = acc4 + x100 * w100 + x101 * w101 + x102 * w102
            acc4 = acc4 + x110 * w110 + x111 * w111 + x112 * w112
            acc4 = acc4 + x120 * w120 + x121 * w121 + x122 * w122
            acc4 = acc4 + x200 * w200 + x201 * w201 + x202 * w202
            acc4 = acc4 + x210 * w210 + x211 * w211 + x212 * w212
            acc4 = acc4 + x220 * w220 + x221 * w221 + x222 * w222

            # Channel 5
            w000 = al.convert(w[5, ic, 0, 0, 0], al.f32)
            w001 = al.convert(w[5, ic, 0, 0, 1], al.f32)
            w002 = al.convert(w[5, ic, 0, 0, 2], al.f32)
            w010 = al.convert(w[5, ic, 0, 1, 0], al.f32)
            w011 = al.convert(w[5, ic, 0, 1, 1], al.f32)
            w012 = al.convert(w[5, ic, 0, 1, 2], al.f32)
            w020 = al.convert(w[5, ic, 0, 2, 0], al.f32)
            w021 = al.convert(w[5, ic, 0, 2, 1], al.f32)
            w022 = al.convert(w[5, ic, 0, 2, 2], al.f32)
            w100 = al.convert(w[5, ic, 1, 0, 0], al.f32)
            w101 = al.convert(w[5, ic, 1, 0, 1], al.f32)
            w102 = al.convert(w[5, ic, 1, 0, 2], al.f32)
            w110 = al.convert(w[5, ic, 1, 1, 0], al.f32)
            w111 = al.convert(w[5, ic, 1, 1, 1], al.f32)
            w112 = al.convert(w[5, ic, 1, 1, 2], al.f32)
            w120 = al.convert(w[5, ic, 1, 2, 0], al.f32)
            w121 = al.convert(w[5, ic, 1, 2, 1], al.f32)
            w122 = al.convert(w[5, ic, 1, 2, 2], al.f32)
            w200 = al.convert(w[5, ic, 2, 0, 0], al.f32)
            w201 = al.convert(w[5, ic, 2, 0, 1], al.f32)
            w202 = al.convert(w[5, ic, 2, 0, 2], al.f32)
            w210 = al.convert(w[5, ic, 2, 1, 0], al.f32)
            w211 = al.convert(w[5, ic, 2, 1, 1], al.f32)
            w212 = al.convert(w[5, ic, 2, 1, 2], al.f32)
            w220 = al.convert(w[5, ic, 2, 2, 0], al.f32)
            w221 = al.convert(w[5, ic, 2, 2, 1], al.f32)
            w222 = al.convert(w[5, ic, 2, 2, 2], al.f32)

            acc5 = acc5 + x000 * w000 + x001 * w001 + x002 * w002
            acc5 = acc5 + x010 * w010 + x011 * w011 + x012 * w012
            acc5 = acc5 + x020 * w020 + x021 * w021 + x022 * w022
            acc5 = acc5 + x100 * w100 + x101 * w101 + x102 * w102
            acc5 = acc5 + x110 * w110 + x111 * w111 + x112 * w112
            acc5 = acc5 + x120 * w120 + x121 * w121 + x122 * w122
            acc5 = acc5 + x200 * w200 + x201 * w201 + x202 * w202
            acc5 = acc5 + x210 * w210 + x211 * w211 + x212 * w212
            acc5 = acc5 + x220 * w220 + x221 * w221 + x222 * w222

            # Channel 6
            w000 = al.convert(w[6, ic, 0, 0, 0], al.f32)
            w001 = al.convert(w[6, ic, 0, 0, 1], al.f32)
            w002 = al.convert(w[6, ic, 0, 0, 2], al.f32)
            w010 = al.convert(w[6, ic, 0, 1, 0], al.f32)
            w011 = al.convert(w[6, ic, 0, 1, 1], al.f32)
            w012 = al.convert(w[6, ic, 0, 1, 2], al.f32)
            w020 = al.convert(w[6, ic, 0, 2, 0], al.f32)
            w021 = al.convert(w[6, ic, 0, 2, 1], al.f32)
            w022 = al.convert(w[6, ic, 0, 2, 2], al.f32)
            w100 = al.convert(w[6, ic, 1, 0, 0], al.f32)
            w101 = al.convert(w[6, ic, 1, 0, 1], al.f32)
            w102 = al.convert(w[6, ic, 1, 0, 2], al.f32)
            w110 = al.convert(w[6, ic, 1, 1, 0], al.f32)
            w111 = al.convert(w[6, ic, 1, 1, 1], al.f32)
            w112 = al.convert(w[6, ic, 1, 1, 2], al.f32)
            w120 = al.convert(w[6, ic, 1, 2, 0], al.f32)
            w121 = al.convert(w[6, ic, 1, 2, 1], al.f32)
            w122 = al.convert(w[6, ic, 1, 2, 2], al.f32)
            w200 = al.convert(w[6, ic, 2, 0, 0], al.f32)
            w201 = al.convert(w[6, ic, 2, 0, 1], al.f32)
            w202 = al.convert(w[6, ic, 2, 0, 2], al.f32)
            w210 = al.convert(w[6, ic, 2, 1, 0], al.f32)
            w211 = al.convert(w[6, ic, 2, 1, 1], al.f32)
            w212 = al.convert(w[6, ic, 2, 1, 2], al.f32)
            w220 = al.convert(w[6, ic, 2, 2, 0], al.f32)
            w221 = al.convert(w[6, ic, 2, 2, 1], al.f32)
            w222 = al.convert(w[6, ic, 2, 2, 2], al.f32)

            acc6 = acc6 + x000 * w000 + x001 * w001 + x002 * w002
            acc6 = acc6 + x010 * w010 + x011 * w011 + x012 * w012
            acc6 = acc6 + x020 * w020 + x021 * w021 + x022 * w022
            acc6 = acc6 + x100 * w100 + x101 * w101 + x102 * w102
            acc6 = acc6 + x110 * w110 + x111 * w111 + x112 * w112
            acc6 = acc6 + x120 * w120 + x121 * w121 + x122 * w122
            acc6 = acc6 + x200 * w200 + x201 * w201 + x202 * w202
            acc6 = acc6 + x210 * w210 + x211 * w211 + x212 * w212
            acc6 = acc6 + x220 * w220 + x221 * w221 + x222 * w222

            # Channel 7
            w000 = al.convert(w[7, ic, 0, 0, 0], al.f32)
            w001 = al.convert(w[7, ic, 0, 0, 1], al.f32)
            w002 = al.convert(w[7, ic, 0, 0, 2], al.f32)
            w010 = al.convert(w[7, ic, 0, 1, 0], al.f32)
            w011 = al.convert(w[7, ic, 0, 1, 1], al.f32)
            w012 = al.convert(w[7, ic, 0, 1, 2], al.f32)
            w020 = al.convert(w[7, ic, 0, 2, 0], al.f32)
            w021 = al.convert(w[7, ic, 0, 2, 1], al.f32)
            w022 = al.convert(w[7, ic, 0, 2, 2], al.f32)
            w100 = al.convert(w[7, ic, 1, 0, 0], al.f32)
            w101 = al.convert(w[7, ic, 1, 0, 1], al.f32)
            w102 = al.convert(w[7, ic, 1, 0, 2], al.f32)
            w110 = al.convert(w[7, ic, 1, 1, 0], al.f32)
            w111 = al.convert(w[7, ic, 1, 1, 1], al.f32)
            w112 = al.convert(w[7, ic, 1, 1, 2], al.f32)
            w120 = al.convert(w[7, ic, 1, 2, 0], al.f32)
            w121 = al.convert(w[7, ic, 1, 2, 1], al.f32)
            w122 = al.convert(w[7, ic, 1, 2, 2], al.f32)
            w200 = al.convert(w[7, ic, 2, 0, 0], al.f32)
            w201 = al.convert(w[7, ic, 2, 0, 1], al.f32)
            w202 = al.convert(w[7, ic, 2, 0, 2], al.f32)
            w210 = al.convert(w[7, ic, 2, 1, 0], al.f32)
            w211 = al.convert(w[7, ic, 2, 1, 1], al.f32)
            w212 = al.convert(w[7, ic, 2, 1, 2], al.f32)
            w220 = al.convert(w[7, ic, 2, 2, 0], al.f32)
            w221 = al.convert(w[7, ic, 2, 2, 1], al.f32)
            w222 = al.convert(w[7, ic, 2, 2, 2], al.f32)

            acc7 = acc7 + x000 * w000 + x001 * w001 + x002 * w002
            acc7 = acc7 + x010 * w010 + x011 * w011 + x012 * w012
            acc7 = acc7 + x020 * w020 + x021 * w021 + x022 * w022
            acc7 = acc7 + x100 * w100 + x101 * w101 + x102 * w102
            acc7 = acc7 + x110 * w110 + x111 * w111 + x112 * w112
            acc7 = acc7 + x120 * w120 + x121 * w121 + x122 * w122
            acc7 = acc7 + x200 * w200 + x201 * w201 + x202 * w202
            acc7 = acc7 + x210 * w210 + x211 * w211 + x212 * w212
            acc7 = acc7 + x220 * w220 + x221 * w221 + x222 * w222

            # Channel 8
            w000 = al.convert(w[8, ic, 0, 0, 0], al.f32)
            w001 = al.convert(w[8, ic, 0, 0, 1], al.f32)
            w002 = al.convert(w[8, ic, 0, 0, 2], al.f32)
            w010 = al.convert(w[8, ic, 0, 1, 0], al.f32)
            w011 = al.convert(w[8, ic, 0, 1, 1], al.f32)
            w012 = al.convert(w[8, ic, 0, 1, 2], al.f32)
            w020 = al.convert(w[8, ic, 0, 2, 0], al.f32)
            w021 = al.convert(w[8, ic, 0, 2, 1], al.f32)
            w022 = al.convert(w[8, ic, 0, 2, 2], al.f32)
            w100 = al.convert(w[8, ic, 1, 0, 0], al.f32)
            w101 = al.convert(w[8, ic, 1, 0, 1], al.f32)
            w102 = al.convert(w[8, ic, 1, 0, 2], al.f32)
            w110 = al.convert(w[8, ic, 1, 1, 0], al.f32)
            w111 = al.convert(w[8, ic, 1, 1, 1], al.f32)
            w112 = al.convert(w[8, ic, 1, 1, 2], al.f32)
            w120 = al.convert(w[8, ic, 1, 2, 0], al.f32)
            w121 = al.convert(w[8, ic, 1, 2, 1], al.f32)
            w122 = al.convert(w[8, ic, 1, 2, 2], al.f32)
            w200 = al.convert(w[8, ic, 2, 0, 0], al.f32)
            w201 = al.convert(w[8, ic, 2, 0, 1], al.f32)
            w202 = al.convert(w[8, ic, 2, 0, 2], al.f32)
            w210 = al.convert(w[8, ic, 2, 1, 0], al.f32)
            w211 = al.convert(w[8, ic, 2, 1, 1], al.f32)
            w212 = al.convert(w[8, ic, 2, 1, 2], al.f32)
            w220 = al.convert(w[8, ic, 2, 2, 0], al.f32)
            w221 = al.convert(w[8, ic, 2, 2, 1], al.f32)
            w222 = al.convert(w[8, ic, 2, 2, 2], al.f32)

            acc8 = acc8 + x000 * w000 + x001 * w001 + x002 * w002
            acc8 = acc8 + x010 * w010 + x011 * w011 + x012 * w012
            acc8 = acc8 + x020 * w020 + x021 * w021 + x022 * w022
            acc8 = acc8 + x100 * w100 + x101 * w101 + x102 * w102
            acc8 = acc8 + x110 * w110 + x111 * w111 + x112 * w112
            acc8 = acc8 + x120 * w120 + x121 * w121 + x122 * w122
            acc8 = acc8 + x200 * w200 + x201 * w201 + x202 * w202
            acc8 = acc8 + x210 * w210 + x211 * w211 + x212 * w212
            acc8 = acc8 + x220 * w220 + x221 * w221 + x222 * w222

            # Channel 9
            w000 = al.convert(w[9, ic, 0, 0, 0], al.f32)
            w001 = al.convert(w[9, ic, 0, 0, 1], al.f32)
            w002 = al.convert(w[9, ic, 0, 0, 2], al.f32)
            w010 = al.convert(w[9, ic, 0, 1, 0], al.f32)
            w011 = al.convert(w[9, ic, 0, 1, 1], al.f32)
            w012 = al.convert(w[9, ic, 0, 1, 2], al.f32)
            w020 = al.convert(w[9, ic, 0, 2, 0], al.f32)
            w021 = al.convert(w[9, ic, 0, 2, 1], al.f32)
            w022 = al.convert(w[9, ic, 0, 2, 2], al.f32)
            w100 = al.convert(w[9, ic, 1, 0, 0], al.f32)
            w101 = al.convert(w[9, ic, 1, 0, 1], al.f32)
            w102 = al.convert(w[9, ic, 1, 0, 2], al.f32)
            w110 = al.convert(w[9, ic, 1, 1, 0], al.f32)
            w111 = al.convert(w[9, ic, 1, 1, 1], al.f32)
            w112 = al.convert(w[9, ic, 1, 1, 2], al.f32)
            w120 = al.convert(w[9, ic, 1, 2, 0], al.f32)
            w121 = al.convert(w[9, ic, 1, 2, 1], al.f32)
            w122 = al.convert(w[9, ic, 1, 2, 2], al.f32)
            w200 = al.convert(w[9, ic, 2, 0, 0], al.f32)
            w201 = al.convert(w[9, ic, 2, 0, 1], al.f32)
            w202 = al.convert(w[9, ic, 2, 0, 2], al.f32)
            w210 = al.convert(w[9, ic, 2, 1, 0], al.f32)
            w211 = al.convert(w[9, ic, 2, 1, 1], al.f32)
            w212 = al.convert(w[9, ic, 2, 1, 2], al.f32)
            w220 = al.convert(w[9, ic, 2, 2, 0], al.f32)
            w221 = al.convert(w[9, ic, 2, 2, 1], al.f32)
            w222 = al.convert(w[9, ic, 2, 2, 2], al.f32)

            acc9 = acc9 + x000 * w000 + x001 * w001 + x002 * w002
            acc9 = acc9 + x010 * w010 + x011 * w011 + x012 * w012
            acc9 = acc9 + x020 * w020 + x021 * w021 + x022 * w022
            acc9 = acc9 + x100 * w100 + x101 * w101 + x102 * w102
            acc9 = acc9 + x110 * w110 + x111 * w111 + x112 * w112
            acc9 = acc9 + x120 * w120 + x121 * w121 + x122 * w122
            acc9 = acc9 + x200 * w200 + x201 * w201 + x202 * w202
            acc9 = acc9 + x210 * w210 + x211 * w211 + x212 * w212
            acc9 = acc9 + x220 * w220 + x221 * w221 + x222 * w222

            # Channel 10
            w000 = al.convert(w[10, ic, 0, 0, 0], al.f32)
            w001 = al.convert(w[10, ic, 0, 0, 1], al.f32)
            w002 = al.convert(w[10, ic, 0, 0, 2], al.f32)
            w010 = al.convert(w[10, ic, 0, 1, 0], al.f32)
            w011 = al.convert(w[10, ic, 0, 1, 1], al.f32)
            w012 = al.convert(w[10, ic, 0, 1, 2], al.f32)
            w020 = al.convert(w[10, ic, 0, 2, 0], al.f32)
            w021 = al.convert(w[10, ic, 0, 2, 1], al.f32)
            w022 = al.convert(w[10, ic, 0, 2, 2], al.f32)
            w100 = al.convert(w[10, ic, 1, 0, 0], al.f32)
            w101 = al.convert(w[10, ic, 1, 0, 1], al.f32)
            w102 = al.convert(w[10, ic, 1, 0, 2], al.f32)
            w110 = al.convert(w[10, ic, 1, 1, 0], al.f32)
            w111 = al.convert(w[10, ic, 1, 1, 1], al.f32)
            w112 = al.convert(w[10, ic, 1, 1, 2], al.f32)
            w120 = al.convert(w[10, ic, 1, 2, 0], al.f32)
            w121 = al.convert(w[10, ic, 1, 2, 1], al.f32)
            w122 = al.convert(w[10, ic, 1, 2, 2], al.f32)
            w200 = al.convert(w[10, ic, 2, 0, 0], al.f32)
            w201 = al.convert(w[10, ic, 2, 0, 1], al.f32)
            w202 = al.convert(w[10, ic, 2, 0, 2], al.f32)
            w210 = al.convert(w[10, ic, 2, 1, 0], al.f32)
            w211 = al.convert(w[10, ic, 2, 1, 1], al.f32)
            w212 = al.convert(w[10, ic, 2, 1, 2], al.f32)
            w220 = al.convert(w[10, ic, 2, 2, 0], al.f32)
            w221 = al.convert(w[10, ic, 2, 2, 1], al.f32)
            w222 = al.convert(w[10, ic, 2, 2, 2], al.f32)

            acc10 = acc10 + x000 * w000 + x001 * w001 + x002 * w002
            acc10 = acc10 + x010 * w010 + x011 * w011 + x012 * w012
            acc10 = acc10 + x020 * w020 + x021 * w021 + x022 * w022
            acc10 = acc10 + x100 * w100 + x101 * w101 + x102 * w102
            acc10 = acc10 + x110 * w110 + x111 * w111 + x112 * w112
            acc10 = acc10 + x120 * w120 + x121 * w121 + x122 * w122
            acc10 = acc10 + x200 * w200 + x201 * w201 + x202 * w202
            acc10 = acc10 + x210 * w210 + x211 * w211 + x212 * w212
            acc10 = acc10 + x220 * w220 + x221 * w221 + x222 * w222

            # Channel 11
            w000 = al.convert(w[11, ic, 0, 0, 0], al.f32)
            w001 = al.convert(w[11, ic, 0, 0, 1], al.f32)
            w002 = al.convert(w[11, ic, 0, 0, 2], al.f32)
            w010 = al.convert(w[11, ic, 0, 1, 0], al.f32)
            w011 = al.convert(w[11, ic, 0, 1, 1], al.f32)
            w012 = al.convert(w[11, ic, 0, 1, 2], al.f32)
            w020 = al.convert(w[11, ic, 0, 2, 0], al.f32)
            w021 = al.convert(w[11, ic, 0, 2, 1], al.f32)
            w022 = al.convert(w[11, ic, 0, 2, 2], al.f32)
            w100 = al.convert(w[11, ic, 1, 0, 0], al.f32)
            w101 = al.convert(w[11, ic, 1, 0, 1], al.f32)
            w102 = al.convert(w[11, ic, 1, 0, 2], al.f32)
            w110 = al.convert(w[11, ic, 1, 1, 0], al.f32)
            w111 = al.convert(w[11, ic, 1, 1, 1], al.f32)
            w112 = al.convert(w[11, ic, 1, 1, 2], al.f32)
            w120 = al.convert(w[11, ic, 1, 2, 0], al.f32)
            w121 = al.convert(w[11, ic, 1, 2, 1], al.f32)
            w122 = al.convert(w[11, ic, 1, 2, 2], al.f32)
            w200 = al.convert(w[11, ic, 2, 0, 0], al.f32)
            w201 = al.convert(w[11, ic, 2, 0, 1], al.f32)
            w202 = al.convert(w[11, ic, 2, 0, 2], al.f32)
            w210 = al.convert(w[11, ic, 2, 1, 0], al.f32)
            w211 = al.convert(w[11, ic, 2, 1, 1], al.f32)
            w212 = al.convert(w[11, ic, 2, 1, 2], al.f32)
            w220 = al.convert(w[11, ic, 2, 2, 0], al.f32)
            w221 = al.convert(w[11, ic, 2, 2, 1], al.f32)
            w222 = al.convert(w[11, ic, 2, 2, 2], al.f32)

            acc11 = acc11 + x000 * w000 + x001 * w001 + x002 * w002
            acc11 = acc11 + x010 * w010 + x011 * w011 + x012 * w012
            acc11 = acc11 + x020 * w020 + x021 * w021 + x022 * w022
            acc11 = acc11 + x100 * w100 + x101 * w101 + x102 * w102
            acc11 = acc11 + x110 * w110 + x111 * w111 + x112 * w112
            acc11 = acc11 + x120 * w120 + x121 * w121 + x122 * w122
            acc11 = acc11 + x200 * w200 + x201 * w201 + x202 * w202
            acc11 = acc11 + x210 * w210 + x211 * w211 + x212 * w212
            acc11 = acc11 + x220 * w220 + x221 * w221 + x222 * w222

            # Channel 12
            w000 = al.convert(w[12, ic, 0, 0, 0], al.f32)
            w001 = al.convert(w[12, ic, 0, 0, 1], al.f32)
            w002 = al.convert(w[12, ic, 0, 0, 2], al.f32)
            w010 = al.convert(w[12, ic, 0, 1, 0], al.f32)
            w011 = al.convert(w[12, ic, 0, 1, 1], al.f32)
            w012 = al.convert(w[12, ic, 0, 1, 2], al.f32)
            w020 = al.convert(w[12, ic, 0, 2, 0], al.f32)
            w021 = al.convert(w[12, ic, 0, 2, 1], al.f32)
            w022 = al.convert(w[12, ic, 0, 2, 2], al.f32)
            w100 = al.convert(w[12, ic, 1, 0, 0], al.f32)
            w101 = al.convert(w[12, ic, 1, 0, 1], al.f32)
            w102 = al.convert(w[12, ic, 1, 0, 2], al.f32)
            w110 = al.convert(w[12, ic, 1, 1, 0], al.f32)
            w111 = al.convert(w[12, ic, 1, 1, 1], al.f32)
            w112 = al.convert(w[12, ic, 1, 1, 2], al.f32)
            w120 = al.convert(w[12, ic, 1, 2, 0], al.f32)
            w121 = al.convert(w[12, ic, 1, 2, 1], al.f32)
            w122 = al.convert(w[12, ic, 1, 2, 2], al.f32)
            w200 = al.convert(w[12, ic, 2, 0, 0], al.f32)
            w201 = al.convert(w[12, ic, 2, 0, 1], al.f32)
            w202 = al.convert(w[12, ic, 2, 0, 2], al.f32)
            w210 = al.convert(w[12, ic, 2, 1, 0], al.f32)
            w211 = al.convert(w[12, ic, 2, 1, 1], al.f32)
            w212 = al.convert(w[12, ic, 2, 1, 2], al.f32)
            w220 = al.convert(w[12, ic, 2, 2, 0], al.f32)
            w221 = al.convert(w[12, ic, 2, 2, 1], al.f32)
            w222 = al.convert(w[12, ic, 2, 2, 2], al.f32)

            acc12 = acc12 + x000 * w000 + x001 * w001 + x002 * w002
            acc12 = acc12 + x010 * w010 + x011 * w011 + x012 * w012
            acc12 = acc12 + x020 * w020 + x021 * w021 + x022 * w022
            acc12 = acc12 + x100 * w100 + x101 * w101 + x102 * w102
            acc12 = acc12 + x110 * w110 + x111 * w111 + x112 * w112
            acc12 = acc12 + x120 * w120 + x121 * w121 + x122 * w122
            acc12 = acc12 + x200 * w200 + x201 * w201 + x202 * w202
            acc12 = acc12 + x210 * w210 + x211 * w211 + x212 * w212
            acc12 = acc12 + x220 * w220 + x221 * w221 + x222 * w222

            # Channel 13
            w000 = al.convert(w[13, ic, 0, 0, 0], al.f32)
            w001 = al.convert(w[13, ic, 0, 0, 1], al.f32)
            w002 = al.convert(w[13, ic, 0, 0, 2], al.f32)
            w010 = al.convert(w[13, ic, 0, 1, 0], al.f32)
            w011 = al.convert(w[13, ic, 0, 1, 1], al.f32)
            w012 = al.convert(w[13, ic, 0, 1, 2], al.f32)
            w020 = al.convert(w[13, ic, 0, 2, 0], al.f32)
            w021 = al.convert(w[13, ic, 0, 2, 1], al.f32)
            w022 = al.convert(w[13, ic, 0, 2, 2], al.f32)
            w100 = al.convert(w[13, ic, 1, 0, 0], al.f32)
            w101 = al.convert(w[13, ic, 1, 0, 1], al.f32)
            w102 = al.convert(w[13, ic, 1, 0, 2], al.f32)
            w110 = al.convert(w[13, ic, 1, 1, 0], al.f32)
            w111 = al.convert(w[13, ic, 1, 1, 1], al.f32)
            w112 = al.convert(w[13, ic, 1, 1, 2], al.f32)
            w120 = al.convert(w[13, ic, 1, 2, 0], al.f32)
            w121 = al.convert(w[13, ic, 1, 2, 1], al.f32)
            w122 = al.convert(w[13, ic, 1, 2, 2], al.f32)
            w200 = al.convert(w[13, ic, 2, 0, 0], al.f32)
            w201 = al.convert(w[13, ic, 2, 0, 1], al.f32)
            w202 = al.convert(w[13, ic, 2, 0, 2], al.f32)
            w210 = al.convert(w[13, ic, 2, 1, 0], al.f32)
            w211 = al.convert(w[13, ic, 2, 1, 1], al.f32)
            w212 = al.convert(w[13, ic, 2, 1, 2], al.f32)
            w220 = al.convert(w[13, ic, 2, 2, 0], al.f32)
            w221 = al.convert(w[13, ic, 2, 2, 1], al.f32)
            w222 = al.convert(w[13, ic, 2, 2, 2], al.f32)

            acc13 = acc13 + x000 * w000 + x001 * w001 + x002 * w002
            acc13 = acc13 + x010 * w010 + x011 * w011 + x012 * w012
            acc13 = acc13 + x020 * w020 + x021 * w021 + x022 * w022
            acc13 = acc13 + x100 * w100 + x101 * w101 + x102 * w102
            acc13 = acc13 + x110 * w110 + x111 * w111 + x112 * w112
            acc13 = acc13 + x120 * w120 + x121 * w121 + x122 * w122
            acc13 = acc13 + x200 * w200 + x201 * w201 + x202 * w202
            acc13 = acc13 + x210 * w210 + x211 * w211 + x212 * w212
            acc13 = acc13 + x220 * w220 + x221 * w221 + x222 * w222

            # Channel 14
            w000 = al.convert(w[14, ic, 0, 0, 0], al.f32)
            w001 = al.convert(w[14, ic, 0, 0, 1], al.f32)
            w002 = al.convert(w[14, ic, 0, 0, 2], al.f32)
            w010 = al.convert(w[14, ic, 0, 1, 0], al.f32)
            w011 = al.convert(w[14, ic, 0, 1, 1], al.f32)
            w012 = al.convert(w[14, ic, 0, 1, 2], al.f32)
            w020 = al.convert(w[14, ic, 0, 2, 0], al.f32)
            w021 = al.convert(w[14, ic, 0, 2, 1], al.f32)
            w022 = al.convert(w[14, ic, 0, 2, 2], al.f32)
            w100 = al.convert(w[14, ic, 1, 0, 0], al.f32)
            w101 = al.convert(w[14, ic, 1, 0, 1], al.f32)
            w102 = al.convert(w[14, ic, 1, 0, 2], al.f32)
            w110 = al.convert(w[14, ic, 1, 1, 0], al.f32)
            w111 = al.convert(w[14, ic, 1, 1, 1], al.f32)
            w112 = al.convert(w[14, ic, 1, 1, 2], al.f32)
            w120 = al.convert(w[14, ic, 1, 2, 0], al.f32)
            w121 = al.convert(w[14, ic, 1, 2, 1], al.f32)
            w122 = al.convert(w[14, ic, 1, 2, 2], al.f32)
            w200 = al.convert(w[14, ic, 2, 0, 0], al.f32)
            w201 = al.convert(w[14, ic, 2, 0, 1], al.f32)
            w202 = al.convert(w[14, ic, 2, 0, 2], al.f32)
            w210 = al.convert(w[14, ic, 2, 1, 0], al.f32)
            w211 = al.convert(w[14, ic, 2, 1, 1], al.f32)
            w212 = al.convert(w[14, ic, 2, 1, 2], al.f32)
            w220 = al.convert(w[14, ic, 2, 2, 0], al.f32)
            w221 = al.convert(w[14, ic, 2, 2, 1], al.f32)
            w222 = al.convert(w[14, ic, 2, 2, 2], al.f32)

            acc14 = acc14 + x000 * w000 + x001 * w001 + x002 * w002
            acc14 = acc14 + x010 * w010 + x011 * w011 + x012 * w012
            acc14 = acc14 + x020 * w020 + x021 * w021 + x022 * w022
            acc14 = acc14 + x100 * w100 + x101 * w101 + x102 * w102
            acc14 = acc14 + x110 * w110 + x111 * w111 + x112 * w112
            acc14 = acc14 + x120 * w120 + x121 * w121 + x122 * w122
            acc14 = acc14 + x200 * w200 + x201 * w201 + x202 * w202
            acc14 = acc14 + x210 * w210 + x211 * w211 + x212 * w212
            acc14 = acc14 + x220 * w220 + x221 * w221 + x222 * w222

            # Channel 15
            w000 = al.convert(w[15, ic, 0, 0, 0], al.f32)
            w001 = al.convert(w[15, ic, 0, 0, 1], al.f32)
            w002 = al.convert(w[15, ic, 0, 0, 2], al.f32)
            w010 = al.convert(w[15, ic, 0, 1, 0], al.f32)
            w011 = al.convert(w[15, ic, 0, 1, 1], al.f32)
            w012 = al.convert(w[15, ic, 0, 1, 2], al.f32)
            w020 = al.convert(w[15, ic, 0, 2, 0], al.f32)
            w021 = al.convert(w[15, ic, 0, 2, 1], al.f32)
            w022 = al.convert(w[15, ic, 0, 2, 2], al.f32)
            w100 = al.convert(w[15, ic, 1, 0, 0], al.f32)
            w101 = al.convert(w[15, ic, 1, 0, 1], al.f32)
            w102 = al.convert(w[15, ic, 1, 0, 2], al.f32)
            w110 = al.convert(w[15, ic, 1, 1, 0], al.f32)
            w111 = al.convert(w[15, ic, 1, 1, 1], al.f32)
            w112 = al.convert(w[15, ic, 1, 1, 2], al.f32)
            w120 = al.convert(w[15, ic, 1, 2, 0], al.f32)
            w121 = al.convert(w[15, ic, 1, 2, 1], al.f32)
            w122 = al.convert(w[15, ic, 1, 2, 2], al.f32)
            w200 = al.convert(w[15, ic, 2, 0, 0], al.f32)
            w201 = al.convert(w[15, ic, 2, 0, 1], al.f32)
            w202 = al.convert(w[15, ic, 2, 0, 2], al.f32)
            w210 = al.convert(w[15, ic, 2, 1, 0], al.f32)
            w211 = al.convert(w[15, ic, 2, 1, 1], al.f32)
            w212 = al.convert(w[15, ic, 2, 1, 2], al.f32)
            w220 = al.convert(w[15, ic, 2, 2, 0], al.f32)
            w221 = al.convert(w[15, ic, 2, 2, 1], al.f32)
            w222 = al.convert(w[15, ic, 2, 2, 2], al.f32)

            acc15 = acc15 + x000 * w000 + x001 * w001 + x002 * w002
            acc15 = acc15 + x010 * w010 + x011 * w011 + x012 * w012
            acc15 = acc15 + x020 * w020 + x021 * w021 + x022 * w022
            acc15 = acc15 + x100 * w100 + x101 * w101 + x102 * w102
            acc15 = acc15 + x110 * w110 + x111 * w111 + x112 * w112
            acc15 = acc15 + x120 * w120 + x121 * w121 + x122 * w122
            acc15 = acc15 + x200 * w200 + x201 * w201 + x202 * w202
            acc15 = acc15 + x210 * w210 + x211 * w211 + x212 * w212
            acc15 = acc15 + x220 * w220 + x221 * w221 + x222 * w222

        # Apply epilogue and write output
        out_stride_b = OC * OD * OH * OW
        out_stride_oc = OD * OH * OW
        out_stride_od = OH * OW
        out_stride_oh = OW
        out_layout = al.make_layout(
            (B, OC, OD, OH, OW),
            (out_stride_b, out_stride_oc, out_stride_od, out_stride_oh, 1),
        )
        out = al.make_tensor(out_ptr, al.bf16, out_layout)

        one_f32 = al.convert(1.0, al.f32)
        zero_f32 = al.convert(0.0, al.f32)

        # Channel 0
        acc0 = acc0 + al.convert(cb[0], al.f32)
        acc0 = acc0 * al.convert(sf[0], al.f32)
        acc0 = al.tanh(acc0)
        acc0 = acc0 * al.convert(bt[0], al.f32)
        neg0 = zero_f32 - acc0
        sig0 = one_f32 / (one_f32 + al.exp(neg0))
        out[b, 0, od, oh, ow] = al.convert(sig0, al.bf16)

        # Channel 1
        acc1 = acc1 + al.convert(cb[1], al.f32)
        acc1 = acc1 * al.convert(sf[1], al.f32)
        acc1 = al.tanh(acc1)
        acc1 = acc1 * al.convert(bt[1], al.f32)
        neg1 = zero_f32 - acc1
        sig1 = one_f32 / (one_f32 + al.exp(neg1))
        out[b, 1, od, oh, ow] = al.convert(sig1, al.bf16)

        # Channel 2
        acc2 = acc2 + al.convert(cb[2], al.f32)
        acc2 = acc2 * al.convert(sf[2], al.f32)
        acc2 = al.tanh(acc2)
        acc2 = acc2 * al.convert(bt[2], al.f32)
        neg2 = zero_f32 - acc2
        sig2 = one_f32 / (one_f32 + al.exp(neg2))
        out[b, 2, od, oh, ow] = al.convert(sig2, al.bf16)

        # Channel 3
        acc3 = acc3 + al.convert(cb[3], al.f32)
        acc3 = acc3 * al.convert(sf[3], al.f32)
        acc3 = al.tanh(acc3)
        acc3 = acc3 * al.convert(bt[3], al.f32)
        neg3 = zero_f32 - acc3
        sig3 = one_f32 / (one_f32 + al.exp(neg3))
        out[b, 3, od, oh, ow] = al.convert(sig3, al.bf16)

        # Channel 4
        acc4 = acc4 + al.convert(cb[4], al.f32)
        acc4 = acc4 * al.convert(sf[4], al.f32)
        acc4 = al.tanh(acc4)
        acc4 = acc4 * al.convert(bt[4], al.f32)
        neg4 = zero_f32 - acc4
        sig4 = one_f32 / (one_f32 + al.exp(neg4))
        out[b, 4, od, oh, ow] = al.convert(sig4, al.bf16)

        # Channel 5
        acc5 = acc5 + al.convert(cb[5], al.f32)
        acc5 = acc5 * al.convert(sf[5], al.f32)
        acc5 = al.tanh(acc5)
        acc5 = acc5 * al.convert(bt[5], al.f32)
        neg5 = zero_f32 - acc5
        sig5 = one_f32 / (one_f32 + al.exp(neg5))
        out[b, 5, od, oh, ow] = al.convert(sig5, al.bf16)

        # Channel 6
        acc6 = acc6 + al.convert(cb[6], al.f32)
        acc6 = acc6 * al.convert(sf[6], al.f32)
        acc6 = al.tanh(acc6)
        acc6 = acc6 * al.convert(bt[6], al.f32)
        neg6 = zero_f32 - acc6
        sig6 = one_f32 / (one_f32 + al.exp(neg6))
        out[b, 6, od, oh, ow] = al.convert(sig6, al.bf16)

        # Channel 7
        acc7 = acc7 + al.convert(cb[7], al.f32)
        acc7 = acc7 * al.convert(sf[7], al.f32)
        acc7 = al.tanh(acc7)
        acc7 = acc7 * al.convert(bt[7], al.f32)
        neg7 = zero_f32 - acc7
        sig7 = one_f32 / (one_f32 + al.exp(neg7))
        out[b, 7, od, oh, ow] = al.convert(sig7, al.bf16)

        # Channel 8
        acc8 = acc8 + al.convert(cb[8], al.f32)
        acc8 = acc8 * al.convert(sf[8], al.f32)
        acc8 = al.tanh(acc8)
        acc8 = acc8 * al.convert(bt[8], al.f32)
        neg8 = zero_f32 - acc8
        sig8 = one_f32 / (one_f32 + al.exp(neg8))
        out[b, 8, od, oh, ow] = al.convert(sig8, al.bf16)

        # Channel 9
        acc9 = acc9 + al.convert(cb[9], al.f32)
        acc9 = acc9 * al.convert(sf[9], al.f32)
        acc9 = al.tanh(acc9)
        acc9 = acc9 * al.convert(bt[9], al.f32)
        neg9 = zero_f32 - acc9
        sig9 = one_f32 / (one_f32 + al.exp(neg9))
        out[b, 9, od, oh, ow] = al.convert(sig9, al.bf16)

        # Channel 10
        acc10 = acc10 + al.convert(cb[10], al.f32)
        acc10 = acc10 * al.convert(sf[10], al.f32)
        acc10 = al.tanh(acc10)
        acc10 = acc10 * al.convert(bt[10], al.f32)
        neg10 = zero_f32 - acc10
        sig10 = one_f32 / (one_f32 + al.exp(neg10))
        out[b, 10, od, oh, ow] = al.convert(sig10, al.bf16)

        # Channel 11
        acc11 = acc11 + al.convert(cb[11], al.f32)
        acc11 = acc11 * al.convert(sf[11], al.f32)
        acc11 = al.tanh(acc11)
        acc11 = acc11 * al.convert(bt[11], al.f32)
        neg11 = zero_f32 - acc11
        sig11 = one_f32 / (one_f32 + al.exp(neg11))
        out[b, 11, od, oh, ow] = al.convert(sig11, al.bf16)

        # Channel 12
        acc12 = acc12 + al.convert(cb[12], al.f32)
        acc12 = acc12 * al.convert(sf[12], al.f32)
        acc12 = al.tanh(acc12)
        acc12 = acc12 * al.convert(bt[12], al.f32)
        neg12 = zero_f32 - acc12
        sig12 = one_f32 / (one_f32 + al.exp(neg12))
        out[b, 12, od, oh, ow] = al.convert(sig12, al.bf16)

        # Channel 13
        acc13 = acc13 + al.convert(cb[13], al.f32)
        acc13 = acc13 * al.convert(sf[13], al.f32)
        acc13 = al.tanh(acc13)
        acc13 = acc13 * al.convert(bt[13], al.f32)
        neg13 = zero_f32 - acc13
        sig13 = one_f32 / (one_f32 + al.exp(neg13))
        out[b, 13, od, oh, ow] = al.convert(sig13, al.bf16)

        # Channel 14
        acc14 = acc14 + al.convert(cb[14], al.f32)
        acc14 = acc14 * al.convert(sf[14], al.f32)
        acc14 = al.tanh(acc14)
        acc14 = acc14 * al.convert(bt[14], al.f32)
        neg14 = zero_f32 - acc14
        sig14 = one_f32 / (one_f32 + al.exp(neg14))
        out[b, 14, od, oh, ow] = al.convert(sig14, al.bf16)

        # Channel 15
        acc15 = acc15 + al.convert(cb[15], al.f32)
        acc15 = acc15 * al.convert(sf[15], al.f32)
        acc15 = al.tanh(acc15)
        acc15 = acc15 * al.convert(bt[15], al.f32)
        neg15 = zero_f32 - acc15
        sig15 = one_f32 / (one_f32 + al.exp(neg15))
        out[b, 15, od, oh, ow] = al.convert(sig15, al.bf16)


def _prepare_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def _avelang_conv3d_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_bias: torch.Tensor,
    scaling_factor: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_contiguous(x)
    w_bf16 = _prepare_bf16_contiguous(weight)
    cb_bf16 = _prepare_bf16_contiguous(conv_bias)
    sf_bf16 = _prepare_bf16_contiguous(scaling_factor)
    b_bf16 = _prepare_bf16_contiguous(bias)

    B = x_bf16.shape[0]
    ID = x_bf16.shape[2]
    IH = x_bf16.shape[3]
    IW = x_bf16.shape[4]
    OC = w_bf16.shape[0]
    OD = ID - 3 + 1
    OH = IH - 3 + 1
    OW = IW - 3 + 1

    total_spatial = B * OD * OH * OW
    num_blocks = (total_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty(
        (B, OC, OD, OH, OW), device=x_bf16.device, dtype=torch.bfloat16
    )

    _conv3d_fused_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, w_bf16, cb_bf16, sf_bf16, b_bf16, out,
        B, OC, ID, IH, IW, OD, OH, OW, total_spatial,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        weight = self.conv.weight.data
        conv_bias = self.conv.bias.data
        sf = self.scaling_factor.data
        b = self.bias.data

        result = _avelang_conv3d_fused(x, weight, conv_bias, sf, b)
        return result.to(x.dtype)


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 64, 64
kernel_size = 3
scaling_factor = 2
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape]
