import torch
import torch.nn as nn
import avelang
import avelang.language as al

@avelang.jit
def conv2d_mfma_kernel(
    X: al.Tensor((8, 64, 512, 256), al.f32),
    W: al.Tensor((128, 64, 5, 7), al.f32),
    Y3: al.Tensor((1024, 508, 250), al.f32),
):
    lane = al.thread_id(0)
    wl = lane % 64
    wl32 = wl // 32
    lane_col = wl % 32
    lane_k_base = wl32 * 4
    wr = lane // 128
    wc = (lane // 64) % 2

    gm128 = al.block_id(0) * 128
    gn128 = al.block_id(1) * 128

    N_SPATIAL = 1016000
    OH = 508
    OW = 250
    OH_OW = 127000
    KH_KW = 35
    KW = 7
    K_TILES = 280

    acc00 = al.full((16,), 0.0, al.f32)
    acc01 = al.full((16,), 0.0, al.f32)
    acc10 = al.full((16,), 0.0, al.f32)
    acc11 = al.full((16,), 0.0, al.f32)

    a_bf16_0 = al.make_local((4,), al.bf16)
    a_bf16_1 = al.make_local((4,), al.bf16)
    b_bf16_0 = al.make_local((4,), al.bf16)
    b_bf16_1 = al.make_local((4,), al.bf16)

    for k_tile in al.range(K_TILES):
        k_base = k_tile * 8

        for e in al.range(4):
            k = k_base + lane_k_base + e
            ic = k // KH_KW
            rem = k - ic * KH_KW
            kh = rem // KW
            kw = rem - kh * KW

            m0 = gm128 + wr * 64 + lane_col
            a_bf16_0[e] = al.convert(W[m0, ic, kh, kw], al.bf16)

            m1 = gm128 + wr * 64 + 32 + lane_col
            a_bf16_1[e] = al.convert(W[m1, ic, kh, kw], al.bf16)

        for e in al.range(4):
            k = k_base + lane_k_base + e
            ic = k // KH_KW
            rem = k - ic * KH_KW
            kh = rem // KW
            kw = rem - kh * KW

            n0 = gn128 + wc * 64 + lane_col
            if n0 < N_SPATIAL:
                batch0 = n0 // OH_OW
                rem_n0 = n0 - batch0 * OH_OW
                oh0 = rem_n0 // OW
                ow0 = rem_n0 - oh0 * OW
                i0_0 = oh0 + kh
                i1_0 = ow0 + kw
                b_bf16_0[e] = al.convert(X[batch0, ic, i0_0, i1_0], al.bf16)
            else:
                b_bf16_0[e] = al.convert(0.0, al.bf16)

            n1 = gn128 + wc * 64 + 32 + lane_col
            if n1 < N_SPATIAL:
                batch1 = n1 // OH_OW
                rem_n1 = n1 - batch1 * OH_OW
                oh1 = rem_n1 // OW
                ow1 = rem_n1 - oh1 * OW
                i0_1 = oh1 + kh
                i1_1 = ow1 + kw
                b_bf16_1[e] = al.convert(X[batch1, ic, i0_1, i1_1], al.bf16)
            else:
                b_bf16_1[e] = al.convert(0.0, al.bf16)

        a_packed_0 = al.view(a_bf16_0, al.Tensor((2,), al.u32))
        a_packed_1 = al.view(a_bf16_1, al.Tensor((2,), al.u32))
        b_packed_0 = al.view(b_bf16_0, al.Tensor((2,), al.u32))
        b_packed_1 = al.view(b_bf16_1, al.Tensor((2,), al.u32))

        acc00 = al.amdgpu.mfma_32x32x8_bf16_f32(b_packed_0, a_packed_0, acc00)
        acc01 = al.amdgpu.mfma_32x32x8_bf16_f32(b_packed_1, a_packed_0, acc01)
        acc10 = al.amdgpu.mfma_32x32x8_bf16_f32(b_packed_0, a_packed_1, acc10)
        acc11 = al.amdgpu.mfma_32x32x8_bf16_f32(b_packed_1, a_packed_1, acc11)

    # Writeback tm=0, tn=0
    for acc_idx in al.range(16):
        col_flat = gn128 + wc * 64 + 0 + lane_col
        row_oc = gm128 + wr * 64 + 0 + 8 * (acc_idx // 4) + wl32 * 4 + (acc_idx % 4)
        if col_flat < N_SPATIAL:
            n_out = col_flat // OH_OW
            rem = col_flat - n_out * OH_OW
            oh_out = rem // OW
            ow_out = rem - oh_out * OW
            Y3[n_out * 128 + row_oc, oh_out, ow_out] = acc00[acc_idx]

    # Writeback tm=0, tn=1
    for acc_idx in al.range(16):
        col_flat = gn128 + wc * 64 + 32 + lane_col
        row_oc = gm128 + wr * 64 + 0 + 8 * (acc_idx // 4) + wl32 * 4 + (acc_idx % 4)
        if col_flat < N_SPATIAL:
            n_out = col_flat // OH_OW
            rem = col_flat - n_out * OH_OW
            oh_out = rem // OW
            ow_out = rem - oh_out * OW
            Y3[n_out * 128 + row_oc, oh_out, ow_out] = acc01[acc_idx]

    # Writeback tm=1, tn=0
    for acc_idx in al.range(16):
        col_flat = gn128 + wc * 64 + 0 + lane_col
        row_oc = gm128 + wr * 64 + 32 + 8 * (acc_idx // 4) + wl32 * 4 + (acc_idx % 4)
        if col_flat < N_SPATIAL:
            n_out = col_flat // OH_OW
            rem = col_flat - n_out * OH_OW
            oh_out = rem // OW
            ow_out = rem - oh_out * OW
            Y3[n_out * 128 + row_oc, oh_out, ow_out] = acc10[acc_idx]

    # Writeback tm=1, tn=1
    for acc_idx in al.range(16):
        col_flat = gn128 + wc * 64 + 32 + lane_col
        row_oc = gm128 + wr * 64 + 32 + 8 * (acc_idx // 4) + wl32 * 4 + (acc_idx % 4)
        if col_flat < N_SPATIAL:
            n_out = col_flat // OH_OW
            rem = col_flat - n_out * OH_OW
            oh_out = rem // OW
            ow_out = rem - oh_out * OW
            Y3[n_out * 128 + row_oc, oh_out, ow_out] = acc11[acc_idx]


class ModelNew(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
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

    def forward(self, x):
        orig_dtype = x.dtype
        x0 = x.contiguous().to(torch.float32)
        w = self.conv2d.weight.to(device=x.device, dtype=torch.float32).contiguous()
        y3 = torch.empty((1024, 508, 250), device=x.device, dtype=torch.float32)
        conv2d_mfma_kernel[lambda: ((1, 7938, 1), (256, 1, 1))](x0, w, y3)
        y = y3.view(8, 128, 508, 250)
        if orig_dtype != torch.float32:
            y = y.to(orig_dtype)
        return y
