import torch
import torch.nn as nn
import avelang
import avelang.language as al


def _ceildiv(a, b):
    return (a + b - 1) // b


@avelang.jit
def conv2d_mfma_kernel(
    X_ptr: al.Pointer(al.bf16),
    W_ptr: al.Pointer(al.bf16),
    Y_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
    KH: al.i32,
    KW: al.i32,
    OH: al.i32,
    OW: al.i32,
    M_TOTAL: al.i32,
    X_stride_b: al.i32,
    X_stride_ic: al.i32,
    X_stride_h: al.i32,
    W_stride_oc: al.i32,
    W_stride_ic: al.i32,
    W_stride_kh: al.i32,
    Y_stride_b: al.i32,
    Y_stride_oc: al.i32,
    Y_stride_oh: al.i32,
):
    X = al.make_tensor(X_ptr, al.bf16, al.make_layout((B, IC, H, W), (X_stride_b, X_stride_ic, X_stride_h, 1)))
    Wgt = al.make_tensor(W_ptr, al.bf16, al.make_layout((OC, IC, KH, KW), (W_stride_oc, W_stride_ic, W_stride_kh, 1)))
    Y = al.make_tensor(Y_ptr, al.bf16, al.make_layout((B, OC, OH, OW), (Y_stride_b, Y_stride_oc, Y_stride_oh, 1)))

    tid = al.thread_id(0)
    bid = al.block_id(0)
    lane = tid % 64
    warp_id = tid // 64
    warp_row = warp_id // 2
    warp_col = warp_id % 2

    lane_col = lane % 32
    lane_k_base = (lane // 32) * 4

    group_m_base = bid * 128
    group_n_base = 0

    K_TOTAL = IC * KH * KW

    acc_00 = al.make_local((16,), al.f32)
    acc_01 = al.make_local((16,), al.f32)
    acc_10 = al.make_local((16,), al.f32)
    acc_11 = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc_00[i] = al.convert(0.0, al.f32)
        acc_01[i] = al.convert(0.0, al.f32)
        acc_10[i] = al.convert(0.0, al.f32)
        acc_11[i] = al.convert(0.0, al.f32)

    OHOW = OH * OW
    KHxKW = KH * KW

    for k_tile in al.range(0, K_TOTAL, 8):
        a0 = al.make_local((4,), al.bf16)
        a1 = al.make_local((4,), al.bf16)
        b0 = al.make_local((4,), al.bf16)
        b1 = al.make_local((4,), al.bf16)

        k_base = k_tile + lane_k_base

        for e in al.range(4):
            k_idx = k_base + e
            if k_idx < K_TOTAL:
                ic_val = k_idx // KHxKW
                rem_k = k_idx % KHxKW
                kh_val = rem_k // KW
                kw_val = rem_k % KW

                m0 = group_m_base + warp_row * 64 + 0 * 32 + lane_col
                if m0 < M_TOTAL:
                    batch0 = m0 // OHOW
                    rem0 = m0 % OHOW
                    oh0 = rem0 // OW
                    ow0 = rem0 % OW
                    a0[e] = X[batch0, ic_val, oh0 + kh_val, ow0 + kw_val]
                else:
                    a0[e] = al.convert(0, al.bf16)

                m1 = group_m_base + warp_row * 64 + 32 + lane_col
                if m1 < M_TOTAL:
                    batch1 = m1 // OHOW
                    rem1 = m1 % OHOW
                    oh1 = rem1 // OW
                    ow1 = rem1 % OW
                    a1[e] = X[batch1, ic_val, oh1 + kh_val, ow1 + kw_val]
                else:
                    a1[e] = al.convert(0, al.bf16)

                n0 = group_n_base + warp_col * 64 + 0 * 32 + lane_col
                b0[e] = Wgt[n0, ic_val, kh_val, kw_val]

                n1 = group_n_base + warp_col * 64 + 32 + lane_col
                b1[e] = Wgt[n1, ic_val, kh_val, kw_val]
            else:
                a0[e] = al.convert(0, al.bf16)
                a1[e] = al.convert(0, al.bf16)
                b0[e] = al.convert(0, al.bf16)
                b1[e] = al.convert(0, al.bf16)

        acc_00 = al.amdgpu.mfma_32x32x8_bf16_f32(
            al.view(a0, al.Tensor((2,), al.u32)),
            al.view(b0, al.Tensor((2,), al.u32)),
            acc_00,
        )
        acc_01 = al.amdgpu.mfma_32x32x8_bf16_f32(
            al.view(a0, al.Tensor((2,), al.u32)),
            al.view(b1, al.Tensor((2,), al.u32)),
            acc_01,
        )
        acc_10 = al.amdgpu.mfma_32x32x8_bf16_f32(
            al.view(a1, al.Tensor((2,), al.u32)),
            al.view(b0, al.Tensor((2,), al.u32)),
            acc_10,
        )
        acc_11 = al.amdgpu.mfma_32x32x8_bf16_f32(
            al.view(a1, al.Tensor((2,), al.u32)),
            al.view(b1, al.Tensor((2,), al.u32)),
            acc_11,
        )

    lane_half = lane // 32
    lane_c = lane % 32

    col_0 = group_n_base + warp_col * 64 + 0 * 32 + lane_c
    col_1 = group_n_base + warp_col * 64 + 32 + lane_c

    for acc_idx in al.range(16):
        row_offset = 8 * (acc_idx // 4) + 4 * lane_half + (acc_idx % 4)
        row_0 = group_m_base + warp_row * 64 + 0 * 32 + row_offset
        row_1 = group_m_base + warp_row * 64 + 32 + row_offset

        if row_0 < M_TOTAL:
            batch_00 = row_0 // OHOW
            rem_00 = row_0 % OHOW
            oh_00 = rem_00 // OW
            ow_00 = rem_00 % OW
            Y[batch_00, col_0, oh_00, ow_00] = al.convert(acc_00[acc_idx], al.bf16)
            Y[batch_00, col_1, oh_00, ow_00] = al.convert(acc_01[acc_idx], al.bf16)

        if row_1 < M_TOTAL:
            batch_10 = row_1 // OHOW
            rem_10 = row_1 % OHOW
            oh_10 = rem_10 // OW
            ow_10 = rem_10 % OW
            Y[batch_10, col_0, oh_10, ow_10] = al.convert(acc_10[acc_idx], al.bf16)
            Y[batch_10, col_1, oh_10, ow_10] = al.convert(acc_11[acc_idx], al.bf16)


class ModelNew(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size),
                                stride=stride, padding=padding, dilation=dilation,
                                groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, ic, h, w_in = x.shape
        w = self.conv2d.weight
        oc = w.shape[0]
        kh = kw = self.conv2d.kernel_size[0]
        stride_h = stride_w = self.conv2d.stride[0]
        pad_h = pad_w = self.conv2d.padding[0]
        dil_h = dil_w = self.conv2d.dilation[0]

        oh = (h + 2 * pad_h - dil_h * (kh - 1) - 1) // stride_h + 1
        ow = (w_in + 2 * pad_w - dil_w * (kw - 1) - 1) // stride_w + 1

        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = w.to(torch.bfloat16).contiguous()
        y = torch.empty(b, oc, oh, ow, device=x.device, dtype=torch.bfloat16)

        m_total = b * oh * ow
        grid_x = _ceildiv(m_total, 128)

        conv2d_mfma_kernel[lambda: ((grid_x, 1, 1), (256, 1, 1))](
            x_bf16, w_bf16, y,
            b, ic, oc, h, w_in, kh, kw, oh, ow, m_total,
            ic * h * w_in, h * w_in, w_in,
            ic * kh * kw, kh * kw, kw,
            oc * oh * ow, oh * ow, ow,
        )
        return y
