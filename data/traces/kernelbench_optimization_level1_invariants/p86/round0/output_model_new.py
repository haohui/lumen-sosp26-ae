import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def fused_conv_kernel(
    X: al.Tensor((16, 64, 512, 512), al.bf16),
    DW: al.Tensor((64, 1, 3, 3), al.bf16),
    PW: al.Tensor((128, 64, 1, 1), al.bf16),
    Y: al.Tensor((16, 128, 512, 512), al.bf16),
):
    # Thread and block identifiers
    tid = al.thread_id(0)
    block_m = al.block_id(0)
    block_n = al.block_id(1)

    lane = tid % 64
    warp_id = tid // 64
    warp_row = warp_id // 2
    warp_col = warp_id % 2

    lane_col = lane % 32
    lane_k_base = (lane // 32) * 4

    group_m_base = block_m * 128
    group_n_base = block_n * 128

    # Allocate persistent fragments
    a_bf16_0 = al.make_local((4,), al.bf16)
    a_bf16_1 = al.make_local((4,), al.bf16)
    b_bf16_0 = al.make_local((4,), al.bf16)
    b_bf16_1 = al.make_local((4,), al.bf16)
    c00 = al.make_local((16,), al.f32)
    c01 = al.make_local((16,), al.f32)
    c10 = al.make_local((16,), al.f32)
    c11 = al.make_local((16,), al.f32)

    # Create i32 views once outside K loop
    a0 = al.view(a_bf16_0, al.Tensor((2,), al.i32))
    a1 = al.view(a_bf16_1, al.Tensor((2,), al.i32))
    b0 = al.view(b_bf16_0, al.Tensor((2,), al.i32))
    b1 = al.view(b_bf16_1, al.Tensor((2,), al.i32))

    # Initialize accumulators to zero
    zero_f32 = al.convert(0.0, al.f32)
    for ai in al.range(16):
        c00[ai] = zero_f32
        c01[ai] = zero_f32
        c10[ai] = zero_f32
        c11[ai] = zero_f32

    # K-loop over MFMA-sized chunks (72 tiles for 64*9=576 K elements)
    for k_tile in al.range(72):
        # --- Load A fragment 0 (tm=0) ---
        m0 = group_m_base + warp_row * 64 + 0 * 32 + lane_col
        batch_idx_0 = m0 // 262144
        pos0 = m0 - batch_idx_0 * 262144
        oh0 = pos0 // 512
        ow0 = pos0 - oh0 * 512

        for e in al.range(4):
            k = k_tile * 8 + lane_k_base + e
            ic = k // 9
            r = k - ic * 9
            kh = r // 3
            kw = r - kh * 3
            ih = oh0 - 1 + kh
            iw = ow0 - 1 + kw
            if ih >= 0 and ih < 512 and iw >= 0 and iw < 512:
                a_bf16_0[e] = X[batch_idx_0, ic, ih, iw]
            else:
                a_bf16_0[e] = al.convert(0.0, al.bf16)

        # --- Load A fragment 1 (tm=1) ---
        m1 = group_m_base + warp_row * 64 + 1 * 32 + lane_col
        batch_idx_1 = m1 // 262144
        pos1 = m1 - batch_idx_1 * 262144
        oh1 = pos1 // 512
        ow1 = pos1 - oh1 * 512

        for e in al.range(4):
            k = k_tile * 8 + lane_k_base + e
            ic = k // 9
            r = k - ic * 9
            kh = r // 3
            kw = r - kh * 3
            ih = oh1 - 1 + kh
            iw = ow1 - 1 + kw
            if ih >= 0 and ih < 512 and iw >= 0 and iw < 512:
                a_bf16_1[e] = X[batch_idx_1, ic, ih, iw]
            else:
                a_bf16_1[e] = al.convert(0.0, al.bf16)

        # --- Load B fragment 0 (tn=0) ---
        n0 = group_n_base + warp_col * 64 + 0 * 32 + lane_col

        for e in al.range(4):
            k = k_tile * 8 + lane_k_base + e
            ic = k // 9
            r = k - ic * 9
            kh = r // 3
            kw = r - kh * 3
            dw_val = al.convert(DW[ic, 0, kh, kw], al.f32)
            pw_val = al.convert(PW[n0, ic, 0, 0], al.f32)
            b_bf16_0[e] = al.convert(dw_val * pw_val, al.bf16)

        # --- Load B fragment 1 (tn=1) ---
        n1 = group_n_base + warp_col * 64 + 1 * 32 + lane_col

        for e in al.range(4):
            k = k_tile * 8 + lane_k_base + e
            ic = k // 9
            r = k - ic * 9
            kh = r // 3
            kw = r - kh * 3
            dw_val = al.convert(DW[ic, 0, kh, kw], al.f32)
            pw_val = al.convert(PW[n1, ic, 0, 0], al.f32)
            b_bf16_1[e] = al.convert(dw_val * pw_val, al.bf16)

        # --- MFMA accumulate ---
        c00 = al.amdgpu.mfma_32x32x8_bf16_f32(a0, b0, c00)
        c01 = al.amdgpu.mfma_32x32x8_bf16_f32(a0, b1, c01)
        c10 = al.amdgpu.mfma_32x32x8_bf16_f32(a1, b0, c10)
        c11 = al.amdgpu.mfma_32x32x8_bf16_f32(a1, b1, c11)

    # --- Writeback: tm=0, tn=0 ---
    trb = group_m_base + warp_row * 64 + 0 * 32
    tcb = group_n_base + warp_col * 64 + 0 * 32
    for ai in al.range(16):
        col = tcb + lane_col
        row = trb + 8 * (ai // 4) + 4 * (lane // 32) + (ai - 4 * (ai // 4))
        batch_idx = row // 262144
        pos = row - batch_idx * 262144
        oh = pos // 512
        ow = pos - oh * 512
        Y[batch_idx, col, oh, ow] = al.convert(c00[ai], al.bf16)

    # --- Writeback: tm=0, tn=1 ---
    trb = group_m_base + warp_row * 64 + 0 * 32
    tcb = group_n_base + warp_col * 64 + 1 * 32
    for ai in al.range(16):
        col = tcb + lane_col
        row = trb + 8 * (ai // 4) + 4 * (lane // 32) + (ai - 4 * (ai // 4))
        batch_idx = row // 262144
        pos = row - batch_idx * 262144
        oh = pos // 512
        ow = pos - oh * 512
        Y[batch_idx, col, oh, ow] = al.convert(c01[ai], al.bf16)

    # --- Writeback: tm=1, tn=0 ---
    trb = group_m_base + warp_row * 64 + 1 * 32
    tcb = group_n_base + warp_col * 64 + 0 * 32
    for ai in al.range(16):
        col = tcb + lane_col
        row = trb + 8 * (ai // 4) + 4 * (lane // 32) + (ai - 4 * (ai // 4))
        batch_idx = row // 262144
        pos = row - batch_idx * 262144
        oh = pos // 512
        ow = pos - oh * 512
        Y[batch_idx, col, oh, ow] = al.convert(c10[ai], al.bf16)

    # --- Writeback: tm=1, tn=1 ---
    trb = group_m_base + warp_row * 64 + 1 * 32
    tcb = group_n_base + warp_col * 64 + 1 * 32
    for ai in al.range(16):
        col = tcb + lane_col
        row = trb + 8 * (ai // 4) + 4 * (lane // 32) + (ai - 4 * (ai // 4))
        batch_idx = row // 262144
        pos = row - batch_idx * 262144
        oh = pos // 512
        ow = pos - oh * 512
        Y[batch_idx, col, oh, ow] = al.convert(c11[ai], al.bf16)


class ModelNew(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   bias=bias)

    def forward(self, x):
        x0 = x.contiguous()
        dw = self.depthwise.weight.to(device=x.device, dtype=x.dtype).contiguous()
        pw = self.pointwise.weight.to(device=x.device, dtype=x.dtype).contiguous()
        y = torch.empty((16, 128, 512, 512), device=x.device, dtype=x.dtype)

        N_POS = 16 * 512 * 512
        GRID_M = (N_POS + 127) // 128

        fused_conv_kernel[lambda: ((GRID_M, 1, 1), (256, 1, 1))](x0, dw, pw, y)
        return y
