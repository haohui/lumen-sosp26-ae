import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_M = 32
TILE_N = 32
TILE_K = 8


@avelang.jit
def conv2d_mfma_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    y_ptr: al.Pointer(al.bf16),
    N: al.i32,
    IC: al.i32,
    H: al.i32,
    W: al.i32,
    OC: al.i32,
    KH: al.i32,
    KW: al.i32,
    OH: al.i32,
    OW: al.i32,
    STRIDE_H: al.i32,
    STRIDE_W: al.i32,
    PAD_H: al.i32,
    PAD_W: al.i32,
    DIL_H: al.i32,
    DIL_W: al.i32,
):
    lane = al.thread_id(0)

    x_layout = al.make_layout((N, IC, H, W), (IC * H * W, H * W, W, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    w_layout = al.make_layout((OC, IC, KH, KW), (IC * KH * KW, KH * KW, KW, 1))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)
    y_layout = al.make_layout((N, OC, OH, OW), (OC * OH * OW, OH * OW, OW, 1))
    y = al.make_tensor(y_ptr, al.bf16, y_layout)

    m_block = al.block_id(0)
    n_block = al.block_id(1)

    oc_start = m_block * TILE_M
    spat_start = n_block * TILE_N

    # Per-thread MFMA registers using the amdgpu_gemm pattern:
    # A: 2 u32 = 4 BF16 packed (4 BF16 = 64 bits = 2 u32 per thread for 32x32x8)
    # B: 2 u32 = 4 BF16 packed
    # C: 16 f32 = accumulator
    a_reg = al.make_local((1, 2), al.u32)
    b_reg = al.make_local((1, 2), al.u32)
    c_reg = al.make_local((1, 16), al.f32)

    for i in al.range(16):
        c_reg[0, i] = al.convert(0.0, al.f32)

    # Thread mapping from MFMA swizzle:
    # A(i, j) -> lane_id = i + (j/4)*32, element = j%4
    # So lane handles: i = lane%32 for rows, K half determined by lane//32
    lane_m = lane % TILE_M
    lane_k_half = lane // TILE_M

    K_TOTAL = IC * KH * KW
    k_tiles = K_TOTAL // TILE_K

    for kt in al.range(k_tiles):
        k_start = kt * TILE_K

        # Load A (weight) as packed u32
        # Thread loads 4 BF16 values from weight into 2 u32 registers
        oc = oc_start + lane_m
        a_u32_0 = al.convert(0, al.u32)
        a_u32_1 = al.convert(0, al.u32)

        for k_off in al.range(4):
            k_idx = k_start + lane_k_half * 4 + k_off
            a_val = al.convert(0.0, al.bf16)
            if k_idx < K_TOTAL and oc < OC:
                kw_idx = k_idx % KW
                khw_idx = k_idx // KW
                kh_idx = khw_idx % KH
                ic_idx = khw_idx // KH
                a_val = al.convert(w[oc, ic_idx, kh_idx, kw_idx], al.bf16)
            a_u16 = al.bitcast(a_val, al.u16)
            a_packed = al.convert(a_u16, al.u32)
            if k_off < 2:
                if k_off == 0:
                    a_u32_0 = a_packed
                else:
                    a_shifted = a_packed * al.convert(65536, al.u32)
                    a_u32_0 = a_u32_0 + a_shifted
            else:
                if k_off == 2:
                    a_u32_1 = a_packed
                else:
                    a_shifted = a_packed * al.convert(65536, al.u32)
                    a_u32_1 = a_u32_1 + a_shifted

        a_reg[0, 0] = a_u32_0
        a_reg[0, 1] = a_u32_1

        # Load B (input) as packed u32
        spat_pos = spat_start + lane_m
        sn = spat_pos // (OH * OW)
        srem = spat_pos % (OH * OW)
        soh = srem // OW
        sow = srem % OW
        b_u32_0 = al.convert(0, al.u32)
        b_u32_1 = al.convert(0, al.u32)

        for k_off in al.range(4):
            k_idx = k_start + lane_k_half * 4 + k_off
            b_val = al.convert(0.0, al.bf16)
            if k_idx < K_TOTAL and sn < N:
                kw_idx = k_idx % KW
                khw_idx = k_idx // KW
                kh_idx = khw_idx % KH
                ic_idx = khw_idx // KH
                ih = soh * STRIDE_H - PAD_H + kh_idx * DIL_H
                iw = sow * STRIDE_W - PAD_W + kw_idx * DIL_W
                if (ih >= 0) and (ih < H) and (iw >= 0) and (iw < W):
                    b_val = al.convert(x[sn, ic_idx, ih, iw], al.bf16)
            b_u16 = al.bitcast(b_val, al.u16)
            b_packed = al.convert(b_u16, al.u32)
            if k_off < 2:
                if k_off == 0:
                    b_u32_0 = b_packed
                else:
                    b_shifted = b_packed * al.convert(65536, al.u32)
                    b_u32_0 = b_u32_0 + b_shifted
            else:
                if k_off == 2:
                    b_u32_1 = b_packed
                else:
                    b_shifted = b_packed * al.convert(65536, al.u32)
                    b_u32_1 = b_u32_1 + b_shifted

        b_reg[0, 0] = b_u32_0
        b_reg[0, 1] = b_u32_1

        c_reg[0] = al.amdgpu.mfma_32x32x8_bf16_f32(a_reg[0], b_reg[0], c_reg[0])

    # Store accumulated output using the MFMA writeback mapping
    lane_group = lane // TILE_N
    lane_col = lane % TILE_N

    for t in al.range(16):
        row_off = (t // 4) * 8 + lane_group * 4 + (t % 4)
        oc = oc_start + row_off
        spat_pos = spat_start + lane_col
        if oc < OC:
            sn = spat_pos // (OH * OW)
            srem = spat_pos % (OH * OW)
            soh = srem // OW
            sow = srem % OW
            if sn < N and soh < OH and sow < OW:
                y[sn, oc, soh, sow] = al.convert(c_reg[0, t], al.bf16)


class ModelNew(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
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
        self._cached_weight_ptr = None
        self._cached_w_bf16 = None

    def forward(self, x):
        N_val, IC_val, H_val, W_val = x.shape
        OC_val = self.conv2d.out_channels
        KH, KW = self.conv2d.kernel_size
        stride_h = self.conv2d.stride[0] if isinstance(self.conv2d.stride, tuple) else self.conv2d.stride
        stride_w = self.conv2d.stride[1] if isinstance(self.conv2d.stride, tuple) else self.conv2d.stride
        pad_h = self.conv2d.padding[0] if isinstance(self.conv2d.padding, tuple) else self.conv2d.padding
        pad_w = self.conv2d.padding[1] if isinstance(self.conv2d.padding, tuple) else self.conv2d.padding
        dil_h = self.conv2d.dilation[0] if isinstance(self.conv2d.dilation, tuple) else self.conv2d.dilation
        dil_w = self.conv2d.dilation[1] if isinstance(self.conv2d.dilation, tuple) else self.conv2d.dilation
        OH_val = (H_val + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
        OW_val = (W_val + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1

        w_tensor = self.conv2d.weight
        w_ptr = w_tensor.data_ptr()
        if self._cached_weight_ptr != w_ptr:
            self._cached_w_bf16 = w_tensor.detach().to(dtype=torch.bfloat16).contiguous()
            self._cached_weight_ptr = w_ptr

        x_bf16 = x.detach().to(device=x.device, dtype=torch.bfloat16).contiguous()
        y = torch.empty((N_val, OC_val, OH_val, OW_val), device=x.device, dtype=torch.bfloat16)

        oc_tiles = (OC_val + TILE_M - 1) // TILE_M
        spat_total = N_val * OH_val * OW_val
        spat_tiles = (spat_total + TILE_N - 1) // TILE_N

        conv2d_mfma_kernel[lambda: ((oc_tiles, spat_tiles, 1), (64, 1, 1))](
            x_bf16,
            self._cached_w_bf16,
            y,
            N_val, IC_val, H_val, W_val,
            OC_val, KH, KW,
            OH_val, OW_val,
            stride_h, stride_w,
            pad_h, pad_w,
            dil_h, dil_w,
        )
        return y
