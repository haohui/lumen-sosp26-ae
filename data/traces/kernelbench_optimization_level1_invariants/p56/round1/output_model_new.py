import torch
import torch.nn as nn
import avelang
import avelang.language as al

SPLIT_K_SLICES = 2
KH_KW = 35
OUT_CHANNELS = 128
BATCH_SIZE = 8
OH = 508
OW = 250
OH_OW = 127000
N_SPATIAL = 1016000
K_PER_SPLIT = 1120
K_TILES_SPLIT = 140
GEMM_N = 128
N_TILES_PER_SPATIAL = (N_SPATIAL + 31) // 32


@avelang.jit
def conv2d_mfma_split_kernel(
    X: al.Tensor((8, 64, 512, 256), al.f32),
    W: al.Tensor((128, 64, 5, 7), al.f32),
    ws_ptr: al.Pointer(al.f32),
):
    ws2d = al.make_layout((SPLIT_K_SLICES * N_SPATIAL, GEMM_N), (GEMM_N, 1))
    ws = al.make_tensor(ws_ptr, al.f32, ws2d)

    lane = al.thread_id(0)
    wr = lane // 64
    wl_in = lane % 64
    wl32 = wl_in // 32
    lane_col = wl_in % 32
    lane_k_base = wl32 * 4

    linear_block_id = al.block_id(0)
    split_k_id = linear_block_id % SPLIT_K_SLICES
    tile_block_id = linear_block_id // SPLIT_K_SLICES

    gm32 = wr * 32
    gn32 = al.block_id(1) * 32

    k_start = split_k_id * K_PER_SPLIT
    spatial_offset = split_k_id * N_SPATIAL

    acc = al.full((16,), 0.0, al.f32)
    a_bf16 = al.make_local((4,), al.bf16)
    b_bf16 = al.make_local((4,), al.bf16)

    for k_tile in al.range(K_TILES_SPLIT):
        k_base = k_start + k_tile * 8
        for e in al.range(4):
            k = k_base + lane_k_base + e
            ic = k // KH_KW
            r = k - ic * KH_KW
            kh = r // 7
            kw = r - kh * 7
            m = gm32 + lane_col
            a_bf16[e] = al.convert(W[m, ic, kh, kw], al.bf16)
        for e in al.range(4):
            k = k_base + lane_k_base + e
            ic = k // KH_KW
            r = k - ic * KH_KW
            kh = r // 7
            kw = r - kh * 7
            n = gn32 + lane_col
            if n < N_SPATIAL:
                bv = n // OH_OW
                rv = n - bv * OH_OW
                oh_v = rv // OW
                ow_v = rv - oh_v * OW
                b_bf16[e] = al.convert(X[bv, ic, oh_v + kh, ow_v + kw], al.bf16)
            else:
                b_bf16[e] = al.convert(0.0, al.bf16)
        ap = al.view(a_bf16, al.Tensor((2,), al.u32))
        bp = al.view(b_bf16, al.Tensor((2,), al.u32))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(ap, bp, acc)

    for acc_idx in al.range(16):
        col_flat = gn32 + lane_col
        row_oc = gm32 + 8 * (acc_idx // 4) + wl32 * 4 + (acc_idx % 4)
        if col_flat < N_SPATIAL:
            ws[spatial_offset + col_flat, row_oc] = acc[acc_idx]


@avelang.jit
def finalize_kernel(
    ws_ptr: al.Pointer(al.f32),
    out: al.Tensor((1024, 508, 250), al.f32),
):
    ws2d = al.make_layout((SPLIT_K_SLICES * N_SPATIAL, GEMM_N), (GEMM_N, 1))
    ws = al.make_tensor(ws_ptr, al.f32, ws2d)

    idx = al.block_id(0) * 256 + al.thread_id(0)
    total = N_SPATIAL * GEMM_N

    if idx < total:
        spatial = idx // GEMM_N
        oc = idx - spatial * GEMM_N

        s0 = ws[0 * N_SPATIAL + spatial, oc]
        s1 = ws[1 * N_SPATIAL + spatial, oc]
        val = s0 + s1

        batch = spatial // OH_OW
        rem_s = spatial - batch * OH_OW
        oh_v = rem_s // OW
        ow_v = rem_s - oh_v * OW

        out[batch * 128 + oc, oh_v, ow_v] = val


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

        workspace = torch.empty(
            SPLIT_K_SLICES * N_SPATIAL, OUT_CHANNELS,
            device=x.device, dtype=torch.float32,
        )
        y3 = torch.empty(1024, 508, 250, device=x.device, dtype=torch.float32)

        conv2d_mfma_split_kernel[lambda: ((SPLIT_K_SLICES, N_TILES_PER_SPATIAL, 1), (256, 1, 1))](
            x0, w, workspace.data_ptr()
        )

        total_elems = N_SPATIAL * GEMM_N
        final_grid = (total_elems + 255) // 256
        finalize_kernel[lambda: ((final_grid, 1, 1), (256, 1, 1))](
            workspace.data_ptr(), y3
        )

        y = y3.view(BATCH_SIZE, OUT_CHANNELS, OH, OW)
        if orig_dtype != torch.float32:
            y = y.to(orig_dtype)
        return y
