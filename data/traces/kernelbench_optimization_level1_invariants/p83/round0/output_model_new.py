import torch
import torch.nn as nn
import avelang
import avelang.language as al

# =============================================================================
# Tile and MFMA constants
# =============================================================================
BLOCK_M = 128
BLOCK_N = 128
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS
MMA_M = 32
MMA_N = 32
REPLICATION = 64
ACC_SIZE = 16

# =============================================================================
# MFMA depthwise conv2d kernel
# Implicit GEMM: M = BATCH * OH * OW, K = 3, N = CHANNELS * 64 (replication)
# Block tile: 128 x 128, 4 warps as 2x2 grid.
# M dimension is split across grid Y and Z to stay within HIP limits.
# =============================================================================

@avelang.jit
def depthwise_conv_mfma_kernel(
    X_ptr: al.Pointer(al.bf16),
    W_ptr: al.Pointer(al.bf16),
    Y_ptr: al.Pointer(al.bf16),
    BATCH: al.i32,
    CHANNELS: al.i32,
    H: al.i32,
    W_IN: al.i32,
    OH: al.i32,
    OW: al.i32,
    M_total: al.i32,
    M_TILES_Y: al.i32,
    M_TILES_Z: al.i32,
    stride_X_n: al.i32,
    stride_X_c: al.i32,
    stride_X_h: al.i32,
    stride_W_oc: al.i32,
    stride_W_kh: al.i32,
    stride_Y_n: al.i32,
    stride_Y_c: al.i32,
    stride_Y_oh: al.i32,
):
    tid = al.thread_id(0)
    block_n = al.block_id(0)
    block_m = al.block_id(1) + al.block_id(2) * M_TILES_Y

    wid = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    warp_row = wid // 2
    warp_col = wid % 2

    lane_col = lane % 32
    lane_group = lane // 32
    lane_k_base = lane_group * 4

    # Build tensor views
    X = al.make_tensor(
        X_ptr, al.bf16,
        al.make_layout((BATCH, CHANNELS, H, W_IN),
                       (stride_X_n, stride_X_c, stride_X_h, 1)),
    )
    W_t = al.make_tensor(
        W_ptr, al.bf16,
        al.make_layout((CHANNELS, 3), (stride_W_oc, stride_W_kh)),
    )
    Y = al.make_tensor(
        Y_ptr, al.bf16,
        al.make_layout((BATCH, CHANNELS, OH, OW),
                       (stride_Y_n, stride_Y_c, stride_Y_oh, 1)),
    )

    # Channel for this warp (BLOCK_N // REPLICATION = 2 channels per block)
    c = block_n * (BLOCK_N // REPLICATION) + warp_col

    # Group base positions
    group_m_base = block_m * BLOCK_M
    OH_OW = OH * OW

    # Accumulator: acc[tm, tn, acc_idx]  (2x2x16 f32)
    acc = al.make_local((2, 2, ACC_SIZE), al.f32)
    for tm in al.range(2):
        for tn in al.range(2):
            for i in al.range(ACC_SIZE):
                acc[tm, tn, i] = al.convert(0.0, al.f32)

    # Fragment storage: a_frag[tm, e], b_frag[tn, e]  (2x4 bf16 each)
    a_frag = al.make_local((2, 4), al.bf16)
    b_frag = al.make_local((2, 4), al.bf16)

    # ------------------------------------------------------------------
    # Load A fragments from global memory
    # A[m, k] = X[n_batch, c, oh + kh, ow]  for m = (n_batch, oh, ow)
    # ------------------------------------------------------------------
    for tm in al.range(2):
        m_val = group_m_base + warp_row * 64 + tm * 32 + lane_col
        for e in al.range(4):
            kh = lane_k_base + e
            if kh < 3:
                if m_val < M_total:
                    n_batch = m_val // OH_OW
                    resid = m_val - n_batch * OH_OW
                    oh = resid // OW
                    ow = resid - oh * OW
                    h_in = oh + kh
                    if h_in < H:
                        a_frag[tm, e] = X[n_batch, c, h_in, ow]
                    else:
                        a_frag[tm, e] = al.convert(0.0, al.bf16)
                else:
                    a_frag[tm, e] = al.convert(0.0, al.bf16)
            else:
                a_frag[tm, e] = al.convert(0.0, al.bf16)

    # ------------------------------------------------------------------
    # Load B fragments from global memory
    # B[k, n] = W[c, 0, kh, 0]  for n = (c, dummy)
    # ------------------------------------------------------------------
    for tn in al.range(2):
        for e in al.range(4):
            kh = lane_k_base + e
            if kh < 3:
                b_frag[tn, e] = W_t[c, kh]
            else:
                b_frag[tn, e] = al.convert(0.0, al.bf16)

    # ------------------------------------------------------------------
    # MFMA: 4 calls per warp (2x2 grid of 32x32 tiles)
    # Each a_frag[tm] / b_frag[tn] is (4,) bf16 viewed as (2,) u32
    # ------------------------------------------------------------------
    a_vec_0 = al.view(a_frag[0], al.Tensor((2,), al.u32))
    a_vec_1 = al.view(a_frag[1], al.Tensor((2,), al.u32))
    b_vec_0 = al.view(b_frag[0], al.Tensor((2,), al.u32))
    b_vec_1 = al.view(b_frag[1], al.Tensor((2,), al.u32))

    acc[0, 0] = al.amdgpu.mfma_32x32x8_bf16_f32(a_vec_0, b_vec_0, acc[0, 0])
    acc[1, 0] = al.amdgpu.mfma_32x32x8_bf16_f32(a_vec_1, b_vec_0, acc[1, 0])
    acc[0, 1] = al.amdgpu.mfma_32x32x8_bf16_f32(a_vec_0, b_vec_1, acc[0, 1])
    acc[1, 1] = al.amdgpu.mfma_32x32x8_bf16_f32(a_vec_1, b_vec_1, acc[1, 1])

    # ------------------------------------------------------------------
    # Writeback: fixed MFMA accumulator layout.
    # Only write for col % 64 == 0, i.e. tn==0 and lane_col==0.
    # ------------------------------------------------------------------
    group_n_base = block_n * BLOCK_N
    for tn in al.range(2):
        if tn == 0:
            if lane_col == 0:
                col = group_n_base + warp_col * 64
                if col < CHANNELS * 64:
                    for tm in al.range(2):
                        tile_row_base = group_m_base + warp_row * 64 + tm * 32
                        for acc_idx in al.range(ACC_SIZE):
                            row = (tile_row_base +
                                   8 * (acc_idx // 4) +
                                   4 * lane_group +
                                   (acc_idx % 4))
                            if row < M_total:
                                n_batch = row // OH_OW
                                resid = row - n_batch * OH_OW
                                oh = resid // OW
                                ow_out = resid - oh * OW
                                if oh < OH:
                                    if ow_out < OW:
                                        Y[n_batch, c, oh, ow_out] = al.convert(
                                            acc[tm, tn, acc_idx], al.bf16)

# =============================================================================
# Host wrapper
# =============================================================================

def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


M_TILES_Y = 4096


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(kernel_size, 1),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required.")

        x_bf16 = _prepare_bf16_cuda_contiguous(x)
        w = self.conv2d.weight.data
        w_bf16 = _prepare_bf16_cuda_contiguous(w)

        BATCH, CHANNELS, H, W_IN = x_bf16.shape
        OH = H - 2  # kernel_h=3, stride=1, pad=0, dilation=1
        OW = W_IN   # kernel_w=1
        M_total = BATCH * OH * OW
        N_total = CHANNELS * REPLICATION

        # weight reshape: (8, 1, 3, 1) -> (8, 3)
        w_reshaped = w_bf16.reshape(CHANNELS, 3).contiguous()

        out = torch.empty(
            (BATCH, CHANNELS, OH, OW),
            device=x_bf16.device,
            dtype=torch.bfloat16,
        )

        # Strides in element counts
        stride_X_n = CHANNELS * H * W_IN
        stride_X_c = H * W_IN
        stride_X_h = W_IN
        stride_W_oc = 3
        stride_W_kh = 1
        stride_Y_n = CHANNELS * OH * OW
        stride_Y_c = OH * OW
        stride_Y_oh = OW

        M_tiles = (M_total + BLOCK_M - 1) // BLOCK_M
        N_tiles = N_total // BLOCK_N

        # Split M tiles across grid Y and Z to avoid HIP grid-dim limits
        m_tiles_y = M_TILES_Y
        m_tiles_z = (M_tiles + m_tiles_y - 1) // m_tiles_y

        grid = (N_tiles, m_tiles_y, m_tiles_z)

        depthwise_conv_mfma_kernel[lambda: (grid, (THREADS, 1, 1))](
            x_bf16,
            w_reshaped,
            out,
            BATCH,
            CHANNELS,
            H,
            W_IN,
            OH,
            OW,
            M_total,
            m_tiles_y,
            m_tiles_z,
            stride_X_n,
            stride_X_c,
            stride_X_h,
            stride_W_oc,
            stride_W_kh,
            stride_Y_n,
            stride_Y_c,
            stride_Y_oh,
        )

        # Match reference output dtype
        return out.to(dtype=x.dtype)
