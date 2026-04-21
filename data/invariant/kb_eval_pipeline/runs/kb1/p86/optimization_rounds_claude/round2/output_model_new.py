import torch
import torch.nn as nn
import substrate
import substrate.language as S

# Constants for MFMA tiling
WARP_SIZE = 64
NUM_WARPS = 4
WARPS_N = 2
GROUP_M = 128
GROUP_N = 128
GROUP_K = 8  # MFMA_K = 8
MFMA_M = 32
MFMA_N = 32
MFMA_K = 8
MFMA_ACC_SIZE = 16
WAVE_REPEAT_M = 2
WAVE_REPEAT_N = 2
WARP_TILE_M = WAVE_REPEAT_M * MFMA_M  # 64
WARP_TILE_N = WAVE_REPEAT_N * MFMA_N  # 64
THREADS = WARP_SIZE * NUM_WARPS  # 256

# Split-K configuration
SPLIT_K_SLICES = 2

INPUT0_SHAPE = (16, 64, 512, 512)
OUTPUT_SHAPE = (16, 128, 512, 512)
DW_WEIGHT_SHAPE = (64, 1, 3, 3)
PW_WEIGHT_SHAPE = (128, 64, 1, 1)

BATCH = 16
IN_CHANNELS = 64
OUT_CHANNELS = 128
HEIGHT = 512
WIDTH = 512
KERNEL_H = 3
KERNEL_W = 3
PAD_H = 1
PAD_W = 1


def _compute_magic_u32_params(divisor: int) -> tuple:
    """Compute host-side magic/shift for q = (mul_hi(magic, n) + n) >> shift."""
    if divisor <= 0 or divisor >= (1 << 32):
        raise ValueError(f"divisor must be in [1, 2^32) (got {divisor})")

    shift = (divisor - 1).bit_length()
    if divisor & (divisor - 1) == 0:
        return 0, shift

    magic = ((1 << (32 + shift)) // divisor) - (1 << 32) + 1
    return magic, shift


@substrate.jit
def _mdiv_u32(numer: S.u32, magic: S.u32, shift: S.u32) -> S.u32:
    prod_hi = S.convert((S.convert(magic, S.u64) * S.convert(numer, S.u64)) >> 32, S.u32)
    return (prod_hi + numer) >> shift


@substrate.jit
def _mdiv_u32_rem(
    numer: S.u32,
    denom: S.u32,
    magic: S.u32,
    shift: S.u32,
) -> (S.u32, S.u32):
    quot = _mdiv_u32(numer, magic, shift)
    rem = numer - quot * denom
    return quot, rem


@substrate.jit
def split_k_conv_kernel(
    X: S.Pointer(S.bf16),
    DW: S.Pointer(S.bf16),
    PW: S.Pointer(S.bf16),
    workspace: S.Pointer(S.f32),
    gemm_m: S.u32,
    gemm_n: S.u32,
    hw_out: S.u32,
    out_w: S.u32,
    in_channels: S.u32,
    kernel_h: S.u32,
    kernel_w: S.u32,
    c_per_split: S.u32,
    hw_out_magic: S.u32,
    hw_out_shift: S.u32,
    out_w_magic: S.u32,
    out_w_shift: S.u32,
    kernel_w_magic: S.u32,
    kernel_w_shift: S.u32,
):
    """Split-K Conv2D kernel using MFMA. Each split computes partial fp32 accumulation."""
    linear_block_id = S.block_id(0)
    n_groups = (gemm_n + GROUP_N - 1) // GROUP_N

    # Block decomposition: linear_block_id = tile_block_id * SPLIT_K_SLICES + split_k_id
    tile_block_id = linear_block_id // SPLIT_K_SLICES
    split_k_id = linear_block_id - tile_block_id * SPLIT_K_SLICES

    group_m = tile_block_id // n_groups
    group_n = tile_block_id - group_m * n_groups

    tid = S.thread_id(0)
    wid = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    warp_row = wid // WARPS_N
    warp_col = wid % WARPS_N

    lane_col = lane % MFMA_N
    lane_k_base = (lane // MFMA_N) * 4

    group_m_base = group_m * GROUP_M
    group_n_base = group_n * GROUP_N

    # Channel partition for this split
    c_start = split_k_id * c_per_split
    c_end = in_channels
    # Manual min: if c_start + c_per_split < in_channels, use c_start + c_per_split
    c_end_if = c_start + c_per_split
    if c_end_if < in_channels:
        c_end = c_end_if

    kernel_area = kernel_h * kernel_w
    # Total K for this split
    split_k_size = (c_end - c_start) * kernel_area

    # Create tensor views for NCHW input
    x_tensor = S.make_tensor(
        X,
        S.bf16,
        S.make_layout((BATCH * HEIGHT * WIDTH * IN_CHANNELS,), (1,)),
    )
    dw_tensor = S.make_tensor(
        DW,
        S.bf16,
        S.make_layout((IN_CHANNELS * KERNEL_H * KERNEL_W,), (1,)),
    )
    pw_tensor = S.make_tensor(
        PW,
        S.bf16,
        S.make_layout((OUT_CHANNELS * IN_CHANNELS,), (1,)),
    )

    # Accumulator: 2x2 array of 32x32 MFMA tiles, each with 16 f32 values
    acc = S.make_local((WAVE_REPEAT_M, WAVE_REPEAT_N, MFMA_ACC_SIZE), S.f32)

    zero_f32 = S.convert(0.0, S.f32)

    # Initialize accumulators
    for tm in S.range(WAVE_REPEAT_M):
        for tn in S.range(WAVE_REPEAT_N):
            for acc_idx in S.range(MFMA_ACC_SIZE):
                acc[tm, tn, acc_idx] = zero_f32

    # K tiles within this split's channel range
    k_tiles = (split_k_size + MFMA_K - 1) // MFMA_K

    # A and B fragments
    a_frag = S.make_local((WAVE_REPEAT_M, 4), S.bf16)
    b_frag = S.make_local((WAVE_REPEAT_N, 4), S.bf16)

    for k_tile in S.range(k_tiles):
        k_base = k_tile * MFMA_K

        # Load A fragments (input tensor)
        for tm in S.range(WAVE_REPEAT_M):
            m = group_m_base + warp_row * WARP_TILE_M + tm * MFMA_M + lane_col
            for e in S.range(4):
                k = k_base + lane_k_base + e
                # Convert m to (batch, hw_idx) and k to (ic, kh, kw)
                if m < gemm_m and k < split_k_size:
                    batch, hw_idx = _mdiv_u32_rem(m, hw_out, hw_out_magic, hw_out_shift)
                    oh, ow = _mdiv_u32_rem(hw_idx, out_w, out_w_magic, out_w_shift)

                    # K linearization: k_idx = c * kernel_area + kh * kernel_w + kw
                    # Within split: c is relative to c_start
                    c_rel = k // kernel_area
                    spatial = k - c_rel * kernel_area
                    kh = spatial // kernel_w
                    kw = spatial - kh * kernel_w

                    # Absolute channel index
                    ic = c_start + c_rel

                    ih = oh + kh - PAD_H
                    iw = ow + kw - PAD_W

                    if ih >= 0 and ih < HEIGHT and iw >= 0 and iw < WIDTH and ic < c_end:
                        # X is NCHW layout
                        x_idx = ((batch * IN_CHANNELS + ic) * HEIGHT + ih) * WIDTH + iw
                        a_frag[tm, e] = x_tensor[x_idx]
                    else:
                        a_frag[tm, e] = S.convert(0.0, S.bf16)
                else:
                    a_frag[tm, e] = S.convert(0.0, S.bf16)

        # Load B fragments (fused weights)
        for tn in S.range(WAVE_REPEAT_N):
            n = group_n_base + warp_col * WARP_TILE_N + tn * MFMA_N + lane_col
            for e in S.range(4):
                k = k_base + lane_k_base + e
                if n < gemm_n and k < split_k_size:
                    # K linearization: k_idx = c * kernel_area + kh * kernel_w + kw
                    c_rel = k // kernel_area
                    spatial = k - c_rel * kernel_area
                    kh = spatial // kernel_w
                    kw = spatial - kh * kernel_w

                    # Absolute channel index
                    ic = c_start + c_rel
                    oc = n

                    if ic < c_end:
                        # Fused weight: DW[ic, 0, kh, kw] * PW[oc, ic, 0, 0]
                        # DW is (64, 1, 3, 3) in OIHW format
                        dw_idx = ic * KERNEL_H * KERNEL_W + kh * KERNEL_W + kw
                        # PW is (128, 64, 1, 1) in OIHW format
                        pw_idx = oc * IN_CHANNELS + ic

                        dw_val = dw_tensor[dw_idx]
                        pw_val = pw_tensor[pw_idx]

                        # Multiply depthwise and pointwise weights
                        b_frag[tn, e] = S.convert(
                            S.convert(dw_val, S.f32) * S.convert(pw_val, S.f32),
                            S.bf16
                        )
                    else:
                        b_frag[tn, e] = S.convert(0.0, S.bf16)
                else:
                    b_frag[tn, e] = S.convert(0.0, S.bf16)

        # Perform MFMA operations
        for tm in S.range(WAVE_REPEAT_M):
            a_frag_view = S.view(a_frag[tm], S.Tensor((1, 4, 1), S.bf16))
            for tn in S.range(WAVE_REPEAT_N):
                b_frag_view = S.view(b_frag[tn], S.Tensor((1, 4, 1), S.bf16))
                acc[tm, tn] = S.amdgpu.mfma_32x32x8_bf16_f32(
                    a_frag_view[0], b_frag_view[0], acc[tm, tn]
                )

    # Writeback to workspace using buffer_atomic_add_f32
    # Workspace layout: GEMM-major linear_idx = row * gemm_n + col
    # row = batch * hw_out + hw_idx, col = out_channel

    for tm in S.range(WAVE_REPEAT_M):
        tile_row_base = group_m_base + warp_row * WARP_TILE_M + tm * MFMA_M
        for tn in S.range(WAVE_REPEAT_N):
            tile_col_base = group_n_base + warp_col * WARP_TILE_N + tn * MFMA_N
            for acc_idx in S.range(MFMA_ACC_SIZE):
                col = tile_col_base + lane_col
                row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // MFMA_N) + (acc_idx % 4)

                if row < gemm_m and col < gemm_n:
                    # GEMM-major linear index for workspace
                    linear_idx = row * gemm_n + col
                    # Atomic add to workspace
                    S.amdgpu.buffer_atomic_add_f32(acc[tm, tn, acc_idx], workspace, linear_idx)


@substrate.jit
def store_kernel(
    workspace: S.Pointer(S.f32),
    Y: S.Pointer(S.bf16),
    gemm_m: S.u32,
    gemm_n: S.u32,
    hw_out: S.u32,
    out_w: S.u32,
    hw_out_magic: S.u32,
    hw_out_shift: S.u32,
    out_w_magic: S.u32,
    out_w_shift: S.u32,
):
    """Convert fp32 workspace back to bf16 NCHW output."""
    linear_block_id = S.block_id(0)
    tid = S.thread_id(0)

    # Each thread handles multiple output elements
    # Grid: (gemm_m * gemm_n + THREADS - 1) // THREADS blocks
    base_idx = linear_block_id * THREADS + tid

    if base_idx < gemm_m * gemm_n:
        row = base_idx // gemm_n
        col = base_idx - row * gemm_n

        # Read from workspace
        val = workspace[base_idx]

        # Convert to NCHW output layout
        batch, hw_idx = _mdiv_u32_rem(row, hw_out, hw_out_magic, hw_out_shift)
        oh, ow = _mdiv_u32_rem(hw_idx, out_w, out_w_magic, out_w_shift)
        oc = col

        y_idx = ((batch * OUT_CHANNELS + oc) * HEIGHT + oh) * WIDTH + ow
        Y[y_idx] = S.convert(val, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        # Precompute magic numbers
        self.hw_out_magic, self.hw_out_shift = _compute_magic_u32_params(HEIGHT * WIDTH)
        self.out_w_magic, self.out_w_shift = _compute_magic_u32_params(WIDTH)
        self.kernel_w_magic, self.kernel_w_shift = _compute_magic_u32_params(KERNEL_W)

        # Channel partition for split-K
        self.c_per_split = (IN_CHANNELS + SPLIT_K_SLICES - 1) // SPLIT_K_SLICES

        # Cache for fused weights and workspace
        self._cached_dw_ptr = None
        self._cached_pw_ptr = None
        self._fused_weights = None
        self._workspace = None
        self._workspace_size = 0

    def _ensure_fused_weights(self, dw, pw, device):
        """Build or return cached fused weights."""
        dw_ptr = dw.data_ptr()
        pw_ptr = pw.data_ptr()

        if self._fused_weights is not None and self._cached_dw_ptr == dw_ptr and self._cached_pw_ptr == pw_ptr:
            return self._fused_weights

        # Store DW and PW separately for the kernel to use
        dw_bf16 = dw.to(torch.bfloat16)
        pw_bf16 = pw.to(torch.bfloat16)

        self._fused_weights = (dw_bf16, pw_bf16)
        self._cached_dw_ptr = dw_ptr
        self._cached_pw_ptr = pw_ptr

        return self._fused_weights

    def _ensure_workspace(self, device, gemm_m, gemm_n):
        """Ensure workspace tensor exists with correct size."""
        workspace_size = gemm_m * gemm_n
        if self._workspace is None or self._workspace_size != workspace_size:
            self._workspace = torch.zeros(workspace_size, device=device, dtype=torch.float32)
            self._workspace_size = workspace_size
        else:
            self._workspace.zero_()
        return self._workspace

    def forward(self, x):
        if tuple(x.shape) != INPUT0_SHAPE or x.dtype != torch.float32:
            raise RuntimeError('This fused kernel only supports the benchmark input shape and dtype.')

        x0 = x.contiguous()
        dw = self.depthwise.weight.to(device=x.device, dtype=torch.bfloat16).contiguous()
        pw = self.pointwise.weight.to(device=x.device, dtype=torch.bfloat16).contiguous()

        # Prepare weights
        dw_bf16, pw_bf16 = self._ensure_fused_weights(dw, pw, x.device)

        # Output tensor
        y = torch.empty(OUTPUT_SHAPE, device=x.device, dtype=torch.bfloat16)

        # Convert input to bf16
        x_bf16 = x0.to(torch.bfloat16)

        # Launch configuration
        gemm_m = BATCH * HEIGHT * WIDTH
        gemm_n = OUT_CHANNELS

        m_groups = (gemm_m + GROUP_M - 1) // GROUP_M
        n_groups = (gemm_n + GROUP_N - 1) // GROUP_N

        # Workspace for partial accumulations
        workspace = self._ensure_workspace(x.device, gemm_m, gemm_n)

        hw_out = HEIGHT * WIDTH

        # Split-K kernel launch: extend grid in x by SPLIT_K_SLICES
        tile_blocks = m_groups * n_groups
        split_k_grid = (tile_blocks * SPLIT_K_SLICES, 1, 1)
        block = (THREADS, 1, 1)

        split_k_conv_kernel[lambda: (split_k_grid, block)](
            x_bf16,
            dw_bf16,
            pw_bf16,
            workspace,
            gemm_m,
            gemm_n,
            hw_out,
            WIDTH,
            IN_CHANNELS,
            KERNEL_H,
            KERNEL_W,
            self.c_per_split,
            self.hw_out_magic,
            self.hw_out_shift,
            self.out_w_magic,
            self.out_w_shift,
            self.kernel_w_magic,
            self.kernel_w_shift,
        )

        # Store kernel launch: convert workspace to final bf16 output
        store_grid = ((gemm_m * gemm_n + THREADS - 1) // THREADS, 1, 1)

        store_kernel[lambda: (store_grid, block)](
            workspace,
            y,
            gemm_m,
            gemm_n,
            hw_out,
            WIDTH,
            self.hw_out_magic,
            self.hw_out_shift,
            self.out_w_magic,
            self.out_w_shift,
        )

        return y.to(torch.float32)
