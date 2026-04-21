"""MFMA-optimized Conv2D kernel with Split-K reduction.

This implementation uses MFMA 32x32x8 bf16->f32 instructions for the computation
with Split-K parallel reduction across the channel dimension.

Split-K approach:
- SPLIT_K_SLICES = 2 splits across input channels
- Each split computes partial fp32 accumulation
- Reduction via buffer_atomic_add_f32 into shared fp32 workspace
- Final kernel converts fp32 workspace to bf16 NCHW output
"""

import torch
import torch.nn as nn
import substrate
import substrate.language as S

# Constants for the fixed kernel shapes
BATCH = 64
IN_CHANNELS = 128
OUT_CHANNELS = 128
IN_H = 256
IN_W = 512
OUT_H = 254
OUT_W = 510
KERNEL_H = 3
KERNEL_W = 3
KERNEL_AREA = KERNEL_H * KERNEL_W  # 9

# Warp and block configuration
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS

# Tile dimensions (128x128 output tile)
GROUP_M = 128
GROUP_N = 128
GROUP_K = 32

# MFMA configuration
MFMA_M = 32
MFMA_N = 32
MFMA_K = 8
MFMA_ACC_SIZE = 16

# Wave tile configuration
WAVE_REPEAT_M = 2
WAVE_REPEAT_N = 2
WARP_TILE_M = WAVE_REPEAT_M * MFMA_M  # 64
WARP_TILE_N = WAVE_REPEAT_N * MFMA_N  # 64

# Warp arrangement within CTA
WARPS_N = 2

# Split-K configuration
SPLIT_K_SLICES = 2

# Byte sizes
BF16_BYTES = 2
F32_BYTES = 4

# K block slices for shared memory
K_BLOCK_SLICES = GROUP_K // MFMA_K  # 4

# Load configuration
LOAD_GROUP_ROWS = GROUP_M
LOAD_GROUP_COUNT = THREADS // LOAD_GROUP_ROWS
LOADS_PER_THREAD = K_BLOCK_SLICES // LOAD_GROUP_COUNT

# Shared memory sizes
A_SHM_WORDS = K_BLOCK_SLICES * GROUP_M * 4
B_SHM_WORDS = K_BLOCK_SLICES * GROUP_N * 4
SHM_TOTAL_WORDS = A_SHM_WORDS + B_SHM_WORDS

# Output transpose configuration
OUTPUT_TRANSPOSE_TILE = 32
OUTPUT_TRANSPOSE_THREADS = 256
OUTPUT_TRANSPOSE_ROWS_PER_THREAD = 4


def _compute_magic_u32_params(divisor: int) -> tuple:
    """Compute host-side magic/shift for unsigned division."""
    if divisor <= 0 or divisor >= (1 << 32):
        raise ValueError(f"divisor must be in [1, 2^32) (got {divisor})")

    shift = (divisor - 1).bit_length()
    if divisor & (divisor - 1) == 0:
        return 0, shift

    magic = ((1 << (32 + shift)) // divisor) - (1 << 32) + 1
    return magic, shift


@substrate.jit
def _mdiv_u32(numer: S.u32, magic: S.u32, shift: S.u32) -> S.u32:
    """Unsigned division using magic number multiplication."""
    prod_hi = S.convert((S.convert(magic, S.u64) * S.convert(numer, S.u64)) >> 32, S.u32)
    return (prod_hi + numer) >> shift


@substrate.jit
def _mdiv_u32_rem(numer: S.u32, denom: S.u32, magic: S.u32, shift: S.u32) -> (S.u32, S.u32):
    """Unsigned division with remainder using magic number multiplication."""
    quot = _mdiv_u32(numer, magic, shift)
    rem = numer - quot * denom
    return quot, rem


@substrate.jit
def _mfma_permute_row(row: S.u32) -> S.u32:
    """Apply MFMA 32x32 row permutation.

    The MFMA 32x32x8 instruction has a specific lane-to-row mapping.
    This function computes the permutation: lane_row -> matrix_row.
    """
    high = (row >> 2) & 7
    rotated = ((high & 1) << 2) | (high >> 1)
    return (row & 3) | (rotated << 2)


@substrate.jit
def _mfma_permute_row_inv(permuted_row: S.u32) -> S.u32:
    """Inverse of MFMA 32x32 row permutation.

    Inverts the rotation on the high 3 bits.
    """
    low_bits = permuted_row & 3
    rotated = (permuted_row >> 2) & 7
    high = ((rotated << 1) & 7) | (rotated >> 2)
    return low_bits | (high << 2)


@substrate.jit
def _igemm_splitk_kernel(
    input_nchw: S.Pointer(S.bf16),
    weight_oihw: S.Pointer(S.bf16),
    out_accum: S.Pointer(S.f32),
    in_h: S.u32,
    in_w: S.u32,
    batch_size: S.u32,
    out_channels: S.u32,
    in_channels: S.u32,
    out_h: S.u32,
    out_w: S.u32,
    kernel_h: S.u32,
    kernel_w: S.u32,
    hw_out_magic: S.u32,
    hw_out_shift: S.u32,
    out_w_magic: S.u32,
    out_w_shift: S.u32,
    split_channels_magic: S.u32,
    split_channels_shift: S.u32,
    kernel_w_magic: S.u32,
    kernel_w_shift: S.u32,
    kernel_area_magic: S.u32,
    kernel_area_shift: S.u32,
    n_groups_magic: S.u32,
    n_groups_shift: S.u32,
):
    """Implicit-GEMM kernel with Split-K reduction using MFMA 32x32x8 bf16."""
    gemm_m = batch_size * out_h * out_w
    gemm_n = out_channels
    kernel_area = kernel_h * kernel_w
    hw_out = out_h * out_w

    linear_block_id = S.block_id(0)
    split_k_id = linear_block_id & (SPLIT_K_SLICES - 1)
    tile_block_id = linear_block_id >> 1
    n_groups = (gemm_n + GROUP_N - 1) // GROUP_N
    group_m, group_n = _mdiv_u32_rem(tile_block_id, n_groups, n_groups_magic, n_groups_shift)

    tid = S.thread_id(0)
    wid = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    warp_row = wid // WARPS_N
    warp_col = wid % WARPS_N
    lane_col = lane % MFMA_N
    group_m_base = group_m * GROUP_M
    group_n_base = group_n * GROUP_N

    # Channel partition for Split-K
    c_per_split = (in_channels + SPLIT_K_SLICES - 1) // SPLIT_K_SLICES
    c_start = split_k_id * c_per_split
    split_channels = c_per_split
    k_slice_span = split_channels * kernel_area

    # Create NCHW tensor view for input
    input_tensor = S.make_tensor(
        input_nchw, S.bf16,
        S.make_layout((batch_size, in_channels, in_h, in_w),
                      (in_channels * in_h * in_w, in_h * in_w, in_w, 1))
    )
    # Create OIHW tensor view for weights
    weight_tensor = S.make_tensor(
        weight_oihw, S.bf16,
        S.make_layout((out_channels, in_channels, kernel_h, kernel_w),
                      (in_channels * kernel_h * kernel_w, kernel_h * kernel_w, kernel_w, 1))
    )

    # Create linear tensor views for buffer operations
    input_linear = S.make_tensor(
        input_nchw, S.bf16,
        S.make_layout((batch_size * in_h * in_w * in_channels,), (1,))
    )
    weight_linear = S.make_tensor(
        weight_oihw, S.bf16,
        S.make_layout((out_channels * in_channels * kernel_area,), (1,))
    )
    out_accum_linear = S.make_tensor(
        out_accum, S.f32,
        S.make_layout((gemm_m * gemm_n,), (1,))
    )

    # Create resource descriptors for buffer operations
    input_rsrc = S.amdgpu.make_rsrc(
        input_linear, batch_size * in_h * in_w * in_channels * BF16_BYTES
    )
    weight_rsrc = S.amdgpu.make_rsrc(
        weight_linear, out_channels * in_channels * kernel_area * BF16_BYTES
    )
    out_accum_rsrc = S.amdgpu.make_rsrc(out_accum_linear, gemm_m * gemm_n * F32_BYTES)

    zero_f32 = S.convert(0.0, S.f32)
    zero_u32 = S.convert(0, S.u32)

    # Initialize accumulators
    acc = S.make_local((WAVE_REPEAT_M, WAVE_REPEAT_N, MFMA_ACC_SIZE), S.f32)
    for tm in S.range(WAVE_REPEAT_M):
        for tn in S.range(WAVE_REPEAT_N):
            for acc_idx in S.range(MFMA_ACC_SIZE):
                acc[tm, tn, acc_idx] = zero_f32

    # Accumulate over K dimension (split channels * kernel_area)
    k_blocks = (k_slice_span + GROUP_K - 1) // GROUP_K

    pack = lane // MFMA_N
    # Base row for this warp (0 or 64 for warp_row 0 or 1)
    row_base = warp_row * WARP_TILE_M
    # Base column for this warp (0 or 64 for warp_col 0 or 1)
    col_base = warp_col * WARP_TILE_N

    # Local storage for fragments - 4 bf16 values per lane for MFMA 32x32x8
    a_frag_bf16 = S.make_local((WAVE_REPEAT_M, 4), S.bf16)
    b_frag_bf16 = S.make_local((WAVE_REPEAT_N, 4), S.bf16)

    for k_block in S.range(k_blocks):
        k_base = k_block * GROUP_K

        # Process K within this block (GROUP_K = 32, but MFMA processes K=8 at a time)
        for k_iter in S.range(K_BLOCK_SLICES):
            k_offset = k_iter * MFMA_K

            # Zero initialize fragments
            for tm in S.range(WAVE_REPEAT_M):
                for ki in S.range(4):
                    a_frag_bf16[tm, ki] = S.convert(0.0, S.bf16)
            for tn in S.range(WAVE_REPEAT_N):
                for ki in S.range(4):
                    b_frag_bf16[tn, ki] = S.convert(0.0, S.bf16)

            # Load A matrix (input) - NCHW layout
            # Each lane loads 4 K values at indices [pack*4, pack*4+1, pack*4+2, pack*4+3]
            # Permutation is applied within each 32-row MFMA block
            for tm in S.range(WAVE_REPEAT_M):
                # Row within this MFMA block (0-31), permuted according to lane
                mfma_row_in_block = _mfma_permute_row(lane_col)
                row_local = row_base + tm * MFMA_M + mfma_row_in_block
                row = group_m_base + row_local

                if row < gemm_m:
                    batch_idx, hw_rem = _mdiv_u32_rem(row, hw_out, hw_out_magic, hw_out_shift)
                    h_out_idx, w_out_idx = _mdiv_u32_rem(hw_rem, out_w, out_w_magic, out_w_shift)

                    for ki in S.range(4):
                        k_idx = k_base + k_offset + pack * 4 + ki
                        if k_idx < k_slice_span:
                            c, spatial = _mdiv_u32_rem(k_idx, kernel_area, kernel_area_magic, kernel_area_shift)
                            kh, kw = _mdiv_u32_rem(spatial, kernel_w, kernel_w_magic, kernel_w_shift)

                            c_global = c_start + c
                            if c_global < in_channels:
                                ih = h_out_idx + kh
                                iw = w_out_idx + kw
                                if ih < in_h and iw < in_w:
                                    # Load single bf16 value via tensor view
                                    a_frag_bf16[tm, ki] = input_tensor[batch_idx, c_global, ih, iw]

            # Load B matrix (weights) - OIHW layout
            for tn in S.range(WAVE_REPEAT_N):
                col_local = col_base + tn * MFMA_N + lane_col
                col = group_n_base + col_local

                if col < gemm_n:
                    for ki in S.range(4):
                        k_idx = k_base + k_offset + pack * 4 + ki
                        if k_idx < k_slice_span:
                            c, spatial = _mdiv_u32_rem(k_idx, kernel_area, kernel_area_magic, kernel_area_shift)
                            kh, kw = _mdiv_u32_rem(spatial, kernel_w, kernel_w_magic, kernel_w_shift)

                            c_global = c_start + c
                            if c_global < in_channels:
                                b_frag_bf16[tn, ki] = weight_tensor[col, c_global, kh, kw]

            # Execute MFMA operations
            # View 4 bf16 values as tensor for MFMA 32x32x8
            for tm in S.range(WAVE_REPEAT_M):
                a_view = S.view(a_frag_bf16[tm], S.Tensor((1, 4, 1), S.bf16))
                for tn in S.range(WAVE_REPEAT_N):
                    b_view = S.view(b_frag_bf16[tn], S.Tensor((1, 4, 1), S.bf16))
                    acc[tm, tn] = S.amdgpu.mfma_32x32x8_bf16_f32(
                        a_view[0], b_view[0], acc[tm, tn]
                    )

    # Writeback: each thread writes its accumulated values to the workspace
    # For MFMA 32x32x8 with 64 lanes, each lane holds 16 f32 values
    # Layout: lane l holds C[16*(l//32) + t, l%32] for t in 0..15
    # No permutation needed for output - stored directly
    for tm in S.range(WAVE_REPEAT_M):
        for tn in S.range(WAVE_REPEAT_N):
            for acc_idx in S.range(MFMA_ACC_SIZE):
                # MFMA 32x32 layout: lane l holds C[16*(l//32) + t, l%32] for t in 0..15
                mfma_row = 16 * (lane // 32) + acc_idx
                mfma_col = lane % 32

                # Map to warp tile coordinates (include warp base offsets)
                row_local = row_base + tm * MFMA_M + mfma_row
                col_local = col_base + tn * MFMA_N + mfma_col

                # Map to global coordinates
                row = group_m_base + row_local
                col = group_n_base + col_local

                if row < gemm_m and col < gemm_n:
                    linear_idx = row * gemm_n + col
                    byte_offset = linear_idx * F32_BYTES
                    S.amdgpu.buffer_atomic_add_f32(
                        acc[tm, tn, acc_idx], out_accum_rsrc, byte_offset, zero_u32, 0
                    )


@substrate.jit
def _store_output_kernel(
    out_accum: S.Pointer(S.f32),
    out: S.Pointer(S.bf16),
    batch_size: S.u32,
    out_channels: S.u32,
    out_h: S.u32,
    out_w: S.u32,
):
    """Convert fp32 GEMM-major workspace to bf16 NCHW output."""
    hw_out = out_h * out_w
    hw_tiles = (hw_out + OUTPUT_TRANSPOSE_TILE - 1) // OUTPUT_TRANSPOSE_TILE
    channel_tiles = (out_channels + OUTPUT_TRANSPOSE_TILE - 1) // OUTPUT_TRANSPOSE_TILE

    linear_block_id = S.block_id(0)
    tiles_per_batch = hw_tiles * channel_tiles
    batch = linear_block_id // tiles_per_batch
    tile_rem = linear_block_id - batch * tiles_per_batch
    tile_hw = tile_rem // channel_tiles
    tile_channel = tile_rem - tile_hw * channel_tiles

    tid = S.thread_id(0)
    local_col = tid & 31
    local_row_base = (tid >> 5) * OUTPUT_TRANSPOSE_ROWS_PER_THREAD
    src_hw_base = tile_hw * OUTPUT_TRANSPOSE_TILE + local_row_base
    src_channel = tile_channel * OUTPUT_TRANSPOSE_TILE + local_col

    out_accum_matrix = S.make_tensor(
        out_accum, S.f32,
        S.make_layout((batch_size * hw_out, out_channels), (out_channels, 1))
    )
    out_tensor = S.make_tensor(
        out, S.bf16,
        S.make_layout((batch_size, out_channels, hw_out), (out_channels * hw_out, hw_out, 1))
    )

    for i in S.range(OUTPUT_TRANSPOSE_ROWS_PER_THREAD):
        src_hw = src_hw_base + i
        if batch < batch_size and src_hw < hw_out and src_channel < out_channels:
            val = out_accum_matrix[batch * hw_out + src_hw, src_channel]
            out_tensor[batch, src_channel, src_hw] = S.convert(val, S.bf16)


def _igemm_launch_config():
    """Compute launch configuration for Split-K GEMM kernel."""
    gemm_m = BATCH * OUT_H * OUT_W
    gemm_n = OUT_CHANNELS
    m_groups = (gemm_m + GROUP_M - 1) // GROUP_M
    n_groups = (gemm_n + GROUP_N - 1) // GROUP_N
    grid = (m_groups * n_groups * SPLIT_K_SLICES, 1, 1)
    block = (THREADS, 1, 1)
    return lambda: (grid, block)


def _store_launch_config():
    """Compute launch configuration for output store kernel."""
    hw_out = OUT_H * OUT_W
    hw_tiles = (hw_out + OUTPUT_TRANSPOSE_TILE - 1) // OUTPUT_TRANSPOSE_TILE
    channel_tiles = (OUT_CHANNELS + OUTPUT_TRANSPOSE_TILE - 1) // OUTPUT_TRANSPOSE_TILE
    grid = (BATCH * hw_tiles * channel_tiles, 1, 1)
    block = (OUTPUT_TRANSPOSE_THREADS, 1, 1)
    return lambda: (grid, block)


class ModelNew(nn.Module):
    """MFMA-optimized Conv2D with Split-K reduction."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                                stride=stride, padding=padding, bias=bias)

        # Cache for cudagraph safety
        self._cached_x_ptr = None
        self._cached_w_ptr = None
        self._cached_workspace = None
        self._cached_y = None

        # Precompute magic division parameters
        hw_out = OUT_H * OUT_W
        self._hw_out_magic, self._hw_out_shift = _compute_magic_u32_params(hw_out)
        self._out_w_magic, self._out_w_shift = _compute_magic_u32_params(OUT_W)
        c_per_split = (IN_CHANNELS + SPLIT_K_SLICES - 1) // SPLIT_K_SLICES
        self._split_channels_magic, self._split_channels_shift = _compute_magic_u32_params(c_per_split)
        self._kernel_w_magic, self._kernel_w_shift = _compute_magic_u32_params(KERNEL_W)
        self._kernel_area_magic, self._kernel_area_shift = _compute_magic_u32_params(KERNEL_AREA)

        gemm_n = OUT_CHANNELS
        n_groups = (gemm_n + GROUP_N - 1) // GROUP_N
        self._n_groups_magic, self._n_groups_shift = _compute_magic_u32_params(n_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape) != (BATCH, IN_CHANNELS, IN_H, IN_W):
            raise RuntimeError(f'This fused kernel only supports shape {(BATCH, IN_CHANNELS, IN_H, IN_W)}.')
        if x.dtype != torch.float32:
            raise RuntimeError('This fused kernel only supports torch.float32 dtype.')

        # Convert to bf16 for computation
        x_bf16 = x.to(dtype=torch.bfloat16)
        w_bf16 = self.conv2d.weight.to(device=x.device, dtype=torch.bfloat16)

        # Check if we need to reallocate
        x_ptr = x_bf16.data_ptr()
        w_ptr = w_bf16.data_ptr()

        if self._cached_workspace is None or self._cached_x_ptr != x_ptr or self._cached_w_ptr != w_ptr:
            # Allocate fp32 workspace for Split-K accumulation
            gemm_m = BATCH * OUT_H * OUT_W
            gemm_n = OUT_CHANNELS
            self._cached_workspace = torch.zeros((gemm_m, gemm_n), device=x.device, dtype=torch.float32)
            self._cached_y = torch.empty((BATCH, OUT_CHANNELS, OUT_H, OUT_W), device=x.device, dtype=torch.bfloat16)
            self._cached_x_ptr = x_ptr
            self._cached_w_ptr = w_ptr
        else:
            # Zero the workspace for reuse
            self._cached_workspace.zero_()

        workspace = self._cached_workspace
        y_bf16 = self._cached_y

        # Run Split-K GEMM kernel
        _igemm_splitk_kernel[_igemm_launch_config()](
            x_bf16, w_bf16, workspace,
            IN_H, IN_W, BATCH, OUT_CHANNELS, IN_CHANNELS,
            OUT_H, OUT_W, KERNEL_H, KERNEL_W,
            self._hw_out_magic, self._hw_out_shift,
            self._out_w_magic, self._out_w_shift,
            self._split_channels_magic, self._split_channels_shift,
            self._kernel_w_magic, self._kernel_w_shift,
            self._kernel_area_magic, self._kernel_area_shift,
            self._n_groups_magic, self._n_groups_shift,
        )

        # Run output store kernel
        _store_output_kernel[_store_launch_config()](
            workspace, y_bf16, BATCH, OUT_CHANNELS, OUT_H, OUT_W
        )

        # Convert back to float32
        return y_bf16.to(dtype=torch.float32)
