import torch
import torch.nn as nn
import substrate
import substrate.language as S

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
NUM_GROUPS = 16
GROUP_SIZE = OUT_FEATURES // NUM_GROUPS
HARDTANH_MIN = -2.0
HARDTANH_MAX = 2.0
EPS = 1e-5

# Tile sizes for 4-warps (2x2 warp grid)
TILE_M = 64
TILE_N = 64
K_TILE = 16  # Each K-tile has 16 K values, 2 MFMA ops per tile
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS
NUM_BUFFERS = 2  # Double buffering


@substrate.jit
def gemm_mfma_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((IN_FEATURES, OUT_FEATURES), S.bf16),  # (K, N) layout
    BIAS0: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    """GEMM kernel using MFMA 32x32x8_bf16_f32.

    Optimizations:
    1. Double buffering: 2 LDS buffers for A and B
    2. Software pipelining: overlap global loads with MFMA computation
    3. K-loop unrolled by 2: process 2 K-tiles per iteration (32 K), 4 MFMA ops
    4. Vectorized loads: raw_buffer_load_x2 for 8-byte loads (4 bf16)
    5. Range-based OOB handling: branches removed, range in buffer ops handles OOB
       - OOB loads return 0 (computations work correctly)
       - OOB stores are discarded
    """
    bx = S.block_id(0)  # M dimension
    by = S.block_id(1)  # N dimension
    tid = S.thread_id(0)
    lane = tid % WARP_SIZE
    warp_id = tid // WARP_SIZE

    # Warp grid: 2x2 warps
    warp_m = warp_id // 2
    warp_n = warp_id % 2

    # Output tile base for this warp
    tile_m_base = bx * TILE_M + warp_m * 32
    tile_n_base = by * TILE_N + warp_n * 32

    # Create resource descriptors with range for OOB handling
    # Range is in bytes - OOB loads return 0, OOB stores are discarded
    a_range = BATCH_SIZE * IN_FEATURES * 2  # Total bytes in X tensor
    b_range = IN_FEATURES * OUT_FEATURES * 2  # Total bytes in W tensor

    a_rsrc = S.amdgpu.make_rsrc(X, a_range)
    b_rsrc = S.amdgpu.make_rsrc(W, b_range)

    # Double-buffered LDS for A and B
    A_shared = S.make_shared((NUM_BUFFERS, TILE_M, K_TILE), S.bf16)
    B_shared = S.make_shared((NUM_BUFFERS, K_TILE, TILE_N), S.bf16)

    # Accumulator for 32x32 output (16 f32 per lane)
    acc = S.full((16,), 0.0, S.f32)

    num_k_tiles = IN_FEATURES // K_TILE

    # Thread mapping for x2 loads (8 bytes = 4 bf16)
    elem_idx = tid * 4
    a_row = elem_idx // K_TILE
    a_col = elem_idx % K_TILE
    b_row = elem_idx // TILE_N
    b_col = elem_idx % TILE_N

    # Global indices (constant across K)
    global_a_row = bx * TILE_M + a_row
    global_b_col = by * TILE_N + b_col

    # LDS access indices
    a_lds_row = warp_m * 32 + (lane % 32)
    a_lds_col = (lane // 32) * 4
    b_lds_col = warp_n * 32 + (lane % 32)

    # ===== Prologue: load first K-tile =====
    k_base = 0
    global_a_col = k_base + a_col
    byte_offset = (global_a_row * IN_FEATURES + global_a_col) * 2
    data_a = S.amdgpu.raw_buffer_load_x2(a_rsrc, byte_offset, 0, 0)
    data_a_bf16 = S.view(data_a, S.Tensor((4,), S.bf16))
    for i in S.range(4):
        A_shared[0, a_row, a_col + i] = data_a_bf16[i]

    global_b_row = k_base + b_row
    byte_offset = (global_b_row * OUT_FEATURES + global_b_col) * 2
    data_b = S.amdgpu.raw_buffer_load_x2(b_rsrc, byte_offset, 0, 0)
    data_b_bf16 = S.view(data_b, S.Tensor((4,), S.bf16))
    for i in S.range(4):
        B_shared[0, b_row, b_col + i] = data_b_bf16[i]

    S.syncthreads()

    # ===== Main loop with K-loop unrolled by 2 =====
    # Branches removed: range parameter handles OOB access automatically
    # - OOB loads return 0 (computations with 0 don't affect result)
    # - Removing branches in the loop is more beneficial than extra computations
    for k_tile_pair in S.range(0, num_k_tiles, 2):
        k_tile_0 = k_tile_pair
        k_tile_1 = k_tile_pair + 1

        buf0 = k_tile_0 % NUM_BUFFERS
        buf1 = k_tile_1 % NUM_BUFFERS

        # ===== Load second K-tile (software pipelining) =====
        # No branch needed - OOB loads return 0, range handles bounds
        k_base_1 = k_tile_1 * K_TILE
        global_a_col_1 = k_base_1 + a_col
        byte_offset_1 = (global_a_row * IN_FEATURES + global_a_col_1) * 2
        data_a_1 = S.amdgpu.raw_buffer_load_x2(a_rsrc, byte_offset_1, 0, 0)
        data_a_bf16_1 = S.view(data_a_1, S.Tensor((4,), S.bf16))
        for i in S.range(4):
            A_shared[buf1, a_row, a_col + i] = data_a_bf16_1[i]

        global_b_row_1 = k_base_1 + b_row
        byte_offset_1 = (global_b_row_1 * OUT_FEATURES + global_b_col) * 2
        data_b_1 = S.amdgpu.raw_buffer_load_x2(b_rsrc, byte_offset_1, 0, 0)
        data_b_bf16_1 = S.view(data_b_1, S.Tensor((4,), S.bf16))
        for i in S.range(4):
            B_shared[buf1, b_row, b_col + i] = data_b_bf16_1[i]

        # ===== Process first K-tile (2 MFMA ops) =====
        # K offset 0
        a_frag0_0 = S.make_local((4,), S.bf16)
        for i in S.range(4):
            a_frag0_0[i] = A_shared[buf0, a_lds_row, a_lds_col + i]

        b_lds_row_0 = (lane // 32) * 4
        b_frag0_0 = S.make_local((4,), S.bf16)
        for i in S.range(4):
            b_frag0_0[i] = B_shared[buf0, b_lds_row_0 + i, b_lds_col]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0_0, b_frag0_0, acc)

        # K offset 8
        a_lds_col_8 = a_lds_col + 8
        a_frag0_8 = S.make_local((4,), S.bf16)
        for i in S.range(4):
            a_frag0_8[i] = A_shared[buf0, a_lds_row, a_lds_col_8 + i]

        b_lds_row_8 = b_lds_row_0 + 8
        b_frag0_8 = S.make_local((4,), S.bf16)
        for i in S.range(4):
            b_frag0_8[i] = B_shared[buf0, b_lds_row_8 + i, b_lds_col]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0_8, b_frag0_8, acc)

        S.syncthreads()

        # ===== Process second K-tile (2 MFMA ops) =====
        # No branch needed - OOB loads return 0, MFMA with 0 is correct
        # K offset 0
        a_frag1_0 = S.make_local((4,), S.bf16)
        for i in S.range(4):
            a_frag1_0[i] = A_shared[buf1, a_lds_row, a_lds_col + i]

        b_frag1_0 = S.make_local((4,), S.bf16)
        for i in S.range(4):
            b_frag1_0[i] = B_shared[buf1, b_lds_row_0 + i, b_lds_col]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1_0, b_frag1_0, acc)

        # K offset 8
        a_frag1_8 = S.make_local((4,), S.bf16)
        for i in S.range(4):
            a_frag1_8[i] = A_shared[buf1, a_lds_row, a_lds_col_8 + i]

        b_frag1_8 = S.make_local((4,), S.bf16)
        for i in S.range(4):
            b_frag1_8[i] = B_shared[buf1, b_lds_row_8 + i, b_lds_col]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1_8, b_frag1_8, acc)

        # ===== Load next pair's first K-tile =====
        # No branch needed - OOB loads return 0, range handles bounds
        next_k_tile = k_tile_pair + 2
        next_buf = next_k_tile % NUM_BUFFERS
        k_base_next = next_k_tile * K_TILE
        global_a_col_next = k_base_next + a_col
        byte_offset_next = (global_a_row * IN_FEATURES + global_a_col_next) * 2
        data_a_next = S.amdgpu.raw_buffer_load_x2(a_rsrc, byte_offset_next, 0, 0)
        data_a_bf16_next = S.view(data_a_next, S.Tensor((4,), S.bf16))
        for i in S.range(4):
            A_shared[next_buf, a_row, a_col + i] = data_a_bf16_next[i]

        global_b_row_next = k_base_next + b_row
        byte_offset_next = (global_b_row_next * OUT_FEATURES + global_b_col) * 2
        data_b_next = S.amdgpu.raw_buffer_load_x2(b_rsrc, byte_offset_next, 0, 0)
        data_b_bf16_next = S.view(data_b_next, S.Tensor((4,), S.bf16))
        for i in S.range(4):
            B_shared[next_buf, b_row, b_col + i] = data_b_bf16_next[i]

        S.syncthreads()

    # ===== Write results =====
    # Output store - tensor indexing with OOB check for global_row
    for acc_idx in S.range(16):
        row_offset = 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        col_offset = lane % 32

        global_row = tile_m_base + row_offset
        global_col = tile_n_base + col_offset

        if global_row < BATCH_SIZE:
            bias_val = S.convert(BIAS0[global_col], S.f32)
            result = acc[acc_idx] + bias_val
            Y[global_row, global_col] = S.convert(result, S.bf16)


@substrate.jit
def group_norm_hardtanh_kernel(
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
    GN_WEIGHT: S.Tensor((OUT_FEATURES,), S.bf16),
    GN_BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
):
    """GroupNorm + HardTanh kernel."""
    sample = S.block_id(0)
    group = S.block_id(1)
    group_start = group * GROUP_SIZE
    tid = S.thread_id(0)

    local_sum = S.convert(0.0, S.f32)
    for i in S.range(GROUP_SIZE // 64):
        idx = i * 64 + tid
        val = S.convert(Y[sample, group_start + idx], S.f32)
        local_sum = local_sum + val

    lds_sums = S.make_shared((64,), S.f32)
    lds_sums[tid] = local_sum
    S.syncthreads()

    for s in S.range(6):
        stride = 1 << (5 - s)
        if tid < stride:
            lds_sums[tid] = lds_sums[tid] + lds_sums[tid + stride]
        S.syncthreads()

    mean = lds_sums[0] / S.convert(GROUP_SIZE, S.f32)

    local_sq_sum = S.convert(0.0, S.f32)
    for i in S.range(GROUP_SIZE // 64):
        idx = i * 64 + tid
        val = S.convert(Y[sample, group_start + idx], S.f32)
        diff = val - mean
        local_sq_sum = local_sq_sum + diff * diff

    lds_sums[tid] = local_sq_sum
    S.syncthreads()

    for s in S.range(6):
        stride = 1 << (5 - s)
        if tid < stride:
            lds_sums[tid] = lds_sums[tid] + lds_sums[tid + stride]
        S.syncthreads()

    var = lds_sums[0] / S.convert(GROUP_SIZE, S.f32)
    denom = S.sqrt(var + S.convert(EPS, S.f32))

    for i in S.range(GROUP_SIZE // 64):
        idx = i * 64 + tid
        col = group_start + idx
        val = S.convert(Y[sample, col], S.f32)
        v = (val - mean) / denom
        v = v * S.convert(GN_WEIGHT[col], S.f32) + S.convert(GN_BIAS[col], S.f32)

        if v < S.convert(HARDTANH_MIN, S.f32):
            v = S.convert(HARDTANH_MIN, S.f32)
        if v > S.convert(HARDTANH_MAX, S.f32):
            v = S.convert(HARDTANH_MAX, S.f32)

        Y[sample, col] = S.convert(v, S.bf16)


def _launch_gemm():
    grid_m = (BATCH_SIZE + TILE_M - 1) // TILE_M
    grid_n = (OUT_FEATURES + TILE_N - 1) // TILE_N
    return ((grid_m, grid_n, 1), (THREADS, 1, 1))


def _launch_gn():
    return ((BATCH_SIZE, NUM_GROUPS, 1), (64, 1, 1))


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)
        self._cached_w = None
        self._cached_bias = None
        self._cached_gn_w = None
        self._cached_gn_b = None

    def _ensure_cached_tensors(self, device):
        w = self.gemm.weight.t().contiguous()  # (K, N) layout
        if (self._cached_w is None or self._cached_w.data_ptr() != w.data_ptr() or self._cached_w.device != device):
            self._cached_w = w.to(device=device, dtype=torch.bfloat16).contiguous()
            self._cached_bias = self.gemm.bias.to(device=device, dtype=torch.bfloat16).contiguous()
            self._cached_gn_w = self.group_norm.weight.to(device=device, dtype=torch.bfloat16).contiguous()
            self._cached_gn_b = self.group_norm.bias.to(device=device, dtype=torch.bfloat16).contiguous()

    def forward(self, x):
        self._ensure_cached_tensors(x.device)
        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=torch.bfloat16)

        gemm_mfma_kernel[_launch_gemm](x.contiguous(), self._cached_w, self._cached_bias, y)
        group_norm_hardtanh_kernel[_launch_gn](y, self._cached_gn_w, self._cached_gn_b)

        return y
