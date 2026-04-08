"""AMDGPU GEMM kernels and helpers."""
import substrate
import substrate.language as S
import torch

WARP_SIZE = 64
NUM_WARPS = 4
GROUP_M = 256
GROUP_N = 128
GROUP_K = 64
VEC_SIZE = 8
THREADS = WARP_SIZE * NUM_WARPS

MMA_M = 16
MMA_N = 16
MMA_K = 16

# For each uint4 loads, a warp gets TILE_M * TILE_N elements
TILE_M = 16
TILE_N = 16

# Work group configuration
WARPS_M = 2  # Number of warps in M dimension per work group
WARPS_N = 2  # Number of warps in N dimension per work group

WARP_PER_ROW = 2
WARP_PER_COL = 2
WARP_MAT_M = GROUP_M // WARP_PER_ROW
WARP_MAT_N = GROUP_N // WARP_PER_COL
M_TILES_PER_WARP = WARP_MAT_M // MMA_M
N_TILES_PER_WARP = WARP_MAT_N // MMA_N
BATCH_K = GROUP_K // MMA_K
MATMUL_K_VEC = GROUP_K // VEC_SIZE
MATMUL_K_TILES = GROUP_K // (VEC_SIZE * BATCH_K)
ROW_STRIDE = MATMUL_K_VEC * 4
COL_STRIDE = 4

# Shared memory padding (shm_layout_config=0 from petit-kernel).
SHM_PAD_ROWS = 4  # kReadRowsAPerStep / kReadRowsBPerStep
SHM_PAD_BYTES = 32  # kPaddingSizeA / kPaddingSizeB
SHM_PAD_U32 = SHM_PAD_BYTES // 4
SHM_A_U32 = GROUP_M * ROW_STRIDE + (GROUP_M // SHM_PAD_ROWS) * SHM_PAD_U32
SHM_B_U32 = GROUP_N * ROW_STRIDE + (GROUP_N // SHM_PAD_ROWS) * SHM_PAD_U32

B_TILE_COLS_U32 = GROUP_N // 2
B_ROWPAIR_GROUPS = GROUP_K // 4
SHM_B_U64 = GROUP_K * GROUP_N // 4
B_LOADS_PER_THREAD = GROUP_K * GROUP_N // 2 // THREADS // 4
B_TRANS_HALF_U32 = B_ROWPAIR_GROUPS * B_TILE_COLS_U32
B_EVEN_SELECTOR = 0x05040100
B_ODD_SELECTOR = 0x07060302


def gemm_1stage_launch_config(m, n):
    m_groups = (m + GROUP_M - 1) // GROUP_M
    n_groups = (n + GROUP_N - 1) // GROUP_N
    grid = (m_groups * n_groups, 1, 1)
    block = (NUM_WARPS * WARP_SIZE, 1, 1)
    return grid, block


def gemm_1stage_batch_launch_config(batch, m, n):
    m_groups = (m + GROUP_M - 1) // GROUP_M
    n_groups = (n + GROUP_N - 1) // GROUP_N
    grid = (m_groups * n_groups, batch, 1)
    block = (NUM_WARPS * WARP_SIZE, 1, 1)
    return grid, block


def gemm_1stage_validate_shape(m, n, k):
    if m % GROUP_M != 0 or n % GROUP_N != 0 or k % GROUP_K != 0:
        raise ValueError(
            f"M must be divided by {GROUP_M}, N by {GROUP_N}, and K by {GROUP_K} "
            f"(got m={m}, n={n}, k={k})."
        )


# TODO: Use buffer_load to load from global to guard against OOB access
@substrate.jit
def load_global_a(
    A: S.Pointer(S.u32),
    m: S.u32,
    k: S.u32,
    tile_a_idx_row: S.u32,
    tile_a_idx_col: S.u32,
    reg_a: S.Tensor((GROUP_M * GROUP_K // VEC_SIZE // THREADS, 4,), S.u32,),
):
    K_VEC = GROUP_K // VEC_SIZE
    k_vec = k // VEC_SIZE
    layout_a = S.make_layout(
        (m, k_vec, 4),
        (k_vec * 4, 4, 1),
    )
    g_a = S.make_tensor(A, S.u32, layout_a)

    tid = S.thread_id(0)
    idx = tid
    for i in S.range(GROUP_M * GROUP_K // VEC_SIZE // THREADS):
        row = idx // K_VEC
        col = idx % K_VEC
        reg_a[i] = g_a[tile_a_idx_row * GROUP_M + row, tile_a_idx_col * K_VEC + col]
        idx = idx + THREADS


@substrate.jit
def load_global_a_batched(
    A: S.Pointer(S.u32),
    batch: S.u32,
    m: S.u32,
    k: S.u32,
    batch_id: S.u32,
    tile_a_idx_row: S.u32,
    tile_a_idx_col: S.u32,
    reg_a: S.Tensor((GROUP_M * GROUP_K // VEC_SIZE // THREADS, 4,), S.u32,),
):
    K_VEC = GROUP_K // VEC_SIZE
    k_vec = k // VEC_SIZE
    layout_a = S.make_layout(
        (batch, m, k_vec, 4),
        (m * k_vec * 4, k_vec * 4, 4, 1),
    )
    g_a = S.make_tensor(A, S.u32, layout_a)

    tid = S.thread_id(0)
    idx = tid
    for i in S.range(GROUP_M * GROUP_K // VEC_SIZE // THREADS):
        row = idx // K_VEC
        col = idx % K_VEC
        reg_a[i] = g_a[batch_id, tile_a_idx_row * GROUP_M + row, tile_a_idx_col * K_VEC + col]
        idx = idx + THREADS

@substrate.jit
def load_global_b_row_major(
    B: S.Pointer(S.u32),
    n: S.u32,
    k: S.u32,
    tile_b_idx_row: S.u32,
    tile_b_idx_col: S.u32,
    reg_b: S.Tensor((B_LOADS_PER_THREAD, 4), S.u32),
):
    layout_b = S.make_layout(
        (k, n // 2),
        (n // 2, 1),
    )
    g_b = S.make_tensor(B, S.u32, layout_b)

    tid = S.thread_id(0)
    idx_u2 = tid
    for i in S.range(B_LOADS_PER_THREAD):
        rowpair_group = idx_u2 // B_TILE_COLS_U32
        col_u32 = tile_b_idx_col * B_TILE_COLS_U32 + (idx_u2 % B_TILE_COLS_U32)
        row0 = tile_b_idx_row * GROUP_K + rowpair_group * 4
        for j in S.range(4):
            reg_b[i, j] = g_b[row0 + j, col_u32]
        idx_u2 = idx_u2 + THREADS


@substrate.jit
def load_global_b_row_major_batched(
    B: S.Pointer(S.u32),
    batch: S.u32,
    n: S.u32,
    k: S.u32,
    batch_id: S.u32,
    tile_b_idx_row: S.u32,
    tile_b_idx_col: S.u32,
    reg_b: S.Tensor((B_LOADS_PER_THREAD, 4), S.u32),
):
    layout_b = S.make_layout(
        (batch, k, n // 2),
        (k * (n // 2), n // 2, 1),
    )
    g_b = S.make_tensor(B, S.u32, layout_b)

    tid = S.thread_id(0)
    idx_u2 = tid
    for i in S.range(B_LOADS_PER_THREAD):
        rowpair_group = idx_u2 // B_TILE_COLS_U32
        col_u32 = tile_b_idx_col * B_TILE_COLS_U32 + (idx_u2 % B_TILE_COLS_U32)
        row0 = tile_b_idx_row * GROUP_K + rowpair_group * 4
        for j in S.range(4):
            reg_b[i, j] = g_b[batch_id, row0 + j, col_u32]
        idx_u2 = idx_u2 + THREADS


@substrate.jit
def load_global_b_transposed(
    B: S.Pointer(S.u32),
    n: S.u32,
    k: S.u32,
    tile_b_idx_row: S.u32,
    tile_b_idx_col: S.u32,
    reg_b: S.Tensor((GROUP_N * GROUP_K // VEC_SIZE // THREADS, 4,), S.u32,),
):
    k_vec = k // VEC_SIZE
    layout_b = S.make_layout(
        (n, k_vec, 4),
        (k_vec * 4, 4, 1),
    )
    g_b = S.make_tensor(B, S.u32, layout_b)

    tid = S.thread_id(0)
    idx = tid
    for i in S.range(GROUP_N * GROUP_K // VEC_SIZE // THREADS):
        row = idx // MATMUL_K_VEC
        col = idx % MATMUL_K_VEC
        reg_b[i] = g_b[tile_b_idx_row * GROUP_N + row, tile_b_idx_col * MATMUL_K_VEC + col]
        idx = idx + THREADS


@substrate.jit
def load_global_b_transposed_batched(
    B: S.Pointer(S.u32),
    batch: S.u32,
    n: S.u32,
    k: S.u32,
    batch_id: S.u32,
    tile_b_idx_row: S.u32,
    tile_b_idx_col: S.u32,
    reg_b: S.Tensor((GROUP_N * GROUP_K // VEC_SIZE // THREADS, 4,), S.u32,),
):
    k_vec = k // VEC_SIZE
    layout_b = S.make_layout(
        (batch, n, k_vec, 4),
        (n * k_vec * 4, k_vec * 4, 4, 1),
    )
    g_b = S.make_tensor(B, S.u32, layout_b)

    tid = S.thread_id(0)
    idx = tid
    for i in S.range(GROUP_N * GROUP_K // VEC_SIZE // THREADS):
        row = idx // MATMUL_K_VEC
        col = idx % MATMUL_K_VEC
        reg_b[i] = g_b[batch_id, tile_b_idx_row * GROUP_N + row, tile_b_idx_col * MATMUL_K_VEC + col]
        idx = idx + THREADS


@substrate.jit
def store_shm_a(
    shm_a: S.Tensor((SHM_A_U32,), S.u32),
    reg_a: S.Tensor((GROUP_M * GROUP_K // VEC_SIZE // THREADS, 4,), S.u32,),
):
    REG_SIZE_A = GROUP_M * GROUP_K // VEC_SIZE // THREADS
    K_VEC = GROUP_K // VEC_SIZE
    layout_sa = S.make_layout(
        ((GROUP_M // SHM_PAD_ROWS, SHM_PAD_ROWS), MATMUL_K_VEC, 4),
        ((ROW_STRIDE * SHM_PAD_ROWS + SHM_PAD_U32, ROW_STRIDE), 4, 1),
    )
    s_a = S.view(shm_a, S.u32, layout_sa)
    tid = S.thread_id(0)
    idx = tid
    for i in S.range(REG_SIZE_A):
        row = idx // K_VEC
        col = idx % K_VEC
        s_a[row, col] = reg_a[i]
        idx = idx + THREADS

@substrate.jit
def store_shm_b_transposed(
    shm_b: S.Tensor((SHM_B_U64,), S.u64),
    reg_b: S.Tensor((B_LOADS_PER_THREAD, 4), S.u32),
):
    shm_b_u32 = S.view(shm_b, S.Tensor((SHM_B_U64 * 2,), S.u32))
    tid = S.thread_id(0)
    idx_u2 = tid
    for i in S.range(B_LOADS_PER_THREAD):
        even0 = S.amdgpu.perm(reg_b[i, 1], reg_b[i, 0], B_EVEN_SELECTOR)
        even1 = S.amdgpu.perm(reg_b[i, 3], reg_b[i, 2], B_EVEN_SELECTOR)
        odd0 = S.amdgpu.perm(reg_b[i, 1], reg_b[i, 0], B_ODD_SELECTOR)
        odd1 = S.amdgpu.perm(reg_b[i, 3], reg_b[i, 2], B_ODD_SELECTOR)
        even_base = idx_u2 * 2
        odd_base = (idx_u2 + B_TRANS_HALF_U32) * 2
        shm_b_u32[even_base] = even0
        shm_b_u32[even_base + 1] = even1
        shm_b_u32[odd_base] = odd0
        shm_b_u32[odd_base + 1] = odd1
        idx_u2 = idx_u2 + THREADS


@substrate.jit
def store_shm_b_plain(
    shm_b: S.Tensor((SHM_B_U32,), S.u32),
    reg_b: S.Tensor((GROUP_N * GROUP_K // VEC_SIZE // THREADS, 4,), S.u32,),
):
    layout_sb = S.make_layout(
        ((GROUP_N // SHM_PAD_ROWS, SHM_PAD_ROWS), MATMUL_K_VEC, 4),
        ((ROW_STRIDE * SHM_PAD_ROWS + SHM_PAD_U32, ROW_STRIDE), 4, 1),
    )
    s_b = S.view(shm_b, S.u32, layout_sb)
    tid = S.thread_id(0)
    idx = tid
    for i in S.range(GROUP_N * GROUP_K // VEC_SIZE // THREADS):
        row = idx // MATMUL_K_VEC
        col = idx % MATMUL_K_VEC
        s_b[row, col] = reg_b[i]
        idx = idx + THREADS


# Start with 2x2 matmul -- they are multiple ways of mapping to warps/threads
@substrate.jit
def load_shm_to_regs(
    shm_a: S.Tensor((SHM_A_U32,), S.u32),
    shm_b: S.Tensor((SHM_B_U64,), S.u64),
    tile_m: S.u32,
    tile_n: S.u32,
    k_tile: S.u32,
    tile_a: S.Tensor((1, 4), S.u32),
    tile_b: S.Tensor((1, 4), S.u32),
):
    tid = S.thread_id(0)
    wid = tid // WARP_SIZE
    wtid = tid % WARP_SIZE
    warp_row = wid // WARP_PER_COL
    warp_col = wid % WARP_PER_COL
    mma_k = wtid % MMA_K
    batch_k = wtid // MMA_K

    layout_sa = S.make_layout(
        ((GROUP_M // SHM_PAD_ROWS, SHM_PAD_ROWS), MATMUL_K_VEC, 4),
        ((ROW_STRIDE * SHM_PAD_ROWS + SHM_PAD_U32, ROW_STRIDE), 4, 1),
    )
    s_a = S.view(shm_a, S.u32, layout_sa)
    shm_b_u32 = S.view(shm_b, S.Tensor((SHM_B_U64 * 2,), S.u32))

    row_a = warp_row * (M_TILES_PER_WARP * TILE_M) + tile_m * TILE_M + mma_k
    col_a = k_tile * BATCH_K + batch_k
    tile_a[0] = s_a[row_a, col_a]

    row_b = warp_col * (N_TILES_PER_WARP * TILE_N) + tile_n * TILE_N + mma_k
    col_b = k_tile * BATCH_K + batch_k
    col_u32 = row_b // 2
    odd_offset = (wtid & 1) * B_TRANS_HALF_U32
    rowpair_group = col_b * 2
    base0 = (odd_offset + rowpair_group * B_TILE_COLS_U32 + col_u32) * 2
    base1 = (odd_offset + (rowpair_group + 1) * B_TILE_COLS_U32 + col_u32) * 2
    tile_b[0, 0] = shm_b_u32[base0]
    tile_b[0, 1] = shm_b_u32[base0 + 1]
    tile_b[0, 2] = shm_b_u32[base1]
    tile_b[0, 3] = shm_b_u32[base1 + 1]


@substrate.jit
def load_shm_to_regs_transposed_b(
    shm_a: S.Tensor((SHM_A_U32,), S.u32),
    shm_b: S.Tensor((SHM_B_U32,), S.u32),
    tile_m: S.u32,
    tile_n: S.u32,
    k_tile: S.u32,
    tile_a: S.Tensor((1, 4), S.u32),
    tile_b: S.Tensor((1, 4), S.u32),
):
    tid = S.thread_id(0)
    wid = tid // WARP_SIZE
    wtid = tid % WARP_SIZE
    warp_row = wid // WARP_PER_COL
    warp_col = wid % WARP_PER_COL
    mma_k = wtid % MMA_K
    batch_k = wtid // MMA_K

    layout_sa = S.make_layout(
        ((GROUP_M // SHM_PAD_ROWS, SHM_PAD_ROWS), MATMUL_K_VEC, 4),
        ((ROW_STRIDE * SHM_PAD_ROWS + SHM_PAD_U32, ROW_STRIDE), 4, 1),
    )
    layout_sb = S.make_layout(
        ((GROUP_N // SHM_PAD_ROWS, SHM_PAD_ROWS), MATMUL_K_VEC, 4),
        ((ROW_STRIDE * SHM_PAD_ROWS + SHM_PAD_U32, ROW_STRIDE), 4, 1),
    )
    s_a = S.view(shm_a, S.u32, layout_sa)
    s_b = S.view(shm_b, S.u32, layout_sb)

    row_a = warp_row * (M_TILES_PER_WARP * TILE_M) + tile_m * TILE_M + mma_k
    col_a = k_tile * BATCH_K + batch_k
    tile_a[0] = s_a[row_a, col_a]

    row_b = warp_col * (N_TILES_PER_WARP * TILE_N) + tile_n * TILE_N + mma_k
    col_b = k_tile * BATCH_K + batch_k
    tile_b[0] = s_b[row_b, col_b]


@substrate.jit
def matmul_from_regs(
    tile_a: S.Tensor((1, 4), S.u32),
    tile_b: S.Tensor((1, 4), S.u32),
    acc: S.Tensor((GROUP_M * GROUP_N // THREADS // 4, 4,), S.f32,),
    acc_idx: S.u32,
):
    r_a = tile_a[0]
    r_b = tile_b[0]
    r_a_bf16 = S.view(r_a, S.Tensor((2, 4, 1), S.bf16))
    r_b_bf16 = S.view(r_b, S.Tensor((2, 4, 1), S.bf16))
    acc[acc_idx] = S.amdgpu.mfma_f32_16x16x16_bf16(r_b_bf16[0], r_a_bf16[0], acc[acc_idx])
    acc[acc_idx] = S.amdgpu.mfma_f32_16x16x16_bf16(r_b_bf16[1], r_a_bf16[1], acc[acc_idx])


@substrate.jit
def write_results(
    id_m: S.u32,
    id_n: S.u32,
    acc: S.Tensor((GROUP_M * GROUP_N // THREADS // 4, 4,), S.f32),
    C: S.Pointer(S.u32),
    m: S.u32,
    n: S.u32,
):
    tid = S.thread_id(0)
    wid = tid // WARP_SIZE
    lane_id = tid % WARP_SIZE

    row_block = TILE_M // (WARP_SIZE // TILE_N)

    layout_c = S.make_layout(
        (
            m // GROUP_M, n // GROUP_N,
            (WARP_PER_COL, WARP_PER_ROW),
            (N_TILES_PER_WARP, M_TILES_PER_WARP),
            (TILE_N, TILE_N // 4),
            2,
        ),
        (
            WARP_PER_ROW * M_TILES_PER_WARP * TILE_M * (n // 2), GROUP_N // 2,
            (N_TILES_PER_WARP * (TILE_N // 2), M_TILES_PER_WARP * TILE_M * (n // 2)),
            (TILE_N // 2, TILE_M * (n // 2)),
            (n // 2, 2),
            1,
        ),
    )
    g_c = S.make_tensor(C, S.u32, layout_c)

    for acc_idx in S.range(M_TILES_PER_WARP * N_TILES_PER_WARP):
        acc_vec = acc[acc_idx]
        r = S.make_local((4,), S.bf16)
        for t in S.range(row_block):
            r[t] = S.convert(acc_vec[t], S.bf16)

        packed = S.view(r, S.Tensor((2,), S.u32))
        g_c[id_m, id_n, wid, acc_idx, lane_id] = packed


@substrate.jit
def write_results_batched(
    batch: S.u32,
    batch_id: S.u32,
    id_m: S.u32,
    id_n: S.u32,
    acc: S.Tensor((GROUP_M * GROUP_N // THREADS // 4, 4,), S.f32),
    C: S.Pointer(S.u32),
    m: S.u32,
    n: S.u32,
):
    tid = S.thread_id(0)
    wid = tid // WARP_SIZE
    lane_id = tid % WARP_SIZE

    row_block = TILE_M // (WARP_SIZE // TILE_N)

    layout_c = S.make_layout(
        (
            batch, m // GROUP_M, n // GROUP_N,
            (WARP_PER_COL, WARP_PER_ROW),
            (N_TILES_PER_WARP, M_TILES_PER_WARP),
            (TILE_N, TILE_N // 4),
            2,
        ),
        (
            (m // GROUP_M) * (n // GROUP_N) * GROUP_N * (GROUP_M // 2),
            WARP_PER_ROW * M_TILES_PER_WARP * TILE_M * (n // 2), GROUP_N // 2,
            (N_TILES_PER_WARP * (TILE_N // 2), M_TILES_PER_WARP * TILE_M * (n // 2)),
            (TILE_N // 2, TILE_M * (n // 2)),
            (n // 2, 2),
            1,
        ),
    )
    g_c = S.make_tensor(C, S.u32, layout_c)

    for acc_idx in S.range(M_TILES_PER_WARP * N_TILES_PER_WARP):
        acc_vec = acc[acc_idx]
        r = S.make_local((4,), S.bf16)
        for t in S.range(row_block):
            r[t] = S.convert(acc_vec[t], S.bf16)

        packed = S.view(r, S.Tensor((2,), S.u32))
        g_c[batch_id, id_m, id_n, wid, acc_idx, lane_id] = packed


@substrate.jit
def wgm_mapping(m: S.u32, n: S.u32) -> (S.u32, S.u32):
    block_id_linear = S.block_id(0)
    linear_group_id = S.convert(block_id_linear, S.u32)
    group_m_size = S.convert(GROUP_M, S.u32)
    group_n_size = S.convert(GROUP_N, S.u32)
    m_groups = m // group_m_size
    n_groups = n // group_n_size

    total_groups = m_groups * n_groups

    cu_count = S.convert(38 * 8, S.u32)
    wgm_xcc = S.convert(8, S.u32)
    workgroup_mapping = S.convert(32, S.u32)

    linear_group_limit = (total_groups // wgm_xcc) * wgm_xcc
    cu_base = (linear_group_id // cu_count) * cu_count
    cu_xcc = (linear_group_id % cu_count) // wgm_xcc
    cu_base = cu_base + cu_xcc

    cu_tail_limit = (total_groups // cu_count) * cu_count
    active_cu = (total_groups % cu_count) if (linear_group_id > cu_tail_limit) else cu_count
    cu_xcc_stride = (active_cu // wgm_xcc) * (linear_group_id % wgm_xcc)
    linear_group_mapped = cu_base + cu_xcc_stride

    linear_group_id = (
        linear_group_mapped if (linear_group_id < linear_group_limit) else linear_group_id
    )

    group_m = linear_group_id // n_groups
    group_n = linear_group_id - group_m * n_groups

    mapping_block = group_m // workgroup_mapping
    mapping_linear = group_n + (group_m % workgroup_mapping) * n_groups
    mapping_groups = m_groups // workgroup_mapping
    mapping_tail = m_groups % workgroup_mapping
    mapping_tail = mapping_tail if (mapping_tail != 0) else workgroup_mapping
    mapping_span = mapping_tail if (mapping_block >= mapping_groups) else workgroup_mapping

    group_n = mapping_linear // mapping_span
    group_m = mapping_linear % mapping_span
    group_m = group_m + mapping_block * workgroup_mapping

    return group_m, group_n


@substrate.jit
def wgm_mapping_batched(m: S.u32, n: S.u32) -> (S.u32, S.u32, S.u32):
    group_m, group_n = wgm_mapping(m, n)
    batch_id = S.convert(S.block_id(1), S.u32)
    return batch_id, group_m, group_n


@substrate.jit
def _gemm_1stage_pipeline_kernel(
    A: S.Pointer(S.u32),
    B: S.Pointer(S.u32),
    C: S.Pointer(S.u32),
    m: S.u32,
    n: S.u32,
    k: S.u32,
):
    id_m, id_n = wgm_mapping(m, n)

    # Store A / B from registers to shared memory (padded layout).
    shm_a = S.make_shared((SHM_A_U32,), S.u32)
    shm_b = S.make_shared((SHM_B_U64,), S.u64)
    acc = S.make_local((GROUP_M * GROUP_N // THREADS // 4, 4,), S.f32)
    for i in S.range(GROUP_M * GROUP_N // THREADS // 4):
        for j in S.range(4):
            acc[i, j] = 0

    k_tiles = k // GROUP_K
    k_tiles_u32 = S.convert(k_tiles, S.u32)
    # Stagger K tiles (wgm_config_id=7) to reduce inter-wavefront contention.
    k_stagger_mask = S.convert(0x7, S.u32)
    k_stagger_stride = S.convert(4, S.u32)
    # kStaggerUMapping=0 -> use gid_n (id_n after workgroup mapping).
    stagger_data = id_n
    k_start = (stagger_data & k_stagger_mask) * k_stagger_stride
    k_start = k_start if (k_start < k_tiles_u32) else S.convert(0, S.u32)

    # 2-stage single-buffer pipeline:
    # stage 0: prefetch next tile into registers
    # stage 1: compute current tile from shared memory
    reg_a = S.make_local((GROUP_M * GROUP_K // VEC_SIZE // THREADS, 4,), S.u32,)
    reg_b = S.make_local((B_LOADS_PER_THREAD, 4), S.u32)
    tile_a0 = S.make_local((1, 4), S.u32)
    tile_b0 = S.make_local((1, 4), S.u32)
    tile_a1 = S.make_local((1, 4), S.u32)
    tile_b1 = S.make_local((1, 4), S.u32)

    # Prime the pipeline with staggered k_tile.
    load_global_a(A, m, k, id_m, k_start, reg_a)
    load_global_b_row_major(B, n, k, k_start, id_n, reg_b)
    store_shm_a(shm_a, reg_a)
    store_shm_b_transposed(shm_b, reg_b)
    S.syncthreads()

    # Main pipeline loop: prefetch next tile while computing current
    for k_iter in S.range(k_tiles - 1):
        k_next = k_start + k_iter + 1
        k_next = k_next - (k_tiles_u32 if (k_next >= k_tiles_u32) else S.convert(0, S.u32))
        load_global_a(A, m, k, id_m, k_next, reg_a)
        load_global_b_row_major(B, n, k, k_next, id_n, reg_b)

        # Compute matmul on current tile in shared memory
        for i in S.range(M_TILES_PER_WARP):
            for j in S.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                # MATMUL_K_TILES is hard-coded to 2: prefetch next then compute.
                load_shm_to_regs(shm_a, shm_b, i, j, 0, tile_a0, tile_b0)
                load_shm_to_regs(shm_a, shm_b, i, j, 1, tile_a1, tile_b1)
                matmul_from_regs(tile_a0, tile_b0, acc, acc_idx)
                matmul_from_regs(tile_a1, tile_b1, acc, acc_idx)
        S.syncthreads()

        # Publish prefetched tile to shared memory for next iteration
        store_shm_a(shm_a, reg_a)
        store_shm_b_transposed(shm_b, reg_b)
        S.syncthreads()

    # Compute the last tile already in shared memory
    for i in S.range(M_TILES_PER_WARP):
        for j in S.range(N_TILES_PER_WARP):
            acc_idx = i * N_TILES_PER_WARP + j
            load_shm_to_regs(shm_a, shm_b, i, j, 0, tile_a0, tile_b0)
            load_shm_to_regs(shm_a, shm_b, i, j, 1, tile_a1, tile_b1)
            matmul_from_regs(tile_a0, tile_b0, acc, acc_idx)
            matmul_from_regs(tile_a1, tile_b1, acc, acc_idx)

    write_results(id_m, id_n, acc, C, m, n)


@substrate.jit
def _gemm_1stage_pipeline_transposed_b_kernel(
    A: S.Pointer(S.u32),
    B: S.Pointer(S.u32),
    C: S.Pointer(S.u32),
    m: S.u32,
    n: S.u32,
    k: S.u32,
):
    id_m, id_n = wgm_mapping(m, n)

    shm_a = S.make_shared((SHM_A_U32,), S.u32)
    shm_b = S.make_shared((SHM_B_U32,), S.u32)
    acc = S.make_local((GROUP_M * GROUP_N // THREADS // 4, 4,), S.f32)
    for i in S.range(GROUP_M * GROUP_N // THREADS // 4):
        for j in S.range(4):
            acc[i, j] = 0

    k_tiles = k // GROUP_K
    k_tiles_u32 = S.convert(k_tiles, S.u32)
    k_stagger_mask = S.convert(0x7, S.u32)
    k_stagger_stride = S.convert(4, S.u32)
    stagger_data = id_n
    k_start = (stagger_data & k_stagger_mask) * k_stagger_stride
    k_start = k_start if (k_start < k_tiles_u32) else S.convert(0, S.u32)

    reg_a = S.make_local((GROUP_M * GROUP_K // VEC_SIZE // THREADS, 4,), S.u32,)
    reg_b = S.make_local((GROUP_N * GROUP_K // VEC_SIZE // THREADS, 4,), S.u32,)
    tile_a0 = S.make_local((1, 4), S.u32)
    tile_b0 = S.make_local((1, 4), S.u32)
    tile_a1 = S.make_local((1, 4), S.u32)
    tile_b1 = S.make_local((1, 4), S.u32)

    load_global_a(A, m, k, id_m, k_start, reg_a)
    load_global_b_transposed(B, n, k, id_n, k_start, reg_b)
    store_shm_a(shm_a, reg_a)
    store_shm_b_plain(shm_b, reg_b)
    S.syncthreads()

    for k_iter in S.range(k_tiles - 1):
        k_next = k_start + k_iter + 1
        k_next = k_next - (k_tiles_u32 if (k_next >= k_tiles_u32) else S.convert(0, S.u32))
        load_global_a(A, m, k, id_m, k_next, reg_a)
        load_global_b_transposed(B, n, k, id_n, k_next, reg_b)

        for i in S.range(M_TILES_PER_WARP):
            for j in S.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                load_shm_to_regs_transposed_b(shm_a, shm_b, i, j, 0, tile_a0, tile_b0)
                load_shm_to_regs_transposed_b(shm_a, shm_b, i, j, 1, tile_a1, tile_b1)
                matmul_from_regs(tile_a0, tile_b0, acc, acc_idx)
                matmul_from_regs(tile_a1, tile_b1, acc, acc_idx)
        S.syncthreads()

        store_shm_a(shm_a, reg_a)
        store_shm_b_plain(shm_b, reg_b)
        S.syncthreads()

    for i in S.range(M_TILES_PER_WARP):
        for j in S.range(N_TILES_PER_WARP):
            acc_idx = i * N_TILES_PER_WARP + j
            load_shm_to_regs_transposed_b(shm_a, shm_b, i, j, 0, tile_a0, tile_b0)
            load_shm_to_regs_transposed_b(shm_a, shm_b, i, j, 1, tile_a1, tile_b1)
            matmul_from_regs(tile_a0, tile_b0, acc, acc_idx)
            matmul_from_regs(tile_a1, tile_b1, acc, acc_idx)

    write_results(id_m, id_n, acc, C, m, n)


@substrate.jit
def _gemm_1stage_pipeline_batched_kernel(
    A: S.Pointer(S.u32),
    B: S.Pointer(S.u32),
    C: S.Pointer(S.u32),
    batch: S.u32,
    m: S.u32,
    n: S.u32,
    k: S.u32,
):
    batch_id, id_m, id_n = wgm_mapping_batched(m, n)

    shm_a = S.make_shared((SHM_A_U32,), S.u32)
    shm_b = S.make_shared((SHM_B_U64,), S.u64)
    acc = S.make_local((GROUP_M * GROUP_N // THREADS // 4, 4,), S.f32)
    for i in S.range(GROUP_M * GROUP_N // THREADS // 4):
        for j in S.range(4):
            acc[i, j] = 0

    k_tiles = k // GROUP_K
    k_tiles_u32 = S.convert(k_tiles, S.u32)
    k_stagger_mask = S.convert(0x7, S.u32)
    k_stagger_stride = S.convert(4, S.u32)
    stagger_data = id_n
    k_start = (stagger_data & k_stagger_mask) * k_stagger_stride
    k_start = k_start if (k_start < k_tiles_u32) else S.convert(0, S.u32)

    reg_a = S.make_local((GROUP_M * GROUP_K // VEC_SIZE // THREADS, 4,), S.u32,)
    reg_b = S.make_local((B_LOADS_PER_THREAD, 4), S.u32)
    tile_a0 = S.make_local((1, 4), S.u32)
    tile_b0 = S.make_local((1, 4), S.u32)
    tile_a1 = S.make_local((1, 4), S.u32)
    tile_b1 = S.make_local((1, 4), S.u32)

    load_global_a_batched(A, batch, m, k, batch_id, id_m, k_start, reg_a)
    load_global_b_row_major_batched(B, batch, n, k, batch_id, k_start, id_n, reg_b)
    store_shm_a(shm_a, reg_a)
    store_shm_b_transposed(shm_b, reg_b)
    S.syncthreads()

    for k_iter in S.range(k_tiles - 1):
        k_next = k_start + k_iter + 1
        k_next = k_next - (k_tiles_u32 if (k_next >= k_tiles_u32) else S.convert(0, S.u32))
        load_global_a_batched(A, batch, m, k, batch_id, id_m, k_next, reg_a)
        load_global_b_row_major_batched(B, batch, n, k, batch_id, k_next, id_n, reg_b)

        for i in S.range(M_TILES_PER_WARP):
            for j in S.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                load_shm_to_regs(shm_a, shm_b, i, j, 0, tile_a0, tile_b0)
                load_shm_to_regs(shm_a, shm_b, i, j, 1, tile_a1, tile_b1)
                matmul_from_regs(tile_a0, tile_b0, acc, acc_idx)
                matmul_from_regs(tile_a1, tile_b1, acc, acc_idx)
        S.syncthreads()

        store_shm_a(shm_a, reg_a)
        store_shm_b_transposed(shm_b, reg_b)
        S.syncthreads()

    for i in S.range(M_TILES_PER_WARP):
        for j in S.range(N_TILES_PER_WARP):
            acc_idx = i * N_TILES_PER_WARP + j
            load_shm_to_regs(shm_a, shm_b, i, j, 0, tile_a0, tile_b0)
            load_shm_to_regs(shm_a, shm_b, i, j, 1, tile_a1, tile_b1)
            matmul_from_regs(tile_a0, tile_b0, acc, acc_idx)
            matmul_from_regs(tile_a1, tile_b1, acc, acc_idx)

    write_results_batched(batch, batch_id, id_m, id_n, acc, C, m, n)


@substrate.jit
def _gemm_1stage_pipeline_transposed_b_batched_kernel(
    A: S.Pointer(S.u32),
    B: S.Pointer(S.u32),
    C: S.Pointer(S.u32),
    batch: S.u32,
    m: S.u32,
    n: S.u32,
    k: S.u32,
):
    batch_id, id_m, id_n = wgm_mapping_batched(m, n)

    shm_a = S.make_shared((SHM_A_U32,), S.u32)
    shm_b = S.make_shared((SHM_B_U32,), S.u32)
    acc = S.make_local((GROUP_M * GROUP_N // THREADS // 4, 4,), S.f32)
    for i in S.range(GROUP_M * GROUP_N // THREADS // 4):
        for j in S.range(4):
            acc[i, j] = 0

    k_tiles = k // GROUP_K
    k_tiles_u32 = S.convert(k_tiles, S.u32)
    k_stagger_mask = S.convert(0x7, S.u32)
    k_stagger_stride = S.convert(4, S.u32)
    stagger_data = id_n
    k_start = (stagger_data & k_stagger_mask) * k_stagger_stride
    k_start = k_start if (k_start < k_tiles_u32) else S.convert(0, S.u32)

    reg_a = S.make_local((GROUP_M * GROUP_K // VEC_SIZE // THREADS, 4,), S.u32,)
    reg_b = S.make_local((GROUP_N * GROUP_K // VEC_SIZE // THREADS, 4,), S.u32,)
    tile_a0 = S.make_local((1, 4), S.u32)
    tile_b0 = S.make_local((1, 4), S.u32)
    tile_a1 = S.make_local((1, 4), S.u32)
    tile_b1 = S.make_local((1, 4), S.u32)

    load_global_a_batched(A, batch, m, k, batch_id, id_m, k_start, reg_a)
    load_global_b_transposed_batched(B, batch, n, k, batch_id, id_n, k_start, reg_b)
    store_shm_a(shm_a, reg_a)
    store_shm_b_plain(shm_b, reg_b)
    S.syncthreads()

    for k_iter in S.range(k_tiles - 1):
        k_next = k_start + k_iter + 1
        k_next = k_next - (k_tiles_u32 if (k_next >= k_tiles_u32) else S.convert(0, S.u32))
        load_global_a_batched(A, batch, m, k, batch_id, id_m, k_next, reg_a)
        load_global_b_transposed_batched(B, batch, n, k, batch_id, id_n, k_next, reg_b)

        for i in S.range(M_TILES_PER_WARP):
            for j in S.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                load_shm_to_regs_transposed_b(shm_a, shm_b, i, j, 0, tile_a0, tile_b0)
                load_shm_to_regs_transposed_b(shm_a, shm_b, i, j, 1, tile_a1, tile_b1)
                matmul_from_regs(tile_a0, tile_b0, acc, acc_idx)
                matmul_from_regs(tile_a1, tile_b1, acc, acc_idx)
        S.syncthreads()

        store_shm_a(shm_a, reg_a)
        store_shm_b_plain(shm_b, reg_b)
        S.syncthreads()

    for i in S.range(M_TILES_PER_WARP):
        for j in S.range(N_TILES_PER_WARP):
            acc_idx = i * N_TILES_PER_WARP + j
            load_shm_to_regs_transposed_b(shm_a, shm_b, i, j, 0, tile_a0, tile_b0)
            load_shm_to_regs_transposed_b(shm_a, shm_b, i, j, 1, tile_a1, tile_b1)
            matmul_from_regs(tile_a0, tile_b0, acc, acc_idx)
            matmul_from_regs(tile_a1, tile_b1, acc, acc_idx)

    write_results_batched(batch, batch_id, id_m, id_n, acc, C, m, n)


def _validate_gemm_inputs(
    A: torch.Tensor,
    B: torch.Tensor,
):
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("A and B must be torch.Tensor")
    if A.ndim != B.ndim or A.ndim not in (2, 3):
        raise ValueError(
            "A and B must both be rank-2 or rank-3 tensors "
            f"(got A.ndim={A.ndim}, B.ndim={B.ndim})"
        )
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        raise TypeError(
            f"A and B must be torch.bfloat16 (got A={A.dtype}, B={B.dtype})"
        )
    if A.device.type != "cuda" or B.device.type != "cuda":
        raise ValueError(
            f"A and B must be CUDA tensors (got A={A.device}, B={B.device})"
        )
    if A.device != B.device:
        raise ValueError(f"A and B must be on the same device (got {A.device} and {B.device})")

def _prepare_out(
    shape: tuple[int, ...],
    device: torch.device,
    out: torch.Tensor | None,
) -> torch.Tensor:
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be torch.Tensor")
    if out is None or out.numel() == 0:
        out = torch.empty(shape, dtype=torch.bfloat16, device=device)
    elif out.ndim != len(shape) or tuple(out.shape) != shape:
        raise ValueError(f"out must have shape {shape} (got {tuple(out.shape)})")
    elif out.dtype != torch.bfloat16:
        raise TypeError(f"out must be torch.bfloat16 (got {out.dtype})")
    elif out.device != device:
        raise ValueError(f"out must be on {device} (got {out.device})")

    return out


def gemm_1stage_pipeline_row_major(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_gemm_inputs(A, B)
    if A.ndim == 2:
        m, k = A.shape
        b_k, n = B.shape

        if k != b_k:
            raise ValueError(
                "K dimension mismatch: B must have standard row-major GEMM shape (K, N) "
                f"(got A.shape={A.shape}, B.shape={B.shape})"
            )

        gemm_1stage_validate_shape(m, n, k)
        out = _prepare_out((m, n), A.device, out)

        grid, block = gemm_1stage_launch_config(m, n)
        _gemm_1stage_pipeline_kernel[lambda: (grid, block)](A, B, out, m, n, k)
        return out

    batch, m, k = A.shape
    b_batch, b_k, n = B.shape

    if batch != b_batch or k != b_k:
        raise ValueError(
            "For batched row-major GEMM, A and B must have shapes (B, M, K) and (B, K, N) "
            f"(got A.shape={A.shape}, B.shape={B.shape})"
        )

    gemm_1stage_validate_shape(m, n, k)
    out = _prepare_out((batch, m, n), A.device, out)

    grid, block = gemm_1stage_batch_launch_config(batch, m, n)
    _gemm_1stage_pipeline_batched_kernel[lambda: (grid, block)](A, B, out, batch, m, n, k)
    return out


def gemm_1stage_pipeline_transposed_b(
    A: torch.Tensor,
    B_t: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_gemm_inputs(A, B_t)
    if A.ndim == 2:
        m, k = A.shape
        n, b_k = B_t.shape

        if k != b_k:
            raise ValueError(
                "K dimension mismatch: transposed B must have shape (N, K) "
                f"(got A.shape={A.shape}, B.shape={B_t.shape})"
            )

        gemm_1stage_validate_shape(m, n, k)
        out = _prepare_out((m, n), A.device, out)

        grid, block = gemm_1stage_launch_config(m, n)
        _gemm_1stage_pipeline_transposed_b_kernel[lambda: (grid, block)](A, B_t, out, m, n, k)
        return out

    batch, m, k = A.shape
    b_batch, n, b_k = B_t.shape

    if batch != b_batch or k != b_k:
        raise ValueError(
            "For batched transposed GEMM, A and B must have shapes (B, M, K) and (B, N, K) "
            f"(got A.shape={A.shape}, B.shape={B_t.shape})"
        )

    gemm_1stage_validate_shape(m, n, k)
    out = _prepare_out((batch, m, n), A.device, out)

    grid, block = gemm_1stage_batch_launch_config(batch, m, n)
    _gemm_1stage_pipeline_transposed_b_batched_kernel[lambda: (grid, block)](
        A, B_t, out, batch, m, n, k
    )
    return out


def gemm_1stage_pipeline(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return gemm_1stage_pipeline_row_major(A, B, out)


__all__ = [
    "gemm_1stage_pipeline",
    "gemm_1stage_pipeline_row_major",
    "gemm_1stage_pipeline_transposed_b",
]
