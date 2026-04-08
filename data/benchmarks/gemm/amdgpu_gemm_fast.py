"""AMDGPU GEMM kernels and helpers."""

import substrate
import substrate.language as S
import torch

WARP_SIZE = 64
NUM_WARPS = 4
GROUP_M = 128
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

SCHED_MASK_MFMA = 0x8
SCHED_MASK_BUFFER_LOAD = 0x20
SCHED_MASK_DS_READ = 0x100
SCHED_MASK_DS_WRITE = 0x200
SCHED_MASK_S_BARRIER = 0x0

# Shared memory padding (shm_layout_config=0 from petit-kernel).
SHM_PAD_ROWS_A = WARP_MAT_M // MMA_M  # kReadRowsAPerStep
SHM_PAD_ROWS_B = WARP_MAT_N // MMA_N  # kReadRowsBPerStep
SHM_PAD_BYTES = 32  # kPaddingSizeA, kPaddingSizeB
SHM_PAD_U32 = SHM_PAD_BYTES // 4
SHM_PAD_INTERVAL_A = SHM_PAD_ROWS_A * GROUP_K * 2
SHM_PAD_INTERVAL_B = SHM_PAD_ROWS_B * GROUP_K * 2
SHM_A_SIZE = GROUP_M * GROUP_K * 2
SHM_B_SIZE = GROUP_N * GROUP_K * 2

SHM_A_U32 = (SHM_A_SIZE + SHM_A_SIZE // SHM_PAD_INTERVAL_A * SHM_PAD_BYTES) // 4
SHM_B_U32 = (SHM_B_SIZE + SHM_B_SIZE // SHM_PAD_INTERVAL_B * SHM_PAD_BYTES) // 4


def gemm_launch_config(m, n):
    m_groups = (m + GROUP_M - 1) // GROUP_M
    n_groups = (n + GROUP_N - 1) // GROUP_N
    grid = (m_groups * n_groups, 1, 1)
    block = (NUM_WARPS * WARP_SIZE, 1, 1)
    return grid, block


def gemm_1stage_validate_shape(m, n, k):
    if m % GROUP_M != 0 or n % GROUP_N != 0 or k % GROUP_K != 0:
        raise ValueError(
            "M and N must be multiples of 128 and K must be a multiple of 64 "
            f"(got m={m}, n={n}, k={k})."
        )


@substrate.jit
def fetch_global_and_store_shm_pipeline(
    rsrc_a: S.Tensor((4,), S.u32),
    rsrc_b: S.Tensor((4,), S.u32),
    k: S.u32,
    tile_a_idx_row: S.u32,
    tile_a_idx_col: S.u32,
    tile_b_idx_row: S.u32,
    tile_b_idx_col: S.u32,
    shm: S.Tensor((SHM_A_U32 + SHM_B_U32,), S.u32),
    reg_a: S.Tensor(
        (
            GROUP_M * GROUP_K // VEC_SIZE // THREADS,
            4,
        ),
        S.u32,
    ),
    reg_b: S.Tensor(
        (
            GROUP_N * GROUP_K // VEC_SIZE // THREADS,
            4,
        ),
        S.u32,
    ),
):
    REG_SIZE_A = GROUP_M * GROUP_K // VEC_SIZE // THREADS
    REG_SIZE_B = GROUP_N * GROUP_K // VEC_SIZE // THREADS
    K_VEC = GROUP_K // VEC_SIZE
    load_store_rows_interval = THREADS // K_VEC

    tid = S.thread_id(0)

    store_base_u32_a = ((tid * 16) + (tid * 16) // SHM_PAD_INTERVAL_A * SHM_PAD_BYTES) // 4
    store_offset_u32_a = (
        (load_store_rows_interval * GROUP_K * 2)
        + (load_store_rows_interval * GROUP_K * 2) // SHM_PAD_INTERVAL_A * SHM_PAD_BYTES
    ) // 4
    store_base_u4_a = store_base_u32_a
    store_offset_u4_a = store_offset_u32_a

    store_base_u32_b = ((tid * 16) + (tid * 16) // SHM_PAD_INTERVAL_B * SHM_PAD_BYTES) // 4
    store_offset_u32_b = (
        (load_store_rows_interval * GROUP_K * 2)
        + (load_store_rows_interval * GROUP_K * 2) // SHM_PAD_INTERVAL_B * SHM_PAD_BYTES
    ) // 4
    store_base_u4_b = store_base_u32_b
    store_offset_u4_b = store_offset_u32_b

    row = tid // K_VEC
    col = tid % K_VEC
    thread_base = row * k * 2 + col * 16

    tile_a_base = tile_a_idx_row * (GROUP_M * k * 2) + tile_a_idx_col * (GROUP_K * 2)
    tile_b_base = tile_b_idx_row * (GROUP_N * k * 2) + tile_b_idx_col * (GROUP_K * 2)

    stride_bytes = load_store_rows_interval * k * 2

    for i in S.range(REG_SIZE_B):
        offset = tile_b_base + i * stride_bytes
        shm[store_base_u4_b + i * store_offset_u4_b] = reg_b[i]
        reg_b[i] = S.amdgpu.raw_buffer_load_x4(rsrc_b, thread_base, offset, 0)

    for i in S.range(REG_SIZE_A):
        offset = tile_a_base + i * stride_bytes
        shm[store_base_u4_a + i * store_offset_u4_a + SHM_B_U32] = reg_a[i]
        reg_a[i] = S.amdgpu.raw_buffer_load_x4(rsrc_a, thread_base, offset, 0)


@substrate.jit
def load_global(
    rsrc_a: S.Tensor((4,), S.u32),
    rsrc_b: S.Tensor((4,), S.u32),
    k: S.u32,
    tile_a_idx_row: S.u32,
    tile_a_idx_col: S.u32,
    tile_b_idx_row: S.u32,
    tile_b_idx_col: S.u32,
    reg_a: S.Tensor(
        (
            GROUP_M * GROUP_K // VEC_SIZE // THREADS,
            4,
        ),
        S.u32,
    ),
    reg_b: S.Tensor(
        (
            GROUP_N * GROUP_K // VEC_SIZE // THREADS,
            4,
        ),
        S.u32,
    ),
):
    REG_SIZE_A = GROUP_M * GROUP_K // VEC_SIZE // THREADS
    REG_SIZE_B = GROUP_N * GROUP_K // VEC_SIZE // THREADS
    K_VEC = GROUP_K // VEC_SIZE
    load_store_rows_interval = THREADS // K_VEC

    tid = S.thread_id(0)

    row = tid // K_VEC
    col = tid % K_VEC
    thread_base = row * k * 2 + col * 16

    tile_a_base = tile_a_idx_row * (GROUP_M * k * 2) + tile_a_idx_col * (GROUP_K * 2)
    tile_b_base = tile_b_idx_row * (GROUP_N * k * 2) + tile_b_idx_col * (GROUP_K * 2)

    stride_bytes = load_store_rows_interval * k * 2

    for i in S.range(REG_SIZE_B):
        reg_b[i] = S.amdgpu.raw_buffer_load_x4(
            rsrc_b, thread_base, tile_b_base + i * stride_bytes, 0
        )

    for i in S.range(REG_SIZE_A):
        reg_a[i] = S.amdgpu.raw_buffer_load_x4(
            rsrc_a, thread_base, tile_a_base + i * stride_bytes, 0
        )


@substrate.jit
def store_shm(
    shm: S.Tensor((SHM_A_U32 + SHM_B_U32,), S.u32),
    reg_a: S.Tensor(
        (
            GROUP_M * GROUP_K // VEC_SIZE // THREADS,
            4,
        ),
        S.u32,
    ),
    reg_b: S.Tensor(
        (
            GROUP_N * GROUP_K // VEC_SIZE // THREADS,
            4,
        ),
        S.u32,
    ),
):
    REG_SIZE_A = GROUP_M * GROUP_K // VEC_SIZE // THREADS
    REG_SIZE_B = GROUP_N * GROUP_K // VEC_SIZE // THREADS
    K_VEC = GROUP_K // VEC_SIZE
    load_store_rows_interval = THREADS // K_VEC

    tid = S.thread_id(0)

    store_base_u32_a = ((tid * 16) + (tid * 16) // SHM_PAD_INTERVAL_A * SHM_PAD_BYTES) // 4
    store_offset_u32_a = (
        (load_store_rows_interval * GROUP_K * 2)
        + (load_store_rows_interval * GROUP_K * 2) // SHM_PAD_INTERVAL_A * SHM_PAD_BYTES
    ) // 4
    store_base_u4_a = store_base_u32_a
    store_offset_u4_a = store_offset_u32_a

    store_base_u32_b = ((tid * 16) + (tid * 16) // SHM_PAD_INTERVAL_B * SHM_PAD_BYTES) // 4
    store_offset_u32_b = (
        (load_store_rows_interval * GROUP_K * 2)
        + (load_store_rows_interval * GROUP_K * 2) // SHM_PAD_INTERVAL_B * SHM_PAD_BYTES
    ) // 4
    store_base_u4_b = store_base_u32_b
    store_offset_u4_b = store_offset_u32_b

    for i in S.range(REG_SIZE_B):
        shm[store_base_u4_b + i * store_offset_u4_b] = reg_b[i]

    for i in S.range(REG_SIZE_A):
        shm[store_base_u4_a + i * store_offset_u4_a + SHM_B_U32] = reg_a[i]


@substrate.jit
def load_shm_to_regs(
    shm: S.Tensor((SHM_A_U32 + SHM_B_U32,), S.u32),
    batch_id: S.u32,
    tile_a: S.Tensor((M_TILES_PER_WARP, 4), S.u32),
    tile_b: S.Tensor((N_TILES_PER_WARP, 4), S.u32),
):
    tid = S.thread_id(0)
    wid = tid // WARP_SIZE
    wtid = tid % WARP_SIZE
    warp_row_a = wid // WARP_PER_COL % WARP_PER_ROW
    warp_row_b = wid % WARP_PER_COL
    mma_k = wtid % MMA_K
    lane_group = wtid // MMA_K

    start_row_a = warp_row_a * (M_TILES_PER_WARP * TILE_M) + mma_k * SHM_PAD_ROWS_A
    start_row_b = warp_row_b * (N_TILES_PER_WARP * TILE_N) + mma_k * SHM_PAD_ROWS_B

    start_row_a_bytes = start_row_a * GROUP_K * 2
    start_row_b_bytes = start_row_b * GROUP_K * 2
    start_row_a_bytes = (
        start_row_a_bytes + (start_row_a_bytes // SHM_PAD_INTERVAL_A) * SHM_PAD_BYTES
    )
    start_row_b_bytes = (
        start_row_b_bytes + (start_row_b_bytes // SHM_PAD_INTERVAL_B) * SHM_PAD_BYTES
    )

    col_bytes = (lane_group * 16) + (batch_id * 64)

    base_a_u4 = (start_row_a_bytes + col_bytes) // 16 + SHM_B_U32 // 4
    base_b_u4 = (start_row_b_bytes + col_bytes) // 16

    step_bytes_a = GROUP_K * 2
    step_bytes_a = step_bytes_a + (step_bytes_a // SHM_PAD_INTERVAL_A) * SHM_PAD_BYTES
    step_u4_a = step_bytes_a // 16

    step_bytes_b = GROUP_K * 2
    step_bytes_b = step_bytes_b + (step_bytes_b // SHM_PAD_INTERVAL_B) * SHM_PAD_BYTES
    step_u4_b = step_bytes_b // 16

    shm_layout = S.make_layout(((SHM_A_U32 + SHM_B_U32) // 4, 4), (4, 1))
    shm_u4 = S.view(shm, S.u32, shm_layout)

    for i in S.range(N_TILES_PER_WARP):
        tile_b[i] = shm_u4[base_b_u4 + i * step_u4_b]

    for i in S.range(M_TILES_PER_WARP):
        tile_a[i] = shm_u4[base_a_u4 + i * step_u4_a]


@substrate.jit
def matmul_from_regs(
    r_a: S.Tensor((M_TILES_PER_WARP, 4), S.u32),
    r_b: S.Tensor((N_TILES_PER_WARP, 4), S.u32),
    acc: S.Tensor(
        (M_TILES_PER_WARP * N_TILES_PER_WARP, 4),
        S.f32,
    ),
):
    for m in S.range(M_TILES_PER_WARP):
        r_a_bf16 = S.view(r_a[m], S.Tensor((2, 4), S.bf16))
        for n in S.range(N_TILES_PER_WARP):
            idx = m * N_TILES_PER_WARP + n
            r_b_bf16 = S.view(r_b[n], S.Tensor((2, 4), S.bf16))
            acc[idx] = S.amdgpu.mfma_f32_16x16x16_bf16(
                r_a_bf16[0], r_b_bf16[0], acc[idx]
            )

    for m in S.range(M_TILES_PER_WARP):
        r_a_bf16 = S.view(r_a[m], S.Tensor((2, 4), S.bf16))
        for n in S.range(N_TILES_PER_WARP):
            idx = m * N_TILES_PER_WARP + n
            r_b_bf16 = S.view(r_b[n], S.Tensor((2, 4), S.bf16))
            acc[idx] = S.amdgpu.mfma_f32_16x16x16_bf16(
                r_a_bf16[1], r_b_bf16[1], acc[idx]
            )


@substrate.jit
def hot_loop_scheduler():
    S.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)

    for _ in S.range(8):
        S.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, 1, 0)
        S.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)

    S.amdgpu.sched_group_barrier(SCHED_MASK_S_BARRIER, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)

    S.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 3, 0)

    S.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 3, 0)

    S.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)

    S.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 3, 0)

    S.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)

    S.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 3, 0)

    S.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)

    S.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
    S.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 3, 0)

    # barrier for hacky LLVM
    S.amdgpu.sched_group_barrier(SCHED_MASK_S_BARRIER, 1, 0)

    for _ in S.range(8):
        S.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, 1, 0)
        S.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)


@substrate.jit
def write_results(
    id_m: S.u32,
    id_n: S.u32,
    acc: S.Tensor(
        (M_TILES_PER_WARP * N_TILES_PER_WARP, 4),
        S.f32,
    ),
    rsrc: S.Tensor((4,), S.u32),
    m: S.u32,
    n: S.u32,
):
    tid = S.thread_id(0)
    wid = tid // WARP_SIZE
    wtid = tid % WARP_SIZE
    wid_m = wid // WARP_PER_COL % WARP_PER_ROW
    wid_n = wid % WARP_PER_COL
    mma_k = wtid % MMA_K
    batch_k = wtid // MMA_K

    warp_step_a = M_TILES_PER_WARP * 16
    warp_step_b = N_TILES_PER_WARP * 16
    large_step_a = SHM_PAD_ROWS_A * 16
    large_step_b = SHM_PAD_ROWS_B * 16

    output = S.make_local((4, N_TILES_PER_WARP), S.u16)

    global_base = ((id_m * GROUP_M // VEC_SIZE) * n + id_n * GROUP_N // VEC_SIZE) * 16

    for m_tile in S.range(M_TILES_PER_WARP):
        for n_tile in S.range(N_TILES_PER_WARP):
            t = S.make_local((4,), S.u16)
            v = S.view(t, S.Tensor((2,), S.u32))
            idx = m_tile * N_TILES_PER_WARP + n_tile

            for l in S.range(2):
                lo = S.bitcast(acc[idx, l * 2 + 0], S.u32)
                hi = S.bitcast(acc[idx, l * 2 + 1], S.u32)
                tmp = S.amdgpu.perm(hi, lo, 0x07060302)
                v[l] = tmp

            for i in S.range(4):
                output[i, n_tile] = t[i]

        row_base = (
            wid_m * warp_step_a
            + batch_k * SHM_PAD_ROWS_A * 4
            + m_tile // SHM_PAD_ROWS_A * large_step_a
            + m_tile % SHM_PAD_ROWS_A
        )
        col_base = wid_n * warp_step_b + mma_k * SHM_PAD_ROWS_B

        base = global_base + (row_base * n + col_base) * 2

        output_packed = S.view(output, S.Tensor((4, N_TILES_PER_WARP // 2), S.u32))

        for i in S.range(4):
            row_offset = i * SHM_PAD_ROWS_A
            for j in S.range(N_TILES_PER_WARP // 4):
                t = j * 4
                col_offset = t // SHM_PAD_ROWS_B * large_step_b + t % SHM_PAD_ROWS_B
                offset = (row_offset * n + col_offset) * 2
                output_slice = S.make_local((2,), S.u32)
                output_slice[0] = output_packed[i, j * 2 + 0]
                output_slice[1] = output_packed[i, j * 2 + 1]
                S.amdgpu.raw_buffer_store_x2(output_slice, rsrc, base, offset, 0)


@substrate.jit
def wgm_mapping(m: S.u32, n: S.u32) -> (S.u32, S.u32):
    linear_group_id = S.block_id(0)
    m_groups = m // GROUP_M
    n_groups = n // GROUP_N

    total_groups = m_groups * n_groups

    cu_count = S.convert(38 * 8, S.u32)
    wgm_xcc = S.convert(8, S.u32)
    workgroup_mapping = S.convert(32, S.u32)

    linear_group_limit = (total_groups // wgm_xcc) * wgm_xcc
    cu_base = (linear_group_id // cu_count) * cu_count
    cu_xcc = (linear_group_id % cu_count) // wgm_xcc
    cu_base = cu_base + cu_xcc

    cu_tail_limit = (total_groups // cu_count) * cu_count
    active_cu = (
        (total_groups % cu_count) if (linear_group_id > cu_tail_limit) else cu_count
    )
    cu_xcc_stride = (active_cu // wgm_xcc) * (linear_group_id % wgm_xcc)
    linear_group_mapped = cu_base + cu_xcc_stride

    linear_group_id = (
        linear_group_mapped
        if (linear_group_id < linear_group_limit)
        else linear_group_id
    )

    group_m = linear_group_id // n_groups
    group_n = linear_group_id - group_m * n_groups

    mapping_block = group_m // workgroup_mapping
    mapping_linear = group_n + (group_m % workgroup_mapping) * n_groups
    mapping_groups = m_groups // workgroup_mapping
    mapping_tail = m_groups % workgroup_mapping
    mapping_tail = workgroup_mapping if (mapping_tail == 0) else mapping_tail

    mapping_span = (
        mapping_tail if (mapping_block >= mapping_groups) else workgroup_mapping
    )

    group_n = mapping_linear // mapping_span
    group_m = mapping_linear % mapping_span
    group_m = group_m + mapping_block * workgroup_mapping

    return group_m, group_n


@substrate.jit
def _gemm_1stage_pipeline_kernel_batch2(
    A: S.Pointer(S.u32),
    B: S.Pointer(S.u32),
    C: S.Pointer(S.u32),
    m: S.u32,
    n: S.u32,
    k: S.u32,
):
    id_m, id_n = wgm_mapping(m, n)
    k_vec = k // VEC_SIZE

    shm = S.make_shared((SHM_A_U32 + SHM_B_U32,), S.u32)

    acc = S.make_local((M_TILES_PER_WARP * N_TILES_PER_WARP, 4), S.f32)
    for i in S.range(M_TILES_PER_WARP * N_TILES_PER_WARP):
        for t in S.range(4):
            acc[i, t] = 0

    rsrc_a = S.make_local((4,), S.u32)
    rsrc_b = S.make_local((4,), S.u32)
    rsrc_c = S.make_local((4,), S.u32)

    layout_a = S.make_layout(
        (m, k_vec, 4),
        (k_vec * 4, 4, 1),
    )
    layout_b = S.make_layout(
        (n, k_vec, 4),
        (k_vec * 4, 4, 1),
    )

    g_a = S.make_tensor(A, S.u32, layout_a)
    g_b = S.make_tensor(B, S.u32, layout_b)
    rsrc_a = S.amdgpu.make_rsrc(g_a, m * k * 2)
    rsrc_b = S.amdgpu.make_rsrc(g_b, n * k * 2)

    layout_c = S.make_layout((m, n // 2), (n // 2, 1))
    g_c = S.make_tensor(C, S.u32, layout_c)
    rsrc_c = S.amdgpu.make_rsrc(g_c, m * n * 2)

    k_total = k // GROUP_K
    k_total_u32 = S.convert(k_total, S.u32)
    k_stagger_mask = S.convert(0x7, S.u32)
    k_stagger_stride = S.convert(4, S.u32)
    stagger_data = id_n
    k_start = (stagger_data & k_stagger_mask) * k_stagger_stride
    k_start = k_start if (k_start < k_total_u32) else S.convert(0, S.u32)

    reg_a = S.make_local(
        (
            GROUP_M * GROUP_K // VEC_SIZE // THREADS,
            4,
        ),
        S.u32,
    )
    reg_b = S.make_local(
        (
            GROUP_N * GROUP_K // VEC_SIZE // THREADS,
            4,
        ),
        S.u32,
    )
    tile_a0 = S.make_local((M_TILES_PER_WARP, 4), S.u32)
    tile_b0 = S.make_local((N_TILES_PER_WARP, 4), S.u32)
    tile_a1 = S.make_local((M_TILES_PER_WARP, 4), S.u32)
    tile_b1 = S.make_local((N_TILES_PER_WARP, 4), S.u32)

    # Prime Stage: Load Tile 0
    load_global(rsrc_a, rsrc_b, k, id_m, k_start, id_n, k_start, reg_a, reg_b)
    # Store
    store_shm(shm, reg_a, reg_b)
    S.syncthreads()

    k_reg = k_start + 1
    k_reg = k_reg - k_total_u32 if (k_reg >= k_total_u32) else k_reg
    load_global(rsrc_a, rsrc_b, k, id_m, k_reg, id_n, k_reg, reg_a, reg_b)

    load_shm_to_regs(shm, 0, tile_a0, tile_b0)

    k_idx = S.convert(0, S.u32)
    while k_idx + 3 < k_total_u32:
        # Step 1
        k_idx = k_idx + 1
        load_shm_to_regs(shm, 1, tile_a1, tile_b1)
        matmul_from_regs(tile_a0, tile_b0, acc)
        S.syncthreads()

        k_load = k_reg + 1
        k_load = k_load - k_total_u32 if (k_load >= k_total_u32) else k_load
        fetch_global_and_store_shm_pipeline(
            rsrc_a, rsrc_b, k, id_m, k_load, id_n, k_load, shm, reg_a, reg_b
        )
        k_reg = k_load
        S.syncthreads()

        load_shm_to_regs(shm, 0, tile_a0, tile_b0)
        matmul_from_regs(tile_a1, tile_b1, acc)

        # Step 2
        k_idx = k_idx + 1
        load_shm_to_regs(shm, 1, tile_a1, tile_b1)
        matmul_from_regs(tile_a0, tile_b0, acc)
        S.syncthreads()

        k_load = k_reg + 1
        k_load = k_load - k_total_u32 if (k_load >= k_total_u32) else k_load
        fetch_global_and_store_shm_pipeline(
            rsrc_a, rsrc_b, k, id_m, k_load, id_n, k_load, shm, reg_a, reg_b
        )
        k_reg = k_load
        S.syncthreads()

        load_shm_to_regs(shm, 0, tile_a0, tile_b0)
        matmul_from_regs(tile_a1, tile_b1, acc)

        hot_loop_scheduler()
        hot_loop_scheduler()

    if k_idx + 2 < k_total_u32:
        k_idx = k_idx + 1
        load_shm_to_regs(shm, 1, tile_a1, tile_b1)
        matmul_from_regs(tile_a0, tile_b0, acc)
        S.syncthreads()

        k_load = k_reg + 1
        k_load = k_load - k_total_u32 if (k_load >= k_total_u32) else k_load
        fetch_global_and_store_shm_pipeline(
            rsrc_a, rsrc_b, k, id_m, k_load, id_n, k_load, shm, reg_a, reg_b
        )
        k_reg = k_load
        S.syncthreads()

        load_shm_to_regs(shm, 0, tile_a0, tile_b0)
        matmul_from_regs(tile_a1, tile_b1, acc)

        hot_loop_scheduler()

    if k_idx + 1 < k_total_u32:
        k_idx = k_idx + 1
        load_shm_to_regs(shm, 1, tile_a1, tile_b1)
        matmul_from_regs(tile_a0, tile_b0, acc)
        S.syncthreads()

        store_shm(shm, reg_a, reg_b)
        S.syncthreads()

        load_shm_to_regs(shm, 0, tile_a0, tile_b0)
        matmul_from_regs(tile_a1, tile_b1, acc)

    if k_idx < k_total_u32:
        load_shm_to_regs(shm, 1, tile_a1, tile_b1)
        matmul_from_regs(tile_a0, tile_b0, acc)
        matmul_from_regs(tile_a1, tile_b1, acc)

    write_results(id_m, id_n, acc, rsrc_c, m, n)


def gemm_1stage_pipeline_transposed_b(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("A and B must be torch.Tensor")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(
            f"A and B must be rank-2 tensors (got A.ndim={A.ndim}, B.ndim={B.ndim})"
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
        raise ValueError(
            f"A and B must be on the same device (got {A.device} and {B.device})"
        )

    m, k = A.shape
    n, b_k = B.shape

    if k != b_k:
        raise ValueError(
            "K dimension mismatch: B must be pre-transposed with shape (N, K) "
            f"(got A.shape={A.shape}, B.shape={B.shape})"
        )

    gemm_1stage_validate_shape(m, n, k)

    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be torch.Tensor")
    if out is None or out.numel() == 0:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)
    elif out.ndim != 2 or out.shape != (m, n):
        raise ValueError(f"out must have shape {(m, n)} (got {tuple(out.shape)})")
    elif out.dtype != torch.bfloat16:
        raise TypeError(f"out must be torch.bfloat16 (got {out.dtype})")
    elif out.device != A.device:
        raise ValueError(f"out must be on {A.device} (got {out.device})")
    
    out[:, :] = 0

    grid, block = gemm_launch_config(m, n)
    _gemm_1stage_pipeline_kernel_batch2[lambda: (grid, block)](A, B, out, m, n, k)
    return out


__all__ = [
    "gemm_1stage_pipeline_transposed_b",
]
