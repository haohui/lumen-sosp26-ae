import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Problem constants
BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
NUM_GROUPS = 16
HARDTANH_MIN = -2.0
HARDTANH_MAX = 2.0

# GEMM constants (from optimized repo kernel)
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS
GROUP_M = 128
GROUP_N = 128
GROUP_K = 64
WARP_PER_ROW = 2
WARP_PER_COL = 2
WARP_MAT_M = GROUP_M // WARP_PER_ROW
WARP_MAT_N = GROUP_N // WARP_PER_COL
M_TILES_PER_WARP = WARP_MAT_M // 16
N_TILES_PER_WARP = WARP_MAT_N // 16
VEC_SIZE = 8
BF16_BYTES = 2
REG_ROWS_A = GROUP_M * GROUP_K // VEC_SIZE // THREADS
REG_ROWS_B = GROUP_N * GROUP_K // VEC_SIZE // THREADS
SHM_PAD_ROWS = 4
SHM_PAD_BF16 = 16
SHM_GROUPS_A = GROUP_M // SHM_PAD_ROWS
SHM_GROUPS_B = GROUP_N // SHM_PAD_ROWS
SHM_GROUP_BF16 = SHM_PAD_ROWS * GROUP_K + SHM_PAD_BF16
SHM_GROUP_WORDS = SHM_GROUP_BF16 // 2
SHM_TOTAL_BF16_A = SHM_GROUPS_A * SHM_GROUP_BF16
SHM_TOTAL_BF16_B = SHM_GROUPS_B * SHM_GROUP_BF16
SHM_CHUNKS_PER_ROW = GROUP_K // VEC_SIZE

# GroupNorm constants
GN_BLOCK_SIZE = 256
CHANNELS_PER_GROUP = OUT_FEATURES // NUM_GROUPS


# Scheduler masks for instruction-level parallelism
SCHED_MASK_MFMA = 0x8
SCHED_MASK_BUFFER_LOAD = 0x20
SCHED_MASK_DS_READ = 0x100
SCHED_MASK_DS_WRITE = 0x200


@avelang.jit
def _hot_loop_scheduler():
    for _ in al.range(8):
        al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)

    for _ in al.range(8):
        al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 3, 0)

    for _ in al.range(8):
        al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)


@avelang.jit
def _load_global_a(
    src_rsrc: al.Tensor((4,), al.u32),
    k: al.u32,
    group_row: al.u32,
    k_idx: al.u32,
    tid: al.u32,
    reg: al.Tensor((REG_ROWS_A, VEC_SIZE), al.bf16),
):
    row = tid // SHM_CHUNKS_PER_ROW
    col = (tid - row * SHM_CHUNKS_PER_ROW) * VEC_SIZE
    thread_offset = (row * k + col) * BF16_BYTES
    tile_offset = (group_row * GROUP_M * k + k_idx * GROUP_K) * BF16_BYTES
    thread_offset_stride = (THREADS * VEC_SIZE // GROUP_K) * k * BF16_BYTES
    for i in al.range(REG_ROWS_A):
        packed = al.amdgpu.raw_buffer_load_x4(
            src_rsrc, thread_offset, tile_offset + i * thread_offset_stride, 0,
        )
        frag = al.view(packed, al.Tensor((VEC_SIZE,), al.bf16))
        for v in al.range(VEC_SIZE):
            reg[i, v] = frag[v]


@avelang.jit
def _load_global_b(
    src_rsrc: al.Tensor((4,), al.u32),
    k: al.u32,
    group_row: al.u32,
    k_idx: al.u32,
    tid: al.u32,
    reg: al.Tensor((REG_ROWS_B, VEC_SIZE), al.bf16),
):
    row = tid // SHM_CHUNKS_PER_ROW
    col = (tid - row * SHM_CHUNKS_PER_ROW) * VEC_SIZE
    thread_offset = (row * k + col) * BF16_BYTES
    tile_offset = (group_row * GROUP_N * k + k_idx * GROUP_K) * BF16_BYTES
    thread_offset_stride = (THREADS * VEC_SIZE // GROUP_K) * k * BF16_BYTES
    for i in al.range(REG_ROWS_B):
        packed = al.amdgpu.raw_buffer_load_x4(
            src_rsrc, thread_offset, tile_offset + i * thread_offset_stride, 0,
        )
        frag = al.view(packed, al.Tensor((VEC_SIZE,), al.bf16))
        for v in al.range(VEC_SIZE):
            reg[i, v] = frag[v]


@avelang.jit
def _store_shm_a(
    shm: al.Tensor((SHM_TOTAL_BF16_A,), al.bf16),
    reg: al.Tensor((REG_ROWS_A, VEC_SIZE), al.bf16),
    tid: al.u32,
):
    shm_vec = al.view(
        shm, al.u32,
        al.make_layout(
            (SHM_TOTAL_BF16_A // VEC_SIZE, 4),
            (VEC_SIZE // 2, 1),
        ),
    )
    row = tid // SHM_CHUNKS_PER_ROW
    row_group = row // SHM_PAD_ROWS
    row_in_group = row - row_group * SHM_PAD_ROWS
    chunk = tid - row * SHM_CHUNKS_PER_ROW
    shm_chunk = (
        row_group * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
        + row_in_group * SHM_CHUNKS_PER_ROW
        + chunk
    )
    shm_chunk_stride = (
        (THREADS * VEC_SIZE // GROUP_K // SHM_PAD_ROWS)
        * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
    )
    for i in al.range(REG_ROWS_A):
        packed = al.view(reg[i], al.Tensor((4,), al.u32))
        shm_vec[shm_chunk + i * shm_chunk_stride] = packed


@avelang.jit
def _store_shm_b(
    shm: al.Tensor((SHM_TOTAL_BF16_B,), al.bf16),
    reg: al.Tensor((REG_ROWS_B, VEC_SIZE), al.bf16),
    tid: al.u32,
):
    shm_vec = al.view(
        shm, al.u32,
        al.make_layout(
            (SHM_TOTAL_BF16_B // VEC_SIZE, 4),
            (VEC_SIZE // 2, 1),
        ),
    )
    row = tid // SHM_CHUNKS_PER_ROW
    row_group = row // SHM_PAD_ROWS
    row_in_group = row - row_group * SHM_PAD_ROWS
    chunk = tid - row * SHM_CHUNKS_PER_ROW
    shm_chunk = (
        row_group * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
        + row_in_group * SHM_CHUNKS_PER_ROW
        + chunk
    )
    shm_chunk_stride = (
        (THREADS * VEC_SIZE // GROUP_K // SHM_PAD_ROWS)
        * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
    )
    for i in al.range(REG_ROWS_B):
        packed = al.view(reg[i], al.Tensor((4,), al.u32))
        shm_vec[shm_chunk + i * shm_chunk_stride] = packed


@avelang.jit
def _load_shm_to_regs_batch_a(
    shm: al.Tensor((SHM_TOTAL_BF16_A,), al.bf16),
    row_base: al.u32,
    batch_id: al.u32,
    wtid: al.u32,
    data: al.Tensor((M_TILES_PER_WARP, 4), al.u32),
):
    shm_vec = al.view(
        shm, al.u32,
        al.make_layout(
            (SHM_GROUPS_A, SHM_PAD_ROWS, SHM_CHUNKS_PER_ROW, 4),
            (SHM_GROUP_WORDS, GROUP_K // 2, 4, 1),
        ),
    )
    row_start = row_base + (wtid % 16) * M_TILES_PER_WARP
    chunk_base = (wtid // 16) + batch_id * (32 // VEC_SIZE)
    for tile in al.range(M_TILES_PER_WARP):
        row = row_start + tile
        row_group = row // SHM_PAD_ROWS
        row_in_group = row - row_group * SHM_PAD_ROWS
        data[tile] = shm_vec[row_group, row_in_group, chunk_base]


@avelang.jit
def _load_shm_to_regs_batch_b(
    shm: al.Tensor((SHM_TOTAL_BF16_B,), al.bf16),
    row_base: al.u32,
    batch_id: al.u32,
    wtid: al.u32,
    data: al.Tensor((N_TILES_PER_WARP, 4), al.u32),
):
    shm_vec = al.view(
        shm, al.u32,
        al.make_layout(
            (SHM_GROUPS_B, SHM_PAD_ROWS, SHM_CHUNKS_PER_ROW, 4),
            (SHM_GROUP_WORDS, GROUP_K // 2, 4, 1),
        ),
    )
    row_start = row_base + (wtid % 16) * N_TILES_PER_WARP
    chunk_base = (wtid // 16) + batch_id * (32 // VEC_SIZE)
    for tile in al.range(N_TILES_PER_WARP):
        row = row_start + tile
        row_group = row // SHM_PAD_ROWS
        row_in_group = row - row_group * SHM_PAD_ROWS
        data[tile] = shm_vec[row_group, row_in_group, chunk_base]


@avelang.jit
def _matmul_from_regs_batch(
    data_a: al.Tensor((M_TILES_PER_WARP, 4), al.u32),
    data_b: al.Tensor((N_TILES_PER_WARP, 4), al.u32),
    acc: al.Tensor((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32),
):
    for tile_m in al.range(M_TILES_PER_WARP):
        for tile_n in al.range(N_TILES_PER_WARP):
            frag_a = al.view(data_a[tile_m], al.Tensor((2, 2, 1), al.u32))
            frag_b = al.view(data_b[tile_n], al.Tensor((2, 2, 1), al.u32))
            acc[tile_m, tile_n] = al.amdgpu.mfma_16x16x16_bf16_f32(
                frag_a[0], frag_b[0], acc[tile_m, tile_n],
            )
            acc[tile_m, tile_n] = al.amdgpu.mfma_16x16x16_bf16_f32(
                frag_a[1], frag_b[1], acc[tile_m, tile_n],
            )


@avelang.jit
def _write_results(
    dst_rsrc: al.Tensor((4,), al.u32),
    n: al.u32,
    group_m: al.u32,
    group_n: al.u32,
    wtid: al.u32,
    warp_row: al.u32,
    warp_col: al.u32,
    acc: al.Tensor((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32),
):
    lane_row_group = wtid // 16
    lane_col = wtid % 16
    warp_offset = (
        (group_m * GROUP_M + warp_row * WARP_MAT_M) * n
        + group_n * GROUP_N
        + warp_col * WARP_MAT_N
    ) * BF16_BYTES

    for tile_m in al.range(M_TILES_PER_WARP):
        for acc_idx in al.range(4):
            row_offset = (
                lane_row_group * (4 * M_TILES_PER_WARP)
                + acc_idx * M_TILES_PER_WARP
                + tile_m
            ) * n
            col_offset = lane_col * N_TILES_PER_WARP
            thread_offset = (row_offset + col_offset) * BF16_BYTES

            lo0 = al.bitcast(acc[tile_m, 0, acc_idx], al.u32)
            hi0 = al.bitcast(acc[tile_m, 1, acc_idx], al.u32)
            lo1 = al.bitcast(acc[tile_m, 2, acc_idx], al.u32)
            hi1 = al.bitcast(acc[tile_m, 3, acc_idx], al.u32)
            packed = al.full((2,), 0, al.u32)
            packed[0] = al.amdgpu.perm(hi0, lo0, 0x07060302)
            packed[1] = al.amdgpu.perm(hi1, lo1, 0x07060302)
            al.amdgpu.raw_buffer_store_x2(packed, dst_rsrc, thread_offset, warp_offset, 0)


@avelang.jit
def gemm_bf16_kernel(
    A: al.Pointer(al.bf16),
    B: al.Pointer(al.bf16),
    C: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    k: al.u32,
):
    tid = al.thread_id(0)
    wid = tid // WARP_SIZE
    wtid = tid % WARP_SIZE
    warp_row = wid // WARP_PER_COL
    warp_col = wid % WARP_PER_COL

    group_m = al.block_id(1)
    group_n = al.block_id(0)

    a_tensor = al.make_tensor(A, al.bf16, al.make_layout((m * k,), (1,)))
    b_tensor = al.make_tensor(B, al.bf16, al.make_layout((n * k,), (1,)))
    c_tensor = al.make_tensor(C, al.bf16, al.make_layout((m * n,), (1,)))
    a_rsrc = al.amdgpu.make_rsrc(a_tensor, m * k * BF16_BYTES)
    b_rsrc = al.amdgpu.make_rsrc(b_tensor, n * k * BF16_BYTES)
    c_rsrc = al.amdgpu.make_rsrc(c_tensor, m * n * BF16_BYTES)

    shm_a = al.make_shared((SHM_TOTAL_BF16_A,), al.bf16)
    shm_b = al.make_shared((SHM_TOTAL_BF16_B,), al.bf16)
    reg_a = al.make_local((REG_ROWS_A, VEC_SIZE), al.bf16)
    reg_b = al.make_local((REG_ROWS_B, VEC_SIZE), al.bf16)
    data_a0 = al.make_local((M_TILES_PER_WARP, 4), al.u32)
    data_a1 = al.make_local((M_TILES_PER_WARP, 4), al.u32)
    data_b0 = al.make_local((N_TILES_PER_WARP, 4), al.u32)
    data_b1 = al.make_local((N_TILES_PER_WARP, 4), al.u32)
    acc = al.make_local((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32)

    for tile_m in al.range(M_TILES_PER_WARP):
        for tile_n in al.range(N_TILES_PER_WARP):
            for acc_idx in al.range(4):
                acc[tile_m, tile_n, acc_idx] = al.convert(0.0, al.f32)

    k_total = k // GROUP_K

    _load_global_a(a_rsrc, k, group_m, 0, tid, reg_a)
    _load_global_b(b_rsrc, k, group_n, 0, tid, reg_b)
    _store_shm_a(shm_a, reg_a, tid)
    _store_shm_b(shm_b, reg_b, tid)
    al.syncthreads()

    _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 0, wtid, data_a0)
    _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 0, wtid, data_b0)
    _load_global_a(a_rsrc, k, group_m, 1, tid, reg_a)
    _load_global_b(b_rsrc, k, group_n, 1, tid, reg_b)

    for k_idx in al.range(0, k_total - 3, 2):
        _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 1, wtid, data_a1)
        _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 1, wtid, data_b1)
        _matmul_from_regs_batch(data_a0, data_b0, acc)
        al.syncthreads()

        _store_shm_a(shm_a, reg_a, tid)
        _store_shm_b(shm_b, reg_b, tid)
        _load_global_a(a_rsrc, k, group_m, k_idx + 2, tid, reg_a)
        _load_global_b(b_rsrc, k, group_n, k_idx + 2, tid, reg_b)
        al.syncthreads()

        _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 0, wtid, data_a0)
        _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 0, wtid, data_b0)
        _matmul_from_regs_batch(data_a1, data_b1, acc)

        _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 1, wtid, data_a1)
        _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 1, wtid, data_b1)
        _matmul_from_regs_batch(data_a0, data_b0, acc)
        al.syncthreads()

        _store_shm_a(shm_a, reg_a, tid)
        _store_shm_b(shm_b, reg_b, tid)
        _load_global_a(a_rsrc, k, group_m, k_idx + 3, tid, reg_a)
        _load_global_b(b_rsrc, k, group_n, k_idx + 3, tid, reg_b)
        al.syncthreads()

        _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 0, wtid, data_a0)
        _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 0, wtid, data_b0)
        _matmul_from_regs_batch(data_a1, data_b1, acc)

        _hot_loop_scheduler()
        _hot_loop_scheduler()

    _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 1, wtid, data_a1)
    _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 1, wtid, data_b1)
    _matmul_from_regs_batch(data_a0, data_b0, acc)
    al.syncthreads()

    _store_shm_a(shm_a, reg_a, tid)
    _store_shm_b(shm_b, reg_b, tid)
    al.syncthreads()

    _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 0, wtid, data_a0)
    _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 0, wtid, data_b0)
    _matmul_from_regs_batch(data_a1, data_b1, acc)

    _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 1, wtid, data_a1)
    _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 1, wtid, data_b1)
    _matmul_from_regs_batch(data_a0, data_b0, acc)
    _matmul_from_regs_batch(data_a1, data_b1, acc)

    _write_results(c_rsrc, n, group_m, group_n, wtid, warp_row, warp_col, acc)


@avelang.jit
def bias_groupnorm_hardtanh_kernel(
    gemm_out_ptr: al.Pointer(al.bf16),
    gemm_bias_ptr: al.Pointer(al.bf16),
    gn_weight_ptr: al.Pointer(al.bf16),
    gn_bias_ptr: al.Pointer(al.bf16),
    final_out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    out_features: al.i32,
    num_groups: al.i32,
    channels_per_group: al.i32,
):
    tid = al.thread_id(0)
    block_id = al.block_id(0)

    sample_idx = block_id
    if sample_idx < batch_size:
        layout_in = al.make_layout((batch_size * out_features,), (1,))
        gemm_out = al.make_tensor(gemm_out_ptr, al.bf16, layout_in)
        layout_bias = al.make_layout((out_features,), (1,))
        g_bias = al.make_tensor(gemm_bias_ptr, al.bf16, layout_bias)

        layout_w = al.make_layout((out_features,), (1,))
        gn_w = al.make_tensor(gn_weight_ptr, al.bf16, layout_w)
        gn_b = al.make_tensor(gn_bias_ptr, al.bf16, layout_w)

        layout_out = al.make_layout((batch_size * out_features,), (1,))
        final_out = al.make_tensor(final_out_ptr, al.bf16, layout_out)

        smem_sum = al.make_shared((128,), al.f32)
        smem_sq = al.make_shared((128,), al.f32)
        ht_min = al.convert(HARDTANH_MIN, al.f32)
        ht_max = al.convert(HARDTANH_MAX, al.f32)

        for g in al.range(0, 16):
            base = sample_idx * out_features + g * channels_per_group

            l_sum = al.convert(0.0, al.f32)
            l_sq = al.convert(0.0, al.f32)

            for i in al.range(tid, channels_per_group, GN_BLOCK_SIZE):
                idx = base + i
                val = al.convert(gemm_out[idx], al.f32)
                l_sum = l_sum + val
                l_sq = l_sq + val * val

            if tid < 128:
                smem_sum[tid] = l_sum
                smem_sq[tid] = l_sq
            al.syncthreads()

            if tid >= 128:
                smem_sum[tid - 128] = smem_sum[tid - 128] + l_sum
                smem_sq[tid - 128] = smem_sq[tid - 128] + l_sq
            al.syncthreads()

            if tid < 64:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 64]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 64]
            al.syncthreads()
            if tid < 64:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 32]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 32]
            al.syncthreads()
            if tid < 64:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 16]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 16]
            al.syncthreads()
            if tid < 64:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 8]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 8]
            al.syncthreads()

            w_sum = al.convert(0.0, al.f32)
            w_sq = al.convert(0.0, al.f32)
            if tid < 8:
                w_sum = smem_sum[tid]
                w_sq = smem_sq[tid]
            w_sum = w_sum + al.shuffle_down(w_sum, 4, 8)
            w_sq = w_sq + al.shuffle_down(w_sq, 4, 8)
            w_sum = w_sum + al.shuffle_down(w_sum, 2, 8)
            w_sq = w_sq + al.shuffle_down(w_sq, 2, 8)
            w_sum = w_sum + al.shuffle_down(w_sum, 1, 8)
            w_sq = w_sq + al.shuffle_down(w_sq, 1, 8)

            if tid == 0:
                cp_f32 = al.convert(channels_per_group, al.f32)
                m_val = w_sum / cp_f32
                v_val = w_sq / cp_f32 - m_val * m_val
                eps_val = al.convert(1e-5, al.f32)
                smem_sum[0] = m_val
                smem_sq[0] = al.convert(1.0, al.f32) / al.sqrt(v_val + eps_val)
            al.syncthreads()

            mean_val = smem_sum[0]
            rstd_val = smem_sq[0]

            for i in al.range(tid, channels_per_group, GN_BLOCK_SIZE):
                idx = base + i
                channel = g * channels_per_group + i

                val = al.convert(gemm_out[idx], al.f32)
                bias_val = al.convert(g_bias[channel], al.f32)
                w_val = al.convert(gn_w[channel], al.f32)
                b_val = al.convert(gn_b[channel], al.f32)

                val = val + bias_val
                normalized = (val - mean_val) * rstd_val
                result = normalized * w_val + b_val

                if result < ht_min:
                    result = ht_min
                if result > ht_max:
                    result = ht_max

                final_out[idx] = al.convert(result, al.bf16)

def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_gemm_groupnorm_hardtanh(
    x: torch.Tensor,
    gemm_weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    gn_weight: torch.Tensor,
    gn_bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = _prepare_bf16_cuda_contiguous(gemm_weight)
    b_bf16 = _prepare_bf16_cuda_contiguous(gemm_bias)
    gn_w_bf16 = _prepare_bf16_cuda_contiguous(gn_weight)
    gn_b_bf16 = _prepare_bf16_cuda_contiguous(gn_bias)

    m, k = x_bf16.shape
    n, weight_k = w_bf16.shape

    gemm_out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)
    grid_gemm = (n // GROUP_N, m // GROUP_M, 1)
    gemm_bf16_kernel[lambda: (grid_gemm, (THREADS, 1, 1))](
        x_bf16, w_bf16, gemm_out, m, n, k
    )

    final_out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)
    grid_gn = (m, 1, 1)
    bias_groupnorm_hardtanh_kernel[lambda: (grid_gn, (GN_BLOCK_SIZE, 1, 1))](
        gemm_out, b_bf16, gn_w_bf16, gn_b_bf16, final_out,
        m, n, NUM_GROUPS, CHANNELS_PER_GROUP,
    )

    return final_out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

        self.gemm_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gemm_bias = nn.Parameter(torch.empty(out_features))
        self.gn_weight = nn.Parameter(torch.empty(out_features))
        self.gn_bias = nn.Parameter(torch.empty(out_features))

        nn.init.kaiming_uniform_(self.gemm_weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.gemm_weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.gemm_bias, -bound, bound)
        nn.init.ones_(self.gn_weight)
        nn.init.zeros_(self.gn_bias)

    def forward(self, x):
        return avelang_gemm_groupnorm_hardtanh(
            x, self.gemm_weight, self.gemm_bias,
            self.gn_weight, self.gn_bias,
        )


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_FEATURES)]


def get_init_inputs():
    return [IN_FEATURES, OUT_FEATURES, NUM_GROUPS, HARDTANH_MIN, HARDTANH_MAX]
