import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Tile configuration matching the reference: 128x128 blocks, K=64
GROUP_M = 128
GROUP_N = 128
GROUP_K = 64
WARP_SIZE = 64
NUM_WARPS = 4
WARP_PER_ROW = 2
WARP_PER_COL = 2
WARP_MAT_M = GROUP_M // WARP_PER_ROW  # 64
WARP_MAT_N = GROUP_N // WARP_PER_COL  # 64
M_TILES_PER_WARP = WARP_MAT_M // 16   # 4 (MFMA 16x16x16)
N_TILES_PER_WARP = WARP_MAT_N // 16   # 4
VEC_SIZE = 8
BF16_BYTES = 2
THREADS = WARP_SIZE * NUM_WARPS

# Shared memory padding: 4-row groups with 16 bf16 padding
SHM_PAD_ROWS = 4
SHM_PAD_BF16 = 16
SHM_GROUPS_A = GROUP_M // SHM_PAD_ROWS  # 32
SHM_GROUPS_B = GROUP_N // SHM_PAD_ROWS  # 32
SHM_GROUP_BF16_A = SHM_PAD_ROWS * GROUP_K + SHM_PAD_BF16  # 4*64+16 = 272
SHM_GROUP_BF16_B = SHM_PAD_ROWS * GROUP_K + SHM_PAD_BF16  # 4*64+16 = 272
SHM_TOTAL_BF16_A = SHM_GROUPS_A * SHM_GROUP_BF16_A  # 32*272 = 8704
SHM_TOTAL_BF16_B = SHM_GROUPS_B * SHM_GROUP_BF16_B  # 32*272 = 8704
SHM_CHUNKS_PER_ROW = GROUP_K // VEC_SIZE  # 8
BATCHES_PER_K = GROUP_K // 32  # 2 batches of 32 K columns each


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
            (SHM_GROUP_BF16_A // 2, GROUP_K // 2, 4, 1),
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
            (SHM_GROUP_BF16_B // 2, GROUP_K // 2, 4, 1),
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
def _matmul_batch(
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
    C_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    group_m: al.u32,
    group_n: al.u32,
    wtid: al.u32,
    warp_row: al.u32,
    warp_col: al.u32,
    acc: al.Tensor((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32),
):
    c_tensor = al.make_tensor(C_ptr, al.bf16, al.make_layout((m * n,), (1,)))
    c_rsrc = al.amdgpu.make_rsrc(c_tensor, m * n * BF16_BYTES)

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
            al.amdgpu.raw_buffer_store_x2(packed, c_rsrc, thread_offset, warp_offset, 0)


@avelang.jit
def gemm_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    k: al.u32,
):
    tid = al.thread_id(0)
    wid = tid // WARP_SIZE
    wtid = tid % WARP_SIZE
    warp_row = wid // WARP_PER_COL
    warp_col = wid % WARP_PER_COL

    group_m = al.block_id(0)
    group_n = al.block_id(1)

    # Global tensor views
    a_tensor = al.make_tensor(A_ptr, al.bf16, al.make_layout((m, k), (k, 1)))
    b_tensor = al.make_tensor(B_ptr, al.bf16, al.make_layout((k, n), (n, 1)))

    # Shared memory (padded, 1D arrays)
    shm_a = al.make_shared((SHM_TOTAL_BF16_A,), al.bf16)
    shm_b = al.make_shared((SHM_TOTAL_BF16_B,), al.bf16)

    # Register fragments
    data_a = al.make_local((M_TILES_PER_WARP, 4), al.u32)
    data_b = al.make_local((N_TILES_PER_WARP, 4), al.u32)
    acc = al.make_local((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32)

    # Zero accumulators
    for tile_m in al.range(M_TILES_PER_WARP):
        for tile_n in al.range(N_TILES_PER_WARP):
            for acc_idx in al.range(4):
                acc[tile_m, tile_n, acc_idx] = al.convert(0.0, al.f32)

    m_start = group_m * GROUP_M
    n_start = group_n * GROUP_N

    # Load A tile into shared memory (padded layout)
    # A is (M, K) = (128, 64) per tile. Store with M as padded dimension.
    total_a = GROUP_M * GROUP_K
    for offs in al.range(tid, total_a, THREADS):
        row = offs // GROUP_K  # M dimension
        col = offs % GROUP_K   # K dimension
        gr = m_start + row
        gc = col
        if gr < m and gc < k:
            row_group = row // SHM_PAD_ROWS
            row_in_group = row % SHM_PAD_ROWS
            idx = row_group * SHM_GROUP_BF16_A + row_in_group * GROUP_K + col
            shm_a[idx] = a_tensor[gr, gc]

    # Load B tile into shared memory (padded layout)
    # B is (K, N) = (64, 128) per tile. Store with N as padded dimension (transposed).
    total_b = GROUP_N * GROUP_K
    for offs in al.range(tid, total_b, THREADS):
        row = offs // GROUP_K   # N dimension
        col = offs % GROUP_K    # K dimension
        gr = col                # K index
        gc = n_start + row      # N index
        if gr < k and gc < n:
            row_group = row // SHM_PAD_ROWS
            row_in_group = row % SHM_PAD_ROWS
            idx = row_group * SHM_GROUP_BF16_B + row_in_group * GROUP_K + col
            shm_b[idx] = b_tensor[gr, gc]

    al.syncthreads()

    # Process K sub-batches
    for batch_id in al.range(BATCHES_PER_K):
        _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, batch_id, wtid, data_a)
        _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, batch_id, wtid, data_b)
        _matmul_batch(data_a, data_b, acc)

    # Write results
    _write_results(C_ptr, m, n, group_m, group_n, wtid, warp_row, warp_col, acc)


def avelang_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA/HIP device."

    A = A.to(torch.bfloat16).contiguous()
    B = B.to(torch.bfloat16).contiguous()

    M, K_a = A.shape
    K_b, N = B.shape
    assert K_a == K_b, f"K dimension mismatch: {K_a} vs {K_b}"
    K = K_a

    C = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)

    grid_m = (M + GROUP_M - 1) // GROUP_M
    grid_n = (N + GROUP_N - 1) // GROUP_N

    gemm_kernel[lambda: ((grid_m, grid_n, 1), (THREADS, 1, 1))](
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        M,
        N,
        K,
    )

    return C


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return avelang_matmul(A, B)
