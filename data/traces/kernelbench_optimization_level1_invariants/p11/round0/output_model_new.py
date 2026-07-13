import torch
import torch.nn as nn
import avelang
import avelang.language as al


WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS
GROUP_M = 128
GROUP_N = 128
GROUP_K = 16
MMA_M = 32
MMA_N = 32
WARPS_M = 2
WARPS_N = 2
M_TILES_PER_WARP = GROUP_M // (WARPS_M * MMA_M)
N_TILES_PER_WARP = GROUP_N // (WARPS_N * MMA_N)

VEC_ELEMS = 8
BF16_BYTES = 2
A_VECS_PER_ROW = GROUP_K // VEC_ELEMS
B_VECS_PER_ROW = GROUP_K // VEC_ELEMS
SHM_A_VECS = GROUP_M * A_VECS_PER_ROW
SHM_B_VECS = GROUP_N * B_VECS_PER_ROW
GLOBAL_LOADS_A = SHM_A_VECS // THREADS
GLOBAL_LOADS_B = SHM_B_VECS // THREADS

ROW_U32 = A_VECS_PER_ROW * 4


@avelang.jit
def _load_global_a_to_shm(
    shm_a: al.Tensor((SHM_A_VECS, 4), al.u32),
    a_rsrc: al.Tensor((4,), al.u32),
    block_m: al.u32,
    k_base: al.u32,
    k: al.u32,
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    idx = tid
    for _ in al.range(GLOBAL_LOADS_A):
        row = idx // A_VECS_PER_ROW
        col_vec = idx % A_VECS_PER_ROW
        off = ((block_m * GROUP_M + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
        shm_a[idx] = al.amdgpu.raw_buffer_load_x4(a_rsrc, zero, off, 0)
        idx += THREADS


@avelang.jit
def _load_global_b_to_shm(
    shm_b: al.Tensor((SHM_B_VECS, 4), al.u32),
    b_rsrc: al.Tensor((4,), al.u32),
    block_n: al.u32,
    k_base: al.u32,
    n: al.u32,
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    idx = tid
    for _ in al.range(GLOBAL_LOADS_B):
        row = idx // B_VECS_PER_ROW
        col_vec = idx % B_VECS_PER_ROW
        off = ((block_n * GROUP_N + row) * n + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
        shm_b[idx] = al.amdgpu.raw_buffer_load_x4(b_rsrc, zero, off, 0)
        idx += THREADS


@avelang.jit
def gemm_4d_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    k: al.u32,
):
    tid = al.thread_id(0)
    block_n = al.block_id(0)
    block_m = al.block_id(1)
    wid = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    warp_row = wid // WARPS_N
    warp_col = wid % WARPS_N

    a_memref = al.make_tensor(a_ptr, al.bf16, al.make_layout((m * k,), (1,)))
    b_memref = al.make_tensor(b_ptr, al.bf16, al.make_layout((n * k,), (1,)))
    c_memref = al.make_tensor(c_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    a_rsrc = al.amdgpu.make_rsrc(a_memref, m * k * BF16_BYTES)
    b_rsrc = al.amdgpu.make_rsrc(b_memref, n * k * BF16_BYTES)

    shm_a = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_B_VECS, 4), al.u32)
    data_a = al.make_local((M_TILES_PER_WARP, 4), al.u32)
    data_b = al.make_local((N_TILES_PER_WARP, 4), al.u32)
    acc = al.make_local((M_TILES_PER_WARP, N_TILES_PER_WARP, 16), al.f32)

    for ti in al.range(M_TILES_PER_WARP):
        for tj in al.range(N_TILES_PER_WARP):
            for r in al.range(16):
                acc[ti, tj, r] = 0

    k_tiles = k // GROUP_K
    for kt in al.range(k_tiles):
        k_base = kt * GROUP_K
        _load_global_a_to_shm(shm_a, a_rsrc, block_m, k_base, k, tid)
        _load_global_b_to_shm(shm_b, b_rsrc, block_n, k_base, k, tid)
        al.syncthreads()

        shm_a_flat = al.view(shm_a, al.Tensor((SHM_A_VECS * 4,), al.u32))
        shm_b_flat = al.view(shm_b, al.Tensor((SHM_B_VECS * 4,), al.u32))

        for ti in al.range(M_TILES_PER_WARP):
            tile_a = warp_row * M_TILES_PER_WARP + ti
            ra = tile_a * MMA_M + (lane % MMA_M)
            kg_a = (lane // MMA_M) * 2
            ba = ra * ROW_U32
            data_a[ti, 0] = shm_a_flat[ba + kg_a]
            data_a[ti, 1] = shm_a_flat[ba + kg_a + 1]
            data_a[ti, 2] = shm_a_flat[ba + 4 + kg_a]
            data_a[ti, 3] = shm_a_flat[ba + 5 + kg_a]

        for tj in al.range(N_TILES_PER_WARP):
            tile_b = warp_col * N_TILES_PER_WARP + tj
            rb = tile_b * MMA_N + (lane % MMA_N)
            kg_b = (lane // MMA_N) * 2
            bb = rb * ROW_U32
            data_b[tj, 0] = shm_b_flat[bb + kg_b]
            data_b[tj, 1] = shm_b_flat[bb + kg_b + 1]
            data_b[tj, 2] = shm_b_flat[bb + 4 + kg_b]
            data_b[tj, 3] = shm_b_flat[bb + 5 + kg_b]

        for ti in al.range(M_TILES_PER_WARP):
            frag_a = al.view(data_a[ti], al.Tensor((2, 2, 1), al.u32))
            for tj in al.range(N_TILES_PER_WARP):
                frag_b = al.view(data_b[tj], al.Tensor((2, 2, 1), al.u32))
                acc[ti, tj] = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a[0], frag_b[0], acc[ti, tj])
                acc[ti, tj] = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a[1], frag_b[1], acc[ti, tj])

        al.syncthreads()

    lane_group = lane // MMA_N
    lane_col = lane % MMA_N
    block_row_base = block_m * GROUP_M
    block_col_base = block_n * GROUP_N

    for tj in al.range(N_TILES_PER_WARP):
        col = block_col_base + (warp_col * N_TILES_PER_WARP + tj) * MMA_N + lane_col
        if col < n:
            for ti in al.range(M_TILES_PER_WARP):
                row_base = block_row_base + (warp_row * M_TILES_PER_WARP + ti) * MMA_M
                for t in al.range(16):
                    row = row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
                    if row < m:
                        c_memref[row, col] = al.convert(acc[ti, tj, t], al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        b_dim, i_dim, j_dim, l_dim = A.shape
        l_dim_b, k_dim = B.shape
        m_flat = b_dim * i_dim * j_dim

        A_flat = A.reshape(m_flat, l_dim).contiguous()
        B_flat = B.T.contiguous()
        C_flat = torch.empty((m_flat, k_dim), device=A.device, dtype=torch.bfloat16)

        m_tiles = (m_flat + GROUP_M - 1) // GROUP_M
        n_tiles = (k_dim + GROUP_N - 1) // GROUP_N
        grid = (n_tiles, m_tiles, 1)

        gemm_4d_kernel[lambda: (grid, (THREADS, 1, 1))](
            A_flat, B_flat, C_flat,
            m_flat, k_dim, l_dim,
        )

        return C_flat.reshape(b_dim, i_dim, j_dim, k_dim)
