import torch
import torch.nn as nn

import avelang
import avelang.language as al


M = 65536
N = 16384

WARP_SIZE = 64
NUM_WARPS = 4
TILE_M = 64
TILE_N = 64
WARP_M = 32
WARP_N = 32
BF16_BYTES = 2
THREADS = WARP_SIZE * NUM_WARPS

ELEMS_PER_THREAD = 16
HALF_ELEMS = 8
NUM_COL_GROUPS = 4
LDS_SIZE = TILE_M * TILE_N


@avelang.jit
def scale_kernel(
    A_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M_val: al.i32,
    N_val: al.i32,
    scalar_f32: al.f32,
):
    tid = al.thread_id(0)
    wid = tid // WARP_SIZE
    wtid = tid % WARP_SIZE
    warp_row = wid // 2
    warp_col = wid % 2

    bid_m = al.block_id(0)
    bid_n = al.block_id(1)

    tile_m_base = bid_m * TILE_M + warp_row * WARP_M
    tile_n_base = bid_n * TILE_N + warp_col * WARP_N

    # Build tensor views and buffer resources
    layout_a = al.make_layout((M_val, N_val), (N_val, 1))
    a = al.make_tensor(A_ptr, al.bf16, layout_a)
    a_rsrc = al.amdgpu.make_rsrc(a, M_val * N_val * BF16_BYTES)

    layout_c = al.make_layout((M_val, N_val), (N_val, 1))
    c = al.make_tensor(C_ptr, al.bf16, layout_c)

    # Shared memory for staging the 64x64 A tile
    shm_a = al.make_shared((LDS_SIZE,), al.bf16)

    # Thread grid: 64 rows x 4 column-groups = 256 threads
    row_t = tid // NUM_COL_GROUPS
    col_t = tid % NUM_COL_GROUPS

    block_row_start = bid_m * TILE_M
    block_col_start = bid_n * TILE_N

    global_row = block_row_start + row_t
    col_base = block_col_start + col_t * ELEMS_PER_THREAD
    row_byte_offset = global_row * N_val * BF16_BYTES

    # Cooperative global->LDS load of A via vectorized raw_buffer_load_x4.
    # Each thread loads 16 bf16 values (2 loads of 16 bytes each).
    col0_byte = col_base * BF16_BYTES
    loaded0 = al.amdgpu.raw_buffer_load_x4(a_rsrc, col0_byte, row_byte_offset, 0)
    frag0 = al.view(loaded0, al.Tensor((HALF_ELEMS,), al.bf16))

    col1_byte = (col_base + HALF_ELEMS) * BF16_BYTES
    loaded1 = al.amdgpu.raw_buffer_load_x4(a_rsrc, col1_byte, row_byte_offset, 0)
    frag1 = al.view(loaded1, al.Tensor((HALF_ELEMS,), al.bf16))

    lds_base = row_t * TILE_N + col_t * ELEMS_PER_THREAD
    for v in al.range(HALF_ELEMS):
        shm_a[lds_base + v] = frag0[v]
        shm_a[lds_base + HALF_ELEMS + v] = frag1[v]

    al.syncthreads()

    # Per-warp computation
    warp_lds_row_base = warp_row * WARP_M
    warp_lds_col_base = warp_col * WARP_N

    lane = wtid
    lane_row_group = lane // 32
    lane_col = lane % 32

    # MFMA C swizzle invariant: each lane owns 16 (row, col) in a 32x32 tile.
    # row = tile_row_base + 8*(acc_idx//4) + 4*(lane//32) + (acc_idx%4)
    # col = tile_col_base + (lane % 32)
    for acc_idx in al.range(16):
        row_in_tile = 8 * (acc_idx // 4) + 4 * lane_row_group + (acc_idx % 4)
        col_in_tile = lane_col

        lds_idx = (warp_lds_row_base + row_in_tile) * TILE_N + (warp_lds_col_base + col_in_tile)
        val = shm_a[lds_idx]
        result = al.convert(val, al.f32) * scalar_f32

        global_r = tile_m_base + row_in_tile
        global_c = tile_n_base + col_in_tile

        if global_r < M_val and global_c < N_val:
            c[global_r, global_c] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, s):
        if A.dtype != torch.bfloat16:
            A = A.to(torch.bfloat16)
        A = A.contiguous()
        C = torch.empty_like(A)

        M_val = A.shape[0]
        N_val = A.shape[1]
        if isinstance(s, torch.Tensor):
            s_val = float(s.item())
        else:
            s_val = float(s)

        grid_m = (M_val + TILE_M - 1) // TILE_M
        grid_n = (N_val + TILE_N - 1) // TILE_N

        scale_kernel[lambda: ((grid_m, grid_n, 1), (THREADS, 1, 1))](
            A, C, M_val, N_val, s_val
        )
        return C
