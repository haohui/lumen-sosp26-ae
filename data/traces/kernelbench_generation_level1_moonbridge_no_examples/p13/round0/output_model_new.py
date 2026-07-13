import torch
import torch.nn as nn
import avelang
import avelang.language as al

GROUP_M = 128
GROUP_N = 128
GROUP_K = 64
WARP_SIZE = 64
NUM_WARPS = 4
WARP_PER_ROW = 2
WARP_PER_COL = 2
WARP_MAT_M = GROUP_M // WARP_PER_ROW
WARP_MAT_N = GROUP_N // WARP_PER_COL
M_TILES = WARP_MAT_M // 16
N_TILES = WARP_MAT_N // 16
THREADS = WARP_SIZE * NUM_WARPS
SHM_PAD_ROWS = 4
SHM_PAD_BF16 = 16
SHM_GROUPS_A = GROUP_M // SHM_PAD_ROWS
SHM_GROUPS_B = GROUP_N // SHM_PAD_ROWS
SHM_GROUP_BF16 = SHM_PAD_ROWS * GROUP_K + SHM_PAD_BF16
SHM_GROUP_WORDS = SHM_GROUP_BF16 // 2
SHM_TOTAL_BF16_A = SHM_GROUPS_A * SHM_GROUP_BF16
SHM_TOTAL_BF16_B = SHM_GROUPS_B * SHM_GROUP_BF16
SHM_CHUNKS_PER_ROW = GROUP_K // 8


@avelang.jit
def gemm_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
    N: al.i32,
):
    tid = al.thread_id(0)
    wid = tid // WARP_SIZE
    wtid = tid % WARP_SIZE
    warp_row = wid // WARP_PER_COL
    warp_col = wid % WARP_PER_COL
    group_m = al.block_id(0)
    group_n = al.block_id(1)

    a_layout = al.make_layout((N, N), (N, 1))
    b_layout = al.make_layout((N, N), (N, 1))
    a = al.make_tensor(a_ptr, al.bf16, a_layout)
    b = al.make_tensor(b_ptr, al.bf16, b_layout)
    c_flat = al.make_tensor(
        c_ptr, al.bf16, al.make_layout((N * N,), (1,)))

    shm_a = al.make_shared((SHM_TOTAL_BF16_A,), al.bf16)
    shm_b = al.make_shared((SHM_TOTAL_BF16_B,), al.bf16)

    acc = al.make_local((M_TILES, N_TILES, 4), al.f32)
    for tm in al.range(M_TILES):
        for tn in al.range(N_TILES):
            for i in al.range(4):
                acc[tm, tn, i] = al.convert(0.0, al.f32)

    shm_a_mfma = al.view(
        shm_a, al.u32,
        al.make_layout(
            (SHM_GROUPS_A, SHM_PAD_ROWS, SHM_CHUNKS_PER_ROW, 4),
            (SHM_GROUP_WORDS, GROUP_K // 2, 4, 1),
        ),
    )
    shm_b_mfma = al.view(
        shm_b, al.u32,
        al.make_layout(
            (SHM_GROUPS_B, SHM_PAD_ROWS, SHM_CHUNKS_PER_ROW, 4),
            (SHM_GROUP_WORDS, GROUP_K // 2, 4, 1),
        ),
    )

    for k_block in al.range(0, N, GROUP_K):
        for idx in al.range(tid, GROUP_M * GROUP_K, THREADS):
            row = idx // GROUP_K
            col = idx % GROUP_K
            g_row = group_m * GROUP_M + row
            g_col = k_block + col
            val = al.convert(0.0, al.bf16)
            if g_row < N and g_col < N:
                val = a[g_row, g_col]
            rg = row // SHM_PAD_ROWS
            ri = row - rg * SHM_PAD_ROWS
            shm_a[rg * SHM_GROUP_BF16 + ri * GROUP_K + col] = val

        for idx in al.range(tid, GROUP_N * GROUP_K, THREADS):
            row = idx // GROUP_K
            col = idx % GROUP_K
            g_row = group_n * GROUP_N + row
            g_col = k_block + col
            val = al.convert(0.0, al.bf16)
            if g_row < N and g_col < N:
                val = b[g_row, g_col]
            rg = row // SHM_PAD_ROWS
            ri = row - rg * SHM_PAD_ROWS
            shm_b[rg * SHM_GROUP_BF16 + ri * GROUP_K + col] = val

        al.syncthreads()

        for tm in al.range(M_TILES):
            r = warp_row * WARP_MAT_M + tm * 16 + (wtid % 16)
            rg = r // SHM_PAD_ROWS
            ri = r - rg * SHM_PAD_ROWS
            a0_data = shm_a_mfma[rg, ri, wtid // 16]
            a1_data = shm_a_mfma[rg, ri, (wtid // 16) + 4]

            fa0 = al.view(a0_data, al.Tensor((2, 2, 1), al.u32))
            fa1 = al.view(a1_data, al.Tensor((2, 2, 1), al.u32))

            for tn in al.range(N_TILES):
                r2 = warp_col * WARP_MAT_N + tn * 16 + (wtid % 16)
                rg2 = r2 // SHM_PAD_ROWS
                ri2 = r2 - rg2 * SHM_PAD_ROWS
                b0_data = shm_b_mfma[rg2, ri2, wtid // 16]
                b1_data = shm_b_mfma[rg2, ri2, (wtid // 16) + 4]

                fb0 = al.view(b0_data, al.Tensor((2, 2, 1), al.u32))
                fb1 = al.view(b1_data, al.Tensor((2, 2, 1), al.u32))

                acc[tm, tn] = al.amdgpu.mfma_16x16x16_bf16_f32(
                    fa0[0], fb0[0], acc[tm, tn])
                acc[tm, tn] = al.amdgpu.mfma_16x16x16_bf16_f32(
                    fa0[1], fb0[1], acc[tm, tn])
                acc[tm, tn] = al.amdgpu.mfma_16x16x16_bf16_f32(
                    fa1[0], fb1[0], acc[tm, tn])
                acc[tm, tn] = al.amdgpu.mfma_16x16x16_bf16_f32(
                    fa1[1], fb1[1], acc[tm, tn])

        al.syncthreads()

    for tm in al.range(M_TILES):
        for tn in al.range(N_TILES):
            rg = wtid // 16
            lc = wtid % 16
            for i in al.range(4):
                r_off = rg * 4 + i
                c_off = lc
                g_row = (group_m * GROUP_M + warp_row * WARP_MAT_M
                         + tm * 16 + r_off)
                g_col = (group_n * GROUP_N + warp_col * WARP_MAT_N
                         + tn * 16 + c_off)
                if g_row < N and g_col < N:
                    c_flat[g_row * N + g_col] = al.convert(
                        acc[tm, tn, i], al.bf16)


def _launch_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    N = A.shape[0]
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty(N, N, dtype=torch.bfloat16, device=A.device)
    grid_m = (N + GROUP_M - 1) // GROUP_M
    grid_n = (N + GROUP_N - 1) // GROUP_N
    gemm_kernel[lambda: ((grid_m, grid_n, 1), (THREADS, 1, 1))](A, B, C, N)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return _launch_gemm(A, B)
