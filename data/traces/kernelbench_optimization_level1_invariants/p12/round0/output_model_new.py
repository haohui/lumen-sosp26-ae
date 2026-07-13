import torch
import torch.nn as nn
import avelang
import avelang.language as al


M = 4096
N = 4096
TILE_M = 64
TILE_N = 64


@avelang.jit
def diag_left_tiled_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    N_val: al.i32,
    M_val: al.i32,
):
    layout_a = al.make_layout((N_val,), (1,))
    A = al.make_tensor(A_ptr, al.bf16, layout_a)
    layout_b = al.make_layout((N_val, M_val), (M_val, 1))
    B = al.make_tensor(B_ptr, al.bf16, layout_b)
    layout_c = al.make_layout((N_val, M_val), (M_val, 1))
    C = al.make_tensor(C_ptr, al.bf16, layout_c)

    block_m = al.block_id(0) * TILE_M
    block_n = al.block_id(1) * TILE_N

    tid = al.thread_id(0)
    num_threads = al.block_dim(0)

    # LDS for A (TILE_M bf16) and B (TILE_M x TILE_N bf16)
    A_lds = al.make_shared((TILE_M,), al.bf16)
    B_lds = al.make_shared((TILE_M, TILE_N), al.bf16)

    # Cooperative load of A into LDS
    for i in al.range(tid, TILE_M, num_threads):
        row = block_m + i
        if row < N_val:
            A_lds[i] = A[row]

    # Cooperative load of B into LDS
    total_b = TILE_M * TILE_N
    for idx in al.range(tid, total_b, num_threads):
        b_row = idx // TILE_N
        b_col = idx % TILE_N
        global_row = block_m + b_row
        global_col = block_n + b_col
        if global_row < N_val:
            if global_col < M_val:
                B_lds[b_row, b_col] = B[global_row, global_col]

    al.syncthreads()

    # Each thread computes multiple output elements
    for idx in al.range(tid, TILE_M * TILE_N, num_threads):
        out_row = idx // TILE_N
        out_col = idx % TILE_N
        global_row = block_m + out_row
        global_col = block_n + out_col
        if global_row < N_val:
            if global_col < M_val:
                a_val = A_lds[out_row]
                b_val = B_lds[out_row, out_col]
                C[global_row, global_col] = a_val * b_val


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise RuntimeError('Input dtypes must be bfloat16 for the AveLang kernel.')
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty_like(B)
        N_val = A.shape[0]
        M_val = B.shape[1]
        grid_m = (N_val + TILE_M - 1) // TILE_M
        grid_n = (M_val + TILE_N - 1) // TILE_N
        diag_left_tiled_kernel[lambda: ((grid_m, grid_n, 1), (256, 1, 1))](
            A.data_ptr(), B.data_ptr(), C.data_ptr(), N_val, M_val,
        )
        return C
