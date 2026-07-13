import torch
import torch.nn as nn
import avelang
import avelang.language as al

_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_K = 32
_THREAD_M = 8
_THREAD_N = 8


@avelang.jit
def tril_matmul_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    N: al.i32,
):
    bm = al.block_id(0)
    bn = al.block_id(1)
    tm = al.thread_id(0)
    tn = al.thread_id(1)

    m_start = bm * 64
    n_start = bn * 64

    m_end = m_start + 64
    if m_end <= n_start:
        return

    acc = al.full((8, 8), al.convert(0.0, al.f32), al.f32)

    A_shared = al.make_shared((64, 32), al.bf16)
    B_shared = al.make_shared((32, 64), al.bf16)

    layout_2d = al.make_layout((N, N), (N, 1))
    A = al.make_tensor(A_ptr, al.bf16, layout_2d)
    B = al.make_tensor(B_ptr, al.bf16, layout_2d)
    C = al.make_tensor(C_ptr, al.bf16, layout_2d)

    total_threads = 64
    tid = tm * 8 + tn

    k_min = n_start
    k_max = N
    if m_end < k_max:
        k_max = m_end
    if k_min < 0:
        k_min = 0

    k_start_aligned = k_min - (k_min % 32)
    if k_start_aligned < 0:
        k_start_aligned = 0

    first_loc_row = tm * 8
    first_loc_col = tn * 8

    for k_start in al.range(k_start_aligned, k_max, 32):
        for i in al.range(tid, 64 * 32, total_threads):
            loc_row = i // 32
            loc_col = i % 32
            g_row = m_start + loc_row
            g_col = k_start + loc_col
            if (g_row < N) and (g_col < N):
                A_shared[loc_row, loc_col] = A[g_row, g_col]
            else:
                A_shared[loc_row, loc_col] = al.convert(0.0, al.bf16)

        for i in al.range(tid, 32 * 64, total_threads):
            loc_row = i // 64
            loc_col = i % 64
            g_row = k_start + loc_row
            g_col = n_start + loc_col
            if (g_row < N) and (g_col < N):
                B_shared[loc_row, loc_col] = B[g_row, g_col]
            else:
                B_shared[loc_row, loc_col] = al.convert(0.0, al.bf16)

        al.syncthreads()

        for ki in al.range(32):
            actual_k = k_start + ki
            if actual_k >= k_max:
                break

            for lr in al.range(8):
                a_val = al.convert(A_shared[first_loc_row + lr, ki], al.f32)
                for lc in al.range(8):
                    b_val = al.convert(B_shared[ki, first_loc_col + lc], al.f32)
                    acc[lr, lc] = acc[lr, lc] + a_val * b_val

        al.syncthreads()

    for lr in al.range(8):
        g_row = m_start + first_loc_row + lr
        if g_row >= N:
            break
        for lc in al.range(8):
            g_col = n_start + first_loc_col + lc
            if g_col >= N:
                break
            if g_row >= g_col:
                C[g_row, g_col] = al.convert(acc[lr, lc], al.bf16)


def _launch_tril_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if not (A.is_cuda and B.is_cuda):
        A = A.cuda()
        B = B.cuda()
    N = A.shape[0]
    A = A.contiguous()
    B = B.contiguous()
    C = torch.zeros(N, N, dtype=torch.bfloat16, device=A.device)
    grid_m = (N + _BLOCK_M - 1) // _BLOCK_M
    grid_n = (N + _BLOCK_N - 1) // _BLOCK_N
    tril_matmul_kernel[lambda: ((grid_m, grid_n, 1), (_THREAD_M, _THREAD_N, 1))](
        A, B, C, N,
    )
    return C


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, A, B):
        return _launch_tril_matmul(A, B)
