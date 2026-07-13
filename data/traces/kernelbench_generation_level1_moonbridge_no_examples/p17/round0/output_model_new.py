import torch
import torch.nn as nn
import avelang
import avelang.language as al

TM = 128
TN = 128
TK = 8


@avelang.jit
def gemm_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    pid_m = al.block_id(1)
    pid_n = al.block_id(0)

    m_start = pid_m * TM
    n_start = pid_n * TN

    a = al.make_tensor(a_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    b = al.make_tensor(b_ptr, al.bf16, al.make_layout((N, K), (K, 1)))
    c = al.make_tensor(c_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    tid = al.thread_id(0)
    num_threads = al.block_dim(0)

    a_sh = al.make_shared((TM, TK), al.bf16)
    b_sh = al.make_shared((TN, TK), al.bf16)

    # Each thread handles one row: 256 threads, 2 per row (each does 64 columns)
    row = tid // 2
    col_start = (tid % 2) * 64

    # Accumulator for this thread's portion of the row (64 FP32 values)
    acc = al.make_local((64,), al.f32)

    if row < TM:
        for j in al.range(64):
            acc[j] = al.convert(0.0, al.f32)

    for k in al.range(0, K, TK):
        # Cooperative load A tile into shared
        for idx in al.range(tid, TM * TK, num_threads):
            r = idx // TK
            col = idx % TK
            a_sh[r, col] = a[m_start + r, k + col]

        # Cooperative load B tile into shared
        for idx in al.range(tid, TN * TK, num_threads):
            r = idx // TK
            col = idx % TK
            b_sh[r, col] = b[n_start + r, k + col]

        al.syncthreads()

        if row < TM:
            # Pre-convert A row values to FP32 (avoids redundant converts in inner loop)
            a_vals = al.make_local((TK,), al.f32)
            for kk in al.range(TK):
                a_vals[kk] = al.convert(a_sh[row, kk], al.f32)

            for j in al.range(64):
                for kk in al.range(TK):
                    b_val = al.convert(b_sh[col_start + j, kk], al.f32)
                    acc[j] = acc[j] + a_vals[kk] * b_val

        al.syncthreads()

    if row < TM:
        for j in al.range(64):
            c[m_start + row, n_start + col_start + j] = al.convert(acc[j], al.bf16)


def gemm_host(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M_val, K_val = A.shape
    N_val, K2_val = B.shape

    A = A.contiguous()
    B = B.contiguous()

    if A.dtype != torch.bfloat16:
        A = A.to(torch.bfloat16)
    if B.dtype != torch.bfloat16:
        B = B.to(torch.bfloat16)

    C = torch.empty(M_val, N_val, dtype=torch.bfloat16, device=A.device)

    grid_x = (N_val + TN - 1) // TN
    grid_y = (M_val + TM - 1) // TM

    gemm_kernel[lambda: ((grid_x, grid_y, 1), (256, 1, 1))](
        A, B, C,
        M_val, N_val, K_val,
    )

    return C


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return gemm_host(A, B)
