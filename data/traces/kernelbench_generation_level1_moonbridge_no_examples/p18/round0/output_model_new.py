import torch
import torch.nn as nn
import avelang
import avelang.language as al

TM = 64
TN = 64
TK = 32
SUB_M = 8
SUB_N = 8


@avelang.jit
def matmul_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    block_m = al.block_id(0)
    block_n = al.block_id(1)
    tid_m = al.thread_id(0)
    tid_n = al.thread_id(1)

    layout_a = al.make_layout((M, K), (K, 1))
    A = al.make_tensor(A_ptr, al.bf16, layout_a)
    layout_b = al.make_layout((K, N), (N, 1))
    B = al.make_tensor(B_ptr, al.bf16, layout_b)

    As = al.make_shared((64, 32), al.bf16)
    Bs = al.make_shared((32, 64), al.bf16)

    a_base_m = block_m * 64
    b_base_n = block_n * 64

    acc = al.make_local((8, 8), al.f32)
    for mi in al.range(8):
        for ni in al.range(8):
            acc[mi, ni] = al.convert(0.0, al.f32)

    for k_block in al.range(0, K, 32):
        # Cooperative load A tile (64 x 32)
        for mi in al.range(0, 64, 8):
            for ki in al.range(0, 32, 8):
                gm = a_base_m + mi + tid_m
                gk = k_block + ki + tid_n
                if gm < M and gk < K:
                    As[mi + tid_m, ki + tid_n] = A[gm, gk]
                else:
                    As[mi + tid_m, ki + tid_n] = al.convert(0.0, al.bf16)

        # Cooperative load B tile (32 x 64)
        for ki in al.range(0, 32, 8):
            for ni in al.range(0, 64, 8):
                gk = k_block + ki + tid_m
                gn = b_base_n + ni + tid_n
                if gk < K and gn < N:
                    Bs[ki + tid_m, ni + tid_n] = B[gk, gn]
                else:
                    Bs[ki + tid_m, ni + tid_n] = al.convert(0.0, al.bf16)

        al.syncthreads()

        # Compute: dot product for 8x8 sub-tile per thread
        for ki in al.range(32):
            for mi in al.range(8):
                a_val = al.convert(As[tid_m * 8 + mi, ki], al.f32)
                for ni in al.range(8):
                    b_val = al.convert(Bs[ki, tid_n * 8 + ni], al.f32)
                    acc[mi, ni] = acc[mi, ni] + a_val * b_val

        al.syncthreads()

    # Store results to global memory
    layout_c = al.make_layout((M, N), (N, 1))
    C = al.make_tensor(C_ptr, al.bf16, layout_c)

    for mi in al.range(8):
        global_m = a_base_m + tid_m * 8 + mi
        for ni in al.range(8):
            global_n = b_base_n + tid_n * 8 + ni
            if global_m < M and global_n < N:
                C[global_m, global_n] = al.convert(acc[mi, ni], al.bf16)


def avelang_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Compute C = A.T @ B.T using AveLang BF16 tiled matmul."""
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA/HIP device."
    K, M = A.shape
    N, K2 = B.shape
    assert K == K2, "Inner dimension mismatch."

    A_t = A.T.contiguous().to(torch.bfloat16)
    B_t = B.T.contiguous().to(torch.bfloat16)

    C_bf16 = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)

    grid_m = (M + TM - 1) // TM
    grid_n = (N + TN - 1) // TN
    grid = (grid_m, grid_n, 1)
    block = (8, 8, 1)

    matmul_kernel[lambda: (grid, block)](
        A_t,
        B_t,
        C_bf16,
        M,
        N,
        K,
    )

    return C_bf16


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return avelang_matmul(A, B)
