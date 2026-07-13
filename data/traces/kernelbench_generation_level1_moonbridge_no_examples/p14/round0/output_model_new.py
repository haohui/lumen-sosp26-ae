import torch
import torch.nn as nn
import avelang
import avelang.language as al

@avelang.jit
def triu_matmul_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    N: al.i32,
):
    tm = al.thread_id(0)
    tn = al.thread_id(1)
    block_m = al.block_id(0)
    block_n = al.block_id(1)

    m_start = block_m * 64
    n_start = block_n * 64

    # Global memory views
    layout_2d = al.make_layout((N, N), (N, 1))
    A = al.make_tensor(A_ptr, al.bf16, layout_2d)
    B = al.make_tensor(B_ptr, al.bf16, layout_2d)
    C = al.make_tensor(C_ptr, al.bf16, layout_2d)

    # Shared memory tiles: A[64x16], B[16x64]
    A_smem = al.make_shared((64, 16), al.bf16)
    B_smem = al.make_shared((16, 64), al.bf16)

    # Register accumulator: 4x4 per thread (16x16 threads for 64x64 tile)
    acc = al.make_local((4, 4), al.f32)
    for i in al.range(4):
        for j in al.range(4):
            acc[i, j] = al.convert(0, al.f32)

    # K-loop: tile the inner dimension by 16
    for k_block in al.range(0, N, 16):
        k_start = k_block

        # Load A[64x16] tile: thread (tm,tn) handles column tn, rows tm*4..tm*4+3
        for i in al.range(4):
            src_row = m_start + tm * 4 + i
            src_col = k_start + tn
            if src_row < N and src_col < N:
                A_smem[tm * 4 + i, tn] = A[src_row, src_col]
            else:
                A_smem[tm * 4 + i, tn] = al.convert(0, al.bf16)

        # Load B[16x64] tile: thread (tm,tn) handles row tm, cols tn*4..tn*4+3
        for j in al.range(4):
            src_row = k_start + tm
            src_col = n_start + tn * 4 + j
            if src_row < N and src_col < N:
                B_smem[tm, tn * 4 + j] = B[src_row, src_col]
            else:
                B_smem[tm, tn * 4 + j] = al.convert(0, al.bf16)

        al.syncthreads()

        # Compute partial products from shared memory
        for i in al.range(4):
            for j in al.range(4):
                for kk in al.range(16):
                    a_val = al.convert(A_smem[tm * 4 + i, kk], al.f32)
                    b_val = al.convert(B_smem[kk, tn * 4 + j], al.f32)
                    acc[i, j] = acc[i, j] + a_val * b_val

        al.syncthreads()

    # Store to global: upper-triangular only (torch.triu semantics)
    for i in al.range(4):
        for j in al.range(4):
            global_i = m_start + tm * 4 + i
            global_j = n_start + tn * 4 + j
            if global_i < N and global_j < N:
                if global_i <= global_j:
                    C[global_i, global_j] = al.convert(acc[i, j], al.bf16)
                else:
                    C[global_i, global_j] = al.convert(0, al.bf16)


def _triu_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA/HIP device."
    N = A.shape[0]
    assert A.shape == (N, N) and B.shape == (N, N), "Expected square matrices."

    A = A.to(torch.bfloat16).contiguous()
    B = B.to(torch.bfloat16).contiguous()

    C = torch.empty((N, N), dtype=torch.bfloat16, device=A.device)

    BM = 64
    BN = 64
    grid_m = (N + BM - 1) // BM
    grid_n = (N + BN - 1) // BN

    triu_matmul_kernel[lambda: ((grid_m, grid_n, 1), (16, 16, 1))](A, B, C, N)

    return C


N = 4096


def get_inputs():
    A = torch.triu(torch.rand(N, N))
    B = torch.triu(torch.rand(N, N))
    return [A, B]


def get_init_inputs():
    return []


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A, B):
        return _triu_matmul(A, B)
