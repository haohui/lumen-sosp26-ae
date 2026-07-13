import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 32
LOAD_PASSES = 2


@avelang.jit
def bmm_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    N: al.i32,
    M: al.i32,
    K: al.i32,
    L: al.i32,
    K_BLOCKS: al.i32,
):
    n = al.block_id(0)
    m_block = al.block_id(1)
    l_block = al.block_id(2)

    tid_m = al.thread_id(0)
    tid_n = al.thread_id(1)

    m = m_block * BLOCK_M + tid_m
    l = l_block * BLOCK_N + tid_n

    NM = N * M
    A_flat = al.make_tensor(A_ptr, al.bf16, al.make_layout((NM, K), (K, 1)))
    B_flat = al.make_tensor(B_ptr, al.bf16, al.make_layout((K, L), (L, 1)))
    C_flat = al.make_tensor(C_ptr, al.bf16, al.make_layout((NM, L), (L, 1)))

    nm_idx = n * M + m_block * BLOCK_M + tid_m

    A_tile = al.make_shared((BLOCK_M, BLOCK_K), al.bf16)
    B_tile = al.make_shared((BLOCK_K, BLOCK_N), al.bf16)

    acc = al.convert(0.0, al.f32)
    z16 = al.convert(0.0, al.bf16)

    for k_block in al.range(K_BLOCKS):
        k_start = k_block * BLOCK_K
        l_glob = l_block * BLOCK_N

        for p in al.range(LOAD_PASSES):
            cn = p * BLOCK_N + tid_n
            kk = k_start + cn
            if cn < BLOCK_K:
                if kk < K:
                    A_tile[tid_m, cn] = A_flat[nm_idx, kk]
                else:
                    A_tile[tid_m, cn] = z16

        for p in al.range(LOAD_PASSES):
            rm = p * BLOCK_M + tid_m
            kk = k_start + rm
            if rm < BLOCK_K:
                if kk < K:
                    B_tile[rm, tid_n] = B_flat[kk, l_glob + tid_n]
                else:
                    B_tile[rm, tid_n] = z16

        al.syncthreads()

        for kk in al.range(BLOCK_K):
            a_val = al.convert(A_tile[tid_m, kk], al.f32)
            b_val = al.convert(B_tile[kk, tid_n], al.f32)
            acc = acc + a_val * b_val

        al.syncthreads()

    if m < M and l < L:
        C_flat[nm_idx, l] = al.convert(acc, al.bf16)


def avelang_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Host-side launcher for the 3D tensor-matrix multiply kernel."""
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA/HIP device."
    N, M, K_input = A.shape
    Kb, L = B.shape
    assert K_input == Kb, f"K dimension mismatch: {K_input} vs {Kb}"
    K_ = K_input

    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty(N, M, L, dtype=torch.bfloat16, device=A.device)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_l = (L + BLOCK_N - 1) // BLOCK_N
    k_blocks = (K_ + BLOCK_K - 1) // BLOCK_K

    bmm_kernel[lambda: ((N, grid_m, grid_l), (BLOCK_M, BLOCK_N, 1))](
        A, B, C,
        N, M, K_, L, k_blocks,
    )

    return C


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A, B):
        return avelang_bmm(A, B)
