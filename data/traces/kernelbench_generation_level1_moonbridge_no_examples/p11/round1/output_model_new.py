import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def ttm_kernel(
    A: al.Pointer(al.bf16),
    B: al.Pointer(al.bf16),
    C: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    """Tiled GEMM kernel: C[M,N] = A[M,K] * B[K,N] in BF16 with FP32 accumulation."""
    layout_A = al.make_layout((M, K), (K, 1))
    A_t = al.make_tensor(A, al.bf16, layout_A)
    layout_B = al.make_layout((K, N), (N, 1))
    B_t = al.make_tensor(B, al.bf16, layout_B)
    layout_C = al.make_layout((M, N), (N, 1))
    C_t = al.make_tensor(C, al.bf16, layout_C)

    block_m = al.block_id(0)
    block_n = al.block_id(1)
    tid = al.thread_id(0)
    thread_m = tid // 16
    thread_n = tid % 16

    start_m = block_m * 64
    start_n = block_n * 64

    As = al.make_shared((64, 32), al.bf16)
    Bs = al.make_shared((32, 64), al.bf16)

    acc = al.make_local((4, 4), al.f32)
    for mi in al.range(4):
        for ni in al.range(4):
            acc[mi, ni] = al.convert(0.0, al.f32)

    for k_block in al.range(0, K, 32):
        # Cooperative load of A tile [BM x BK] into shared memory
        for mi in al.range(4):
            row = thread_m + mi * 16
            for ki in al.range(2):
                col = thread_n + ki * 16
                gr = start_m + row
                gc = k_block + col
                if gr < M and gc < K:
                    As[row, col] = A_t[gr, gc]
                else:
                    As[row, col] = al.convert(0.0, al.bf16)

        # Cooperative load of B tile [BK x BN] into shared memory
        for ki in al.range(2):
            row = thread_m + ki * 16
            for ni in al.range(4):
                col = thread_n + ni * 16
                gr = k_block + row
                gc = start_n + col
                if gr < K and gc < N:
                    Bs[row, col] = B_t[gr, gc]
                else:
                    Bs[row, col] = al.convert(0.0, al.bf16)

        al.syncthreads()

        # Compute: each thread handles a 4x4 sub-tile
        for k in al.range(32):
            a0 = al.convert(As[thread_m * 4 + 0, k], al.f32)
            a1 = al.convert(As[thread_m * 4 + 1, k], al.f32)
            a2 = al.convert(As[thread_m * 4 + 2, k], al.f32)
            a3 = al.convert(As[thread_m * 4 + 3, k], al.f32)

            b0 = al.convert(Bs[k, thread_n * 4 + 0], al.f32)
            b1 = al.convert(Bs[k, thread_n * 4 + 1], al.f32)
            b2 = al.convert(Bs[k, thread_n * 4 + 2], al.f32)
            b3 = al.convert(Bs[k, thread_n * 4 + 3], al.f32)

            acc[0, 0] = acc[0, 0] + a0 * b0
            acc[0, 1] = acc[0, 1] + a0 * b1
            acc[0, 2] = acc[0, 2] + a0 * b2
            acc[0, 3] = acc[0, 3] + a0 * b3
            acc[1, 0] = acc[1, 0] + a1 * b0
            acc[1, 1] = acc[1, 1] + a1 * b1
            acc[1, 2] = acc[1, 2] + a1 * b2
            acc[1, 3] = acc[1, 3] + a1 * b3
            acc[2, 0] = acc[2, 0] + a2 * b0
            acc[2, 1] = acc[2, 1] + a2 * b1
            acc[2, 2] = acc[2, 2] + a2 * b2
            acc[2, 3] = acc[2, 3] + a2 * b3
            acc[3, 0] = acc[3, 0] + a3 * b0
            acc[3, 1] = acc[3, 1] + a3 * b1
            acc[3, 2] = acc[3, 2] + a3 * b2
            acc[3, 3] = acc[3, 3] + a3 * b3

        al.syncthreads()

    # Store results to global memory
    for mi in al.range(4):
        gr = start_m + thread_m * 4 + mi
        for ni in al.range(4):
            gc = start_n + thread_n * 4 + ni
            if gr < M and gc < N:
                C_t[gr, gc] = al.convert(acc[mi, ni], al.bf16)


def avelang_ttm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Host wrapper: 4D tensor-matrix multiply via AveLang tiled GEMM."""
    b, i, j, l = A.shape
    l2, k = B.shape
    assert l == l2, f"Contracting dimension mismatch: {l} vs {l2}"

    A_bf16 = A.to(torch.bfloat16).contiguous()
    B_bf16 = B.to(torch.bfloat16).contiguous()

    # Flatten A: (b, i, j, l) -> (b*i*j, l)
    A_flat = A_bf16.reshape(b * i * j, l).contiguous()
    M = b * i * j
    N = k
    K = l

    C_flat = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)

    BM = 64
    BN = 64
    grid_m = (M + BM - 1) // BM
    grid_n = (N + BN - 1) // BN

    ttm_kernel[lambda: ((grid_m, grid_n, 1), (256, 1, 1))](
        A_flat, B_bf16, C_flat, M, N, K,
    )

    return C_flat.reshape(b, i, j, k)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return avelang_ttm(A, B)
