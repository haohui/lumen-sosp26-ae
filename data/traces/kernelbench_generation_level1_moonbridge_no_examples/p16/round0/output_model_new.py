import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def matmul_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
    BLOCK_M: al.constexpr,
    BLOCK_N: al.constexpr,
    BLOCK_K: al.constexpr,
):
    """
    Tiled BF16 matrix multiplication C = A^T @ B.
    A is stored as (K, M), we read it transposed as (M, K).
    B is stored as (K, N).
    Uses shared memory tiling with FP32 accumulation.
    One wavefront per workgroup; each thread handles one full output row.
    """
    a_layout = al.make_layout((M, K), (1, M))
    b_layout = al.make_layout((K, N), (N, 1))
    c_layout = al.make_layout((M, N), (N, 1))

    a = al.make_tensor(a_ptr, al.bf16, a_layout)
    b = al.make_tensor(b_ptr, al.bf16, b_layout)
    c = al.make_tensor(c_ptr, al.bf16, c_layout)

    pid_m = al.block_id(0)
    pid_n = al.block_id(1)
    tid = al.thread_id(0)

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    # Shared memory tiles
    a_sh = al.make_shared((BLOCK_M, BLOCK_K), al.bf16)
    b_sh = al.make_shared((BLOCK_K, BLOCK_N), al.bf16)

    NUM_THREADS = 64

    # Each thread holds one full output row as FP32 accumulator
    my_row = tid
    acc = al.make_local((BLOCK_N,), al.f32)
    for j in al.range(BLOCK_N):
        acc[j] = al.convert(0.0, al.f32)

    # Number of elements each thread loads from global to shared
    num_a_loads = BLOCK_M * BLOCK_K // NUM_THREADS
    num_b_loads = BLOCK_K * BLOCK_N // NUM_THREADS

    # Main K loop
    for k_block in al.range(K // BLOCK_K):
        k_off = k_block * BLOCK_K

        # Collaborative load of A tile into shared memory
        for i in al.range(num_a_loads):
            idx = tid * num_a_loads + i
            row = idx // BLOCK_K
            col = idx % BLOCK_K
            a_sh[row, col] = a[offs_m + row, k_off + col]

        # Collaborative load of B tile into shared memory
        for i in al.range(num_b_loads):
            idx = tid * num_b_loads + i
            row = idx // BLOCK_N
            col = idx % BLOCK_N
            b_sh[row, col] = b[k_off + row, offs_n + col]

        al.syncthreads()

        # Compute: dot product of my row of A with all columns of B
        for j in al.range(BLOCK_N):
            for k_val in al.range(BLOCK_K):
                a_val = al.convert(a_sh[my_row, k_val], al.f32)
                b_val = al.convert(b_sh[k_val, j], al.f32)
                acc[j] = acc[j] + a_val * b_val

        al.syncthreads()

    # Store output row: convert FP32 to BF16 and write to C
    for j in al.range(BLOCK_N):
        c[offs_m + my_row, offs_n + j] = al.convert(acc[j], al.bf16)


def avelang_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA/HIP device."
    assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D tensors."

    K, M = A.shape
    K2, N = B.shape
    assert K == K2, f"K dimension mismatch: {K} vs {K2}"

    A_bf16 = A.to(torch.bfloat16).contiguous()
    B_bf16 = B.to(torch.bfloat16).contiguous()
    C_bf16 = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 16

    grid = (M // BLOCK_M, N // BLOCK_N, 1)
    block = (64, 1, 1)

    matmul_kernel[lambda: (grid, block)](
        A_bf16, B_bf16, C_bf16,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return C_bf16


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return avelang_matmul(A, B)
