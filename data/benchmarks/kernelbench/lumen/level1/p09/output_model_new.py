"""
Outer product kernel for tall-and-skinny matrix multiplication.
A: (M, K) where M=32768, K=32
B: (K, M) where K=32, M=32768
C = A @ B: (M, M) = (32768, 32768)
"""
import torch
import substrate
import substrate.language as S

# Tile sizes
TILE_M = 16
TILE_N = 16
WARP_SIZE = 64
K_DIM = 32


@substrate.jit
def outer_product_kernel(
    A_ptr: S.Pointer(S.bf16),
    B_ptr: S.Pointer(S.bf16),
    C_ptr: S.Pointer(S.bf16),
    M: S.i32,
    K: S.i32,
):
    """
    Compute C = A @ B where A is (M, K), B is (K, M), C is (M, M).
    Each block computes a TILE_M x TILE_N output tile.
    """
    block_m = S.block_id(0)
    block_n = S.block_id(1)
    tid = S.thread_id(0)

    # Row and column bases for this tile
    row_base = block_m * TILE_M
    col_base = block_n * TILE_N

    # Shared memory for A and B tiles
    shm_a = S.make_shared((TILE_M, K_DIM), S.bf16)
    shm_b = S.make_shared((K_DIM, TILE_N), S.bf16)

    # Create tensor views for global memory
    layout_a = S.make_layout((M, K), (K, 1))
    layout_b = S.make_layout((K, M), (M, 1))
    A = S.make_tensor(A_ptr, S.bf16, layout_a)
    B = S.make_tensor(B_ptr, S.bf16, layout_b)

    # Load A tile into shared memory (cooperative loading)
    # Total: 16 * 32 = 512 elements, 8 per thread
    for i in S.range(8):
        idx = tid + i * WARP_SIZE
        local_row = idx // K_DIM
        local_col = idx % K_DIM
        global_row = row_base + local_row
        shm_a[local_row, local_col] = A[global_row, local_col]

    # Load B tile into shared memory
    # Total: 32 * 16 = 512 elements, 8 per thread
    for i in S.range(8):
        idx = tid + i * WARP_SIZE
        local_row = idx // TILE_N
        local_col = idx % TILE_N
        global_col = col_base + local_col
        shm_b[local_row, local_col] = B[local_row, global_col]

    S.syncthreads()

    # Each thread computes 4 output elements
    # Distribute: thread (i) computes rows (i//16) + j*4 for j=0..3, column i%16
    thread_group = tid // TILE_N  # 0-15
    thread_col = tid % TILE_N     # 0-15

    # Compute 4 output elements per thread
    for out_idx in S.range(4):
        out_row = (thread_group % 4) + out_idx * 4

        # Accumulate dot product over K dimension
        # Use a scalar accumulator (not local tensor for single value)
        acc = S.convert(0.0, S.f32)

        for k in S.range(K_DIM):
            a_val = shm_a[out_row, k]
            b_val = shm_b[k, thread_col]
            # Convert to f32 for accumulation
            a_f32 = S.convert(a_val, S.f32)
            b_f32 = S.convert(b_val, S.f32)
            acc = acc + a_f32 * b_f32

        # Convert back to bf16 and write to global memory
        out_bf16 = S.convert(acc, S.bf16)
        global_row = row_base + out_row
        global_col = col_base + thread_col

        layout_c = S.make_layout((M, M), (M, 1))
        C = S.make_tensor(C_ptr, S.bf16, layout_c)
        C[global_row, global_col] = out_bf16


class ModelNew(torch.nn.Module):
    """
    Optimized matrix multiplication using Substrate DSL kernels.
    Handles outer product case where K is small (32).
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute C = A @ B where:
        - A: (M, K) = (32768, 32)
        - B: (K, M) = (32, 32768)
        - C: (M, M) = (32768, 32768)
        """
        # Ensure inputs are on GPU and contiguous
        if not A.is_cuda or not B.is_cuda:
            raise ValueError("Inputs must be CUDA tensors")

        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K_B, M_B = B.shape

        # Validate shapes
        if K != K_B:
            raise ValueError(f"K dimension mismatch: A has K={K}, B has K={K_B}")

        # Convert to BF16 if needed
        if A.dtype != torch.bfloat16:
            A = A.to(torch.bfloat16)
        if B.dtype != torch.bfloat16:
            B = B.to(torch.bfloat16)

        # Allocate output
        C = torch.empty((M, M), dtype=torch.bfloat16, device=A.device)

        # Launch kernel
        grid_m = (M + TILE_M - 1) // TILE_M
        grid_n = (M + TILE_N - 1) // TILE_N

        outer_product_kernel[lambda: ((grid_m, grid_n, 1), (WARP_SIZE, 1, 1))](
            A, B, C, M, K
        )

        return C
