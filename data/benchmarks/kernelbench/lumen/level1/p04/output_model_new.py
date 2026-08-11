import torch
import substrate
import substrate.language as S

# Configuration constants
BLOCK_SIZE: S.constexpr = 256
WARP_SIZE = 64


@substrate.jit
def gemv_kernel(
    A_ptr: S.Pointer(S.bf16),
    B_ptr: S.Pointer(S.bf16),
    C_ptr: S.Pointer(S.bf16),
    M: S.i32,
    K: S.i32,
):
    """
    Matrix-vector multiplication: C = A @ B
    where A is (M, K), B is (K, 1), C is (M, 1)
    """
    tid = S.thread_id(0)
    bid = S.block_id(0)

    # Each block handles one row
    row_idx = bid

    if row_idx < M:
        # Create tensor layouts
        layout_a = S.make_layout((M, K), (K, 1))
        layout_b = S.make_layout((K,), (1,))
        layout_c = S.make_layout((M, 1), (1, 1))

        A_tensor = S.make_tensor(A_ptr, S.bf16, layout_a)
        B_tensor = S.make_tensor(B_ptr, S.bf16, layout_b)
        C_tensor = S.make_tensor(C_ptr, S.bf16, layout_c)

        # Each thread accumulates partial sum
        # Thread tid processes elements at indices: tid, tid + BLOCK_SIZE, tid + 2*BLOCK_SIZE, ...
        partial_sum = S.convert(0.0, S.f32)

        k_iter_start = tid
        for k_offset in S.range((K + BLOCK_SIZE - 1) // BLOCK_SIZE):
            k_idx = k_iter_start + k_offset * BLOCK_SIZE
            if k_idx < K:
                a_val = A_tensor[row_idx, k_idx]
                b_val = B_tensor[k_idx]
                a_f32 = S.convert(a_val, S.f32)
                b_f32 = S.convert(b_val, S.f32)
                partial_sum = partial_sum + a_f32 * b_f32

        # Shuffle reduction within warps
        # AMD warp size is 64
        for offset in S.range(6):  # log2(64) = 6 iterations
            shuffle_offset = S.convert(1 << (5 - offset), S.i32)
            other = S.shuffle_down(partial_sum, shuffle_offset, WARP_SIZE)
            partial_sum = partial_sum + other

        # Only lane 0 of each warp has the complete sum for its warp
        # Need to reduce across warps using shared memory
        shm = S.make_shared((4,), S.f32)  # 4 warps per block

        wid = tid // WARP_SIZE
        lane_id = tid % WARP_SIZE

        if lane_id == 0:
            shm[wid] = partial_sum

        S.syncthreads()

        # Warp 0, lane 0 loads all warp sums and reduces
        if tid == 0:
            total_sum = S.convert(0.0, S.f32)
            for w in S.range(4):
                total_sum = total_sum + shm[w]
            C_tensor[row_idx, 0] = S.convert(total_sum, S.bf16)


def substrate_gemv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Matrix-vector multiplication using Substrate kernel.
    A: (M, K) matrix
    B: (K,) or (K, 1) vector
    Returns: (M, 1) result
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA/HIP device."
    assert A.dtype == torch.bfloat16, f"A must be bfloat16, got {A.dtype}"
    assert B.dtype == torch.bfloat16, f"B must be bfloat16, got {B.dtype}"

    # Handle B shape: (K, 1) or (K,)
    if B.ndim == 2 and B.shape[1] == 1:
        B = B.squeeze(1)
    elif B.ndim != 1:
        raise ValueError(f"B must be a vector of shape (K,) or (K, 1), got {B.shape}")

    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape

    if K != B.shape[0]:
        raise ValueError(f"K dimension mismatch: A has K={K}, B has K={B.shape[0]}")

    # Output shape matches PyTorch: (M, 1)
    C = torch.empty((M, 1), dtype=torch.bfloat16, device=A.device)

    # One block per row
    num_blocks = M
    gemv_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](A, B, C, M, K)

    return C


class ModelNew(torch.nn.Module):
    """
    Optimized model that performs matrix-vector multiplication using Substrate DSL.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return substrate_gemv(A, B)
