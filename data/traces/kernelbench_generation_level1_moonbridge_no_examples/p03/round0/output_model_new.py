import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def bmm_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M: al.i32,
    K: al.i32,
    N: al.i32,
    batch: al.i32,
    stride_ab: al.i32,
    stride_am: al.i32,
    stride_bb: al.i32,
    stride_bn: al.i32,
    stride_cb: al.i32,
    stride_cm: al.i32,
):
    # Build 1-D tensor views over the full flattened buffers
    total_A = batch * stride_ab
    total_B = batch * stride_bb
    total_C = batch * stride_cb

    a_layout = al.make_layout((total_A,), (1,))
    b_layout = al.make_layout((total_B,), (1,))
    c_layout = al.make_layout((total_C,), (1,))

    A = al.make_tensor(A_ptr, al.bf16, a_layout)
    B = al.make_tensor(B_ptr, al.bf16, b_layout)
    C = al.make_tensor(C_ptr, al.bf16, c_layout)

    tile_n_id = al.block_id(0)
    tile_m_id = al.block_id(1)
    batch_id = al.block_id(2)

    tid = al.thread_id(0)

    tile_m_start = tile_m_id * 64
    tile_n_start = tile_n_id * 64

    # Shared-memory tiles for A and B panels
    As = al.make_shared((64, 32), al.bf16)
    Bs = al.make_shared((32, 64), al.bf16)

    # Per-thread f32 accumulators -- 16 output elements per thread
    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = al.convert(0.0, al.f32)

    num_k_blocks = K // 32

    for k_block in al.range(num_k_blocks):
        k_start = k_block * 32

        # Cooperative load of A tile (64 x 32) from global to shared
        for i in al.range(8):
            idx = tid + i * 256
            row = idx // 32
            col = idx % 32
            g_idx = batch_id * stride_ab + (tile_m_start + row) * stride_am + (k_start + col)
            As[row, col] = A[g_idx]

        # Cooperative load of B tile (32 x 64) from global to shared
        for i in al.range(8):
            idx = tid + i * 256
            row = idx // 64
            col = idx % 64
            g_idx = batch_id * stride_bb + (k_start + row) * stride_bn + (tile_n_start + col)
            Bs[row, col] = B[g_idx]

        al.syncthreads()

        # Accumulate partial dot products
        for i in al.range(16):
            idx = tid + i * 256
            out_row = idx // 64
            out_col = idx % 64
            for ki in al.range(32):
                a_val = al.convert(As[out_row, ki], al.f32)
                b_val = al.convert(Bs[ki, out_col], al.f32)
                acc[i] = acc[i] + a_val * b_val

        al.syncthreads()

    # Write accumulated results back to global memory as bf16
    for i in al.range(16):
        idx = tid + i * 256
        out_row = idx // 64
        out_col = idx % 64
        g_idx = batch_id * stride_cb + (tile_m_start + out_row) * stride_cm + (tile_n_start + out_col)
        C[g_idx] = al.convert(acc[i], al.bf16)


def avelang_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Launch the tiled AveLang bmm kernel."""
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA/HIP device."
    batch, M, K = A.shape
    batch_b, K_b, N = B.shape
    assert batch == batch_b and K == K_b, "Incompatible batch or K dimensions."

    A = A.contiguous()
    B = B.contiguous()

    C = torch.empty(batch, M, N, dtype=A.dtype, device=A.device)

    grid_x = N // 64
    grid_y = M // 64
    grid_z = batch

    bmm_kernel[lambda: ((grid_x, grid_y, grid_z), (256, 1, 1))](
        A, B, C,
        M, K, N, batch,
        M * K, K,
        K * N, N,
        M * N, N,
    )

    return C


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return avelang_bmm(A, B)
