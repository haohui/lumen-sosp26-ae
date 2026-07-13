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
):
    block_m = al.block_id(0)
    block_n = al.block_id(1)
    tid = al.thread_id(0)

    # 256 threads in a 16×16 grid. Each thread handles a 2×2 output sub-block.
    row_t = tid >> 4       # tid // 16  (0-15)
    col_t = tid & 15       # tid % 16   (0-15)

    global_row = block_m * 32 + row_t * 2
    global_col = block_n * 32 + col_t * 2

    a_layout = al.make_layout((M, K), (K, 1))
    a = al.make_tensor(a_ptr, al.bf16, a_layout)
    b_layout = al.make_layout((K, N), (N, 1))
    b = al.make_tensor(b_ptr, al.bf16, b_layout)
    c_layout = al.make_layout((M, N), (N, 1))
    c = al.make_tensor(c_ptr, al.bf16, c_layout)

    a_smem = al.make_shared((32, 128), al.bf16)
    b_smem = al.make_shared((128, 32), al.bf16)

    acc00 = al.convert(0.0, al.f32)
    acc01 = al.convert(0.0, al.f32)
    acc10 = al.convert(0.0, al.f32)
    acc11 = al.convert(0.0, al.f32)

    for k_block in al.range(0, K, 128):
        for i in al.range(16):
            idx = i * 256 + tid
            row = idx // 128
            col = idx % 128
            a_smem[row, col] = a[block_m * 32 + row, k_block + col]

        for i in al.range(16):
            idx = i * 256 + tid
            row = idx // 32
            col = idx % 32
            b_smem[row, col] = b[k_block + row, block_n * 32 + col]

        al.syncthreads()

        for k in al.range(128):
            a0 = al.convert(a_smem[row_t * 2,     k], al.f32)
            a1 = al.convert(a_smem[row_t * 2 + 1, k], al.f32)
            b0 = al.convert(b_smem[k, col_t * 2],     al.f32)
            b1 = al.convert(b_smem[k, col_t * 2 + 1], al.f32)
            acc00 = acc00 + a0 * b0
            acc01 = acc01 + a0 * b1
            acc10 = acc10 + a1 * b0
            acc11 = acc11 + a1 * b1

        al.syncthreads()

    c[global_row,     global_col]     = al.convert(acc00, al.bf16)
    c[global_row,     global_col + 1] = al.convert(acc01, al.bf16)
    c[global_row + 1, global_col]     = al.convert(acc10, al.bf16)
    c[global_row + 1, global_col + 1] = al.convert(acc11, al.bf16)


def avelang_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA/HIP device."
    M, K_a = A.shape
    K_b, N = B.shape
    assert K_a == K_b, "Inner dimension mismatch."

    A_bf16 = A.to(torch.bfloat16).contiguous()
    B_bf16 = B.to(torch.bfloat16).contiguous()
    C = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)

    grid = ((M + 31) // 32, (N + 31) // 32, 1)
    block = (256, 1, 1)

    matmul_kernel[lambda: (grid, block)](A_bf16, B_bf16, C, M, N, K_a)

    return C


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return avelang_matmul(A, B)
