import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def row_scale_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
    N: al.i32,
    M: al.i32,
    BLOCK_M: al.constexpr,
    BLOCK_N: al.constexpr,
):
    a_layout = al.make_layout((N,), (1,))
    a = al.make_tensor(a_ptr, al.bf16, a_layout)
    b_layout = al.make_layout((N, M), (M, 1))
    b = al.make_tensor(b_ptr, al.bf16, b_layout)
    c_layout = al.make_layout((N, M), (M, 1))
    c = al.make_tensor(c_ptr, al.bf16, c_layout)

    row = al.block_id(0) * BLOCK_M + al.thread_id(0)
    col = al.block_id(1) * BLOCK_N + al.thread_id(1)

    if row < N and col < M:
        a_val = al.convert(a[row], al.f32)
        b_val = al.convert(b[row, col], al.f32)
        result = a_val * b_val
        c[row, col] = al.convert(result, al.bf16)


def avelang_row_scale(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA/HIP device."
    assert A.dim() == 1 and B.dim() == 2, f"Expected A 1D and B 2D, got {A.dim()}D and {B.dim()}D."
    N = A.shape[0]
    M = B.shape[1]
    assert B.shape[0] == N, f"Shape mismatch: A ({N},) and B ({B.shape[0]}, {M})."

    orig_dtype = B.dtype

    A_bf16 = A.contiguous().to(torch.bfloat16)
    B_bf16 = B.contiguous().to(torch.bfloat16)
    C_bf16 = torch.empty(N, M, dtype=torch.bfloat16, device=A.device)

    BLOCK_M = 16
    BLOCK_N = 16
    grid_m = (N + BLOCK_M - 1) // BLOCK_M
    grid_n = (M + BLOCK_N - 1) // BLOCK_N

    row_scale_kernel[lambda: ((grid_m, grid_n, 1), (BLOCK_M, BLOCK_N, 1))](
        A_bf16, B_bf16, C_bf16,
        N, M, BLOCK_M, BLOCK_N,
    )

    return C_bf16.to(orig_dtype)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return avelang_row_scale(A, B)


M = 4096
N = 4096


def get_inputs():
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]


def get_init_inputs():
    return []
