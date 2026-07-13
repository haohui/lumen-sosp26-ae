import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_N = 128
BLOCK_M = 256
THREADS = 256


@avelang.jit
def diag_scale_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    n: al.i32,
    m: al.i32,
):
    tid = al.thread_id(0)
    bid_col = al.block_id(0)
    bid_row = al.block_id(1)

    row_start = bid_row * BLOCK_N
    col_start = bid_col * BLOCK_M

    a = al.make_tensor(a_ptr, al.bf16, al.make_layout((n,), (1,)))
    b = al.make_tensor(b_ptr, al.bf16, al.make_layout((n, m), (m, 1)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((n, m), (m, 1)))

    for r in al.range(BLOCK_N):
        row = row_start + r
        if row >= n:
            break
        a_val = al.convert(a[row], al.f32)
        for c in al.range(tid, BLOCK_M, THREADS):
            col = col_start + c
            if col < m:
                b_val = al.convert(b[row, col], al.f32)
                out[row, col] = al.convert(a_val * b_val, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_diag_scale(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    a_bf16 = _prepare_bf16_cuda_contiguous(A)
    b_bf16 = _prepare_bf16_cuda_contiguous(B)

    n = a_bf16.shape[0]
    m = b_bf16.shape[1]

    if b_bf16.shape[0] != n:
        raise ValueError(
            f"Shape mismatch: A has {n}, B has shape {b_bf16.shape}"
        )

    out = torch.empty((n, m), device=a_bf16.device, dtype=torch.bfloat16)

    grid_x = (m + BLOCK_M - 1) // BLOCK_M
    grid_y = (n + BLOCK_N - 1) // BLOCK_N

    diag_scale_kernel[lambda: ((grid_x, grid_y, 1), (THREADS, 1, 1))](
        a_bf16, b_bf16, out, n, m
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return avelang_diag_scale(A, B)
