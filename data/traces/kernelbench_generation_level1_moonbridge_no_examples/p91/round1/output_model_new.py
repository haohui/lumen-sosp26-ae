import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def reverse_cumsum_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N_ROWS: al.constexpr,
    N_COLS: al.constexpr,
    BLOCK_SIZE: al.constexpr,
):
    row = al.block_id(0) * BLOCK_SIZE + al.thread_id(0)

    if row >= N_ROWS:
        return

    total_elems = N_ROWS * N_COLS
    layout = al.make_layout((total_elems,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    row_start = row * N_COLS

    # Process this row from right to left, accumulating in bf16
    running_sum = al.convert(0, al.bf16)
    k = N_COLS - 1
    for step in al.range(N_COLS):
        gidx = row_start + k
        x_bf16 = x[gidx]
        next_sum = running_sum + x_bf16
        running_sum = next_sum
        out[gidx] = running_sum
        k = k - 1


def avelang_reverse_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    assert dim == 1, "Only dim=1 is supported for this kernel."

    n_rows, n_cols = x.shape
    x = x.contiguous()
    out = torch.empty_like(x)

    BLOCK_SIZE = 256
    grid_rows = (n_rows + BLOCK_SIZE - 1) // BLOCK_SIZE

    reverse_cumsum_kernel[lambda: ((grid_rows, 1, 1), (BLOCK_SIZE, 1, 1))](
        x.data_ptr(),
        out.data_ptr(),
        n_rows,
        n_cols,
        BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return avelang_reverse_cumsum(x, self.dim)
