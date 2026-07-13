import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def max_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    dim: al.i32,
    B: al.i32,
    D1: al.i32,
    D2: al.i32,
    reduce_size: al.i32,
    out_rows: al.i32,
    out_cols: al.i32,
    BLOCK_SIZE: al.i32,
):
    out_idx = al.block_id(0) * BLOCK_SIZE + al.thread_id(0)

    if out_idx < out_rows * out_cols:
        out_row = out_idx // out_cols
        out_col = out_idx % out_cols

        total_elems = B * D1 * D2
        x_1d_layout = al.make_layout((total_elems,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, x_1d_layout)

        out_total = out_rows * out_cols
        out_1d_layout = al.make_layout((out_total,), (1,))
        out = al.make_tensor(out_ptr, al.bf16, out_1d_layout)

        base = out_row * D2 + out_col
        stride = D1 * D2
        if dim == 1:
            base = out_row * D1 * D2 + out_col
            stride = D2
        if dim == 2:
            base = out_row * D1 * D2 + out_col * D2
            stride = 1

        max_val = al.convert(x[base], al.f32)

        for k in al.range(1, reduce_size):
            val = al.convert(x[base + k * stride], al.f32)
            if val > max_val:
                max_val = val

        out[out_row * out_cols + out_col] = al.convert(max_val, al.bf16)


def avelang_max_reduce(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    B, D1, D2 = x.shape

    orig_dtype = x.dtype
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    if dim == 0:
        out_shape = (D1, D2)
        reduce_size = B
    elif dim == 1:
        out_shape = (B, D2)
        reduce_size = D1
    else:
        out_shape = (B, D1)
        reduce_size = D2

    out = torch.empty(out_shape, dtype=torch.bfloat16, device=x.device)
    out_rows, out_cols = out_shape

    total_out = out_rows * out_cols
    BLOCK_SIZE = 256
    grid_x = (total_out + BLOCK_SIZE - 1) // BLOCK_SIZE

    max_reduce_kernel[lambda: ((grid_x, 1, 1), (BLOCK_SIZE, 1, 1))](
        x, out, dim, B, D1, D2, reduce_size, out_rows, out_cols, BLOCK_SIZE
    )

    if orig_dtype != torch.bfloat16:
        out = out.to(orig_dtype)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_max_reduce(x, self.dim)
