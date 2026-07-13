import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def sum_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    total_in_elems: al.i32,
    total_out_elems: al.i32,
    reduce_size: al.i32,
    num_cols: al.i32,
    stride_outer: al.i32,
    stride_reduce: al.i32,
    stride_inner: al.i32,
    out_stride_outer: al.i32,
    out_stride_inner: al.i32,
):
    outer = al.block_id(0)
    inner_block = al.block_id(1)
    tid = al.thread_id(0)

    # Each block handles 8 columns, each column reduced by 32 threads
    col_id = tid // 32
    local_tid = tid - col_id * 32

    col = inner_block * 8 + col_id

    if col < num_cols:
        layout_in = al.make_layout((total_in_elems,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        base = outer * stride_outer + col * stride_inner

        acc = al.convert(0.0, al.f32)

        for i in al.range(local_tid, reduce_size, 32):
            idx = base + i * stride_reduce
            val = al.convert(x[idx], al.f32)
            acc = acc + val

        val = acc
        val = val + al.shuffle_down(val, 16, 32)
        val = val + al.shuffle_down(val, 8, 32)
        val = val + al.shuffle_down(val, 4, 32)
        val = val + al.shuffle_down(val, 2, 32)
        val = val + al.shuffle_down(val, 1, 32)

        if local_tid == 0:
            layout_out = al.make_layout((total_out_elems,), (1,))
            out = al.make_tensor(out_ptr, al.bf16, layout_out)
            out_idx = outer * out_stride_outer + col * out_stride_inner
            out[out_idx] = al.convert(val, al.bf16)


def avelang_sum_reduce(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    assert x.ndim == 3, "This kernel expects 3D input."

    if not x.is_contiguous():
        x = x.contiguous()

    batch_size, d1, d2 = x.shape
    total_in_elems = batch_size * d1 * d2

    out_shape = list(x.shape)
    out_shape[dim] = 1
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    total_out_elems = out.numel()

    COLS_PER_BLOCK = 8
    THREADS_PER_COL = 32
    BLOCK_SIZE = COLS_PER_BLOCK * THREADS_PER_COL

    if dim == 0:
        reduce_size = batch_size
        num_cols = d2
        grid_x = d1
        grid_y = (d2 + COLS_PER_BLOCK - 1) // COLS_PER_BLOCK
        stride_outer = d2
        stride_reduce = d1 * d2
        stride_inner = 1
        out_stride_outer = d2
        out_stride_inner = 1
    elif dim == 1:
        reduce_size = d1
        num_cols = d2
        grid_x = batch_size
        grid_y = (d2 + COLS_PER_BLOCK - 1) // COLS_PER_BLOCK
        stride_outer = d1 * d2
        stride_reduce = d2
        stride_inner = 1
        out_stride_outer = d2
        out_stride_inner = 1
    elif dim == 2:
        reduce_size = d2
        num_cols = d1
        grid_x = batch_size
        grid_y = (d1 + COLS_PER_BLOCK - 1) // COLS_PER_BLOCK
        stride_outer = d1 * d2
        stride_reduce = 1
        stride_inner = d2
        out_stride_outer = d1
        out_stride_inner = 1
    else:
        raise ValueError(f"Reduction dim {dim} out of range for 3D tensor.")

    sum_reduce_kernel[
        lambda: ((grid_x, grid_y, 1), (BLOCK_SIZE, 1, 1))
    ](
        x.data_ptr(),
        out.data_ptr(),
        total_in_elems,
        total_out_elems,
        reduce_size,
        num_cols,
        stride_outer,
        stride_reduce,
        stride_inner,
        out_stride_outer,
        out_stride_inner,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_sum_reduce(x, self.dim)
