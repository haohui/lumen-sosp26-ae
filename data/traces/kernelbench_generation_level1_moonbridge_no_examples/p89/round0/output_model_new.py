import torch
import torch.nn as nn
import avelang
import avelang.language as al

THREADS_PER_BLOCK = 256


@avelang.jit
def cumsum_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    row_len: al.i32,
    num_rows: al.i32,
):
    """One thread per row: sequential FP32 accumulation, BF16 output."""
    row = al.block_id(0) * THREADS_PER_BLOCK + al.thread_id(0)
    if row >= num_rows:
        return

    layout_2d = al.make_layout((num_rows, row_len), (row_len, 1))
    x = al.make_tensor(x_ptr, al.bf16, layout_2d)
    out = al.make_tensor(out_ptr, al.bf16, layout_2d)

    acc = al.convert(0.0, al.f32)
    for col in al.range(row_len):
        acc = acc + al.convert(x[row, col], al.f32)
        out[row, col] = al.convert(acc, al.bf16)


def avelang_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on a CUDA/HIP device."

    # Move the scan dimension to the last axis.
    if dim != x.ndim - 1 and dim != -1:
        x = x.transpose(dim, -1).contiguous()
        transposed = True
    else:
        transposed = False

    # Flatten leading dims into a single batch dimension.
    original_shape = x.shape
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])

    num_rows = x.shape[0]
    row_len = x.shape[1]

    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    x = x.contiguous()

    out = torch.empty_like(x)
    num_blocks = (num_rows + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cumsum_kernel[lambda: ((num_blocks, 1, 1), (THREADS_PER_BLOCK, 1, 1))](
        x.data_ptr(), out.data_ptr(),
        row_len, num_rows,
    )

    out = out.reshape(original_shape)
    if transposed:
        out = out.transpose(dim, -1).contiguous()

    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return avelang_cumsum(x, self.dim)


# Preserved from input_model.py
batch_size = 32768
input_shape = (32768,)
dim = 1


def get_inputs():
    return [torch.rand(batch_size, *input_shape)]


def get_init_inputs():
    return [dim]
