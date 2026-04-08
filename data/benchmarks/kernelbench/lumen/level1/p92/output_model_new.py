import torch
import substrate
import substrate.language as S

THREADS_X: S.constexpr = 32
TILE_SIZE: S.constexpr = THREADS_X * 2
LOG_THREADS_X: S.constexpr = 5


@substrate.jit
def exclusive_cumsum_row_kernel(
    input_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    row_size: S.i32,
):
    tid = S.thread_id(0)
    row = S.block_id(0)
    out_rows = batch_size - 1
    out_row_size = row_size + 1
    if row >= out_rows:
        return

    x = S.make_tensor(input_ptr, S.bf16, S.make_layout((batch_size, row_size), (row_size, 1)))
    out = S.make_tensor(output_ptr, S.bf16, S.make_layout((out_rows, out_row_size), (out_row_size, 1)))
    row_buf = S.make_shared((TILE_SIZE,), S.bf16)

    zero = S.convert(0.0, S.bf16)
    block_total = zero
    num_tiles = (out_row_size + TILE_SIZE - 1) // TILE_SIZE

    for tile in S.range(num_tiles):
        base_col = tile * TILE_SIZE
        col1 = base_col + tid
        col2 = base_col + THREADS_X + tid

        if col1 == 0:
            row_buf[tid] = zero
        elif col1 < out_row_size:
            row_buf[tid] = x[row, col1 - 1]
        else:
            row_buf[tid] = zero

        if col2 == 0:
            row_buf[THREADS_X + tid] = zero
        elif col2 < out_row_size:
            row_buf[THREADS_X + tid] = x[row, col2 - 1]
        else:
            row_buf[THREADS_X + tid] = zero

        if tid == 0:
            row_buf[0] = row_buf[0] + block_total
        S.syncthreads()

        for m in S.range(LOG_THREADS_X + 1):
            s = 1 << m
            a = ((tid >> m) << (m + 1)) | s
            ti = a + (tid % s)
            si = a - 1
            row_buf[ti] = row_buf[ti] + row_buf[si]
            S.syncthreads()

        if col1 < out_row_size:
            out[row, col1] = row_buf[tid]
        if col2 < out_row_size:
            out[row, col2] = row_buf[THREADS_X + tid]

        block_total = row_buf[TILE_SIZE - 1]
        S.syncthreads()


def substrate_exclusive_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dim() == 2 and dim == 1, "This kernel only supports 2D exclusive cumsum along dim=1."

    x_bf16 = x.to(dtype=torch.bfloat16, device=x.device).contiguous()
    batch_size, row_size = x_bf16.shape
    out = torch.empty((batch_size - 1, row_size + 1), dtype=torch.bfloat16, device=x.device)

    exclusive_cumsum_row_kernel[lambda: ((batch_size - 1, 1, 1), (THREADS_X, 1, 1))](
        x_bf16,
        out,
        batch_size,
        row_size,
    )
    return out.to(dtype=x.dtype)


class ModelNew(torch.nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return substrate_exclusive_cumsum(x, self.dim)
