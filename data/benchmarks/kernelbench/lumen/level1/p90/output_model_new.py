import torch
import substrate
import substrate.language as S

THREADS_X: S.constexpr = 128
TILE_SIZE: S.constexpr = THREADS_X * 2
LOG_THREADS_X: S.constexpr = 7


@substrate.jit
def cumprod_row_kernel(
    input_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    num_rows: S.i32,
    row_size: S.i32,
):
    tid = S.thread_id(0)
    row = S.block_id(0)
    if row >= num_rows:
        return

    x = S.make_tensor(input_ptr, S.bf16, S.make_layout((num_rows, row_size), (row_size, 1)))
    out = S.make_tensor(output_ptr, S.bf16, S.make_layout((num_rows, row_size), (row_size, 1)))
    tile_buf = S.make_shared((TILE_SIZE,), S.bf16)

    one = S.convert(1.0, S.bf16)
    block_total = one
    num_tiles = (row_size + TILE_SIZE - 1) // TILE_SIZE

    for tile in S.range(num_tiles):
        base_col = tile * TILE_SIZE
        col0 = base_col + tid
        col1 = col0 + THREADS_X

        tile_buf[tid] = x[row, col0] if col0 < row_size else one
        tile_buf[THREADS_X + tid] = x[row, col1] if col1 < row_size else one

        if tid == 0:
            tile_buf[0] = tile_buf[0] * block_total
        S.syncthreads()

        for m in S.range(LOG_THREADS_X + 1):
            step = 1 << m
            anchor = ((tid >> m) << (m + 1)) | step
            target_idx = anchor + (tid % step)
            source_idx = anchor - 1
            tile_buf[target_idx] = tile_buf[target_idx] * tile_buf[source_idx]
            S.syncthreads()

        if col0 < row_size:
            out[row, col0] = tile_buf[tid]
        if col1 < row_size:
            out[row, col1] = tile_buf[THREADS_X + tid]

        block_total = tile_buf[TILE_SIZE - 1]
        S.syncthreads()


def substrate_cumprod(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA/HIP device."
    assert x.dim() == 2 and dim == 1, "This kernel only supports 2D cumprod along dim=1."

    x_bf16 = x.to(dtype=torch.bfloat16, device=x.device).contiguous()
    num_rows, row_size = x_bf16.shape
    out = torch.empty_like(x_bf16)

    cumprod_row_kernel[lambda: ((num_rows, 1, 1), (THREADS_X, 1, 1))](
        x_bf16,
        out,
        num_rows,
        row_size,
        num_warps=4,
    )
    return out.to(dtype=x.dtype)


class ModelNew(torch.nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return substrate_cumprod(x, self.dim)
