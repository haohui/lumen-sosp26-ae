import torch
import torch.nn as nn
import substrate
import substrate.language as S

SUBGROUP_THREADS: S.constexpr = 16
ROWS_PER_BLOCK: S.constexpr = 8
BLOCK_THREADS: S.constexpr = SUBGROUP_THREADS * ROWS_PER_BLOCK
TILE_SIZE: S.constexpr = SUBGROUP_THREADS * 2
LOG_THREADS_X: S.constexpr = 4


@substrate.jit
def masked_cumsum_row_kernel(
    x_ptr: S.Pointer(S.bf16),
    mask_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    num_rows: S.i32,
    row_size: S.i32,
):
    tid = S.thread_id(0)
    subgroup = tid // SUBGROUP_THREADS
    lane = tid % SUBGROUP_THREADS
    row = S.block_id(0) * ROWS_PER_BLOCK + subgroup

    x = S.make_tensor(x_ptr, S.bf16, S.make_layout((num_rows, row_size), (row_size, 1)))
    mask = S.make_tensor(mask_ptr, S.bf16, S.make_layout((num_rows, row_size), (row_size, 1)))
    out = S.make_tensor(output_ptr, S.bf16, S.make_layout((num_rows, row_size), (row_size, 1)))
    row_buf = S.make_shared((ROWS_PER_BLOCK, TILE_SIZE), S.bf16)

    zero = S.convert(0.0, S.bf16)
    block_total = zero
    num_tiles = (row_size + TILE_SIZE - 1) // TILE_SIZE

    for tile in S.range(num_tiles):
        base_col = tile * TILE_SIZE
        col0 = base_col + lane
        col1 = base_col + SUBGROUP_THREADS + lane

        if row < num_rows:
            row_buf[subgroup, lane] = x[row, col0] * mask[row, col0] if col0 < row_size else zero
            row_buf[subgroup, SUBGROUP_THREADS + lane] = x[row, col1] * mask[row, col1] if col1 < row_size else zero
        else:
            row_buf[subgroup, lane] = zero
            row_buf[subgroup, SUBGROUP_THREADS + lane] = zero

        if lane == 0:
            row_buf[subgroup, 0] = row_buf[subgroup, 0] + block_total
        S.syncthreads()

        for m in S.range(LOG_THREADS_X + 1):
            step = 1 << m
            anchor = ((lane >> m) << (m + 1)) | step
            target_idx = anchor + (lane % step)
            source_idx = anchor - 1
            row_buf[subgroup, target_idx] = row_buf[subgroup, target_idx] + row_buf[subgroup, source_idx]
            S.syncthreads()

        if row < num_rows:
            if col0 < row_size:
                out[row, col0] = row_buf[subgroup, lane]
            if col1 < row_size:
                out[row, col1] = row_buf[subgroup, SUBGROUP_THREADS + lane]

        block_total = row_buf[subgroup, TILE_SIZE - 1]
        S.syncthreads()


def substrate_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dim() == 2 and dim == 1, "This kernel only supports 2D masked cumsum along dim=1."

    x_bf16 = x.to(dtype=torch.bfloat16, device=x.device).contiguous()
    mask_bf16 = mask.to(dtype=torch.bfloat16, device=mask.device).contiguous()
    num_rows, row_size = x_bf16.shape
    out = torch.empty_like(x_bf16)
    num_blocks = (num_rows + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK

    masked_cumsum_row_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_THREADS, 1, 1))](
        x_bf16,
        mask_bf16,
        out,
        num_rows,
        row_size,
        num_warps=2,
    )
    return out.to(dtype=x.dtype)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return substrate_masked_cumsum(x, mask, self.dim)
