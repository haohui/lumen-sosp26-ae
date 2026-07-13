import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
MAX_TILES: al.constexpr = 256


@avelang.jit
def local_scan_kernel(
    x_ptr: al.Pointer(al.bf16),
    intermediate_ptr: al.Pointer(al.f32),
    tile_sums_ptr: al.Pointer(al.f32),
    num_rows: al.i32,
    num_cols: al.i32,
    num_tiles: al.i32,
):
    tid = al.thread_id(0)
    tile_idx = al.block_id(0)
    row_idx = al.block_id(1)

    if row_idx < num_rows:
        tile_start = tile_idx * BLOCK_SIZE
        smem = al.make_shared((BLOCK_SIZE,), al.f32)

        layout_in = al.make_layout((num_rows, num_cols), (num_cols, 1))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        idx = tile_start + tid
        if idx < num_cols:
            smem[tid] = al.convert(x[row_idx, idx], al.f32)
        else:
            smem[tid] = al.convert(0.0, al.f32)
        al.syncthreads()

        # Serial inclusive scan by thread 0
        if tid == 0:
            acc = al.convert(0.0, al.f32)
            for i in al.range(0, BLOCK_SIZE):
                acc = acc + smem[i]
                smem[i] = acc
        al.syncthreads()

        layout_int = al.make_layout((num_rows, num_cols), (num_cols, 1))
        inter = al.make_tensor(intermediate_ptr, al.f32, layout_int)
        if idx < num_cols:
            inter[row_idx, idx] = smem[tid]

        # Tile sum
        if tid == BLOCK_SIZE - 1:
            layout_ts = al.make_layout((num_rows, num_tiles), (num_tiles, 1))
            ts = al.make_tensor(tile_sums_ptr, al.f32, layout_ts)
            ts[row_idx, tile_idx] = smem[tid]


def _avelang_inclusive_scan(x_bf16: torch.Tensor) -> torch.Tensor:
    """Compute inclusive cumulative sum along dim 1 in BF16 with FP32 accumulation."""
    assert x_bf16.is_cuda, "Tensor must be on CUDA/HIP device."
    assert x_bf16.dtype == torch.bfloat16, "Tensor must be bfloat16"

    num_rows, num_cols = x_bf16.shape
    num_tiles = (num_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    intermediate = torch.empty((num_rows, num_cols), dtype=torch.float32, device=x_bf16.device)
    tile_sums = torch.empty((num_rows, num_tiles), dtype=torch.float32, device=x_bf16.device)
    output = torch.empty((num_rows, num_cols), dtype=torch.bfloat16, device=x_bf16.device)

    grid = (num_tiles, num_rows, 1)
    block = (BLOCK_SIZE, 1, 1)

    local_scan_kernel[lambda: (grid, block)](
        x_bf16, intermediate, tile_sums, num_rows, num_cols, num_tiles
    )

    # Compute exclusive prefix on GPU using torch (only on ~129 tile sums per row)
    inclusive_prefix = torch.cumsum(tile_sums, dim=1)
    prefix = torch.zeros_like(tile_sums)
    prefix[:, 1:] = inclusive_prefix[:, :-1]

    # Add prefix to intermediate and convert to BF16
    for t in range(num_tiles):
        t_start = t * BLOCK_SIZE
        t_end = min(t_start + BLOCK_SIZE, num_cols)
        output[:, t_start:t_end] = (intermediate[:, t_start:t_end] + prefix[:, t:t+1]).to(torch.bfloat16)

    return output


def avelang_exclusive_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA/HIP device."

    x_bf16 = x.contiguous().to(dtype=torch.bfloat16)
    N, M = x_bf16.shape

    zeros_shape = (N, 1) if dim == 1 else (1, M)
    zeros = torch.zeros(zeros_shape, dtype=torch.bfloat16, device=x_bf16.device)
    padded = torch.cat([zeros, x_bf16], dim=dim)
    trimmed = padded[:-1].contiguous()

    if dim == 0:
        trimmed = trimmed.t().contiguous()

    result = _avelang_inclusive_scan(trimmed)

    if dim == 0:
        result = result.t().contiguous()

    return result


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return avelang_exclusive_cumsum(x, self.dim)


batch_size = 32768
input_shape = (32768,)
dim = 1


def get_inputs():
    return [torch.rand(batch_size, *input_shape)]


def get_init_inputs():
    return [dim]
