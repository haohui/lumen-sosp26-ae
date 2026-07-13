import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256
TILE_SIZE = 32768
_NEGATIVE_SLOPE = 0.01


@avelang.jit
def leaky_relu_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    dim: al.i32,
):
    tid = al.thread_id(0)
    tile_id = al.block_id(0)
    batch_id = al.block_id(1)

    row_start = batch_id * dim + tile_id * TILE_SIZE
    tile_end = TILE_SIZE
    if tile_id * TILE_SIZE + TILE_SIZE > dim:
        tile_end = dim - tile_id * TILE_SIZE

    total_elems = batch_size * dim
    layout = al.make_layout((total_elems,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    zero = al.convert(0.0, al.f32)
    slope = al.convert(_NEGATIVE_SLOPE, al.f32)

    for offset in al.range(tid, tile_end, BLOCK_SIZE):
        idx = row_start + offset
        val = al.convert(x[idx], al.f32)
        if val >= zero:
            out[idx] = al.convert(val, al.bf16)
        else:
            out[idx] = al.convert(val * slope, al.bf16)


def avelang_leaky_relu(x: torch.Tensor, negative_slope: float) -> torch.Tensor:
    global _NEGATIVE_SLOPE
    _NEGATIVE_SLOPE = negative_slope

    if not x.is_cuda:
        x = x.cuda()
    x_bf16 = x.contiguous().to(dtype=torch.bfloat16)

    batch_size_val = x_bf16.shape[0]
    dim_val = x_bf16.shape[1]
    tiles_per_row = (dim_val + TILE_SIZE - 1) // TILE_SIZE

    out = torch.empty_like(x_bf16)

    leaky_relu_kernel[lambda: ((tiles_per_row, batch_size_val, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, out, batch_size_val, dim_val
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_leaky_relu(x, self.negative_slope)


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]


def get_init_inputs():
    return []
