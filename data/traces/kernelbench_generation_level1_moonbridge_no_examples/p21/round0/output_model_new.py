import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def sigmoid_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    n_elements: al.i32,
    dim: al.i32,
    BLOCK_SIZE: al.constexpr,
    VEC_SIZE: al.constexpr,
):
    layout = al.make_layout((n_elements,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    row = al.block_id(1)
    tid = al.thread_id(0)
    col_start = al.block_id(0) * (BLOCK_SIZE * VEC_SIZE) + tid
    row_offset = row * dim

    for v in al.range(VEC_SIZE):
        col = col_start + v * BLOCK_SIZE
        if col < dim:
            idx = row_offset + col
            x_val = x[idx]
            x_f32 = al.convert(x_val, al.f32)
            exp_val = al.exp(-x_f32)
            one = al.convert(1.0, al.f32)
            result_f32 = one / (one + exp_val)
            out[idx] = al.convert(result_f32, al.bf16)


def avelang_sigmoid(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA/HIP device."

    x = x.contiguous()
    batch_size, dim = x.shape
    n_elements = x.numel()
    out = torch.empty_like(x)

    BLOCK_SIZE = 256
    VEC_SIZE = 4
    tile_size = BLOCK_SIZE * VEC_SIZE
    grid_x = (dim + tile_size - 1) // tile_size
    grid_y = batch_size

    sigmoid_kernel[lambda: ((grid_x, grid_y, 1), (BLOCK_SIZE, 1, 1))](
        x, out, n_elements, dim, BLOCK_SIZE, VEC_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_sigmoid(x)
