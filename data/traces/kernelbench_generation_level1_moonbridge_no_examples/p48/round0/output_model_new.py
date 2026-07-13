import torch
import torch.nn as nn
import avelang
import avelang.language as al

D2_TILE = 256


@avelang.jit
def mean_reduce_dim1_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    dim1: al.i32,
    dim2: al.i32,
):
    batch = al.block_id(0)
    d2_block = al.block_id(1)
    tid = al.thread_id(0)

    d2 = d2_block * D2_TILE + tid
    if d2 >= dim2:
        return

    # Input layout: (batch_size, dim1, dim2) row-major
    x_strides = (dim1 * dim2, dim2, 1)
    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((batch_size, dim1, dim2), x_strides))

    # Each thread independently accumulates along dim1 in FP32 for numerical stability
    acc = al.convert(0.0, al.f32)
    for i in al.range(0, dim1):
        val = x[batch, i, d2]
        acc = acc + al.convert(val, al.f32)

    mean = acc / al.convert(dim1, al.f32)

    out_strides = (dim2, 1)
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((batch_size, dim2), out_strides))
    out[batch, d2] = al.convert(mean, al.bf16)


def avelang_mean_reduce(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    assert dim == 1, "This kernel only supports dim=1 reduction."

    x = x.contiguous()
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    batch_size, dim1_size, dim2_size = x.shape

    out = torch.empty(batch_size, dim2_size, dtype=torch.bfloat16, device=x.device)

    grid_y = (dim2_size + D2_TILE - 1) // D2_TILE
    mean_reduce_dim1_kernel[lambda: ((batch_size, grid_y, 1), (D2_TILE, 1, 1))](
        x.data_ptr(), out.data_ptr(), batch_size, dim1_size, dim2_size,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_mean_reduce(x, self.dim)
