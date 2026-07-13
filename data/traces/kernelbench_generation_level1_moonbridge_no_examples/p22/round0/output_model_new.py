import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def tanh_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    total: al.i32,
):
    layout = al.make_layout((total,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    idx = al.block_id(0) * al.block_dim(0) + al.thread_id(0)
    stride = al.grid_dim(0) * al.block_dim(0)

    for i in al.range(idx, total, stride):
        val = al.convert(x[i], al.f32)
        out[i] = al.convert(al.tanh(val), al.bf16)


def avelang_tanh(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."

    original_dtype = x.dtype
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    x = x.contiguous()

    m, n = x.shape
    out = torch.empty_like(x)
    total = m * n

    BLOCK_SIZE = 256
    MAX_GRID = 65536
    grid_x = min(MAX_GRID, max(1, (total + BLOCK_SIZE - 1) // BLOCK_SIZE))
    grid = (grid_x, 1, 1)
    block = (BLOCK_SIZE, 1, 1)

    tanh_kernel[lambda: (grid, block)](x, out, total)

    if original_dtype != torch.bfloat16:
        out = out.to(original_dtype)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_tanh(x)
