import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256


@avelang.jit
def gelu_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    bdim = al.block_dim(0)
    gdim = al.grid_dim(0)
    grid_stride = bdim * gdim
    idx = bid * bdim + tid

    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((N,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((N,), (1,)))

    sqrt2 = al.sqrt(al.convert(2.0, al.f32))
    half = al.convert(0.5, al.f32)
    one = al.convert(1.0, al.f32)

    for i in al.range(idx, N, grid_stride):
        val_bf16 = x[i]
        val_f32 = al.convert(val_bf16, al.f32)
        gelu_f32 = val_f32 * half * (one + al.erf(val_f32 / sqrt2))
        out[i] = al.convert(gelu_f32, al.bf16)


def avelang_gelu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    N = x_contig.numel()
    grid_size = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    if grid_size > 65536:
        grid_size = 65536

    gelu_kernel[lambda: ((grid_size, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contig, out, N,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_gelu(x)
