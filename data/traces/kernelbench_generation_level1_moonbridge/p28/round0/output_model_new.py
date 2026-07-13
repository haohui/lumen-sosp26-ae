import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
TILE_SIZE: al.constexpr = 32768


@avelang.jit
def hardsigmoid_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    numel: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    layout = al.make_layout((numel,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    tile_start = bid * TILE_SIZE
    tile_end = tile_start + TILE_SIZE
    if tile_end > numel:
        tile_end = numel

    sixth = al.convert(0.1666666716337204, al.f32)
    half = al.convert(0.5, al.f32)
    zero_f32 = al.convert(0.0, al.f32)
    one_f32 = al.convert(1.0, al.f32)

    for idx in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
        val = al.convert(x[idx], al.f32)
        result = val * sixth + half
        if result < zero_f32:
            result = zero_f32
        if result > one_f32:
            result = one_f32
        out[idx] = al.convert(result, al.bf16)


def avelang_hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."

    x_bf16 = x.contiguous().to(torch.bfloat16)
    numel = x_bf16.numel()
    out = torch.empty_like(x_bf16)

    grid = (numel + TILE_SIZE - 1) // TILE_SIZE

    hardsigmoid_kernel[lambda: ((grid, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, out, numel
    )

    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = avelang_hardsigmoid(x)
        return result.to(x.dtype)
