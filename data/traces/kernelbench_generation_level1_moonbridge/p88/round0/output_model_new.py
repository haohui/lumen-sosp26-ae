import torch
import torch.nn as nn
import avelang
import avelang.language as al
import math

BLOCK_SIZE: al.constexpr = 256
ELEMS_PER_THREAD: al.constexpr = 8


@avelang.jit
def gelu_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    n_elements: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    block_start = bid * BLOCK_SIZE * ELEMS_PER_THREAD

    layout = al.make_layout((n_elements,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    sqrt_2_over_pi = al.convert(0.7978845608028654, al.f32)
    coeff = al.convert(0.044715, al.f32)
    half = al.convert(0.5, al.f32)
    one = al.convert(1.0, al.f32)

    for e in al.range(ELEMS_PER_THREAD):
        idx = block_start + tid + e * BLOCK_SIZE
        if idx < n_elements:
            val = al.convert(x[idx], al.f32)
            x3 = val * val * val
            inner = sqrt_2_over_pi * (val + coeff * x3)
            result = half * val * (one + al.tanh(inner))
            out[idx] = al.convert(result, al.bf16)


def avelang_gelu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    x_bf16 = x.to(dtype=torch.bfloat16).contiguous()
    n_elements = x_bf16.numel()
    out = torch.empty_like(x_bf16)

    grid_x = (n_elements + BLOCK_SIZE * ELEMS_PER_THREAD - 1) // (BLOCK_SIZE * ELEMS_PER_THREAD)

    gelu_kernel[lambda: ((grid_x, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, out, n_elements
    )
    return out.to(x.dtype)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return avelang_gelu(x)
