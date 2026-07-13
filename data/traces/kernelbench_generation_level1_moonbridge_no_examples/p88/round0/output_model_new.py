import torch
import torch.nn as nn
import avelang
import avelang.language as al
import math

# GELU constants wrapped as compile-time values
C_SQRT_2_OVER_PI = al.constexpr(math.sqrt(2.0 / math.pi))
C_COEFF = al.constexpr(0.044715)
C_HALF = al.constexpr(0.5)
C_ONE = al.constexpr(1.0)
C_TWO = al.constexpr(2.0)


@avelang.jit
def gelu_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    num_elements: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_size = al.block_dim(0)
    grid_size = al.grid_dim(0)

    global_tid = bid * block_size + tid
    step = grid_size * block_size

    layout = al.make_layout((num_elements,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    for idx in al.range(global_tid, num_elements, step):
        # Load BF16 and upcast to FP32
        val_bf16 = x[idx]
        val = al.convert(val_bf16, al.f32)

        # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x3 = val * val * val
        inner = val + C_COEFF * x3
        scaled = C_SQRT_2_OVER_PI * inner
        # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        two_scaled = C_TWO * scaled
        e2x = al.exp(two_scaled)
        t = (e2x - C_ONE) / (e2x + C_ONE)
        one_plus_t = C_ONE + t
        result = C_HALF * val * one_plus_t

        # Downcast to BF16 and store
        out[idx] = al.convert(result, al.bf16)


def avelang_gelu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA/HIP device."

    num_elements = x.numel()
    x_flat = x.reshape(-1).contiguous()
    out_flat = torch.empty_like(x_flat)

    BLOCK_SIZE = 256
    # Cap grid to keep launch overhead reasonable; grid-stride loop handles the rest
    num_blocks = min(2048, (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE)

    gelu_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](x_flat, out_flat, num_elements)

    return out_flat.reshape(x.shape)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return avelang_gelu(x)
