import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def softsign_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    numel: al.i32,
    BLOCK_SIZE: al.constexpr,
):
    """Softsign activation: out = x / (1 + |x|), in BF16 with FP32 accumulation."""
    tid = al.thread_id(0)
    bid = al.block_id(0)

    # Build 1-D tensor views over the flat buffer with dynamic extent.
    layout = al.make_layout((numel,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    stride = al.grid_dim(0) * BLOCK_SIZE
    for idx in al.range(bid * BLOCK_SIZE + tid, numel, stride):
        # Load BF16, promote to FP32 for the arithmetic.
        val_f32 = al.convert(x[idx], al.f32)
        abs_val = al.abs(val_f32)
        denom = al.convert(1.0, al.f32) + abs_val
        result = val_f32 / denom
        out[idx] = al.convert(result, al.bf16)


def avelang_softsign(x: torch.Tensor) -> torch.Tensor:
    """Launch the Softsign kernel on contiguous GPU tensor."""
    assert x.is_cuda, "Input must be on CUDA/HIP device."
    x = x.contiguous()
    out = torch.empty_like(x)

    numel = x.numel()
    BLOCK_SIZE = 256
    # Cap grid at AMD HIP max of 65535 blocks along dim 0; grid-stride loop
    # handles cases where numel > grid * block.
    grid_x = min(65535, (numel + BLOCK_SIZE - 1) // BLOCK_SIZE)

    softsign_kernel[lambda: ((grid_x, 1, 1), (BLOCK_SIZE, 1, 1))](
        x.data_ptr(),
        out.data_ptr(),
        numel,
        BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_softsign(x)
