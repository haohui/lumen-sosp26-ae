import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256
GRID_SIZE = 65536


@avelang.jit
def swish_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    num_elements: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    layout = al.make_layout((num_elements,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    one = al.convert(1.0, al.f32)
    zero_f32 = al.convert(0.0, al.f32)

    idx = bid * BLOCK_SIZE + tid
    stride = al.grid_dim(0) * BLOCK_SIZE

    for i in al.range(idx, num_elements, stride):
        x_val = al.convert(x[i], al.f32)
        neg_x = zero_f32 - x_val
        exp_val = al.exp(neg_x)
        out[i] = al.convert(x_val / (one + exp_val), al.bf16)


def avelang_swish(x: torch.Tensor) -> torch.Tensor:
    x_bf16 = x.contiguous().to(torch.bfloat16)
    num_elements = x_bf16.numel()
    grid_x = min(GRID_SIZE, (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE)
    out = torch.empty_like(x_bf16)
    swish_kernel[lambda: ((grid_x, 1, 1), (BLOCK_SIZE, 1, 1))](x_bf16, out, num_elements)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_swish(x)
