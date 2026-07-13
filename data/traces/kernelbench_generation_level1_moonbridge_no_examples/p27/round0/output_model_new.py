import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def selu_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    numel: al.i32,
):
    # Create 1D tensor views over the flattened runtime buffers.
    layout = al.make_layout((numel,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    tid = al.thread_id(0)
    bid = al.block_id(0)
    bdim = al.block_dim(0)

    idx = bid * bdim + tid
    stride = bdim * al.grid_dim(0)

    # SELU constants in FP32.
    alpha = al.convert(1.6732632423543772, al.f32)
    scale = al.convert(1.0507009873554805, al.f32)
    one = al.convert(1.0, al.f32)
    zero = al.convert(0.0, al.f32)

    for i in al.range(idx, numel, stride):
        val = al.convert(x[i], al.f32)

        # max(0, val)
        pos = val
        if val < zero:
            pos = zero

        exp_val = al.exp(val)
        # alpha * (exp(val) - 1)
        neg = alpha * (exp_val - one)
        # min(0, neg)
        if neg > zero:
            neg = zero

        result = scale * (pos + neg)
        out[i] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty_like(x)
        numel = x.numel()

        BLOCK_SIZE = 256
        # Cover the full element range via grid-stride loop; cap grid to a
        # safe hardware limit.
        full_grid = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid_size = int(full_grid) if full_grid < 65536 else 65535
        if grid_size < 1:
            grid_size = 1

        selu_kernel[lambda: ((grid_size, 1, 1), (BLOCK_SIZE, 1, 1))](
            x.data_ptr(),
            out.data_ptr(),
            numel,
        )
        return out
