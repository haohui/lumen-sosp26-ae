import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256
ELEMS_PER_THREAD = 8


@avelang.jit
def softsign_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    numel: al.u32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((numel,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((numel,), (1,)))

    # Each thread processes ELEMS_PER_THREAD contiguous elements.
    # Adjacent threads access adjacent memory → warp-coalesced loads.
    idx = (bid * BLOCK_SIZE + tid) * ELEMS_PER_THREAD

    for _ in al.range(ELEMS_PER_THREAD):
        if idx < numel:
            val_bf16 = x[idx]
            val_f32 = al.convert(val_bf16, al.f32)
            abs_val = al.abs(val_f32)
            one = al.convert(1.0, al.f32)
            result = val_f32 / (one + abs_val)
            out[idx] = al.convert(result, al.bf16)
        idx = idx + al.convert(1, al.u32)


def avelang_softsign(x: torch.Tensor) -> torch.Tensor:
    x_bf16 = x.contiguous().cuda().to(dtype=torch.bfloat16)
    numel = x_bf16.numel()
    out = torch.empty_like(x_bf16)

    threads_per_block = BLOCK_SIZE
    elems_per_block = BLOCK_SIZE * ELEMS_PER_THREAD
    num_blocks = (numel + elems_per_block - 1) // elems_per_block

    grid = (num_blocks, 1, 1)
    softsign_kernel[lambda: (grid, (threads_per_block, 1, 1))](
        x_bf16, out, numel
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_softsign(x)
