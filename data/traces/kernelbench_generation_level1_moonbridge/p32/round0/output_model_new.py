import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
ITERS_PER_THREAD: al.constexpr = 128
TILE_SIZE: al.constexpr = BLOCK_SIZE * ITERS_PER_THREAD


@avelang.jit
def hardtanh_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    layout = al.make_layout((N,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    block_start = bid * TILE_SIZE
    block_end = block_start + TILE_SIZE
    if block_end > N:
        block_end = N

    one = al.convert(1.0, al.bf16)
    neg_one = al.convert(-1.0, al.bf16)

    for i in al.range(block_start + tid, block_end, BLOCK_SIZE):
        val = x[i]
        result = val
        if val > one:
            result = one
        if val < neg_one:
            result = neg_one
        out[i] = result


def avelang_hardtanh(x: torch.Tensor) -> torch.Tensor:
    original_dtype = x.dtype
    x_contig = x.to(dtype=torch.bfloat16, device=x.device).contiguous()
    N = x_contig.numel()

    num_blocks = (N + TILE_SIZE - 1) // TILE_SIZE

    out = torch.empty_like(x_contig)

    hardtanh_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contig, out, N
    )

    return out.to(original_dtype)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_hardtanh(x)
