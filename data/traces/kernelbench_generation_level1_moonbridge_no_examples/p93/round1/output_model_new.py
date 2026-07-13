import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256


@avelang.jit
def masked_mul_kernel(
    x_ptr: al.Pointer(al.bf16),
    mask_ptr: al.Pointer(al.u8),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
):
    row = al.block_id(0)
    tile_idx = al.block_id(1)
    tid = al.thread_id(0)

    if row < M:
        total_elems = M * N
        layout = al.make_layout((total_elems,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout)
        mask = al.make_tensor(mask_ptr, al.u8, layout)
        out = al.make_tensor(out_ptr, al.bf16, layout)

        base = row * N
        col = tile_idx * 256 + tid
        if col < N:
            idx = base + col
            x_val = x[idx]
            m_val = al.convert(mask[idx], al.bf16)
            out[idx] = x_val * m_val


def _run_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    if not x.is_cuda:
        x = x.cuda()
    if not mask.is_cuda:
        mask = mask.cuda()

    assert x.shape == mask.shape, "x and mask must have same shape."
    assert x.dim() == 2, "Input must be 2D."
    assert dim == 1, "Only dim=1 supported."

    M, N = x.shape

    orig_dtype = x.dtype
    x_bf16 = x.to(torch.bfloat16).contiguous()
    mask_u8 = mask.to(torch.uint8).contiguous()
    out_bf16 = torch.empty_like(x_bf16)

    num_tiles = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (M, num_tiles, 1)
    block = (BLOCK_SIZE, 1, 1)

    masked_mul_kernel[lambda: (grid, block)](
        x_bf16, mask_u8, out_bf16, M, N
    )

    result = torch.cumsum(out_bf16, dim=dim)
    return result.to(orig_dtype)


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        return _run_masked_cumsum(x, mask, self.dim)
