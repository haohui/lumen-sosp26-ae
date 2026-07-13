import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def cumsum_dim1_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    num_rows: al.i32,
    num_cols: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    row = bid * BLOCK_SIZE + tid

    if row < num_rows:
        layout = al.make_layout((num_rows, num_cols), (num_cols, 1))
        inp = al.make_tensor(input_ptr, al.bf16, layout)
        out = al.make_tensor(output_ptr, al.bf16, layout)

        acc = al.convert(inp[row, 0], al.f32)
        out[row, 0] = al.convert(acc, al.bf16)

        for col in al.range(1, num_cols):
            val = al.convert(inp[row, col], al.f32)
            acc = acc + val
            out[row, col] = al.convert(acc, al.bf16)


@avelang.jit
def cumsum_dim0_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    num_rows: al.i32,
    num_cols: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    col = bid * BLOCK_SIZE + tid

    if col < num_cols:
        layout = al.make_layout((num_rows, num_cols), (num_cols, 1))
        inp = al.make_tensor(input_ptr, al.bf16, layout)
        out = al.make_tensor(output_ptr, al.bf16, layout)

        acc = al.convert(inp[0, col], al.f32)
        out[0, col] = al.convert(acc, al.bf16)

        for r in al.range(1, num_rows):
            val = al.convert(inp[r, col], al.f32)
            acc = acc + val
            out[r, col] = al.convert(acc, al.bf16)


def avelang_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    if not x.is_cuda:
        x = x.cuda()

    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    x = x.contiguous()
    output = torch.empty_like(x)

    if x.dim() != 2:
        raise ValueError(f"cumsum kernel only supports 2D tensors, got {x.dim()}D")

    num_rows = x.shape[0]
    num_cols = x.shape[1]

    if dim == 1:
        num_blocks = (num_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
        cumsum_dim1_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
            x, output, num_rows, num_cols
        )
    elif dim == 0:
        num_blocks = (num_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
        cumsum_dim0_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
            x, output, num_rows, num_cols
        )
    else:
        raise ValueError(f"Unsupported dim: {dim}")

    return output


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return avelang_cumsum(x, self.dim)
