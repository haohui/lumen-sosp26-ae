import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def cumsum_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    num_rows: al.i32,
    num_cols: al.i32,
):
    bid = al.block_id(0)

    if bid < num_rows:
        layout = al.make_layout((num_rows, num_cols), (num_cols, 1))
        input_tensor = al.make_tensor(input_ptr, al.bf16, layout)
        output_tensor = al.make_tensor(output_ptr, al.bf16, layout)

        running = al.convert(0.0, al.f32)
        for col in al.range(num_cols):
            val_bf16 = input_tensor[bid, col]
            val_f32 = al.convert(val_bf16, al.f32)
            running = running + val_f32
            output_tensor[bid, col] = al.convert(running, al.bf16)


def avelang_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    x = x.contiguous()
    output = torch.empty_like(x)
    cumsum_kernel[lambda: ((x.shape[0], 1, 1), (1, 1, 1))](
        x, output, x.shape[0], x.shape[1]
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return avelang_cumsum(x, self.dim)
