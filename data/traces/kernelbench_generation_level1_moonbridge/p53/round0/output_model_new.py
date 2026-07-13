import torch
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def min_reduction_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    dim1: al.i32,
    dim2: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    global_id = bid * BLOCK_SIZE + tid
    total_output = batch_size * dim2

    if global_id < total_output:
        batch_idx = global_id // dim2
        dim2_idx = global_id - batch_idx * dim2

        layout_in = al.make_layout(
            (batch_size, dim1, dim2),
            (dim1 * dim2, dim2, 1),
        )
        input_tensor = al.make_tensor(input_ptr, al.bf16, layout_in)

        current_min = input_tensor[batch_idx, 0, dim2_idx]

        for i in al.range(1, dim1):
            val = input_tensor[batch_idx, i, dim2_idx]
            current_min = val if val < current_min else current_min

        layout_out = al.make_layout(
            (batch_size, dim2),
            (dim2, 1),
        )
        output_tensor = al.make_tensor(output_ptr, al.bf16, layout_out)
        output_tensor[batch_idx, dim2_idx] = current_min


def avelang_min(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Input tensor must be bfloat16"

    batch_size = x.shape[0]
    dim1 = x.shape[1]
    dim2 = x.shape[2]

    x_contiguous = x.contiguous()
    output = torch.empty((batch_size, dim2), dtype=torch.bfloat16, device=x.device)

    total_output = batch_size * dim2
    num_blocks = (total_output + BLOCK_SIZE - 1) // BLOCK_SIZE

    min_reduction_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contiguous, output, batch_size, dim1, dim2
    )

    return output


class ModelNew(torch.nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_min(x, self.dim)
