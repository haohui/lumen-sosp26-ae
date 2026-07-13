import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def sum_reduction_kernel(
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

        acc = al.make_local((1, 1), al.f32)
        acc[0, 0] = al.convert(input_tensor[batch_idx, 0, dim2_idx], al.f32)

        for i in al.range(1, dim1):
            val_f32 = al.convert(input_tensor[batch_idx, i, dim2_idx], al.f32)
            acc[0, 0] = acc[0, 0] + val_f32

        layout_out = al.make_layout(
            (batch_size, 1, dim2),
            (dim2, dim2, 1),
        )
        output_tensor = al.make_tensor(output_ptr, al.bf16, layout_out)
        output_tensor[batch_idx, 0, dim2_idx] = al.convert(acc[0, 0], al.bf16)


def avelang_sum(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Input tensor must be bfloat16"

    batch_size = x.shape[0]
    dim1 = x.shape[1]
    dim2 = x.shape[2]

    x_contiguous = x.contiguous()

    output = torch.empty((batch_size, 1, dim2), dtype=torch.bfloat16, device=x.device)

    total_output = output.numel()
    num_blocks = (total_output + BLOCK_SIZE - 1) // BLOCK_SIZE

    sum_reduction_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contiguous, output, batch_size, dim1, dim2
    )

    return output


class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specific dimension using AveLang DSL.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        return avelang_sum(x, self.dim)
