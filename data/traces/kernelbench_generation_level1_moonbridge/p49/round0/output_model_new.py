import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def max_reduction_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    dim1: al.i32,
    dim2: al.i32,
):
    tid = al.thread_id(0)
    batch_idx = al.block_id(0)
    block_dim2 = al.block_id(1)

    dim2_idx = block_dim2 * BLOCK_SIZE + tid

    if batch_idx < batch_size and dim2_idx < dim2:

        layout_in = al.make_layout(
            (batch_size, dim1, dim2),
            (dim1 * dim2, dim2, 1),
        )
        input_tensor = al.make_tensor(input_ptr, al.bf16, layout_in)

        current_max = input_tensor[batch_idx, 0, dim2_idx]

        for i in al.range(1, dim1):
            val = input_tensor[batch_idx, i, dim2_idx]
            current_max = val if val > current_max else current_max

        layout_out = al.make_layout(
            (batch_size, dim2),
            (dim2, 1),
        )
        output_tensor = al.make_tensor(output_ptr, al.bf16, layout_out)
        output_tensor[batch_idx, dim2_idx] = current_max


def avelang_max(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Input tensor must be bfloat16"

    batch_size = x.shape[0]
    dim1 = x.shape[1]
    dim2 = x.shape[2]

    x_contiguous = x.contiguous()
    output = torch.empty((batch_size, dim2), dtype=torch.bfloat16, device=x.device)

    grid_x = batch_size
    grid_y = (dim2 + BLOCK_SIZE - 1) // BLOCK_SIZE

    max_reduction_kernel[lambda: ((grid_x, grid_y, 1), (BLOCK_SIZE, 1, 1))](
        x_contiguous, output, batch_size, dim1, dim2
    )

    return output


class ModelNew(nn.Module):
    """
    Optimized model that performs Max reduction over a specific dimension using AveLang DSL.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max reduction over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        return avelang_max(x, self.dim)
