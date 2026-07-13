import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def argmin_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.i64),
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
        current_idx = al.convert(0, al.i64)

        for i in al.range(1, dim1):
            val = input_tensor[batch_idx, i, dim2_idx]
            if val < current_min:
                current_min = val
                current_idx = al.convert(i, al.i64)

        layout_out = al.make_layout(
            (batch_size, dim2),
            (dim2, 1),
        )
        output_tensor = al.make_tensor(output_ptr, al.i64, layout_out)
        output_tensor[batch_idx, dim2_idx] = current_idx


def avelang_argmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."

    x_bf16 = x.contiguous()
    if x_bf16.dtype != torch.bfloat16:
        x_bf16 = x_bf16.to(torch.bfloat16)

    if dim == 1:
        a, b, c = x_bf16.shape
    elif dim == 0:
        x_bf16 = x_bf16.permute(1, 0, 2).contiguous()
        a, b, c = x_bf16.shape
    elif dim == 2:
        x_bf16 = x_bf16.permute(0, 2, 1).contiguous()
        a, b, c = x_bf16.shape
    else:
        raise ValueError(f"Unsupported dim={dim}, expected 0, 1, or 2 for 3D tensor")

    batch_size = a
    dim1 = b
    dim2 = c

    output = torch.empty((batch_size, dim2), dtype=torch.int64, device=x.device)

    total_output = batch_size * dim2
    num_blocks = (total_output + BLOCK_SIZE - 1) // BLOCK_SIZE

    argmin_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, output, batch_size, dim1, dim2
    )

    return output


class ModelNew(nn.Module):
    """
    Optimized model that finds the index of the minimum value along a specified
    dimension using AveLang DSL.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Finds the index of the minimum value along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor containing the indices of the minimum values
                along the specified dimension.
        """
        return avelang_argmin(x, self.dim)
