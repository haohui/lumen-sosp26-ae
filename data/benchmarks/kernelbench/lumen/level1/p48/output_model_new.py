import torch
import substrate
import substrate.language as S

BLOCK_SIZE: S.constexpr = 256


@substrate.jit
def mean_reduction_kernel(
    input_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    dim1: S.i32,
    dim2: S.i32,
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    global_id = bid * BLOCK_SIZE + tid
    total_output = batch_size * dim2

    if global_id < total_output:
        batch_idx = global_id // dim2
        dim2_idx = global_id - batch_idx * dim2

        layout_in = S.make_layout(
            (batch_size, dim1, dim2),
            (dim1 * dim2, dim2, 1),
        )
        input_tensor = S.make_tensor(input_ptr, S.bf16, layout_in)

        # Use FP32 for accumulation to maintain precision
        current_sum = S.convert(0.0, S.f32)

        for i in S.range(dim1):
            val = input_tensor[batch_idx, i, dim2_idx]
            current_sum = current_sum + S.convert(val, S.f32)

        # Compute mean by dividing by dim1
        inv_dim1 = S.amdgpu.rcp(S.convert(dim1, S.f32))
        mean_val = current_sum * inv_dim1

        layout_out = S.make_layout(
            (batch_size, dim2),
            (dim2, 1),
        )
        output_tensor = S.make_tensor(output_ptr, S.bf16, layout_out)
        output_tensor[batch_idx, dim2_idx] = S.convert(mean_val, S.bf16)


def substrate_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Input tensor must be bfloat16"
    assert dim == 1, "Kernel optimized for dim=1 reduction"

    batch_size = x.shape[0]
    dim1 = x.shape[1]
    dim2 = x.shape[2]

    x_contiguous = x.contiguous()
    output = torch.empty((batch_size, dim2), dtype=torch.bfloat16, device=x.device)

    total_output = batch_size * dim2
    num_blocks = (total_output + BLOCK_SIZE - 1) // BLOCK_SIZE

    mean_reduction_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contiguous, output, batch_size, dim1, dim2
    )

    return output


class ModelNew(torch.nn.Module):
    """
    Optimized model that performs mean reduction over a specific dimension using Substrate DSL.
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
        Reduces the input tensor along the specified dimension by taking the mean.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension.
        """
        x_bf16 = x.to(torch.bfloat16)
        return substrate_mean(x_bf16, self.dim)
