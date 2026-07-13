import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def argmax_dim0_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.i64),
    s0: al.i32,
    s1: al.i32,
    s2: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    global_id = bid * BLOCK_SIZE + tid
    total_output = s1 * s2

    if global_id < total_output:
        dim1_idx = global_id // s2
        dim2_idx = global_id - dim1_idx * s2

        stride0 = s1 * s2
        stride1 = s2
        layout_in = al.make_layout(
            (s0, s1, s2),
            (stride0, stride1, 1),
        )
        input_tensor = al.make_tensor(input_ptr, al.bf16, layout_in)

        best_val = input_tensor[0, dim1_idx, dim2_idx]
        best_idx = al.convert(0, al.i64)

        for i in al.range(1, s0):
            val = input_tensor[i, dim1_idx, dim2_idx]
            if val > best_val:
                best_val = val
                best_idx = al.convert(i, al.i64)

        layout_out = al.make_layout(
            (s1, s2),
            (s2, 1),
        )
        output_tensor = al.make_tensor(output_ptr, al.i64, layout_out)
        output_tensor[dim1_idx, dim2_idx] = best_idx


@avelang.jit
def argmax_dim1_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.i64),
    s0: al.i32,
    s1: al.i32,
    s2: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    global_id = bid * BLOCK_SIZE + tid
    total_output = s0 * s2

    if global_id < total_output:
        batch_idx = global_id // s2
        dim2_idx = global_id - batch_idx * s2

        stride0 = s1 * s2
        stride1 = s2
        layout_in = al.make_layout(
            (s0, s1, s2),
            (stride0, stride1, 1),
        )
        input_tensor = al.make_tensor(input_ptr, al.bf16, layout_in)

        best_val = input_tensor[batch_idx, 0, dim2_idx]
        best_idx = al.convert(0, al.i64)

        for i in al.range(1, s1):
            val = input_tensor[batch_idx, i, dim2_idx]
            if val > best_val:
                best_val = val
                best_idx = al.convert(i, al.i64)

        layout_out = al.make_layout(
            (s0, s2),
            (s2, 1),
        )
        output_tensor = al.make_tensor(output_ptr, al.i64, layout_out)
        output_tensor[batch_idx, dim2_idx] = best_idx


@avelang.jit
def argmax_dim2_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.i64),
    s0: al.i32,
    s1: al.i32,
    s2: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    global_id = bid * BLOCK_SIZE + tid
    total_output = s0 * s1

    if global_id < total_output:
        batch_idx = global_id // s1
        dim1_idx = global_id - batch_idx * s1

        stride0 = s1 * s2
        stride1 = s2
        layout_in = al.make_layout(
            (s0, s1, s2),
            (stride0, stride1, 1),
        )
        input_tensor = al.make_tensor(input_ptr, al.bf16, layout_in)

        best_val = input_tensor[batch_idx, dim1_idx, 0]
        best_idx = al.convert(0, al.i64)

        for i in al.range(1, s2):
            val = input_tensor[batch_idx, dim1_idx, i]
            if val > best_val:
                best_val = val
                best_idx = al.convert(i, al.i64)

        layout_out = al.make_layout(
            (s0, s1),
            (s1, 1),
        )
        output_tensor = al.make_tensor(output_ptr, al.i64, layout_out)
        output_tensor[batch_idx, dim1_idx] = best_idx


def avelang_argmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.ndim == 3, "Input tensor must be 3D"

    s0 = x.shape[0]
    s1 = x.shape[1]
    s2 = x.shape[2]

    x_contiguous = x.contiguous()
    if x_contiguous.dtype != torch.bfloat16:
        x_contiguous = x_contiguous.to(torch.bfloat16)

    if dim == 0:
        total_output = s1 * s2
        output = torch.empty((s1, s2), dtype=torch.int64, device=x.device)
        num_blocks = (total_output + BLOCK_SIZE - 1) // BLOCK_SIZE
        argmax_dim0_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_contiguous, output, s0, s1, s2
        )
    elif dim == 1:
        total_output = s0 * s2
        output = torch.empty((s0, s2), dtype=torch.int64, device=x.device)
        num_blocks = (total_output + BLOCK_SIZE - 1) // BLOCK_SIZE
        argmax_dim1_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_contiguous, output, s0, s1, s2
        )
    elif dim == 2:
        total_output = s0 * s1
        output = torch.empty((s0, s1), dtype=torch.int64, device=x.device)
        num_blocks = (total_output + BLOCK_SIZE - 1) // BLOCK_SIZE
        argmax_dim2_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_contiguous, output, s0, s1, s2
        )
    else:
        raise ValueError(f"Unsupported dim={dim} for 3D tensor")

    return output


class ModelNew(nn.Module):
    """
    Optimized model that performs Argmax over a specified dimension using AveLang DSL.
    """
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_argmax(x, self.dim)
