import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256
MAX_GRID_SIZE = 131072


@avelang.jit
def tanh_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    numel: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_dim = al.block_dim(0)
    grid_dim = al.grid_dim(0)

    layout = al.make_layout((numel,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    start = bid * block_dim + tid
    stride = block_dim * grid_dim

    for idx in al.range(start, numel, stride):
        val = al.convert(x[idx], al.f32)
        out[idx] = al.convert(al.tanh(val), al.bf16)


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_tanh(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."

    original_dtype = x.dtype
    x_bf16 = _to_bf16_contiguous(x)

    numel = x_bf16.numel()
    out = torch.empty_like(x_bf16)

    num_blocks = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    if num_blocks > MAX_GRID_SIZE:
        num_blocks = MAX_GRID_SIZE

    tanh_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, out, numel
    )

    if original_dtype != torch.bfloat16:
        return out.to(dtype=original_dtype)
    return out


class ModelNew(nn.Module):
    """
    Simple model that performs a Tanh activation (AveLang BF16 kernel).
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Tanh activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Tanh applied, same shape as input.
        """
        return avelang_tanh(x)
