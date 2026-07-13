import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256
NUM_BLOCKS = 65536


@avelang.jit
def sigmoid_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    total_elements: al.i32,
    stride: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    idx = bid * BLOCK_SIZE + tid

    layout = al.make_layout((total_elements,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    one = al.convert(1.0, al.f32)
    log2e = al.convert(1.44269504089, al.f32)

    for i in al.range(idx, total_elements, stride):
        val = al.convert(x[i], al.f32)
        exp_val = al.exp2(-val * log2e)
        denom = one + exp_val
        result = al.amdgpu.rcp(denom)
        out[i] = al.convert(result, al.bf16)


def avelang_sigmoid(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."

    orig_dtype = x.dtype
    x_bf16 = x.contiguous().to(dtype=torch.bfloat16)
    total_elements = x_bf16.numel()
    out = torch.empty_like(x_bf16)

    num_blocks = NUM_BLOCKS
    if num_blocks > total_elements:
        num_blocks = total_elements
    if num_blocks == 0:
        num_blocks = 1
    stride = num_blocks * BLOCK_SIZE

    sigmoid_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, out, total_elements, stride
    )

    return out.to(orig_dtype)


class ModelNew(nn.Module):
    """
    Simple model that performs a Sigmoid activation via AveLang DSL.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Sigmoid activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Sigmoid applied, same shape as input.
        """
        return avelang_sigmoid(x)
