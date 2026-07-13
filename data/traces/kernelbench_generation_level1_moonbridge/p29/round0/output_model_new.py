import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256


@avelang.jit
def softplus_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    total_elems: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_dim = al.block_dim(0)

    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((total_elems,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((total_elems,), (1,)))

    idx = bid * block_dim + tid

    threshold = al.convert(20.0, al.f32)
    one = al.convert(1.0, al.f32)

    if idx < total_elems:
        val = al.convert(x[idx], al.f32)
        result = val
        if val < threshold:
            result = al.log(one + al.exp(val))
        out[idx] = al.convert(result, al.bf16)


def avelang_softplus(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Input tensor must be bfloat16"

    x_contig = x.contiguous()
    total_elems = x_contig.numel()
    num_blocks = (total_elems + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x_contig)
    grid = (num_blocks, 1, 1)
    softplus_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](x_contig, out, total_elems)
    return out


class ModelNew(nn.Module):
    """
    Simple model that performs a Softplus activation (AveLang DSL optimized).
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x_bf16 = x.to(dtype=torch.bfloat16, device=x.device).contiguous()
        result = avelang_softplus(x_bf16)
        return result.to(original_dtype)
