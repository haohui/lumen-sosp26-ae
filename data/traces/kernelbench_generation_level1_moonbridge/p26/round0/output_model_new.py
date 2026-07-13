import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def gelu_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    n: al.i32,
):
    """Grid-stride GELU kernel with tanh approximation via exp2.

    GELU(x) = x / (1 + exp(-2 * sqrt(2/pi) * (x + 0.044715 * x^3)))
              = x / (1 + exp2(c * (x + 0.044715 * x^3)))
    where c = -2*sqrt(2/pi)/ln(2) ≈ -2.302585.
    """
    block_id = al.block_id(0)
    thread_id = al.thread_id(0)

    start = block_id * BLOCK_SIZE + thread_id
    stride = al.grid_dim(0) * BLOCK_SIZE

    layout = al.make_layout((n,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    for idx in al.range(start, n, stride):
        val = al.convert(x[idx], al.f32)

        x3 = val * val * val
        exp2_arg = -2.3025850929940455 * (val + 0.044715 * x3)
        result = val / (1.0 + al.exp2(exp2_arg))

        out[idx] = al.convert(result, al.bf16)


def avelang_gelu(x: torch.Tensor) -> torch.Tensor:
    """Host wrapper: converts to BF16, launches kernel, converts back."""
    if not x.is_cuda:
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_contig = x.contiguous()
    orig_dtype = x_contig.dtype
    x_bf16 = x_contig.to(torch.bfloat16)

    n = x_bf16.numel()
    out_bf16 = torch.empty_like(x_bf16)

    num_blocks = 65536

    gelu_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, out_bf16, n
    )

    return out_bf16.to(orig_dtype)


class ModelNew(nn.Module):
    """Optimized GELU activation using AveLang DSL BF16 kernel."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_gelu(x)
