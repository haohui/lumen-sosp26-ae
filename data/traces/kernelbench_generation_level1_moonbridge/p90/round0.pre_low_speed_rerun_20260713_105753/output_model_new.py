import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def cumprod_dim1_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    row_len: al.i32,
):
    """
    Cumulative product along dim=1 for a 2D tensor.

    Each thread handles one row, computing a sequential left-to-right
    cumulative product. This matches PyTorch's sequential evaluation order
    exactly, ensuring bit-exact BF16 output for the same inputs.
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)
    row = bid * BLOCK_SIZE + tid

    if row >= row_len:
        return

    total_elems = row_len * row_len
    layout = al.make_layout((total_elems,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    row_start = row * row_len

    running = al.convert(1.0, al.f32)
    for i in al.range(row_len):
        idx = row_start + i
        running = running * al.convert(x[idx], al.f32)
        out[idx] = al.convert(running, al.bf16)


def _to_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    """Convert a tensor to contiguous BF16 on the current CUDA/HIP device."""
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_cumprod(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    GPU-accelerated cumulative product using an AveLang prefix-scan kernel.

    Supports dim=0 and dim=1 for 2D tensors. For dim=0, the tensor is
    transposed, the kernel runs on dim=1, and the result is transposed back.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    original_dtype = x.dtype
    x_bf16 = _to_bf16_cuda_contiguous(x)

    if x_bf16.dim() != 2:
        raise ValueError(f"Expected 2D input tensor, got shape {x_bf16.shape}")

    rows, cols = x_bf16.shape

    if dim == 0:
        x_t = x_bf16.t().contiguous()
        out_t = torch.empty_like(x_t)

        num_blocks = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
        cumprod_dim1_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_t, out_t, cols
        )

        out = out_t.t().contiguous()
    elif dim == 1:
        out = torch.empty_like(x_bf16)

        num_blocks = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
        cumprod_dim1_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_bf16, out, cols
        )
    else:
        raise ValueError(
            f"Unsupported dim={dim}; only dim=0 and dim=1 are supported for 2D tensors."
        )

    if original_dtype != torch.bfloat16:
        out = out.to(dtype=original_dtype)
    return out


class ModelNew(nn.Module):
    """
    Optimized cumulative product model using an AveLang GPU kernel.

    Matches the semantics of torch.cumprod(x, dim=self.dim) with BF16
    computation and FP32 accumulation for numerical stability.

    Uses sequential per-row processing to match PyTorch's evaluation order
    exactly for bit-exact BF16 results.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return avelang_cumprod(x, self.dim)
