import torch
import torch.nn as nn
import avelang
import avelang.language as al


THREADS = 256
ITERS = 256
TILE_SIZE = THREADS * ITERS


@avelang.jit
def relu_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    num_elements: al.u32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_dim = al.block_dim(0)

    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((num_elements,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((num_elements,), (1,)))

    zero_bf16 = al.convert(0.0, al.bf16)

    idx = bid * TILE_SIZE + tid

    for _ in al.range(ITERS):
        if idx < num_elements:
            val = x[idx]
            if val < zero_bf16:
                val = zero_bf16
            out[idx] = val
        idx += block_dim


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_relu(x: torch.Tensor) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)

    num_elements = x_bf16.numel()
    out = torch.empty_like(x_bf16)

    num_blocks = (num_elements + TILE_SIZE - 1) // TILE_SIZE
    grid = (num_blocks, 1, 1)

    relu_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, out, num_elements
    )
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_relu(x)
