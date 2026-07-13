import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def swish_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    num_elements: al.i64,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_size = al.block_dim(0)

    layout = al.make_layout((num_elements,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    gid = bid * block_size + tid
    stride = block_size * al.grid_dim(0)

    for i in al.range(gid, num_elements, stride):
        # Load BF16, upcast to FP32 for stable numerics
        val = al.convert(x[i], al.f32)

        # swish(x) = x / (1 + exp(-x))
        neg_val = -val
        exp_val = al.exp(neg_val)
        denom = exp_val + al.convert(1.0, al.f32)
        result = val / denom

        # Downcast back to BF16 and store
        out[i] = al.convert(result, al.bf16)


def avelang_swish(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device"
    x = x.contiguous()
    out = torch.empty_like(x)
    num_elements = x.numel()

    BLOCK_SIZE = 256
    # Cap grid to avoid excessive launch overhead while keeping enough
    # parallelism; grid-stride loop handles the rest.
    MAX_GRID = 65535
    num_blocks = min((num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, MAX_GRID)

    swish_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x, out, num_elements
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_swish(x)
