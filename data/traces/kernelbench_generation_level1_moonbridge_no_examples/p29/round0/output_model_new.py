import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def softplus_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    n_elements: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_size = al.block_dim(0)
    grid_size = al.grid_dim(0)

    stride = block_size * grid_size
    idx = bid * block_size + tid

    layout = al.make_layout((n_elements,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    for i in al.range(idx, n_elements, stride):
        val = al.convert(x[i], al.f32)
        if val > al.convert(20.0, al.f32):
            out[i] = al.convert(val, al.bf16)
        else:
            exp_val = al.exp(val)
            log_val = al.log(al.convert(1.0, al.f32) + exp_val)
            out[i] = al.convert(log_val, al.bf16)


def avelang_softplus(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA/HIP device"
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()

    BLOCK_SIZE = 256
    grid_size = min((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, 65535)

    softplus_kernel[lambda: ((grid_size, 1, 1), (BLOCK_SIZE, 1, 1))](
        x, out, n_elements
    )
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return avelang_softplus(x)
