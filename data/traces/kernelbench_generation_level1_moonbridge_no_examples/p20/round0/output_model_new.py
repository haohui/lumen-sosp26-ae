import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def leaky_relu_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    negative_slope: al.f32,
):
    layout = al.make_layout((N,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    bdim = al.block_dim(0)
    gdim = al.grid_dim(0)
    tid = al.thread_id(0)
    start = al.block_id(0) * bdim + tid

    slope_bf16 = al.convert(negative_slope, al.bf16)
    zero_bf16 = al.convert(0.0, al.bf16)

    for idx in al.range(start, N, bdim * gdim):
        val = x[idx]
        if val >= zero_bf16:
            out[idx] = val
        else:
            out[idx] = val * slope_bf16


def avelang_leaky_relu(x: torch.Tensor, negative_slope: float) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA/HIP device."
    x = x.contiguous()
    N = x.numel()
    out = torch.empty_like(x)

    BLOCK_SIZE = 256
    num_sms = 304
    grid = num_sms * 32

    leaky_relu_kernel[lambda: ((grid, 1, 1), (BLOCK_SIZE, 1, 1))](
        x.data_ptr(), out.data_ptr(), N, negative_slope
    , num_warps=4)
    return out


class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_leaky_relu(x, self.negative_slope)
