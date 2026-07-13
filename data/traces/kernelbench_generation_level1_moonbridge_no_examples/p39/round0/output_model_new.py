import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def l2norm_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    dim: al.i32,
):
    row = al.block_id(0)
    if row >= batch_size:
        return

    tid = al.thread_id(0)

    total = batch_size * dim
    layout = al.make_layout((total,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    base = row * dim

    sum_sq = al.convert(0.0, al.f32)
    for i in al.range(0, dim, 256):
        idx = tid + i
        if idx < dim:
            val = al.convert(x[base + idx], al.f32)
            sum_sq = sum_sq + val * val

    shared = al.make_shared((256,), al.f32)
    shared[tid] = sum_sq
    al.syncthreads()

    stride_val = 128
    for s in al.range(8):
        if tid < stride_val:
            shared[tid] = shared[tid] + shared[tid + stride_val]
        al.syncthreads()
        stride_val = stride_val // 2

    if tid == 0:
        shared[0] = al.sqrt(shared[0])
    al.syncthreads()

    norm = shared[0]

    for i in al.range(0, dim, 256):
        idx = tid + i
        if idx < dim:
            val = al.convert(x[base + idx], al.f32)
            out[base + idx] = al.convert(val / norm, al.bf16)


def avelang_l2norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA/HIP device."
    x = x.contiguous()
    batch_size, dim = x.shape

    out = torch.empty_like(x)

    grid = (batch_size, 1, 1)
    block = (256, 1, 1)

    l2norm_kernel[lambda: (grid, block)](x, out, batch_size, dim)

    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_l2norm(x)
