import torch
import torch.nn as nn
import avelang
import avelang.language as al
import math


@avelang.jit
def reduce_abs_kernel(
    x_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    dim: al.i32,
    REDUCE_BLOCK: al.constexpr,
    N_STEPS: al.constexpr,
):
    row = al.block_id(0)
    tid = al.thread_id(0)

    if row < batch_size:
        x_layout = al.make_layout((batch_size, dim), (dim, 1))
        x = al.make_tensor(x_ptr, al.bf16, x_layout)

        acc = al.convert(0.0, al.f32)
        for idx in al.range(tid, dim, REDUCE_BLOCK):
            val = al.convert(x[row, idx], al.f32)
            acc = acc + al.abs(val)

        smem = al.make_shared((REDUCE_BLOCK,), al.f32)
        smem[tid] = acc
        al.syncthreads()

        stride_val = REDUCE_BLOCK // 2
        for _ in al.range(0, N_STEPS):
            if tid < stride_val:
                smem[tid] = smem[tid] + smem[tid + stride_val]
            stride_val = stride_val // 2
            al.syncthreads()

        if tid == 0:
            dim_f32 = al.convert(dim, al.f32)
            mean_val = smem[0] / dim_f32
            mean_layout = al.make_layout((batch_size,), (1,))
            mean = al.make_tensor(mean_ptr, al.bf16, mean_layout)
            mean[row] = al.convert(mean_val, al.bf16)


@avelang.jit
def normalize_kernel(
    x_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    dim: al.i32,
    NORM_BLOCK: al.constexpr,
):
    row = al.block_id(0)
    col_block = al.block_id(1)
    tid = al.thread_id(0)

    col = col_block * NORM_BLOCK + tid

    x_layout = al.make_layout((batch_size, dim), (dim, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    out = al.make_tensor(out_ptr, al.bf16, x_layout)

    mean_layout = al.make_layout((batch_size,), (1,))
    mean = al.make_tensor(mean_ptr, al.bf16, mean_layout)

    if row < batch_size and col < dim:
        mean_val = al.convert(mean[row], al.f32)
        x_val = al.convert(x[row, col], al.f32)
        out[row, col] = al.convert(x_val / mean_val, al.bf16)


def avelang_l1_normalize(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    batch_size, dim = x.shape

    x_bf16 = x.to(torch.bfloat16).contiguous()

    mean = torch.empty(batch_size, dtype=torch.bfloat16, device=x.device)
    out = torch.empty_like(x_bf16)

    REDUCE_BLOCK = 256
    NORM_BLOCK = 256
    N_STEPS = int(math.log2(REDUCE_BLOCK))
    num_blocks_col = (dim + NORM_BLOCK - 1) // NORM_BLOCK

    reduce_abs_kernel[lambda: ((batch_size, 1, 1), (REDUCE_BLOCK, 1, 1))](
        x_bf16, mean, batch_size, dim, REDUCE_BLOCK, N_STEPS
    )

    normalize_kernel[lambda: ((batch_size, num_blocks_col, 1), (NORM_BLOCK, 1, 1))](
        x_bf16, mean, out, batch_size, dim, NORM_BLOCK
    )

    return out.to(x.dtype)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_l1_normalize(x)
