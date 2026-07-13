import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256
ELEMS_PER_THREAD = 512


@avelang.jit
def reduce_sum_sq_kernel(
    x_ptr: al.Pointer(al.bf16),
    partial_sums_ptr: al.Pointer(al.f32),
    num_elements: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    x_layout = al.make_layout((num_elements,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    block_start = bid * 256 * 512
    acc = al.convert(0.0, al.f32)
    for i in al.range(512):
        idx = block_start + i * 256 + tid
        if idx < num_elements:
            val = al.convert(x[idx], al.f32)
            acc = acc + val * val

    shared = al.make_shared((256,), al.f32)
    shared[tid] = acc
    al.syncthreads()

    if tid < 128:
        shared[tid] = shared[tid] + shared[tid + 128]
    al.syncthreads()
    if tid < 64:
        shared[tid] = shared[tid] + shared[tid + 64]
    al.syncthreads()
    if tid < 32:
        shared[tid] = shared[tid] + shared[tid + 32]
    al.syncthreads()
    if tid < 16:
        shared[tid] = shared[tid] + shared[tid + 16]
    al.syncthreads()
    if tid < 8:
        shared[tid] = shared[tid] + shared[tid + 8]
    al.syncthreads()
    if tid < 4:
        shared[tid] = shared[tid] + shared[tid + 4]
    al.syncthreads()
    if tid < 2:
        shared[tid] = shared[tid] + shared[tid + 2]
    al.syncthreads()
    if tid < 1:
        shared[tid] = shared[tid] + shared[tid + 1]
    al.syncthreads()

    if tid == 0:
        partial_layout = al.make_layout((al.grid_dim(0),), (1,))
        partial = al.make_tensor(partial_sums_ptr, al.f32, partial_layout)
        partial[bid] = shared[0]


@avelang.jit
def final_reduce_kernel(
    partial_sums_ptr: al.Pointer(al.f32),
    num_partial_sums: al.i32,
):
    tid = al.thread_id(0)

    partial_layout = al.make_layout((num_partial_sums,), (1,))
    partial = al.make_tensor(partial_sums_ptr, al.f32, partial_layout)

    acc = al.convert(0.0, al.f32)
    for i in al.range(tid, num_partial_sums, 256):
        acc = acc + partial[i]

    shared = al.make_shared((256,), al.f32)
    shared[tid] = acc
    al.syncthreads()

    if tid < 128:
        shared[tid] = shared[tid] + shared[tid + 128]
    al.syncthreads()
    if tid < 64:
        shared[tid] = shared[tid] + shared[tid + 64]
    al.syncthreads()
    if tid < 32:
        shared[tid] = shared[tid] + shared[tid + 32]
    al.syncthreads()
    if tid < 16:
        shared[tid] = shared[tid] + shared[tid + 16]
    al.syncthreads()
    if tid < 8:
        shared[tid] = shared[tid] + shared[tid + 8]
    al.syncthreads()
    if tid < 4:
        shared[tid] = shared[tid] + shared[tid + 4]
    al.syncthreads()
    if tid < 2:
        shared[tid] = shared[tid] + shared[tid + 2]
    al.syncthreads()
    if tid < 1:
        shared[tid] = shared[tid] + shared[tid + 1]
    al.syncthreads()

    if tid == 0:
        partial[0] = al.sqrt(shared[0])


@avelang.jit
def normalize_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    partial_sums_ptr: al.Pointer(al.f32),
    num_elements: al.i32,
    num_partial_sums: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    partial_layout = al.make_layout((num_partial_sums,), (1,))
    partial = al.make_tensor(partial_sums_ptr, al.f32, partial_layout)
    norm_val = partial[tid - tid]

    x_layout = al.make_layout((num_elements,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    out = al.make_tensor(out_ptr, al.bf16, x_layout)

    block_start = bid * 256 * 512
    for i in al.range(512):
        idx = block_start + i * 256 + tid
        if idx < num_elements:
            val = al.convert(x[idx], al.f32)
            result = val / norm_val
            out[idx] = al.convert(result, al.bf16)


def avelang_frobenius_norm(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    original_dtype = x.dtype
    x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x

    num_elements = x.numel()
    tile_size = BLOCK_SIZE * ELEMS_PER_THREAD
    num_blocks = (num_elements + tile_size - 1) // tile_size

    partial_sums = torch.empty(num_blocks, dtype=torch.float32, device=x.device)
    out_bf16 = torch.empty_like(x_bf16)

    reduce_sum_sq_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16.data_ptr(),
        partial_sums.data_ptr(),
        num_elements,
    )

    final_reduce_kernel[lambda: ((1, 1, 1), (BLOCK_SIZE, 1, 1))](
        partial_sums.data_ptr(),
        num_blocks,
    )

    normalize_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16.data_ptr(),
        out_bf16.data_ptr(),
        partial_sums.data_ptr(),
        num_elements,
        num_blocks,
    )

    if original_dtype != torch.bfloat16:
        return out_bf16.to(original_dtype)
    return out_bf16


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_frobenius_norm(x)
