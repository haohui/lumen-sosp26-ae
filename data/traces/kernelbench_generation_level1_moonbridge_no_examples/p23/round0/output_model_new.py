import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256


@avelang.jit
def find_row_max_kernel(
    x_ptr: al.Pointer(al.bf16),
    row_max_ptr: al.Pointer(al.f32),
    M: al.i32,
    D: al.i32,
    BLK: al.constexpr,
):
    row = al.block_id(0)
    tid = al.thread_id(0)

    x_layout = al.make_layout((M, D), (D, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    rm_layout = al.make_layout((M,), (1,))
    row_max = al.make_tensor(row_max_ptr, al.f32, rm_layout)

    # Each thread finds its local max over strided chunk
    local_max = al.convert(-1.0e30, al.f32)
    for i in al.range(tid, D, BLK):
        val = al.convert(x[row, i], al.f32)
        if val > local_max:
            local_max = val

    # Block-level max reduction via shared memory (256 -> 128 -> 64 -> ... -> 1)
    shared = al.make_shared((BLK,), al.f32)
    shared[tid] = local_max
    al.syncthreads()

    if tid < 128:
        other = shared[tid + 128]
        if other > shared[tid]:
            shared[tid] = other
    al.syncthreads()
    if tid < 64:
        other = shared[tid + 64]
        if other > shared[tid]:
            shared[tid] = other
    al.syncthreads()
    if tid < 32:
        other = shared[tid + 32]
        if other > shared[tid]:
            shared[tid] = other
    al.syncthreads()
    if tid < 16:
        other = shared[tid + 16]
        if other > shared[tid]:
            shared[tid] = other
    al.syncthreads()
    if tid < 8:
        other = shared[tid + 8]
        if other > shared[tid]:
            shared[tid] = other
    al.syncthreads()
    if tid < 4:
        other = shared[tid + 4]
        if other > shared[tid]:
            shared[tid] = other
    al.syncthreads()
    if tid < 2:
        other = shared[tid + 2]
        if other > shared[tid]:
            shared[tid] = other
    al.syncthreads()
    if tid == 0:
        other = shared[1]
        if other > shared[0]:
            shared[0] = other
    al.syncthreads()

    if tid == 0:
        row_max[row] = shared[0]


@avelang.jit
def softmax_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    row_max_ptr: al.Pointer(al.f32),
    M: al.i32,
    D: al.i32,
    BLK: al.constexpr,
):
    row = al.block_id(0)
    tid = al.thread_id(0)

    x_layout = al.make_layout((M, D), (D, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    out_layout = al.make_layout((M, D), (D, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)
    rm_layout = al.make_layout((M,), (1,))
    row_max_view = al.make_tensor(row_max_ptr, al.f32, rm_layout)

    row_max_val = row_max_view[row]

    # Pass 1: compute per-thread sum of exp(x - max)
    local_sum = al.convert(0.0, al.f32)
    for i in al.range(tid, D, BLK):
        val = al.convert(x[row, i], al.f32)
        diff = val - row_max_val
        exp_val = al.exp(diff)
        local_sum = local_sum + exp_val

    # Block-level sum reduction via shared memory
    shared = al.make_shared((BLK,), al.f32)
    shared[tid] = local_sum
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
    if tid == 0:
        shared[0] = shared[0] + shared[1]
    al.syncthreads()

    total_sum = shared[0]
    inv_sum = al.convert(1.0, al.f32) / total_sum
    shared[0] = inv_sum
    al.syncthreads()
    inv_sum = shared[0]

    # Pass 2: recompute exp(x - max) and normalize, write output
    for i in al.range(tid, D, BLK):
        val = al.convert(x[row, i], al.f32)
        diff = val - row_max_val
        exp_val = al.exp(diff)
        result = exp_val * inv_sum
        out[row, i] = al.convert(result, al.bf16)


def avelang_softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax along dim=1 using AveLang GPU kernels."""
    if not x.is_cuda:
        x = x.cuda()
    x = x.contiguous()
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    M = x.shape[0]
    D = x.shape[1]

    row_max = torch.empty((M,), dtype=torch.float32, device=x.device)
    find_row_max_kernel[lambda: ((M, 1, 1), (BLOCK_SIZE, 1, 1))](
        x, row_max, M, D, BLOCK_SIZE
    )

    out = torch.empty_like(x)
    softmax_kernel[lambda: ((M, 1, 1), (BLOCK_SIZE, 1, 1))](
        x, out, row_max, M, D, BLOCK_SIZE
    )

    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_softmax(x)
