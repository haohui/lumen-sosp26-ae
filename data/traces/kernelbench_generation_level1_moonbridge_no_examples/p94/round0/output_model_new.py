import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def mse_partial_kernel(
    pred_ptr: al.Pointer(al.bf16),
    targ_ptr: al.Pointer(al.bf16),
    partial_out: al.Pointer(al.f32),
    N: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    grid_size = al.grid_dim(0)
    block_size = al.block_dim(0)

    pred = al.make_tensor(pred_ptr, al.bf16, al.make_layout((N,), (1,)))
    targ = al.make_tensor(targ_ptr, al.bf16, al.make_layout((N,), (1,)))
    out = al.make_tensor(partial_out, al.f32, al.make_layout((grid_size,), (1,)))

    acc = al.convert(0.0, al.f32)
    step = grid_size * block_size
    idx = bid * block_size + tid
    for i in al.range(idx, N, step):
        diff = al.convert(pred[i], al.f32) - al.convert(targ[i], al.f32)
        acc = acc + diff * diff

    smem = al.make_shared((256,), al.f32)
    smem[tid] = acc
    al.syncthreads()

    if tid < 128:
        smem[tid] = smem[tid] + smem[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem[tid] = smem[tid] + smem[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem[tid] = smem[tid] + smem[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem[tid] = smem[tid] + smem[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem[tid] = smem[tid] + smem[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem[tid] = smem[tid] + smem[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem[tid] = smem[tid] + smem[tid + 2]
    al.syncthreads()
    if tid == 0:
        smem[0] = smem[0] + smem[1]
    al.syncthreads()

    if tid == 0:
        out[bid] = smem[0]


@avelang.jit
def mse_final_reduce_kernel(
    partial_in: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    num_partials: al.i32,
    N: al.i32,
):
    tid = al.thread_id(0)
    block_size = al.block_dim(0)

    p_in = al.make_tensor(partial_in, al.f32, al.make_layout((num_partials,), (1,)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((1,), (1,)))

    acc = al.convert(0.0, al.f32)
    for i in al.range(tid, num_partials, block_size):
        acc = acc + p_in[i]

    smem = al.make_shared((256,), al.f32)
    smem[tid] = acc
    al.syncthreads()

    if tid < 128:
        smem[tid] = smem[tid] + smem[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem[tid] = smem[tid] + smem[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem[tid] = smem[tid] + smem[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem[tid] = smem[tid] + smem[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem[tid] = smem[tid] + smem[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem[tid] = smem[tid] + smem[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem[tid] = smem[tid] + smem[tid + 2]
    al.syncthreads()
    if tid == 0:
        smem[0] = smem[0] + smem[1]
    al.syncthreads()

    if tid == 0:
        out[0] = smem[0] / al.convert(N, al.f32)


def avelang_mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA/HIP device."
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."

    pred_bf16 = predictions.contiguous().to(torch.bfloat16)
    targ_bf16 = targets.contiguous().to(torch.bfloat16)

    N = predictions.numel()
    block_size = 256
    max_grid = 65535
    grid_size = min((N + block_size - 1) // block_size, max_grid)

    partial_sums = torch.empty(grid_size, dtype=torch.float32, device=predictions.device)
    mse_partial_kernel[lambda: ((grid_size, 1, 1), (block_size, 1, 1))](
        pred_bf16, targ_bf16, partial_sums, N
    )

    result = torch.empty(1, dtype=torch.float32, device=predictions.device)
    mse_final_reduce_kernel[lambda: ((1, 1, 1), (block_size, 1, 1))](
        partial_sums, result, grid_size, N
    )

    return result.squeeze().to(predictions.dtype)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, predictions, targets):
        return avelang_mse(predictions, targets)
