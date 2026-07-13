import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def smooth_l1_reduce_kernel(
    pred_ptr: al.Pointer(al.bf16),
    tgt_ptr: al.Pointer(al.bf16),
    partial_out_ptr: al.Pointer(al.f32),
    N: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    blk_sz = al.block_dim(0)
    grd_sz = al.grid_dim(0)

    pred = al.make_tensor(pred_ptr, al.bf16, al.make_layout((N,), (1,)))
    tgt = al.make_tensor(tgt_ptr, al.bf16, al.make_layout((N,), (1,)))

    gid = bid * blk_sz + tid
    total = grd_sz * blk_sz
    block_sum = al.convert(0.0, al.f32)
    one = al.convert(1.0, al.f32)
    half = al.convert(0.5, al.f32)

    for i in al.range(gid, N, total):
        p = al.convert(pred[i], al.f32)
        t = al.convert(tgt[i], al.f32)
        diff = p - t
        ad = al.abs(diff)
        if ad < one:
            block_sum = block_sum + half * diff * diff
        else:
            block_sum = block_sum + ad - half

    smem = al.make_shared((256,), al.f32)
    smem[tid] = block_sum
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
    if tid < 1:
        smem[tid] = smem[tid] + smem[tid + 1]
    al.syncthreads()

    if tid == 0:
        partial_out = al.make_tensor(partial_out_ptr, al.f32, al.make_layout((grd_sz,), (1,)))
        partial_out[bid] = smem[0]


@avelang.jit
def final_reduce_kernel(
    partial_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    num_blocks: al.i32,
    N: al.i32,
):
    tid = al.thread_id(0)
    blk_sz = al.block_dim(0)

    partials = al.make_tensor(partial_ptr, al.f32, al.make_layout((num_blocks,), (1,)))

    thread_sum = al.convert(0.0, al.f32)
    for i in al.range(tid, num_blocks, blk_sz):
        thread_sum = thread_sum + partials[i]

    smem = al.make_shared((256,), al.f32)
    smem[tid] = thread_sum
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
    if tid < 1:
        smem[tid] = smem[tid] + smem[tid + 1]
    al.syncthreads()

    if tid == 0:
        out = al.make_tensor(out_ptr, al.f32, al.make_layout((1,), (1,)))
        out[0] = smem[0] / al.convert(N, al.f32)


def avelang_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    N = predictions.numel()
    pred_bf16 = predictions.contiguous().to(torch.bfloat16)
    tgt_bf16 = targets.contiguous().to(torch.bfloat16)

    BLOCK_SIZE = 256
    ELEMS_PER_THREAD = 256
    grid_size = max(1, (N + BLOCK_SIZE * ELEMS_PER_THREAD - 1) // (BLOCK_SIZE * ELEMS_PER_THREAD))

    partial_sums = torch.empty(grid_size, dtype=torch.float32, device=pred_bf16.device)

    smooth_l1_reduce_kernel[lambda: ((grid_size, 1, 1), (BLOCK_SIZE, 1, 1))](
        pred_bf16.data_ptr(), tgt_bf16.data_ptr(), partial_sums.data_ptr(), N
    )

    result = torch.empty(1, dtype=torch.float32, device=pred_bf16.device)
    final_reduce_kernel[lambda: ((1, 1, 1), (BLOCK_SIZE, 1, 1))](
        partial_sums.data_ptr(), result.data_ptr(), grid_size, N
    )

    return result.squeeze().to(torch.bfloat16)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return avelang_smooth_l1_loss(predictions, targets)
