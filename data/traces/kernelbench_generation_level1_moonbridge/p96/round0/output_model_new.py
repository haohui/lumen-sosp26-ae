import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
TILE_SIZE: al.constexpr = 32768


@avelang.jit
def smooth_l1_reduce_kernel(
    pred_ptr: al.Pointer(al.bf16),
    target_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    N: al.i32,
    num_tiles: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < num_tiles:
        smem = al.make_shared((BLOCK_SIZE,), al.f32)

        tile_start = bid * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > N:
            tile_end = N

        layout_in = al.make_layout((N,), (1,))
        pred = al.make_tensor(pred_ptr, al.bf16, layout_in)
        tgt = al.make_tensor(target_ptr, al.bf16, layout_in)

        local_sum = al.convert(0.0, al.f32)

        for i in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            p = al.convert(pred[i], al.f32)
            t = al.convert(tgt[i], al.f32)
            diff = p - t
            abs_diff = al.abs(diff)
            one = al.convert(1.0, al.f32)
            half = al.convert(0.5, al.f32)

            if abs_diff < one:
                local_sum = local_sum + half * diff * diff
            else:
                local_sum = local_sum + abs_diff - half

        smem[tid] = local_sum
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

        if tid == 0:
            layout_ps = al.make_layout((num_tiles,), (1,))
            ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
            ps[bid] = smem[0]


@avelang.jit
def smooth_l1_aggregate_kernel(
    partial_sum_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    num_tiles: al.i32,
    N: al.i32,
):
    tid = al.thread_id(0)

    smem = al.make_shared((BLOCK_SIZE,), al.f32)

    layout_ps = al.make_layout((num_tiles,), (1,))
    ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)

    total_sum = al.convert(0.0, al.f32)

    chunk_start = al.convert(0, al.i32)
    for _ in al.range(0, 130):
        if chunk_start >= num_tiles:
            break

        local_sum = al.convert(0.0, al.f32)
        idx = chunk_start + tid
        if idx < num_tiles:
            local_sum = ps[idx]

        smem[tid] = local_sum
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

        if tid == 0:
            total_sum = total_sum + smem[0]

        chunk_start = chunk_start + BLOCK_SIZE
        al.syncthreads()

    if tid == 0:
        N_f32 = al.convert(N, al.f32)
        mean = total_sum / N_f32
        layout_out = al.make_layout((1,), (1,))
        out = al.make_tensor(out_ptr, al.f32, layout_out)
        out[0] = mean


def avelang_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA/HIP device."
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape."

    pred_contig = predictions.contiguous().to(torch.bfloat16)
    tgt_contig = targets.contiguous().to(torch.bfloat16)

    N = pred_contig.numel()
    num_tiles = (N + TILE_SIZE - 1) // TILE_SIZE

    partial_sum = torch.empty((num_tiles,), dtype=torch.float32, device=predictions.device)
    smooth_l1_reduce_kernel[lambda: ((num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        pred_contig, tgt_contig, partial_sum, N, num_tiles
    )

    out = torch.empty((1,), dtype=torch.float32, device=predictions.device)
    smooth_l1_aggregate_kernel[lambda: ((1, 1, 1), (BLOCK_SIZE, 1, 1))](
        partial_sum, out, num_tiles, N
    )

    result = out.squeeze()
    return result.to(predictions.dtype)


class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return avelang_smooth_l1_loss(predictions, targets)
