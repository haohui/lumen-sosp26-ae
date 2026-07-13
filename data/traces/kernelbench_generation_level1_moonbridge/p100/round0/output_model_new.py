import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
TILE_SIZE: al.constexpr = 262144
MAX_TILES_PER_ROUND: al.constexpr = 256


@avelang.jit
def hinge_reduce_kernel(
    pred_ptr: al.Pointer(al.bf16),
    tgt_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    M: al.i32,
    K: al.i32,
    num_tiles: al.i32,
):
    """
    Phase 1: compute per-element hinge loss over a tile of the flattened (M*K) tensor
    and reduce to a partial sum via shared-memory tree reduction.
    Launch: grid = (num_tiles, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < num_tiles:
        smem = al.make_shared((BLOCK_SIZE,), al.f32)

        N = M * K
        tile_start = bid * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > N:
            tile_end = N

        layout_pred = al.make_layout((N,), (1,))
        pred = al.make_tensor(pred_ptr, al.bf16, layout_pred)
        layout_tgt = al.make_layout((M,), (1,))
        tgt = al.make_tensor(tgt_ptr, al.bf16, layout_tgt)

        one = al.convert(1.0, al.f32)
        zero = al.convert(0.0, al.f32)
        local_sum = al.convert(0.0, al.f32)

        for i in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            row = i // K
            p = al.convert(pred[i], al.f32)
            t = al.convert(tgt[row], al.f32)
            hinge_val = one - p * t
            if hinge_val < zero:
                hinge_val = zero
            local_sum = local_sum + hinge_val

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
def hinge_aggregate_kernel(
    partial_sum_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    num_tiles: al.i32,
    M: al.i32,
    K: al.i32,
):
    """
    Phase 2: aggregate partial tile sums into a final mean.
    Launch: grid = (1, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)

    smem = al.make_shared((BLOCK_SIZE,), al.f32)

    layout_ps = al.make_layout((num_tiles,), (1,))
    ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)

    local_sum = al.convert(0.0, al.f32)

    chunk_start = al.convert(0, al.i32)
    for _ in al.range(0, 32):
        if chunk_start >= num_tiles:
            break

        smem_sum = al.convert(0.0, al.f32)
        tile_idx = chunk_start + tid
        if tile_idx < num_tiles:
            smem_sum = ps[tile_idx]

        smem[tid] = smem_sum
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
            local_sum = local_sum + smem[0]

        chunk_start = chunk_start + MAX_TILES_PER_ROUND
        al.syncthreads()

    if tid == 0:
        N = M * K
        N_f32 = al.convert(N, al.f32)
        mean_val = local_sum / N_f32
        layout_out = al.make_layout((1,), (1,))
        out = al.make_tensor(out_ptr, al.bf16, layout_out)
        out[0] = al.convert(mean_val, al.bf16)


def avelang_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA/HIP device."

    pred_bf16 = predictions.contiguous().to(torch.bfloat16)
    tgt_bf16 = targets.contiguous().to(torch.bfloat16)

    M = pred_bf16.shape[0]
    K = pred_bf16.shape[1]
    N = M * K
    num_tiles = (N + TILE_SIZE - 1) // TILE_SIZE

    partial_sum = torch.empty((num_tiles,), dtype=torch.float32, device=pred_bf16.device)

    hinge_reduce_kernel[lambda: ((num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        pred_bf16, tgt_bf16, partial_sum, M, K, num_tiles
    )

    out = torch.empty((1,), dtype=torch.bfloat16, device=pred_bf16.device)

    hinge_aggregate_kernel[lambda: ((1, 1, 1), (BLOCK_SIZE, 1, 1))](
        partial_sum, out, num_tiles, M, K
    )

    return out.squeeze()


class ModelNew(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks using AveLang DSL.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return avelang_hinge_loss(predictions, targets)
