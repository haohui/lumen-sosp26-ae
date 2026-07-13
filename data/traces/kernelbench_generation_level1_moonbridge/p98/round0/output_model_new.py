import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
TILE_SIZE: al.constexpr = 8192
EPS: al.constexpr = 1e-12


@avelang.jit
def kl_div_reduce_kernel(
    pred_ptr: al.Pointer(al.bf16),
    target_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    total_elems: al.i32,
    num_tiles: al.i32,
):
    """
    Phase 1: reduce each tile into a partial KL-divergence sum.
    Launch: grid = (num_tiles, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < num_tiles:
        smem = al.make_shared((BLOCK_SIZE,), al.f32)

        tile_start = bid * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > total_elems:
            tile_end = total_elems

        layout = al.make_layout((total_elems,), (1,))
        pred = al.make_tensor(pred_ptr, al.bf16, layout)
        target = al.make_tensor(target_ptr, al.bf16, layout)

        eps_f32 = al.convert(EPS, al.f32)
        local_sum = al.convert(0.0, al.f32)

        for i in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            p_val = al.convert(pred[i], al.f32)
            t_val = al.convert(target[i], al.f32)
            # Add epsilon to avoid log(0); negligible for non-zero values
            log_p = al.log(p_val + eps_f32)
            log_t = al.log(t_val + eps_f32)
            diff = log_t - log_p
            local_sum = local_sum + t_val * diff

        smem[tid] = local_sum
        al.syncthreads()

        # Shared-memory tree reduction
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
def kl_div_aggregate_kernel(
    partial_sum_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.bf16),
    num_tiles: al.i32,
    batch_size: al.i32,
):
    """
    Phase 2: aggregate tile-level partial sums, divide by batch_size.
    Launch: grid = (1, 1, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)

    smem = al.make_shared((BLOCK_SIZE,), al.f32)

    layout_ps = al.make_layout((num_tiles,), (1,))
    ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)

    local_sum = al.convert(0.0, al.f32)

    for i in al.range(tid, num_tiles, BLOCK_SIZE):
        local_sum = local_sum + ps[i]

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
        total = smem[0]
        bs_f32 = al.convert(batch_size, al.f32)
        result = total / bs_f32
        layout_out = al.make_layout((1,), (1,))
        out = al.make_tensor(output_ptr, al.bf16, layout_out)
        out[0] = al.convert(result, al.bf16)


def avelang_kl_div(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence with batchmean reduction using AveLang kernels."""
    assert predictions.is_cuda, "Tensors must be on CUDA/HIP device."
    assert targets.is_cuda, "Tensors must be on CUDA/HIP device."
    assert predictions.shape == targets.shape, "Input tensors must have the same shape."

    batch_size = predictions.shape[0]
    N = predictions.shape[1]
    total_elems = batch_size * N

    pred_contig = predictions.contiguous()
    target_contig = targets.contiguous()

    num_tiles = (total_elems + TILE_SIZE - 1) // TILE_SIZE

    # Phase 1: tile-level reduction
    partial_sum = torch.empty(
        (num_tiles,), dtype=torch.float32, device=predictions.device
    )

    kl_div_reduce_kernel[lambda: ((num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        pred_contig, target_contig, partial_sum, total_elems, num_tiles
    )

    # Phase 2: aggregate partials and divide by batch_size
    output = torch.empty((1,), dtype=torch.bfloat16, device=predictions.device)

    kl_div_aggregate_kernel[lambda: ((1, 1, 1), (BLOCK_SIZE, 1, 1))](
        partial_sum, output, num_tiles, batch_size
    )

    return output.reshape(())


class ModelNew(nn.Module):
    """
    Optimized model that computes Kullback-Leibler Divergence using AveLang DSL.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return avelang_kl_div(predictions, targets)
