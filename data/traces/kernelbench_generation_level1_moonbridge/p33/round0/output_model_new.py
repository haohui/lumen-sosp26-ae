import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
TILE_SIZE: al.constexpr = 4096
MAX_TILES_PER_ROUND: al.constexpr = 256


@avelang.jit
def batchnorm_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    num_tiles: al.i32,
    spatial: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    channel_idx = bid // num_tiles
    tile_idx = bid - channel_idx * num_tiles

    if channel_idx < C:
        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        total = batch_size * C * H * W
        layout_in = al.make_layout((total,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        local_sum = al.convert(0.0, al.f32)
        local_sq = al.convert(0.0, al.f32)

        C_HW = C * H * W
        HW = H * W

        tile_start = tile_idx * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > spatial:
            tile_end = spatial

        for s in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            b = s // HW
            hw = s - b * HW
            flat_idx = b * C_HW + channel_idx * HW + hw
            val = al.convert(x[flat_idx], al.f32)
            local_sum = local_sum + val
            local_sq = local_sq + val * val

        smem_sum[tid] = local_sum
        smem_sq[tid] = local_sq
        al.syncthreads()

        if tid < 128:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 128]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 64]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 32]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 16]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 8]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 4]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 2]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 1]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 1]

        if tid == 0:
            layout_ps = al.make_layout((C, num_tiles), (num_tiles, 1))
            ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
            psq = al.make_tensor(partial_sq_ptr, al.f32, layout_ps)
            ps[channel_idx, tile_idx] = smem_sum[0]
            psq[channel_idx, tile_idx] = smem_sq[0]


@avelang.jit
def batchnorm_aggregate_kernel(
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    mean_out_ptr: al.Pointer(al.f32),
    rstd_out_ptr: al.Pointer(al.f32),
    C: al.i32,
    num_tiles: al.i32,
    spatial: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < C:
        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        layout_ps = al.make_layout((C, num_tiles), (num_tiles, 1))
        ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
        psq = al.make_tensor(partial_sq_ptr, al.f32, layout_ps)

        total_sum = al.convert(0.0, al.f32)
        total_sq = al.convert(0.0, al.f32)

        chunk_start = al.convert(0, al.i32)
        for _ in al.range(0, 16):
            if chunk_start >= num_tiles:
                break

            local_sum = al.convert(0.0, al.f32)
            local_sq = al.convert(0.0, al.f32)

            tile_idx = chunk_start + tid
            if tile_idx < num_tiles:
                local_sum = ps[bid, tile_idx]
                local_sq = psq[bid, tile_idx]

            smem_sum[tid] = local_sum
            smem_sq[tid] = local_sq
            al.syncthreads()

            if tid < 128:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 128]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 128]
            al.syncthreads()
            if tid < 64:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 64]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 64]
            al.syncthreads()
            if tid < 32:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 32]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 32]
            al.syncthreads()
            if tid < 16:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 16]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 16]
            al.syncthreads()
            if tid < 8:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 8]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 8]
            al.syncthreads()
            if tid < 4:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 4]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 4]
            al.syncthreads()
            if tid < 2:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 2]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 2]
            al.syncthreads()
            if tid < 1:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 1]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 1]

            if tid == 0:
                total_sum = total_sum + smem_sum[0]
                total_sq = total_sq + smem_sq[0]

            chunk_start = chunk_start + MAX_TILES_PER_ROUND
            al.syncthreads()

        if tid == 0:
            spatial_f32 = al.convert(spatial, al.f32)
            mean = total_sum / spatial_f32
            var = total_sq / spatial_f32 - mean * mean
            eps_val = al.convert(1e-5, al.f32)
            rstd = al.convert(1.0, al.f32) / al.sqrt(var + eps_val)

            layout_out = al.make_layout((C,), (1,))
            mo = al.make_tensor(mean_out_ptr, al.f32, layout_out)
            ro = al.make_tensor(rstd_out_ptr, al.f32, layout_out)
            mo[bid] = mean
            ro[bid] = rstd


@avelang.jit
def batchnorm_apply_kernel(
    x_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    rstd_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    num_tiles: al.i32,
    spatial: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    channel_idx = bid // num_tiles
    tile_idx = bid - channel_idx * num_tiles

    if channel_idx < C:
        total = batch_size * C * H * W
        C_HW = C * H * W
        HW = H * W

        layout_in = al.make_layout((total,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        layout_out = al.make_layout((total,), (1,))
        ot = al.make_tensor(out_ptr, al.bf16, layout_out)

        layout_wb = al.make_layout((C,), (1,))
        wt = al.make_tensor(weight_ptr, al.bf16, layout_wb)
        bt = al.make_tensor(bias_ptr, al.bf16, layout_wb)

        layout_mean = al.make_layout((C,), (1,))
        mt = al.make_tensor(mean_ptr, al.f32, layout_mean)
        rt = al.make_tensor(rstd_ptr, al.f32, layout_mean)

        mean = mt[channel_idx]
        rstd = rt[channel_idx]
        w_val = al.convert(wt[channel_idx], al.f32)
        b_val = al.convert(bt[channel_idx], al.f32)

        tile_start = tile_idx * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > spatial:
            tile_end = spatial

        for s in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            b = s // HW
            hw = s - b * HW
            flat_idx = b * C_HW + channel_idx * HW + hw

            x_val = al.convert(x[flat_idx], al.f32)
            normalized = (x_val - mean) * rstd
            result = normalized * w_val + b_val
            ot[flat_idx] = al.convert(result, al.bf16)


def avelang_batchnorm2d(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dim() == 4, "Input must be 4D (B, C, H, W)."

    batch_size = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    spatial = batch_size * H * W

    x_contig = x.contiguous()
    w_bf16 = weight.contiguous().to(torch.bfloat16)
    b_bf16 = bias.contiguous().to(torch.bfloat16)

    eps = 1e-5
    num_tiles = (spatial + TILE_SIZE - 1) // TILE_SIZE

    partial_sum = torch.empty((C, num_tiles), dtype=torch.float32, device=x.device)
    partial_sq = torch.empty((C, num_tiles), dtype=torch.float32, device=x.device)

    batchnorm_reduce_kernel[lambda: ((C * num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contig, partial_sum, partial_sq, batch_size, C, H, W, num_tiles, spatial
    )

    mean_out = torch.empty((C,), dtype=torch.float32, device=x.device)
    rstd_out = torch.empty((C,), dtype=torch.float32, device=x.device)

    batchnorm_aggregate_kernel[lambda: ((C, 1, 1), (BLOCK_SIZE, 1, 1))](
        partial_sum, partial_sq, mean_out, rstd_out, C, num_tiles, spatial
    )

    out = torch.empty_like(x_contig)

    batchnorm_apply_kernel[lambda: ((C * num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contig, w_bf16, b_bf16, out, mean_out, rstd_out,
        batch_size, C, H, W, num_tiles, spatial
    )

    return out


def _avelang_batchnorm2d_eval(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
    running_mean: torch.Tensor, running_var: torch.Tensor,
) -> torch.Tensor:
    """Eval-mode: use running mean/var instead of batch statistics."""
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dim() == 4, "Input must be 4D (B, C, H, W)."

    batch_size = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    spatial = batch_size * H * W

    x_contig = x.contiguous()
    w_bf16 = weight.contiguous().to(torch.bfloat16)
    b_bf16 = bias.contiguous().to(torch.bfloat16)

    eps = 1e-5
    num_tiles = (spatial + TILE_SIZE - 1) // TILE_SIZE

    # Compute mean and rstd from running stats on host
    mean_cpu = running_mean.to(torch.float32)
    var_cpu = running_var.to(torch.float32)
    rstd_cpu = 1.0 / torch.sqrt(var_cpu + eps)

    mean_out = mean_cpu.to(device=x.device, dtype=torch.float32)
    rstd_out = rstd_cpu.to(device=x.device, dtype=torch.float32)

    out = torch.empty_like(x_contig)

    batchnorm_apply_kernel[lambda: ((C * num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contig, w_bf16, b_bf16, out, mean_out, rstd_out,
        batch_size, C, H, W, num_tiles, spatial
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Batch Normalization using AveLang DSL.
    """
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.bn.weight.data
        bias = self.bn.bias.data
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var

        x_bf16 = x.to(dtype=torch.bfloat16, device=x.device).contiguous()

        if self.training:
            result = avelang_batchnorm2d(x_bf16, weight, bias)
        else:
            result = _avelang_batchnorm2d_eval(
                x_bf16, weight, bias, running_mean, running_var
            )
        return result.to(x.dtype)
