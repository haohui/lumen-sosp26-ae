import torch
import torch.nn as nn
import avelang
import avelang.language as al
import math

# InstanceNorm tile configuration
IN_BLOCK_SIZE: al.constexpr = 256
IN_TILE_SIZE: al.constexpr = 4096
IN_MAX_TILES: al.constexpr = 256


@avelang.jit
def instancenorm_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    N: al.i32,
    num_tiles: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    batch_idx = bid // num_tiles
    tile_idx = bid - batch_idx * num_tiles

    if batch_idx < batch_size:
        smem_sum = al.make_shared((IN_BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((IN_BLOCK_SIZE,), al.f32)

        base = batch_idx * N
        tile_start = tile_idx * IN_TILE_SIZE
        tile_end = tile_start + IN_TILE_SIZE
        if tile_end > N:
            tile_end = N

        layout_in = al.make_layout((batch_size * N,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        local_sum = al.convert(0.0, al.f32)
        local_sq = al.convert(0.0, al.f32)

        for i in al.range(tile_start + tid, tile_end, IN_BLOCK_SIZE):
            idx = base + i
            val = al.convert(x[idx], al.f32)
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
            layout_ps = al.make_layout((batch_size, num_tiles), (num_tiles, 1))
            ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
            psq = al.make_tensor(partial_sq_ptr, al.f32, layout_ps)
            ps[batch_idx, tile_idx] = smem_sum[0]
            psq[batch_idx, tile_idx] = smem_sq[0]


@avelang.jit
def instancenorm_aggregate_kernel(
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    mean_out_ptr: al.Pointer(al.f32),
    rstd_out_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    num_tiles: al.i32,
    N: al.i32,
    eps: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < batch_size:
        smem_sum = al.make_shared((IN_BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((IN_BLOCK_SIZE,), al.f32)

        layout_ps = al.make_layout((batch_size, num_tiles), (num_tiles, 1))
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

            chunk_start = chunk_start + IN_MAX_TILES
            al.syncthreads()

        if tid == 0:
            N_f32 = al.convert(N, al.f32)
            mean = total_sum / N_f32
            var = total_sq / N_f32 - mean * mean
            rstd = al.convert(1.0, al.f32) / al.sqrt(var + eps)

            layout_out = al.make_layout((batch_size,), (1,))
            mo = al.make_tensor(mean_out_ptr, al.f32, layout_out)
            ro = al.make_tensor(rstd_out_ptr, al.f32, layout_out)
            mo[bid] = mean
            ro[bid] = rstd


@avelang.jit
def instancenorm_apply_add_mul_kernel(
    x_ptr: al.Pointer(al.bf16),
    y_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    rstd_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    N: al.i32,
    num_tiles: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    batch_idx = bid // num_tiles
    tile_idx = bid - batch_idx * num_tiles

    if batch_idx < batch_size:
        base = batch_idx * N
        tile_start = tile_idx * IN_TILE_SIZE
        tile_end = tile_start + IN_TILE_SIZE
        if tile_end > N:
            tile_end = N

        layout_in = al.make_layout((batch_size * N,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)
        y = al.make_tensor(y_ptr, al.bf16, layout_in)
        ot = al.make_tensor(out_ptr, al.bf16, layout_in)

        layout_mean = al.make_layout((batch_size,), (1,))
        mt = al.make_tensor(mean_ptr, al.f32, layout_mean)
        rt = al.make_tensor(rstd_ptr, al.f32, layout_mean)

        mean = mt[batch_idx]
        rstd = rt[batch_idx]

        for i in al.range(tile_start + tid, tile_end, IN_BLOCK_SIZE):
            idx = base + i
            x_val = al.convert(x[idx], al.f32)
            y_val = al.convert(y[idx], al.f32)

            normalized = (x_val - mean) * rstd
            added = normalized + y_val
            result = added * y_val
            ot[idx] = al.convert(result, al.bf16)


def _prepare_bf16_cuda(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_instancenorm_add_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda(x)
    y_bf16 = _prepare_bf16_cuda(y)

    B, N = x_bf16.shape

    # Phase 1: tile-level reduction
    num_tiles = (N + IN_TILE_SIZE - 1) // IN_TILE_SIZE

    partial_sum = torch.empty((B, num_tiles), dtype=torch.float32, device=x_bf16.device)
    partial_sq = torch.empty((B, num_tiles), dtype=torch.float32, device=x_bf16.device)

    instancenorm_reduce_kernel[lambda: ((B * num_tiles, 1, 1), (IN_BLOCK_SIZE, 1, 1))](
        x_bf16, partial_sum, partial_sq, B, N, num_tiles
    )

    # Phase 2: aggregate partial sums
    mean_out = torch.empty((B,), dtype=torch.float32, device=x_bf16.device)
    rstd_out = torch.empty((B,), dtype=torch.float32, device=x_bf16.device)

    eps_f32 = float(eps)
    instancenorm_aggregate_kernel[lambda: ((B, 1, 1), (IN_BLOCK_SIZE, 1, 1))](
        partial_sum, partial_sq, mean_out, rstd_out, B, num_tiles, N, eps_f32
    )

    # Phase 3: apply norm + add y + multiply by y
    out = torch.empty_like(x_bf16)

    instancenorm_apply_add_mul_kernel[lambda: ((B * num_tiles, 1, 1), (IN_BLOCK_SIZE, 1, 1))](
        x_bf16, y_bf16, out, mean_out, rstd_out, B, N, num_tiles
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.eps = eps

    def forward(self, x, y):
        # Stage 1: Linear (uses PyTorch nn.Linear, matching reference)
        x = self.bmm(x)
        # Stages 2-4: InstanceNorm + add + multiply via AveLang kernels
        x = avelang_instancenorm_add_mul(x, y, self.eps)
        return x


def get_inputs():
    return [torch.rand(1024, 8192), torch.rand(1024, 8192)]


def get_init_inputs():
    return [8192, 8192]
