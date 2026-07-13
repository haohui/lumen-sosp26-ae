import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── Compile-time tile constants ──────────────────────────────────────────────
NORM_THREADS = 256

# ═══════════════════════════════════════════════════════════════════════════════
# InstanceNorm2d reduction kernel — compute mean and variance per (N, C)
# ═══════════════════════════════════════════════════════════════════════════════

@avelang.jit
def instancenorm_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    sum_out_ptr: al.Pointer(al.f32),
    sq_out_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    channels: al.i32,
    spatial_size: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    c_idx = bid % channels
    n_idx = bid // channels

    if n_idx < batch_size:
        smem_sum = al.make_shared((NORM_THREADS,), al.f32)
        smem_sq = al.make_shared((NORM_THREADS,), al.f32)

        x_layout = al.make_layout((batch_size, channels, spatial_size), (channels * spatial_size, spatial_size, 1))
        x = al.make_tensor(x_ptr, al.bf16, x_layout)

        local_sum = al.convert(0.0, al.f32)
        local_sq = al.convert(0.0, al.f32)

        for i in al.range(tid, spatial_size, NORM_THREADS):
            val = al.convert(x[n_idx, c_idx, i], al.f32)
            local_sum = local_sum + val
            local_sq = local_sq + val * val

        smem_sum[tid] = local_sum
        smem_sq[tid] = local_sq
        al.syncthreads()

        # Tree reduction: 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
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
            smem_sum[0] = smem_sum[0] + smem_sum[1]
            smem_sq[0] = smem_sq[0] + smem_sq[1]

        if tid == 0:
            s_layout = al.make_layout((batch_size, channels), (channels, 1))
            s_out = al.make_tensor(sum_out_ptr, al.f32, s_layout)
            sq_out = al.make_tensor(sq_out_ptr, al.f32, s_layout)
            N_spatial = al.convert(spatial_size, al.f32)
            s_out[n_idx, c_idx] = smem_sum[0] / N_spatial
            sq_out[n_idx, c_idx] = smem_sq[0] / N_spatial


# ═══════════════════════════════════════════════════════════════════════════════
# InstanceNorm2d apply kernel — normalize, scale, bias, and divide
# ═══════════════════════════════════════════════════════════════════════════════

@avelang.jit
def instancenorm_apply_divide_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    sqmean_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    channels: al.i32,
    spatial_size: al.i32,
    eps: al.f32,
    div_by: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    c_idx = bid % channels
    n_idx = bid // channels

    if n_idx < batch_size:
        x_layout = al.make_layout((batch_size, channels, spatial_size), (channels * spatial_size, spatial_size, 1))
        x = al.make_tensor(x_ptr, al.bf16, x_layout)

        out_layout = al.make_layout((batch_size, channels, spatial_size), (channels * spatial_size, spatial_size, 1))
        out = al.make_tensor(out_ptr, al.bf16, out_layout)

        m_layout = al.make_layout((batch_size, channels), (channels, 1))
        mean_t = al.make_tensor(mean_ptr, al.f32, m_layout)
        sqm_t = al.make_tensor(sqmean_ptr, al.f32, m_layout)

        mean_val = mean_t[n_idx, c_idx]
        sqmean_val = sqm_t[n_idx, c_idx]
        var_val = sqmean_val - mean_val * mean_val
        zero = al.convert(0.0, al.f32)
        if var_val < zero:
            var_val = zero
        rstd_val = al.convert(1.0, al.f32) / al.sqrt(var_val + eps)

        for i in al.range(tid, spatial_size, NORM_THREADS):
            x_val = al.convert(x[n_idx, c_idx, i], al.f32)
            normed = (x_val - mean_val) * rstd_val
            result = normed / div_by
            out[n_idx, c_idx, i] = al.convert(result, al.bf16)


# ═══════════════════════════════════════════════════════════════════════════════
# Host wrappers
# ═══════════════════════════════════════════════════════════════════════════════

def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is contiguous on CUDA in BF16."""
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_instancenorm(
    x: torch.Tensor,
    divide_by: float,
) -> torch.Tensor:
    """Run InstanceNorm2d + divide_by using AveLang kernels."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device required.")

    x_bf16 = _to_bf16_contiguous(x)
    batch_size, channels, H, W = x_bf16.shape
    spatial_size = H * W
    eps = 1e-5
    div_by = float(divide_by)

    total_blocks = batch_size * channels

    # Flatten spatial dims — views are created inside kernels from the 3D layout
    x_flat = x_bf16.reshape(batch_size, channels, spatial_size)
    mean_out = torch.empty((batch_size, channels), dtype=torch.float32, device=x_bf16.device)
    sqmean_out = torch.empty((batch_size, channels), dtype=torch.float32, device=x_bf16.device)

    # Phase 1: reduce — compute mean and mean of squares per (N, C)
    instancenorm_reduce_kernel[lambda: ((total_blocks, 1, 1), (NORM_THREADS, 1, 1))](
        x_flat, mean_out, sqmean_out,
        batch_size, channels, spatial_size,
    )

    # Phase 2: apply normalization + scale + bias + divide
    out_flat = torch.empty((batch_size, channels, spatial_size), dtype=torch.bfloat16, device=x_bf16.device)
    instancenorm_apply_divide_kernel[lambda: ((total_blocks, 1, 1), (NORM_THREADS, 1, 1))](
        x_flat, out_flat, mean_out, sqmean_out,
        batch_size, channels, spatial_size, eps, div_by,
    )

    return out_flat.reshape(batch_size, channels, H, W)


# ═══════════════════════════════════════════════════════════════════════════════
# ModelNew entrypoint — matches Model interface
# ═══════════════════════════════════════════════════════════════════════════════

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by

    def forward(self, x):
        # Conv2d via PyTorch (exact match to reference)
        x = self.conv(x)
        # InstanceNorm2d + divide via AveLang kernel
        x = avelang_instancenorm(x, self.divide_by)
        return x
