import torch
import torch.nn as nn
import torch.nn.functional as F
import avelang
import avelang.language as al


# ---------------------------------------------------------------------------
# Kernel 1: Conv3D
# ---------------------------------------------------------------------------

@avelang.jit
def conv3d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    total_x: al.i32,
    total_w: al.i32,
    total_o: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    KD: al.i32,
    KH: al.i32,
    KW: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    spatial_total: al.i32,
    batch: al.i32,
):
    b_idx = al.block_id(0)
    spatial_block = al.block_id(1)
    oc = al.thread_id(0)
    tid = al.thread_id(1)

    spatial_idx = spatial_block * al.block_dim(1) + tid

    if (b_idx >= batch) or (oc >= C_out) or (spatial_idx >= spatial_total):
        return

    # Strides
    stride_x_c = D * H * W
    stride_x_d = H * W
    stride_x_h = W
    stride_w_ic = KD * KH * KW
    stride_w_kd = KH * KW
    stride_w_kh = KW
    stride_o_c = D_out * H_out * W_out
    stride_o_d = H_out * W_out

    # Flat 1D tensor views
    x_flat_layout = al.make_layout((total_x,), (1,))
    x_flat = al.make_tensor(x_ptr, al.bf16, x_flat_layout)
    w_flat_layout = al.make_layout((total_w,), (1,))
    w_flat = al.make_tensor(w_ptr, al.bf16, w_flat_layout)
    b_flat_layout = al.make_layout((C_out,), (1,))
    b_flat = al.make_tensor(b_ptr, al.f32, b_flat_layout)
    o_flat_layout = al.make_layout((total_o,), (1,))
    o_flat = al.make_tensor(out_ptr, al.bf16, o_flat_layout)

    # Decode spatial index
    d = spatial_idx // stride_o_d
    rem = spatial_idx % stride_o_d
    h = rem // W_out
    w = rem % W_out

    # Base offsets
    x_base = b_idx * C_in * stride_x_c
    w_base = oc * C_in * stride_w_ic
    o_idx = b_idx * C_out * stride_o_c + oc * stride_o_c + d * stride_o_d + h * W_out + w

    # Accumulate in FP32
    acc = b_flat[oc]

    for ic in al.range(C_in):
        x_ic_off = x_base + ic * stride_x_c
        w_ic_off = w_base + ic * stride_w_ic
        for kd in al.range(KD):
            x_kd_off = x_ic_off + (d + kd) * stride_x_d
            w_kd_off = w_ic_off + kd * stride_w_kd
            for kh in al.range(KH):
                x_kh_off = x_kd_off + (h + kh) * stride_x_h
                w_kh_off = w_kd_off + kh * stride_w_kh
                for kw in al.range(KW):
                    x_idx = x_kh_off + (w + kw)
                    w_idx = w_kh_off + kw
                    x_val = al.convert(x_flat[x_idx], al.f32)
                    w_val = al.convert(w_flat[w_idx], al.f32)
                    acc = acc + x_val * w_val

    o_flat[o_idx] = al.convert(acc, al.bf16)


# ---------------------------------------------------------------------------
# Kernel 2: GroupNorm + spatial mean pooling (fused)
# ---------------------------------------------------------------------------

@avelang.jit
def groupnorm_mean_kernel(
    x_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    total_x: al.i32,
    total_o: al.i32,
    C_out: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    num_groups: al.i32,
    channels_per_group: al.i32,
    spatial_size: al.i32,
    batch: al.i32,
):
    b_idx = al.block_id(0)
    group = al.block_id(1)
    tid = al.thread_id(0)

    if (b_idx >= batch) or (group >= num_groups):
        return

    chan_start = group * channels_per_group
    group_total = channels_per_group * spatial_size

    stride_chn = D * H * W
    stride_d = H * W
    stride_h = W

    x_flat_layout = al.make_layout((total_x,), (1,))
    x_flat = al.make_tensor(x_ptr, al.bf16, x_flat_layout)
    gamma_flat_layout = al.make_layout((C_out,), (1,))
    gamma_flat = al.make_tensor(gamma_ptr, al.bf16, gamma_flat_layout)
    beta_flat_layout = al.make_layout((C_out,), (1,))
    beta_flat = al.make_tensor(beta_ptr, al.bf16, beta_flat_layout)
    o_flat_layout = al.make_layout((total_o,), (1,))
    o_flat = al.make_tensor(out_ptr, al.bf16, o_flat_layout)

    x_base = b_idx * C_out * stride_chn + chan_start * stride_chn

    # Shared memory
    smem_sum = al.make_shared((256,), al.f32)
    smem_sum_sq = al.make_shared((256,), al.f32)
    smem_chn = al.make_shared((256,), al.f32)

    # Per-thread accumulators
    local_sum = al.convert(0.0, al.f32)
    local_sum_sq = al.convert(0.0, al.f32)
    chn0 = al.convert(0.0, al.f32)
    chn1 = al.convert(0.0, al.f32)
    chn2 = al.convert(0.0, al.f32)
    chn3 = al.convert(0.0, al.f32)

    for idx in al.range(tid, group_total, 256):
        chn_local = idx // spatial_size
        spat = idx % spatial_size
        sd = spat // stride_d
        rem_s = spat % stride_d
        sh = rem_s // stride_h
        sw = rem_s % stride_h

        x_off = x_base + chn_local * stride_chn + sd * stride_d + sh * stride_h + sw
        val = al.convert(x_flat[x_off], al.f32)

        local_sum = local_sum + val
        local_sum_sq = local_sum_sq + val * val

        if chn_local == 0:
            chn0 = chn0 + val
        else:
            if chn_local == 1:
                chn1 = chn1 + val
            else:
                if chn_local == 2:
                    chn2 = chn2 + val
                else:
                    chn3 = chn3 + val

    # Reduce group sum
    smem_sum[tid] = local_sum
    al.syncthreads()
    if tid < 128:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 1]
    al.syncthreads()
    group_sum = smem_sum[0]

    # Reduce group sum-of-squares
    smem_sum_sq[tid] = local_sum_sq
    al.syncthreads()
    if tid < 128:
        smem_sum_sq[tid] = smem_sum_sq[tid] + smem_sum_sq[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem_sum_sq[tid] = smem_sum_sq[tid] + smem_sum_sq[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem_sum_sq[tid] = smem_sum_sq[tid] + smem_sum_sq[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem_sum_sq[tid] = smem_sum_sq[tid] + smem_sum_sq[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem_sum_sq[tid] = smem_sum_sq[tid] + smem_sum_sq[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem_sum_sq[tid] = smem_sum_sq[tid] + smem_sum_sq[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem_sum_sq[tid] = smem_sum_sq[tid] + smem_sum_sq[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem_sum_sq[tid] = smem_sum_sq[tid] + smem_sum_sq[tid + 1]
    al.syncthreads()
    group_sum_sq = smem_sum_sq[0]

    # Group statistics
    n_elements = al.convert(group_total, al.f32)
    mu = group_sum / n_elements
    sigma_sq = group_sum_sq / n_elements - mu * mu
    one = al.convert(1.0, al.f32)
    eps = al.convert(1e-5, al.f32)
    inv_std = one / al.sqrt(sigma_sq + eps)

    # Reduce per-channel sums: channel 0
    smem_chn[tid] = chn0
    al.syncthreads()
    if tid < 128:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 1]
    al.syncthreads()
    chn0_sum = smem_chn[0]

    # Channel 1
    smem_chn[tid] = chn1
    al.syncthreads()
    if tid < 128:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 1]
    al.syncthreads()
    chn1_sum = smem_chn[0]

    # Channel 2
    smem_chn[tid] = chn2
    al.syncthreads()
    if tid < 128:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 1]
    al.syncthreads()
    chn2_sum = smem_chn[0]

    # Channel 3
    smem_chn[tid] = chn3
    al.syncthreads()
    if tid < 128:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem_chn[tid] = smem_chn[tid] + smem_chn[tid + 1]
    al.syncthreads()
    chn3_sum = smem_chn[0]

    # Compute output values
    spat_f = al.convert(spatial_size, al.f32)

    if tid == 0:
        chn_mean0 = chn0_sum / spat_f
        g0 = al.convert(gamma_flat[chan_start], al.f32)
        b0 = al.convert(beta_flat[chan_start], al.f32)
        o_flat[b_idx * C_out + chan_start] = al.convert(g0 * (chn_mean0 - mu) * inv_std + b0, al.bf16)

        chn_mean1 = chn1_sum / spat_f
        g1 = al.convert(gamma_flat[chan_start + 1], al.f32)
        b1 = al.convert(beta_flat[chan_start + 1], al.f32)
        o_flat[b_idx * C_out + chan_start + 1] = al.convert(g1 * (chn_mean1 - mu) * inv_std + b1, al.bf16)

        chn_mean2 = chn2_sum / spat_f
        g2 = al.convert(gamma_flat[chan_start + 2], al.f32)
        b2 = al.convert(beta_flat[chan_start + 2], al.f32)
        o_flat[b_idx * C_out + chan_start + 2] = al.convert(g2 * (chn_mean2 - mu) * inv_std + b2, al.bf16)

        chn_mean3 = chn3_sum / spat_f
        g3 = al.convert(gamma_flat[chan_start + 3], al.f32)
        b3 = al.convert(beta_flat[chan_start + 3], al.f32)
        o_flat[b_idx * C_out + chan_start + 3] = al.convert(g3 * (chn_mean3 - mu) * inv_std + b3, al.bf16)


# ---------------------------------------------------------------------------
# ModelNew: host wrapper
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups

    def forward(self, x):
        B, C_in, D, H, W = x.shape
        C_out = self.out_channels

        if isinstance(self.kernel_size, int):
            KD = KH = KW = self.kernel_size
        else:
            KD, KH, KW = self.kernel_size

        D_out = D - KD + 1
        H_out = H - KH + 1
        W_out = W - KW + 1
        spatial_size = D_out * H_out * W_out
        channels_per_group = C_out // self.num_groups

        # Pre-compute total element counts
        total_x = B * C_in * D * H * W
        total_w = C_out * C_in * KD * KH * KW
        total_o_conv = B * C_out * D_out * H_out * W_out
        total_o_gn = B * C_out

        # Extract weights and convert to BF16
        w_bf16 = self.conv.weight.data.to(torch.bfloat16).contiguous()
        b_f32 = self.conv.bias.data.to(torch.float32).contiguous()
        gamma_bf16 = self.group_norm.weight.data.to(torch.bfloat16).contiguous()
        beta_bf16 = self.group_norm.bias.data.to(torch.bfloat16).contiguous()
        x_bf16 = x.to(torch.bfloat16).contiguous()

        # Intermediate buffer for conv output
        intermediate = torch.empty(
            B, C_out, D_out, H_out, W_out,
            dtype=torch.bfloat16, device=x.device,
        )

        # Output buffer
        out_bf16 = torch.empty(B, C_out, dtype=torch.bfloat16, device=x.device)

        # --- Launch conv3d kernel ---
        SPATIAL_TILE = 16
        grid_conv = (
            B,
            (spatial_size + SPATIAL_TILE - 1) // SPATIAL_TILE,
            1,
        )
        block_conv = (C_out, SPATIAL_TILE, 1)

        conv3d_kernel[lambda: (grid_conv, block_conv)](
            x_bf16, w_bf16, b_f32, intermediate,
            total_x, total_w, total_o_conv,
            C_in, C_out, D, H, W,
            KD, KH, KW,
            D_out, H_out, W_out,
            spatial_size, B,
        )

        # --- HardSwish via PyTorch ---
        intermediate = F.hardswish(intermediate)

        # --- Launch groupnorm + mean kernel ---
        BLOCK_SIZE = 256
        grid_gn = (B, self.num_groups, 1)
        block_gn = (BLOCK_SIZE, 1, 1)

        groupnorm_mean_kernel[lambda: (grid_gn, block_gn)](
            intermediate, gamma_bf16, beta_bf16, out_bf16,
            total_o_conv, total_o_gn,
            C_out, D_out, H_out, W_out,
            self.num_groups, channels_per_group, spatial_size,
            B,
        )

        return out_bf16
