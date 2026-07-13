import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
TILE_SIZE: al.constexpr = 4096
MAX_TILES_PER_ROUND: al.constexpr = 256


@avelang.jit
def groupnorm_reduce_kernel(
    x_ptr: al.Pointer(al.f32),
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    N: al.i32,
    C_out: al.i32,
    groups: al.i32,
    channels_per_group: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    num_tiles: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    spatial = D_out * H_out * W_out
    elements_per_group = channels_per_group * spatial
    total_ng = N * groups

    ng_idx = bid // num_tiles
    tile_idx = bid - ng_idx * num_tiles

    if ng_idx < total_ng:
        n = ng_idx // groups
        g = ng_idx - n * groups

        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        base = n * C_out * spatial + g * channels_per_group * spatial

        tile_start = tile_idx * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > elements_per_group:
            tile_end = elements_per_group

        total_flat = N * C_out * spatial
        layout_x = al.make_layout((total_flat,), (1,))
        x = al.make_tensor(x_ptr, al.f32, layout_x)

        local_sum = al.convert(0.0, al.f32)
        local_sq = al.convert(0.0, al.f32)

        for i in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            idx = base + i
            val = x[idx]
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
            layout_ps = al.make_layout((total_ng, num_tiles), (num_tiles, 1))
            ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
            psq = al.make_tensor(partial_sq_ptr, al.f32, layout_ps)
            ps[ng_idx, tile_idx] = smem_sum[0]
            psq[ng_idx, tile_idx] = smem_sq[0]


@avelang.jit
def groupnorm_aggregate_kernel(
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    mean_out_ptr: al.Pointer(al.f32),
    rstd_out_ptr: al.Pointer(al.f32),
    N: al.i32,
    groups: al.i32,
    channels_per_group: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    num_tiles: al.i32,
    eps: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    total_ng = N * groups

    if bid < total_ng:
        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        layout_ps = al.make_layout((total_ng, num_tiles), (num_tiles, 1))
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

            t_idx = chunk_start + tid
            if t_idx < num_tiles:
                local_sum = ps[bid, t_idx]
                local_sq = psq[bid, t_idx]

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
            spatial = D_out * H_out * W_out
            elements_per_group = channels_per_group * spatial
            count_f = al.convert(elements_per_group, al.f32)
            mean = total_sum / count_f
            var = total_sq / count_f - mean * mean
            rstd = al.convert(1.0, al.f32) / al.sqrt(var + eps)

            layout_out = al.make_layout((total_ng,), (1,))
            mo = al.make_tensor(mean_out_ptr, al.f32, layout_out)
            ro = al.make_tensor(rstd_out_ptr, al.f32, layout_out)
            mo[bid] = mean
            ro[bid] = rstd


@avelang.jit
def groupnorm_apply_kernel(
    x_ptr: al.Pointer(al.f32),
    gamma_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    rstd_ptr: al.Pointer(al.f32),
    N: al.i32,
    C_out: al.i32,
    groups: al.i32,
    channels_per_group: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    out_spatial = D_out * H_out * W_out
    total_out = N * C_out * out_spatial

    flat_idx = bid * BLOCK_SIZE + tid

    if flat_idx < total_out:
        out_hw = H_out * W_out

        n = flat_idx // (C_out * out_spatial)
        rem = flat_idx - n * (C_out * out_spatial)
        c = rem // out_spatial
        rem = rem - c * out_spatial
        d = rem // out_hw
        rem = rem - d * out_hw
        h = rem // W_out
        w = rem - h * W_out

        g = c // channels_per_group
        ng_idx = n * groups + g

        layout_x = al.make_layout((total_out,), (1,))
        x = al.make_tensor(x_ptr, al.f32, layout_x)

        layout_g = al.make_layout((C_out,), (1,))
        gamma = al.make_tensor(gamma_ptr, al.f32, layout_g)
        beta_t = al.make_tensor(beta_ptr, al.f32, layout_g)

        layout_out = al.make_layout((total_out,), (1,))
        out = al.make_tensor(out_ptr, al.bf16, layout_out)

        layout_mr = al.make_layout((N * groups,), (1,))
        mean_t = al.make_tensor(mean_ptr, al.f32, layout_mr)
        rstd_t = al.make_tensor(rstd_ptr, al.f32, layout_mr)

        mean_val = mean_t[ng_idx]
        rstd_val = rstd_t[ng_idx]

        x_idx = flat_idx
        x_val = x[x_idx]
        g_val = gamma[c]
        b_val = beta_t[c]

        normalized = (x_val - mean_val) * rstd_val
        result = normalized * g_val + b_val
        out[x_idx] = al.convert(result, al.bf16)


def avelang_groupnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    N, C_out, D_out, H_out, W_out = x.shape
    channels_per_group = C_out // groups
    spatial = D_out * H_out * W_out
    elements_per_group = channels_per_group * spatial
    num_tiles = (elements_per_group + TILE_SIZE - 1) // TILE_SIZE
    total_ng = N * groups
    total_out = N * C_out * spatial

    x_flat = x.float().contiguous().view(-1)
    w_f32 = weight.float().contiguous()
    b_f32 = bias.float().contiguous()

    # Phase 1: tile-level reduction
    partial_sum = torch.empty((total_ng, num_tiles), dtype=torch.float32, device=x.device)
    partial_sq = torch.empty((total_ng, num_tiles), dtype=torch.float32, device=x.device)
    groupnorm_reduce_kernel[lambda: ((total_ng * num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_flat, partial_sum, partial_sq,
        N, C_out, groups, channels_per_group, D_out, H_out, W_out, num_tiles,
    )

    # Phase 2: aggregate across tiles
    mean_out = torch.empty((total_ng,), dtype=torch.float32, device=x.device)
    rstd_out = torch.empty((total_ng,), dtype=torch.float32, device=x.device)
    eps = 1e-5
    groupnorm_aggregate_kernel[lambda: ((total_ng, 1, 1), (BLOCK_SIZE, 1, 1))](
        partial_sum, partial_sq, mean_out, rstd_out,
        N, groups, channels_per_group, D_out, H_out, W_out, num_tiles, eps,
    )

    # Phase 3: apply normalization -> output bf16
    final_out = torch.empty((total_out,), dtype=torch.bfloat16, device=x.device)
    num_blocks_apply = (total_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    groupnorm_apply_kernel[lambda: ((num_blocks_apply, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_flat, w_f32, b_f32, final_out,
        mean_out, rstd_out,
        N, C_out, groups, channels_per_group, D_out, H_out, W_out,
    )

    result = final_out.view(N, C_out, D_out, H_out, W_out)
    return result


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups

        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.relu(x)
        x = avelang_groupnorm(x, self.group_norm.weight.data, self.group_norm.bias.data, self.groups)
        return x


batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 32, 32, 32
kernel_size = 3
groups = 8
bias = False


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, bias]
