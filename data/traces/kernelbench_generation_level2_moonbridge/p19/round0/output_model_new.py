import torch
import torch.nn as nn
import avelang
import torch.nn.functional as F
import avelang.language as al

C_TOTAL = 64
NUM_GROUPS = 8
C_PER_GROUP = C_TOTAL // NUM_GROUPS
OUT_H = 258
OUT_W = 258
OUT_SPATIAL = OUT_H * OUT_W
GN_N = C_PER_GROUP * OUT_H * OUT_W
GN_BLOCK_SIZE = 256
GN_TILE_SIZE = 4096
GN_NUM_TILES = (GN_N + GN_TILE_SIZE - 1) // GN_TILE_SIZE
OUT_TOTAL = 128 * C_TOTAL * OUT_H * OUT_W


@avelang.jit
def groupnorm_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    batch_size: al.u32,
    num_tiles: al.u32,
):
    tid = al.thread_id(0)
    bid_x = al.block_id(0)
    bid_y = al.block_id(1)

    batch_idx = bid_y // NUM_GROUPS
    group_idx = bid_y - batch_idx * NUM_GROUPS
    tile_idx = bid_x

    if batch_idx < batch_size:
        smem_sum = al.make_shared((GN_BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((GN_BLOCK_SIZE,), al.f32)

        group_c_start = group_idx * C_PER_GROUP
        tile_start = tile_idx * GN_TILE_SIZE
        tile_end = tile_start + GN_TILE_SIZE
        if tile_end > GN_N:
            tile_end = GN_N

        x_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((OUT_TOTAL,), (1,)))

        local_sum = al.convert(0.0, al.f32)
        local_sq = al.convert(0.0, al.f32)

        batch_offset = batch_idx * C_TOTAL * OUT_SPATIAL
        group_offset = group_c_start * OUT_SPATIAL
        base_offset = batch_offset + group_offset

        for li in al.range(tile_start + tid, tile_end, GN_BLOCK_SIZE):
            c_g = li // OUT_SPATIAL
            rest = li - c_g * OUT_SPATIAL
            h = rest // OUT_W
            w = rest - h * OUT_W
            gidx = base_offset + c_g * OUT_SPATIAL + h * OUT_W + w
            val = al.convert(x_flat[gidx], al.f32)
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
            layout_ps = al.make_layout((batch_size * NUM_GROUPS, num_tiles), (num_tiles, 1))
            ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
            psq = al.make_tensor(partial_sq_ptr, al.f32, layout_ps)
            ps[bid_y, tile_idx] = smem_sum[0]
            psq[bid_y, tile_idx] = smem_sq[0]


@avelang.jit
def groupnorm_aggregate_kernel(
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    mean_out_ptr: al.Pointer(al.f32),
    rstd_out_ptr: al.Pointer(al.f32),
    batch_size: al.u32,
    num_tiles: al.u32,
    eps: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    total_pairs = batch_size * NUM_GROUPS
    if bid < total_pairs:
        smem_sum = al.make_shared((GN_BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((GN_BLOCK_SIZE,), al.f32)

        layout_ps = al.make_layout((total_pairs, num_tiles), (num_tiles, 1))
        ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
        psq = al.make_tensor(partial_sq_ptr, al.f32, layout_ps)

        local_sum = al.convert(0.0, al.f32)
        local_sq = al.convert(0.0, al.f32)

        t = tid
        iters = (num_tiles + GN_BLOCK_SIZE - 1) // GN_BLOCK_SIZE
        for _ in al.range(iters):
            if t < num_tiles:
                local_sum = local_sum + ps[bid, t]
                local_sq = local_sq + psq[bid, t]
            t = t + GN_BLOCK_SIZE

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
            N_f32 = al.convert(GN_N, al.f32)
            total_sum = smem_sum[0]
            total_sq = smem_sq[0]
            mean = total_sum / N_f32
            var = total_sq / N_f32 - mean * mean
            rstd = al.convert(1.0, al.f32) / al.sqrt(var + eps)

            layout_out = al.make_layout((total_pairs,), (1,))
            mo = al.make_tensor(mean_out_ptr, al.f32, layout_out)
            ro = al.make_tensor(rstd_out_ptr, al.f32, layout_out)
            mo[bid] = mean
            ro[bid] = rstd


@avelang.jit
def groupnorm_apply_kernel(
    x_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    rstd_ptr: al.Pointer(al.f32),
    batch_size: al.u32,
    num_tiles: al.u32,
):
    tid = al.thread_id(0)
    bid_x = al.block_id(0)
    bid_y = al.block_id(1)

    batch_idx = bid_y // NUM_GROUPS
    group_idx = bid_y - batch_idx * NUM_GROUPS
    tile_idx = bid_x

    if batch_idx < batch_size:
        group_c_start = group_idx * C_PER_GROUP
        tile_start = tile_idx * GN_TILE_SIZE
        tile_end = tile_start + GN_TILE_SIZE
        if tile_end > GN_N:
            tile_end = GN_N

        x_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((OUT_TOTAL,), (1,)))
        wt_flat = al.make_tensor(weight_ptr, al.bf16, al.make_layout((C_TOTAL,), (1,)))
        bt_flat = al.make_tensor(bias_ptr, al.bf16, al.make_layout((C_TOTAL,), (1,)))
        out_flat = al.make_tensor(out_ptr, al.bf16, al.make_layout((OUT_TOTAL,), (1,)))

        layout_stat = al.make_layout((batch_size * NUM_GROUPS,), (1,))
        mt = al.make_tensor(mean_ptr, al.f32, layout_stat)
        rt = al.make_tensor(rstd_ptr, al.f32, layout_stat)

        mean = mt[bid_y]
        rstd = rt[bid_y]

        batch_offset = batch_idx * C_TOTAL * OUT_SPATIAL
        group_offset = group_c_start * OUT_SPATIAL
        base_offset = batch_offset + group_offset

        for li in al.range(tile_start + tid, tile_end, GN_BLOCK_SIZE):
            c_g = li // OUT_SPATIAL
            rest = li - c_g * OUT_SPATIAL
            h = rest // OUT_W
            w = rest - h * OUT_W
            global_c = group_c_start + c_g
            gidx = base_offset + c_g * OUT_SPATIAL + h * OUT_W + w

            x_val = al.convert(x_flat[gidx], al.f32)
            w_val = al.convert(wt_flat[global_c], al.f32)
            b_val = al.convert(bt_flat[global_c], al.f32)

            normalized = (x_val - mean) * rstd
            result = normalized * w_val + b_val
            out_flat[gidx] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):

        x = self.conv_transpose(x)
        batch_size_val = x.shape[0]
        x = F.gelu(x)
        x_bf16 = x.contiguous().to(dtype=torch.bfloat16)
        gn_wt_bf16 = self.group_norm.weight.data.contiguous().to(dtype=torch.bfloat16)
        gn_bias_bf16 = self.group_norm.bias.data.contiguous().to(dtype=torch.bfloat16)

        total_pairs = batch_size_val * NUM_GROUPS
        partial_sum = torch.empty((total_pairs, GN_NUM_TILES), dtype=torch.float32, device=x.device)
        partial_sq = torch.empty((total_pairs, GN_NUM_TILES), dtype=torch.float32, device=x.device)

        grid_reduce = (GN_NUM_TILES, total_pairs, 1)
        groupnorm_reduce_kernel[lambda: (grid_reduce, (GN_BLOCK_SIZE, 1, 1))](
            x_bf16, partial_sum, partial_sq, batch_size_val, GN_NUM_TILES
        )

        mean_out = torch.empty((total_pairs,), dtype=torch.float32, device=x.device)
        rstd_out = torch.empty((total_pairs,), dtype=torch.float32, device=x.device)

        eps = 1e-5
        grid_agg = (total_pairs, 1, 1)
        groupnorm_aggregate_kernel[lambda: (grid_agg, (GN_BLOCK_SIZE, 1, 1))](
            partial_sum, partial_sq, mean_out, rstd_out, batch_size_val, GN_NUM_TILES, eps
        )

        out_bf16 = torch.empty_like(x_bf16)
        grid_apply = (GN_NUM_TILES, total_pairs, 1)
        groupnorm_apply_kernel[lambda: (grid_apply, (GN_BLOCK_SIZE, 1, 1))](
            x_bf16, gn_wt_bf16, gn_bias_bf16, out_bf16, mean_out, rstd_out, batch_size_val, GN_NUM_TILES
        )

        return out_bf16.to(x.dtype)


batch_size = 128
in_channels = 64
out_channels = 64
height = 256
width = 256
kernel_size = 3
stride = 1
groups = 8
num_groups = 8


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]
