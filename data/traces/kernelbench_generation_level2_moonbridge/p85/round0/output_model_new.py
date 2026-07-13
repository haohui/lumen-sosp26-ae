import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Problem constants
BATCH_SIZE = 128
IN_CHANNELS = 8
OUT_CHANNELS = 64
H, W = 128, 128
KH, KW = 3, 3
OH, OW = H - KH + 1, W - KW + 1  # 126, 126

NUM_GROUPS = 16
CH_PER_GROUP = OUT_CHANNELS // NUM_GROUPS  # 4
NORM_ELEMS = CH_PER_GROUP * OH * OW  # 63504

POOL_SIZE = 4
POOL_OH = OH // POOL_SIZE  # 31
POOL_OW = OW // POOL_SIZE  # 31

CLAMP_MIN = 0.0
CLAMP_MAX = 1.0
EPS = 1e-5

# Convolution tiling
OC_TILE = 4
OC_TILES = OUT_CHANNELS // OC_TILE  # 16
OH_TILE = 8
OW_TILE = 8
CONV_THREADS = OC_TILE * OH_TILE * OW_TILE  # 256
WEIGHT_ELEMS_PER_OC = IN_CHANNELS * KH * KW  # 72
SHM_WEIGHT_ELEMS = OC_TILE * WEIGHT_ELEMS_PER_OC  # 288

# GroupNorm tiling
BLOCK_SIZE = 256
TILE_SIZE = 4096
NUM_NORM_TILES = (NORM_ELEMS + TILE_SIZE - 1) // TILE_SIZE  # 16
MAX_TILES_PER_ROUND = 256

# Pool+clamp tiling
POOL_THREADS = 256


@avelang.jit
def conv2d_bf16_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
):
    tid = al.thread_id(0)
    bid_x = al.block_id(0)
    bid_y = al.block_id(1)
    bid_z = al.block_id(2)

    batch = bid_x // OC_TILES
    oc_tile_idx = bid_x - batch * OC_TILES

    # Stage weight for this OC tile into shared memory
    shm_w = al.make_shared((SHM_WEIGHT_ELEMS,), al.f32)
    base_oc = oc_tile_idx * OC_TILE
    weight_layout = al.make_layout(
        (OUT_CHANNELS, IN_CHANNELS, KH, KW),
        (IN_CHANNELS * KH * KW, KH * KW, KW, 1),
    )
    wgt = al.make_tensor(weight_ptr, al.bf16, weight_layout)
    for i in al.range(tid, SHM_WEIGHT_ELEMS, CONV_THREADS):
        oc_off = i // WEIGHT_ELEMS_PER_OC
        rem = i - oc_off * WEIGHT_ELEMS_PER_OC
        ic = rem // (KH * KW)
        rem2 = rem - ic * (KH * KW)
        kh = rem2 // KW
        kw = rem2 - kh * KW
        oc_global = base_oc + oc_off
        if oc_global < OUT_CHANNELS:
            shm_w[i] = al.convert(wgt[oc_global, ic, kh, kw], al.f32)

    al.syncthreads()

    spat_per_oc = OH_TILE * OW_TILE
    oc_local = tid // spat_per_oc
    spat = tid - oc_local * spat_per_oc
    oh_local = spat // OW_TILE
    ow_local = spat - oh_local * OW_TILE

    oc = oc_tile_idx * OC_TILE + oc_local
    oh = bid_y * OH_TILE + oh_local
    ow = bid_z * OW_TILE + ow_local

    if oc < OUT_CHANNELS and oh < OH and ow < OW:
        layout_in = al.make_layout(
            (BATCH_SIZE, IN_CHANNELS, H, W),
            (IN_CHANNELS * H * W, H * W, W, 1),
        )
        inp = al.make_tensor(input_ptr, al.bf16, layout_in)

        layout_b = al.make_layout((OUT_CHANNELS,), (1,))
        bias = al.make_tensor(bias_ptr, al.bf16, layout_b)

        layout_out = al.make_layout(
            (BATCH_SIZE, OUT_CHANNELS, OH, OW),
            (OUT_CHANNELS * OH * OW, OH * OW, OW, 1),
        )
        out = al.make_tensor(output_ptr, al.bf16, layout_out)

        acc = al.convert(bias[oc], al.f32)

        # Use shared memory weight
        w_base = oc_local * WEIGHT_ELEMS_PER_OC
        widx = w_base
        for ic in al.range(IN_CHANNELS):
            for kh in al.range(KH):
                for kw in al.range(KW):
                    ih = oh + kh
                    iw = ow + kw
                    ival = al.convert(inp[batch, ic, ih, iw], al.f32)
                    wval = shm_w[widx]
                    acc = acc + ival * wval
                    widx = widx + 1

        out[batch, oc, oh, ow] = al.convert(acc, al.bf16)


@avelang.jit
def groupnorm_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    total_instances = BATCH_SIZE * NUM_GROUPS

    instance_idx = bid // NUM_NORM_TILES
    tile_idx = bid - instance_idx * NUM_NORM_TILES

    if instance_idx < total_instances:
        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        batch_idx = instance_idx // NUM_GROUPS
        group_idx = instance_idx - batch_idx * NUM_GROUPS

        ch_start = group_idx * CH_PER_GROUP

        tile_start = tile_idx * TILE_SIZE
        tile_end = tile_start + TILE_SIZE
        if tile_end > NORM_ELEMS:
            tile_end = NORM_ELEMS

        layout_in = al.make_layout(
            (BATCH_SIZE, OUT_CHANNELS, OH, OW),
            (OUT_CHANNELS * OH * OW, OH * OW, OW, 1),
        )
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        local_sum = al.convert(0.0, al.f32)
        local_sq = al.convert(0.0, al.f32)

        spat_per_ch = OH * OW

        for elem in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            ch_offset = elem // spat_per_ch
            spat = elem - ch_offset * spat_per_ch
            h_idx = spat // OW
            w_idx = spat - h_idx * OW

            if ch_offset < CH_PER_GROUP:
                c_global = ch_start + ch_offset
                val = al.convert(x[batch_idx, c_global, h_idx, w_idx], al.f32)
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
        if tid == 0:
            smem_sum[0] = smem_sum[0] + smem_sum[1]
            smem_sq[0] = smem_sq[0] + smem_sq[1]

        if tid == 0:
            layout_ps = al.make_layout(
                (total_instances, NUM_NORM_TILES),
                (NUM_NORM_TILES, 1),
            )
            ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
            psq = al.make_tensor(partial_sq_ptr, al.f32, layout_ps)
            ps[instance_idx, tile_idx] = smem_sum[0]
            psq[instance_idx, tile_idx] = smem_sq[0]


@avelang.jit
def groupnorm_aggregate_kernel(
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    mean_out_ptr: al.Pointer(al.f32),
    rstd_out_ptr: al.Pointer(al.f32),
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    total_instances = BATCH_SIZE * NUM_GROUPS
    eps_f = al.convert(EPS, al.f32)

    if bid < total_instances:
        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        layout_ps = al.make_layout(
            (total_instances, NUM_NORM_TILES),
            (NUM_NORM_TILES, 1),
        )
        ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
        psq = al.make_tensor(partial_sq_ptr, al.f32, layout_ps)

        total_sum = al.convert(0.0, al.f32)
        total_sq = al.convert(0.0, al.f32)

        chunk_start = 0
        for _ in al.range(0, 16):
            if chunk_start >= NUM_NORM_TILES:
                break

            local_sum = al.convert(0.0, al.f32)
            local_sq = al.convert(0.0, al.f32)

            tile_idx = chunk_start + tid
            if tile_idx < NUM_NORM_TILES:
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
            if tid == 0:
                smem_sum[0] = smem_sum[0] + smem_sum[1]
                smem_sq[0] = smem_sq[0] + smem_sq[1]

            if tid == 0:
                total_sum = total_sum + smem_sum[0]
                total_sq = total_sq + smem_sq[0]

            chunk_start = chunk_start + MAX_TILES_PER_ROUND
            al.syncthreads()

        if tid == 0:
            norm_elems_f = al.convert(NORM_ELEMS, al.f32)
            mean = total_sum / norm_elems_f
            var = total_sq / norm_elems_f - mean * mean
            rstd = al.convert(1.0, al.f32) / al.sqrt(var + eps_f)

            layout_out = al.make_layout((total_instances,), (1,))
            mo = al.make_tensor(mean_out_ptr, al.f32, layout_out)
            ro = al.make_tensor(rstd_out_ptr, al.f32, layout_out)
            mo[bid] = mean
            ro[bid] = rstd


@avelang.jit
def groupnorm_apply_pool_clamp_kernel(
    x_ptr: al.Pointer(al.bf16),
    gn_weight_ptr: al.Pointer(al.bf16),
    gn_bias_ptr: al.Pointer(al.bf16),
    scale_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    rstd_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    pool_per_ch = POOL_OH * POOL_OW
    total_pool_per_batch = OUT_CHANNELS * pool_per_ch
    total_pool_positions = BATCH_SIZE * total_pool_per_batch
    global_offset = bid * POOL_THREADS + tid

    if global_offset < total_pool_positions:
        batch = global_offset // total_pool_per_batch
        rem = global_offset - batch * total_pool_per_batch
        oc = rem // pool_per_ch
        spat = rem - oc * pool_per_ch
        ph = spat // POOL_OW
        pw = spat - ph * POOL_OW
        if oc < OUT_CHANNELS:
            layout_in = al.make_layout(
                (BATCH_SIZE, OUT_CHANNELS, OH, OW),
                (OUT_CHANNELS * OH * OW, OH * OW, OW, 1),
            )
            x = al.make_tensor(x_ptr, al.bf16, layout_in)

            layout_gn_w = al.make_layout((OUT_CHANNELS,), (1,))
            gn_w = al.make_tensor(gn_weight_ptr, al.bf16, layout_gn_w)
            gn_b = al.make_tensor(gn_bias_ptr, al.bf16, layout_gn_w)

            layout_s = al.make_layout((OUT_CHANNELS,), (1,))
            scale = al.make_tensor(scale_ptr, al.bf16, layout_s)

            total_instances = BATCH_SIZE * NUM_GROUPS
            layout_m = al.make_layout((total_instances,), (1,))
            mean_t = al.make_tensor(mean_ptr, al.f32, layout_m)
            rstd_t = al.make_tensor(rstd_ptr, al.f32, layout_m)

            layout_out = al.make_layout(
                (BATCH_SIZE, OUT_CHANNELS, POOL_OH, POOL_OW),
                (OUT_CHANNELS * POOL_OH * POOL_OW, POOL_OH * POOL_OW, POOL_OW, 1),
            )
            out = al.make_tensor(out_ptr, al.bf16, layout_out)

            group = oc // CH_PER_GROUP
            instance_idx = batch * NUM_GROUPS + group
            mean_val = mean_t[instance_idx]
            rstd_val = rstd_t[instance_idx]

            gn_w_val = al.convert(gn_w[oc], al.f32)
            gn_b_val = al.convert(gn_b[oc], al.f32)
            scale_val = al.convert(scale[oc], al.f32)

            oh_start = ph * POOL_SIZE
            ow_start = pw * POOL_SIZE

            max_val = al.convert(-1e30, al.f32)

            for dh in al.range(POOL_SIZE):
                oh_idx = oh_start + dh
                if oh_idx < OH:
                    for dw in al.range(POOL_SIZE):
                        ow_idx = ow_start + dw
                        if ow_idx < OW:
                            x_val = al.convert(x[batch, oc, oh_idx, ow_idx], al.f32)
                            norm_val = (x_val - mean_val) * rstd_val * gn_w_val + gn_b_val
                            scaled = norm_val * scale_val
                            if scaled > max_val:
                                max_val = scaled

            clamp_min_f = al.convert(CLAMP_MIN, al.f32)
            clamp_max_f = al.convert(CLAMP_MAX, al.f32)
            if max_val < clamp_min_f:
                max_val = clamp_min_f
            if max_val > clamp_max_f:
                max_val = clamp_max_f

            out[batch, oc, ph, pw] = al.convert(max_val, al.bf16)


def _prepare_bf16_cuda(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    x_bf16 = _prepare_bf16_cuda(x)
    w_bf16 = _prepare_bf16_cuda(weight)
    b_bf16 = _prepare_bf16_cuda(bias)

    out = torch.empty(
        (BATCH_SIZE, OUT_CHANNELS, OH, OW),
        dtype=torch.bfloat16,
        device=x_bf16.device,
    )

    oh_tiles = (OH + OH_TILE - 1) // OH_TILE
    ow_tiles = (OW + OW_TILE - 1) // OW_TILE
    grid = (BATCH_SIZE * OC_TILES, oh_tiles, ow_tiles)

    conv2d_bf16_kernel[lambda: (grid, (CONV_THREADS, 1, 1))](
        x_bf16, w_bf16, b_bf16, out,
    )
    return out


def avelang_groupnorm_scale_pool_clamp(
    x: torch.Tensor,
    gn_weight: torch.Tensor,
    gn_bias: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.bfloat16
    device = x.device

    total_instances = BATCH_SIZE * NUM_GROUPS

    # Phase 1: tile-level reduction
    partial_sum = torch.empty(
        (total_instances, NUM_NORM_TILES), dtype=torch.float32, device=device
    )
    partial_sq = torch.empty(
        (total_instances, NUM_NORM_TILES), dtype=torch.float32, device=device
    )

    groupnorm_reduce_kernel[
        lambda: ((total_instances * NUM_NORM_TILES, 1, 1), (BLOCK_SIZE, 1, 1))
    ](x, partial_sum, partial_sq)

    # Phase 2: aggregate
    mean_out = torch.empty(
        (total_instances,), dtype=torch.float32, device=device
    )
    rstd_out = torch.empty(
        (total_instances,), dtype=torch.float32, device=device
    )

    groupnorm_aggregate_kernel[
        lambda: ((total_instances, 1, 1), (BLOCK_SIZE, 1, 1))
    ](partial_sum, partial_sq, mean_out, rstd_out)

    # Phase 3: apply + scale + maxpool + clamp
    gn_w = _prepare_bf16_cuda(gn_weight)
    gn_b = _prepare_bf16_cuda(gn_bias)
    scale_bf16 = _prepare_bf16_cuda(scale)

    total_pool_positions = BATCH_SIZE * OUT_CHANNELS * POOL_OH * POOL_OW
    total_blocks = (total_pool_positions + POOL_THREADS - 1) // POOL_THREADS

    out = torch.empty(
        (BATCH_SIZE, OUT_CHANNELS, POOL_OH, POOL_OW),
        dtype=torch.bfloat16,
        device=device,
    )

    groupnorm_apply_pool_clamp_kernel[
        lambda: ((total_blocks, 1, 1), (POOL_THREADS, 1, 1))
    ](x, gn_w, gn_b, scale_bf16, mean_out, rstd_out, out)

    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_groups,
        scale_shape,
        maxpool_kernel_size,
        clamp_min,
        clamp_max,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = x.to(dtype=torch.bfloat16, device=x.device).contiguous()

        conv_out = avelang_conv2d(x, self.conv.weight, self.conv.bias)
        result = avelang_groupnorm_scale_pool_clamp(
            conv_out,
            self.group_norm.weight,
            self.group_norm.bias,
            self.scale,
        )
        return result


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_CHANNELS, H, W)]


def get_init_inputs():
    return [
        IN_CHANNELS,
        OUT_CHANNELS,
        KH,
        NUM_GROUPS,
        (OUT_CHANNELS, 1, 1),
        POOL_SIZE,
        CLAMP_MIN,
        CLAMP_MAX,
    ]
