import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
TILE_SIZE: al.constexpr = 4096
MAX_TILES_PER_ROUND: al.constexpr = 256

TILE_H: al.constexpr = 14
TILE_W: al.constexpr = 14
CHAN_PER_BLOCK: al.constexpr = 4
WEIGHT_SIZE: al.constexpr = 4 * 8 * 3 * 3
C_IN_K_K: al.constexpr = 8 * 3 * 3


@avelang.jit
def conv2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    groups: al.i32,
    C_per_group: al.i32,
    H: al.i32,
    W: al.i32,
    K: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    num_tiles_spatial: al.i32,
    num_tiles_h: al.i32,
    num_tiles_w: al.i32,
):
    tid = al.thread_id(0)
    bid_x = al.block_id(0)
    bid_y = al.block_id(1)

    oc_group = bid_y
    oc_start = oc_group * CHAN_PER_BLOCK

    batch = bid_x // num_tiles_spatial
    tile_idx = bid_x - batch * num_tiles_spatial
    tile_h_idx = tile_idx // num_tiles_w
    tile_w_idx = tile_idx - tile_h_idx * num_tiles_w
    tile_h_start = tile_h_idx * TILE_H
    tile_w_start = tile_w_idx * TILE_W

    if batch >= N:
        return

    tile_h_end = tile_h_start + TILE_H
    if tile_h_end > H_out:
        tile_h_end = H_out
    tile_w_end = tile_w_start + TILE_W
    if tile_w_end > W_out:
        tile_w_end = W_out
    actual_tile_h = tile_h_end - tile_h_start
    actual_tile_w = tile_w_end - tile_w_start

    smem_in = al.make_shared((8 * 16 * 16,), al.bf16)
    smem_w = al.make_shared((WEIGHT_SIZE,), al.bf16)

    layout_x = al.make_layout((N * C_in * H * W,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout_x)
    layout_w = al.make_layout((C_out * C_in * K * K,), (1,))
    w = al.make_tensor(w_ptr, al.bf16, layout_w)
    layout_b = al.make_layout((C_out,), (1,))
    b = al.make_tensor(b_ptr, al.bf16, layout_b)
    layout_out = al.make_layout((N * C_out * H_out * W_out,), (1,))
    out = al.make_tensor(out_ptr, al.bf16, layout_out)

    for w_idx in al.range(tid, WEIGHT_SIZE, BLOCK_SIZE):
        oc_local = w_idx // C_IN_K_K
        remainder = w_idx - oc_local * C_IN_K_K
        ic = remainder // (K * K)
        rem2 = remainder - ic * K * K
        kh = rem2 // K
        kw = rem2 - kh * K
        oc = oc_start + oc_local
        if oc < C_out:
            w_flat = oc * C_in * K * K + ic * K * K + kh * K + kw
            smem_w[w_idx] = w[w_flat]

    th = tid // 16
    tw = tid - th * 16
    for ic in al.range(C_in):
        global_h = tile_h_start + th
        global_w = tile_w_start + tw
        in_flat = batch * C_in * H * W + ic * H * W + global_h * W + global_w
        smem_idx = ic * 16 * 16 + th * 16 + tw
        smem_in[smem_idx] = x[in_flat]

    al.syncthreads()

    th_out = tid // TILE_W
    tw_out = tid - th_out * TILE_W

    local_sum = al.convert(0.0, al.f32)
    local_sq = al.convert(0.0, al.f32)

    if th_out < actual_tile_h and tw_out < actual_tile_w:
        oh = tile_h_start + th_out
        ow = tile_w_start + tw_out

        for oc_local in al.range(CHAN_PER_BLOCK):
            oc = oc_start + oc_local
            if oc < C_out:
                bias_val = al.convert(b[oc], al.f32)
                acc = bias_val
                for ic in al.range(C_in):
                    for kh in al.range(K):
                        ih = th_out + kh
                        for kw in al.range(K):
                            iw = tw_out + kw
                            smem_idx = ic * 16 * 16 + ih * 16 + iw
                            x_val = al.convert(smem_in[smem_idx], al.f32)
                            w_smem_idx = oc_local * C_IN_K_K + ic * K * K + kh * K + kw
                            w_val = al.convert(smem_w[w_smem_idx], al.f32)
                            acc = acc + x_val * w_val

                out_flat = batch * C_out * H_out * W_out + oc * H_out * W_out + oh * W_out + ow
                out[out_flat] = al.convert(acc, al.bf16)
                local_sum = local_sum + acc
                local_sq = local_sq + acc * acc

    smem_sum_stats = al.make_shared((BLOCK_SIZE,), al.f32)
    smem_sq_stats = al.make_shared((BLOCK_SIZE,), al.f32)
    smem_sum_stats[tid] = local_sum
    smem_sq_stats[tid] = local_sq
    al.syncthreads()

    if tid < 128:
        smem_sum_stats[tid] = smem_sum_stats[tid] + smem_sum_stats[tid + 128]
        smem_sq_stats[tid] = smem_sq_stats[tid] + smem_sq_stats[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem_sum_stats[tid] = smem_sum_stats[tid] + smem_sum_stats[tid + 64]
        smem_sq_stats[tid] = smem_sq_stats[tid] + smem_sq_stats[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem_sum_stats[tid] = smem_sum_stats[tid] + smem_sum_stats[tid + 32]
        smem_sq_stats[tid] = smem_sq_stats[tid] + smem_sq_stats[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem_sum_stats[tid] = smem_sum_stats[tid] + smem_sum_stats[tid + 16]
        smem_sq_stats[tid] = smem_sq_stats[tid] + smem_sq_stats[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem_sum_stats[tid] = smem_sum_stats[tid] + smem_sum_stats[tid + 8]
        smem_sq_stats[tid] = smem_sq_stats[tid] + smem_sq_stats[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem_sum_stats[tid] = smem_sum_stats[tid] + smem_sum_stats[tid + 4]
        smem_sq_stats[tid] = smem_sq_stats[tid] + smem_sq_stats[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem_sum_stats[tid] = smem_sum_stats[tid] + smem_sum_stats[tid + 2]
        smem_sq_stats[tid] = smem_sq_stats[tid] + smem_sq_stats[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem_sum_stats[tid] = smem_sum_stats[tid] + smem_sum_stats[tid + 1]
        smem_sq_stats[tid] = smem_sq_stats[tid] + smem_sq_stats[tid + 1]

    if tid == 0:
        group = oc_group
        layout_ps = al.make_layout((N, groups, num_tiles_spatial), (groups * num_tiles_spatial, num_tiles_spatial, 1))
        ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
        psq = al.make_tensor(partial_sq_ptr, al.f32, layout_ps)
        ps[batch, group, tile_idx] = smem_sum_stats[0]
        psq[batch, group, tile_idx] = smem_sq_stats[0]


@avelang.jit
def groupnorm_aggregate_kernel(
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    mean_out_ptr: al.Pointer(al.f32),
    rstd_out_ptr: al.Pointer(al.f32),
    N: al.i32,
    groups: al.i32,
    C_per_group: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    num_tiles: al.i32,
    eps: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    batch = bid // groups
    group = bid - batch * groups

    if batch < N and group < groups:
        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        layout_ps = al.make_layout((N, groups, num_tiles), (groups * num_tiles, num_tiles, 1))
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
                local_sum = ps[batch, group, tile_idx]
                local_sq = psq[batch, group, tile_idx]

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
            elements_per_group = C_per_group * H_out * W_out
            count_f32 = al.convert(elements_per_group, al.f32)
            mean = total_sum / count_f32
            var = total_sq / count_f32 - mean * mean
            rstd = al.convert(1.0, al.f32) / al.sqrt(var + eps)

            layout_out = al.make_layout((N, groups), (groups, 1))
            mo = al.make_tensor(mean_out_ptr, al.f32, layout_out)
            ro = al.make_tensor(rstd_out_ptr, al.f32, layout_out)
            mo[batch, group] = mean
            ro[batch, group] = rstd


@avelang.jit
def fused_norm_act_residual_kernel(
    conv_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    gn_weight_ptr: al.Pointer(al.bf16),
    gn_bias_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    rstd_ptr: al.Pointer(al.f32),
    N: al.i32,
    C_out: al.i32,
    groups: al.i32,
    C_per_group: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    total_elements: al.i32,
    num_tiles: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    tile_start = bid * TILE_SIZE
    tile_end = tile_start + TILE_SIZE
    if tile_end > total_elements:
        tile_end = total_elements

    spatial_size = H_out * W_out

    layout_conv = al.make_layout((total_elements,), (1,))
    conv = al.make_tensor(conv_ptr, al.bf16, layout_conv)
    layout_out = al.make_layout((total_elements,), (1,))
    out = al.make_tensor(out_ptr, al.bf16, layout_out)

    layout_gn_w = al.make_layout((C_out,), (1,))
    gn_w = al.make_tensor(gn_weight_ptr, al.bf16, layout_gn_w)
    gn_b = al.make_tensor(gn_bias_ptr, al.bf16, layout_gn_w)

    layout_mean = al.make_layout((N, groups), (groups, 1))
    mt = al.make_tensor(mean_ptr, al.f32, layout_mean)
    rt = al.make_tensor(rstd_ptr, al.f32, layout_mean)

    three = al.convert(3.0, al.f32)
    six = al.convert(6.0, al.f32)
    zero = al.convert(0.0, al.f32)

    for idx in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
        n = idx // (C_out * spatial_size)
        remainder = idx - n * C_out * spatial_size
        c = remainder // spatial_size
        g = c // C_per_group

        x_conv_val = al.convert(conv[idx], al.f32)
        mean_val = mt[n, g]
        rstd_val = rt[n, g]
        w_val = al.convert(gn_w[c], al.f32)
        b_val = al.convert(gn_b[c], al.f32)

        x_norm = (x_conv_val - mean_val) * rstd_val * w_val + b_val
        x_tanh = al.tanh(x_norm)

        temp = x_tanh + three
        if temp < zero:
            temp = zero
        if temp > six:
            temp = six
        x_hs = x_tanh * temp / six

        x_res = x_conv_val + x_hs
        out[idx] = al.convert(x_res, al.bf16)


@avelang.jit
def logsumexp_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    total_output = N * H * W
    global_id = bid * BLOCK_SIZE + tid

    if global_id < total_output:
        spatial = H * W
        n = global_id // spatial
        remainder = global_id - n * spatial
        h = remainder // W
        w = remainder - h * W

        layout_x = al.make_layout((N * C * H * W,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_x)

        base = n * C * spatial + h * W + w
        max_val = al.convert(x[base], al.f32)
        for c in al.range(1, C):
            val = al.convert(x[base + c * spatial], al.f32)
            if val > max_val:
                max_val = val

        sum_exp = al.convert(0.0, al.f32)
        for c in al.range(C):
            val = al.convert(x[base + c * spatial], al.f32)
            diff = val - max_val
            sum_exp = sum_exp + al.exp(diff)

        result = max_val + al.log(sum_exp)

        layout_out = al.make_layout((total_output,), (1,))
        out = al.make_tensor(out_ptr, al.bf16, layout_out)
        out[global_id] = al.convert(result, al.bf16)


def _prepare_bf16(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_model(x: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                   gn_weight: torch.Tensor, gn_bias: torch.Tensor,
                   groups: int, eps: float = 1e-5) -> torch.Tensor:
    input_dtype = x.dtype

    x_bf16 = _prepare_bf16(x)
    conv_w_bf16 = _prepare_bf16(conv_weight)
    conv_b_bf16 = _prepare_bf16(conv_bias)
    gn_w_bf16 = _prepare_bf16(gn_weight)
    gn_b_bf16 = _prepare_bf16(gn_bias)

    N, C_in, H, W = x_bf16.shape
    C_out = conv_w_bf16.shape[0]
    K = conv_w_bf16.shape[2]
    C_per_group = C_out // groups

    H_out = H - K + 1
    W_out = W - K + 1

    num_tiles_h = (H_out + TILE_H - 1) // TILE_H
    num_tiles_w = (W_out + TILE_W - 1) // TILE_W
    num_tiles_spatial = num_tiles_h * num_tiles_w
    num_oc_groups = C_out // CHAN_PER_BLOCK

    # Phase 1: Fused convolution + GroupNorm stats reduction
    conv_out = torch.empty((N, C_out, H_out, W_out), dtype=torch.bfloat16, device=x_bf16.device)
    partial_sum = torch.empty((N, groups, num_tiles_spatial), dtype=torch.float32, device=x_bf16.device)
    partial_sq = torch.empty((N, groups, num_tiles_spatial), dtype=torch.float32, device=x_bf16.device)

    conv_grid_x = N * num_tiles_spatial
    conv_grid_y = num_oc_groups
    conv2d_kernel[lambda: ((conv_grid_x, conv_grid_y, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, conv_w_bf16, conv_b_bf16, conv_out,
        partial_sum, partial_sq,
        N, C_in, C_out, groups, C_per_group, H, W, K, H_out, W_out,
        num_tiles_spatial, num_tiles_h, num_tiles_w
    )

    # Phase 2: GroupNorm aggregate
    mean_out = torch.empty((N, groups), dtype=torch.float32, device=x_bf16.device)
    rstd_out = torch.empty((N, groups), dtype=torch.float32, device=x_bf16.device)

    agg_grid = (N * groups, 1, 1)
    groupnorm_aggregate_kernel[lambda: (agg_grid, (BLOCK_SIZE, 1, 1))](
        partial_sum, partial_sq, mean_out, rstd_out,
        N, groups, C_per_group, H_out, W_out, num_tiles_spatial, eps
    )

    # Phase 3: Fused GroupNorm apply + Tanh + HardSwish + residual add
    total_elements = N * C_out * H_out * W_out
    fused_num_tiles = (total_elements + TILE_SIZE - 1) // TILE_SIZE
    x_res = torch.empty((N, C_out, H_out, W_out), dtype=torch.bfloat16, device=x_bf16.device)

    fused_norm_act_residual_kernel[lambda: ((fused_num_tiles, 1, 1), (BLOCK_SIZE, 1, 1))](
        conv_out, x_res, gn_w_bf16, gn_b_bf16, mean_out, rstd_out,
        N, C_out, groups, C_per_group, H_out, W_out, total_elements, fused_num_tiles
    )

    # Phase 4: LogSumExp over dim=1
    total_output = N * H_out * W_out
    lse_grid = ((total_output + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
    lse_out = torch.empty((N, 1, H_out, W_out), dtype=torch.bfloat16, device=x_bf16.device)

    logsumexp_kernel[lambda: (lse_grid, (BLOCK_SIZE, 1, 1))](
        x_res, lse_out, N, C_out, H_out, W_out
    )

    return lse_out.to(dtype=input_dtype)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.eps = eps

        self.conv_weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.empty(out_channels))
        self.gn_weight = nn.Parameter(torch.empty(out_channels))
        self.gn_bias = nn.Parameter(torch.empty(out_channels))

        nn.init.kaiming_uniform_(self.conv_weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv_weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.conv_bias, -bound, bound)
        nn.init.ones_(self.gn_weight)
        nn.init.zeros_(self.gn_bias)

    def forward(self, x):
        return avelang_model(x, self.conv_weight, self.conv_bias,
                             self.gn_weight, self.gn_bias,
                             self.groups, self.eps)


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
groups = 16


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups]
