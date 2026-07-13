import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
BN_TILE_SIZE: al.constexpr = 4096
BN_MAX_TILES_PER_ROUND: al.constexpr = 256
ELEMS_PER_THREAD: al.constexpr = 4


@avelang.jit
def conv_transpose3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    K: al.i32,
    stride_val: al.i32,
    pad_val: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    num_blocks: al.i32,
    total_elements: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    grid_total = num_blocks * BLOCK_SIZE

    in_layout = al.make_layout(
        (B, C_in, D_in, H_in, W_in),
        (C_in * D_in * H_in * W_in, D_in * H_in * W_in, H_in * W_in, W_in, 1),
    )
    inp = al.make_tensor(input_ptr, al.bf16, in_layout)

    w_layout = al.make_layout(
        (C_in, C_out, K, K, K),
        (C_out * K * K * K, K * K * K, K * K, K, 1),
    )
    wgt = al.make_tensor(weight_ptr, al.bf16, w_layout)

    bias_t = al.make_tensor(bias_ptr, al.bf16, al.make_layout((C_out,), (1,)))

    out_layout = al.make_layout(
        (B, C_out, D_out, H_out, W_out),
        (C_out * D_out * H_out * W_out, D_out * H_out * W_out, H_out * W_out, W_out, 1),
    )
    out = al.make_tensor(output_ptr, al.bf16, out_layout)

    zero_i32 = al.convert(0, al.i32)

    for gid in al.range(bid * BLOCK_SIZE + tid, total_elements, grid_total):
        tmp = gid
        w_out = tmp % W_out
        tmp = tmp // W_out
        h_out = tmp % H_out
        tmp = tmp // H_out
        d_out = tmp % D_out
        tmp = tmp // D_out
        c_out = tmp % C_out
        b = tmp // C_out

        acc = al.convert(bias_t[c_out], al.f32)

        for ci in al.range(C_in):
            for kd in al.range(K):
                d_val = d_out + pad_val - kd
                if (d_val % stride_val) == zero_i32:
                    d_in = d_val // stride_val
                    if d_in >= zero_i32:
                        if d_in < D_in:
                            for kh in al.range(K):
                                h_val = h_out + pad_val - kh
                                if (h_val % stride_val) == zero_i32:
                                    h_in = h_val // stride_val
                                    if h_in >= zero_i32:
                                        if h_in < H_in:
                                            for kw in al.range(K):
                                                w_val = w_out + pad_val - kw
                                                if (w_val % stride_val) == zero_i32:
                                                    w_in = w_val // stride_val
                                                    if w_in >= zero_i32:
                                                        if w_in < W_in:
                                                            in_val = al.convert(inp[b, ci, d_in, h_in, w_in], al.f32)
                                                            wv = al.convert(wgt[ci, c_out, kd, kh, kw], al.f32)
                                                            acc = acc + in_val * wv

        out[b, c_out, d_out, h_out, w_out] = al.convert(acc, al.bf16)


@avelang.jit
def bn_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    B: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    num_tiles: al.i32,
    spatial_total: al.i32,
    HW: al.i32,
    stride_b: al.i32,
    stride_c: al.i32,
    stride_d: al.i32,
    stride_h: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    channel = bid // num_tiles
    tile_idx = bid - channel * num_tiles

    if channel < C:
        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        tile_start = tile_idx * BN_TILE_SIZE
        tile_end = tile_start + BN_TILE_SIZE
        if tile_end > spatial_total:
            tile_end = spatial_total

        x = al.make_tensor(x_ptr, al.bf16, al.make_layout(
            (B, C, D, H, W),
            (stride_b, stride_c, stride_d, stride_h, 1),
        ))

        local_sum = al.convert(0.0, al.f32)
        local_sq = al.convert(0.0, al.f32)

        for linear_idx in al.range(tile_start + tid, tile_end, BLOCK_SIZE):
            b_idx = linear_idx // (D * H * W)
            rem = linear_idx - b_idx * (D * H * W)
            d_idx = rem // HW
            rem2 = rem - d_idx * HW
            h_idx = rem2 // W
            w_idx = rem2 - h_idx * W
            val = al.convert(x[b_idx, channel, d_idx, h_idx, w_idx], al.f32)
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
            ps = al.make_tensor(partial_sum_ptr, al.f32, al.make_layout((C, num_tiles), (num_tiles, 1)))
            psq = al.make_tensor(partial_sq_ptr, al.f32, al.make_layout((C, num_tiles), (num_tiles, 1)))
            ps[channel, tile_idx] = smem_sum[0]
            psq[channel, tile_idx] = smem_sq[0]


@avelang.jit
def bn_aggregate_kernel(
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    mean_out_ptr: al.Pointer(al.f32),
    rstd_out_ptr: al.Pointer(al.f32),
    C: al.i32,
    num_tiles: al.i32,
    N: al.i32,
    eps: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < C:
        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        ps = al.make_tensor(partial_sum_ptr, al.f32, al.make_layout((C, num_tiles), (num_tiles, 1)))
        psq = al.make_tensor(partial_sq_ptr, al.f32, al.make_layout((C, num_tiles), (num_tiles, 1)))

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

            chunk_start = chunk_start + BN_MAX_TILES_PER_ROUND
            al.syncthreads()

        if tid == 0:
            N_f32 = al.convert(N, al.f32)
            mean_val = total_sum / N_f32
            var_val = total_sq / N_f32 - mean_val * mean_val
            if var_val < al.convert(0.0, al.f32):
                var_val = al.convert(0.0, al.f32)
            rstd_val = al.convert(1.0, al.f32) / al.sqrt(var_val + eps)

            mo = al.make_tensor(mean_out_ptr, al.f32, al.make_layout((C,), (1,)))
            ro = al.make_tensor(rstd_out_ptr, al.f32, al.make_layout((C,), (1,)))
            mo[bid] = mean_val
            ro[bid] = rstd_val


@avelang.jit
def bn_apply_pool_kernel(
    x_ptr: al.Pointer(al.bf16),
    bn_weight_ptr: al.Pointer(al.bf16),
    bn_bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    rstd_ptr: al.Pointer(al.f32),
    B: al.i32,
    C: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    stride_b: al.i32,
    stride_c: al.i32,
    stride_d: al.i32,
    stride_h: al.i32,
    num_blocks: al.i32,
    total_elements: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    grid_total = num_blocks * BLOCK_SIZE

    x = al.make_tensor(x_ptr, al.bf16, al.make_layout(
        (B, C, D_in, H_in, W_in),
        (stride_b, stride_c, stride_d, stride_h, 1),
    ))
    bn_w = al.make_tensor(bn_weight_ptr, al.bf16, al.make_layout((C,), (1,)))
    bn_b = al.make_tensor(bn_bias_ptr, al.bf16, al.make_layout((C,), (1,)))
    mn = al.make_tensor(mean_ptr, al.f32, al.make_layout((C,), (1,)))
    rs = al.make_tensor(rstd_ptr, al.f32, al.make_layout((C,), (1,)))

    out_layout = al.make_layout(
        (B, C, D_out, H_out, W_out),
        (C * D_out * H_out * W_out, D_out * H_out * W_out, H_out * W_out, W_out, 1),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    four = al.convert(4, al.i32)

    for gid in al.range(bid * BLOCK_SIZE + tid, total_elements, grid_total):
        tmp = gid
        w_p = tmp % W_out
        tmp = tmp // W_out
        h_p = tmp % H_out
        tmp = tmp // H_out
        d_p = tmp % D_out
        tmp = tmp // D_out
        c = tmp % C
        b = tmp // C

        mean_val = mn[c]
        rstd_val = rs[c]
        w_val = al.convert(bn_w[c], al.f32)
        b_val = al.convert(bn_b[c], al.f32)

        acc = al.convert(0.0, al.f32)
        d_start = d_p * four
        h_start = h_p * four
        w_start = w_p * four

        for doff in al.range(four):
            din = d_start + doff
            for hoff in al.range(four):
                hin = h_start + hoff
                for woff in al.range(four):
                    win = w_start + woff
                    val = al.convert(x[b, c, din, hin, win], al.f32)
                    val = (val - mean_val) * rstd_val
                    val = val * w_val + b_val
                    acc = acc + val

        result = acc / al.convert(64.0, al.f32)
        out[b, c, d_p, h_p, w_p] = al.convert(result, al.bf16)


def _prepare_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    return t.contiguous().to(dtype=torch.bfloat16)


def avelang_model_forward(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor | None = None,
    bn_running_var: torch.Tensor | None = None,
) -> torch.Tensor:
    x_bf16 = _prepare_bf16_contiguous(x)
    conv_w_bf16 = _prepare_bf16_contiguous(conv_weight)
    conv_b_bf16 = _prepare_bf16_contiguous(conv_bias)
    bn_w_bf16 = _prepare_bf16_contiguous(bn_weight)
    bn_b_bf16 = _prepare_bf16_contiguous(bn_bias)

    B = x_bf16.shape[0]
    C_in = x_bf16.shape[1]
    D_in = x_bf16.shape[2]
    H_in = x_bf16.shape[3]
    W_in = x_bf16.shape[4]
    C_out = conv_w_bf16.shape[1]
    K = conv_w_bf16.shape[2]
    stride = 2
    pad = 1

    D_out = (D_in - 1) * stride - 2 * pad + K
    H_out = (H_in - 1) * stride - 2 * pad + K
    W_out = (W_in - 1) * stride - 2 * pad + K

    total_conv_out = B * C_out * D_out * H_out * W_out
    conv_out = torch.empty(
        (B, C_out, D_out, H_out, W_out),
        dtype=torch.bfloat16,
        device=x_bf16.device,
    )

    num_blocks_conv = (total_conv_out + BLOCK_SIZE * ELEMS_PER_THREAD - 1) // (BLOCK_SIZE * ELEMS_PER_THREAD)
    conv_transpose3d_kernel[lambda: ((num_blocks_conv, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, conv_w_bf16, conv_b_bf16, conv_out,
        B, C_in, C_out, D_in, H_in, W_in, K, stride, pad,
        D_out, H_out, W_out, num_blocks_conv, total_conv_out,
    )

    eps = 1e-5
    is_training = bn_running_mean is None or bn_running_var is None

    stride_bn_b = C_out * D_out * H_out * W_out
    stride_bn_c = D_out * H_out * W_out
    stride_bn_d = H_out * W_out
    stride_bn_h = W_out

    if is_training:
        spatial_total = B * D_out * H_out * W_out
        HW = H_out * W_out
        num_tiles_bn = (spatial_total + BN_TILE_SIZE - 1) // BN_TILE_SIZE

        partial_sum = torch.empty(
            (C_out, num_tiles_bn), dtype=torch.float32, device=x_bf16.device
        )
        partial_sq = torch.empty(
            (C_out, num_tiles_bn), dtype=torch.float32, device=x_bf16.device
        )

        bn_reduce_kernel[lambda: ((C_out * num_tiles_bn, 1, 1), (BLOCK_SIZE, 1, 1))](
            conv_out, partial_sum, partial_sq,
            B, C_out, D_out, H_out, W_out,
            num_tiles_bn, spatial_total, HW,
            stride_bn_b, stride_bn_c, stride_bn_d, stride_bn_h,
        )

        mean_out = torch.empty((C_out,), dtype=torch.float32, device=x_bf16.device)
        rstd_out = torch.empty((C_out,), dtype=torch.float32, device=x_bf16.device)

        bn_aggregate_kernel[lambda: ((C_out, 1, 1), (BLOCK_SIZE, 1, 1))](
            partial_sum, partial_sq, mean_out, rstd_out,
            C_out, num_tiles_bn, spatial_total, eps,
        )
    else:
        mean_out = bn_running_mean.to(
            dtype=torch.float32, device=x_bf16.device
        ).contiguous()
        var = bn_running_var.to(
            dtype=torch.float32, device=x_bf16.device
        ).contiguous()
        rstd_out = 1.0 / torch.sqrt(var + eps)

    D_pool = D_out // 4
    H_pool = H_out // 4
    W_pool = W_out // 4
    total_pooled = B * C_out * D_pool * H_pool * W_pool

    final_out = torch.empty(
        (B, C_out, D_pool, H_pool, W_pool),
        dtype=torch.bfloat16,
        device=x_bf16.device,
    )

    num_blocks_pool = (total_pooled + BLOCK_SIZE * ELEMS_PER_THREAD - 1) // (BLOCK_SIZE * ELEMS_PER_THREAD)
    bn_apply_pool_kernel[lambda: ((num_blocks_pool, 1, 1), (BLOCK_SIZE, 1, 1))](
        conv_out, bn_w_bf16, bn_b_bf16, final_out,
        mean_out, rstd_out,
        B, C_out, D_out, H_out, W_out,
        D_pool, H_pool, W_pool,
        stride_bn_b, stride_bn_c, stride_bn_d, stride_bn_h,
        num_blocks_pool, total_pooled,
    )

    return final_out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        conv_weight = self.conv_transpose.weight.data
        conv_bias = self.conv_transpose.bias.data
        bn_weight = self.batch_norm.weight.data
        bn_bias = self.batch_norm.bias.data

        x_bf16 = x.to(dtype=torch.bfloat16, device=x.device).contiguous()

        if self.training:
            result = avelang_model_forward(
                x_bf16, conv_weight, conv_bias, bn_weight, bn_bias,
            )
        else:
            result = avelang_model_forward(
                x_bf16, conv_weight, conv_bias, bn_weight, bn_bias,
                bn_running_mean=self.batch_norm.running_mean,
                bn_running_var=self.batch_norm.running_var,
            )

        return result.to(x.dtype)
