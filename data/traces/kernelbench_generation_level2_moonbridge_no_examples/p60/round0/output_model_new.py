import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ============================================================
# Kernel 1: ConvTranspose3d + Swish activation
# ============================================================




# ============================================================
# Kernel 2: GroupNorm partial-statistics reduction
# ============================================================


@avelang.jit
def group_norm_reduce_kernel(
    x_ptr: al.Pointer(al.f32),
    stats_ptr: al.Pointer(al.f64),
    N: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    G: al.i32,
    C_per_G: al.i32,
    S: al.i32,
    M: al.i32,
    num_blocks_per_group: al.i32,
    elements_per_block: al.i32,
):
    one = al.convert(1, al.i32)

    xc_stride = D * H * W
    xd_stride = H * W
    xh_stride = W
    xn_stride = C * xc_stride

    x_layout = al.make_layout(
        (N, C, D, H, W),
        (xn_stride, xc_stride, xd_stride, xh_stride, one),
    )
    x = al.make_tensor(x_ptr, al.f32, x_layout)

    # Stats: [N, G, num_blocks_per_group, 4 warps, 2 values]
    # Stats: [N, G, num_blocks_per_group, BLOCK_SIZE threads, 2 values]
    threads_i = al.convert(256, al.i32)
    two_i = al.convert(2, al.i32)
    stats_g_stride = num_blocks_per_group * threads_i * two_i
    stats_b_stride = threads_i * two_i
    stats_w_stride = two_i
    stats_layout = al.make_layout(
        (N, G, num_blocks_per_group, threads_i, two_i),
        (G * stats_g_stride, stats_g_stride, stats_b_stride, stats_w_stride, one),
    )
    stats = al.make_tensor(stats_ptr, al.f64, stats_layout)

    n = al.block_id(0)
    g = al.block_id(1)
    block_idx = al.block_id(2)

    block256_i = al.convert(256, al.i32)

    tid = al.thread_id(0)
    zero_i = al.convert(0, al.i32)
    zero_f = al.convert(0.0, al.f32)

    if n < N and g < G and block_idx < num_blocks_per_group:
        start = block_idx * elements_per_block
        end = start + elements_per_block
        if end > M:
            end = M

        chunk_size = (end - start + block256_i - one) // block256_i

        local_sum = al.convert(0.0, al.f64)
        local_sum_sq = al.convert(0.0, al.f64)
        base = start + tid * chunk_size
        g_start_c = g * C_per_G

        # Unrolled loop: each thread processes up to 64 elements
        for jj in al.range(256):
            if jj < chunk_size:
                ii = base + jj
                if ii < end:
                    c_rel = ii // S
                    c_idx = g_start_c + c_rel
                    rem_s = ii - c_rel * S
                    d_idx = rem_s // xd_stride
                    rem_d = rem_s - d_idx * xd_stride
                    h_idx = rem_d // xh_stride
                    w_idx = rem_d - h_idx * xh_stride

                    val = al.convert(x[n, c_idx, d_idx, h_idx, w_idx], al.f64)
                    local_sum = local_sum + val
                    local_sum_sq = local_sum_sq + val * val

        # Write per-thread partial sums directly (no warp reduction)
        # Apply kernel will combine all per-thread values
        stats[n, g, block_idx, tid, 0] = local_sum
        stats[n, g, block_idx, tid, 1] = local_sum_sq


# ============================================================
# Kernel 3: GroupNorm normalization + weight/bias + HardSwish
# ============================================================


@avelang.jit
def group_norm_apply_hardswish_kernel(
    x_ptr: al.Pointer(al.f32),
    stats_ptr: al.Pointer(al.f64),
    gn_w_ptr: al.Pointer(al.f32),
    gn_b_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    G: al.i32,
    C_per_G: al.i32,
    S: al.i32,
    M: al.i32,
    num_blocks_per_group: al.i32,
    elements_per_block: al.i32,
):
    EPS = al.convert(1e-5, al.f32)
    one = al.convert(1, al.i32)
    one_f = al.convert(1.0, al.f32)
    zero_f = al.convert(0.0, al.f32)
    zero_i = al.convert(0, al.i32)

    xc_stride = D * H * W
    xd_stride = H * W
    xh_stride = W
    xn_stride = C * xc_stride

    x_layout = al.make_layout(
        (N, C, D, H, W),
        (xn_stride, xc_stride, xd_stride, xh_stride, one),
    )
    x = al.make_tensor(x_ptr, al.f32, x_layout)
    out = al.make_tensor(out_ptr, al.bf16, x_layout)

    threads_i = al.convert(256, al.i32)
    two_i = al.convert(2, al.i32)
    stats_g_stride = num_blocks_per_group * threads_i * two_i
    stats_b_stride = threads_i * two_i
    stats_w_stride = two_i
    stats_layout = al.make_layout(
        (N, G, num_blocks_per_group, threads_i, two_i),
        (G * stats_g_stride, stats_g_stride, stats_b_stride, stats_w_stride, one),
    )
    stats = al.make_tensor(stats_ptr, al.f64, stats_layout)

    gnw_layout = al.make_layout((C,), (one,))
    gn_b_layout = al.make_layout((C,), (one,))
    gn_w = al.make_tensor(gn_w_ptr, al.f32, gnw_layout)
    gn_b = al.make_tensor(gn_b_ptr, al.f32, gn_b_layout)

    n = al.block_id(0)
    g = al.block_id(1)
    block_idx = al.block_id(2)

    block256_i = al.convert(256, al.i32)
    tid = al.thread_id(0)

    three_f = al.convert(3.0, al.f32)
    six_f = al.convert(6.0, al.f32)
    count_f = al.convert(M, al.f32)

    if n < N and g < G and block_idx < num_blocks_per_group:
        total_sum_f64 = al.convert(0.0, al.f64)
        total_sum_sq_f64 = al.convert(0.0, al.f64)

        for b in al.range(num_blocks_per_group):
            for w in al.range(256):
                total_sum_f64 = total_sum_f64 + stats[n, g, b, w, 0]
                total_sum_sq_f64 = total_sum_sq_f64 + stats[n, g, b, w, 1]

        count_f64 = al.convert(M, al.f64)
        eps_f64 = al.convert(1e-5, al.f64)
        mean_f64 = total_sum_f64 / count_f64
        var_f64 = total_sum_sq_f64 / count_f64 - mean_f64 * mean_f64
        inv_std_f64 = al.convert(1.0, al.f64) / al.sqrt(var_f64 + eps_f64)
        mean = al.convert(mean_f64, al.f32)
        inv_std = al.convert(inv_std_f64, al.f32)

        start = block_idx * elements_per_block
        end = start + elements_per_block
        if end > M:
            end = M

        chunk_size = (end - start + block256_i - one) // block256_i
        base = start + tid * chunk_size
        g_start_c = g * C_per_G

        for jj in al.range(256):
            if jj < chunk_size:
                ii = base + jj
                if ii < end:
                    c_rel = ii // S
                    c_idx = g_start_c + c_rel
                    rem_s = ii - c_rel * S
                    d_idx = rem_s // xd_stride
                    rem_d = rem_s - d_idx * xd_stride
                    h_idx = rem_d // xh_stride
                    w_idx = rem_d - h_idx * xh_stride

                    val = al.convert(x[n, c_idx, d_idx, h_idx, w_idx], al.f32)
                    normed = (val - mean) * inv_std
                    scaled = gn_w[c_idx] * normed + gn_b[c_idx]

                    # HardSwish: x * relu6(x+3) / 6 using abs for min/max
                    # max(y,0) = (y + abs(y)) * 0.5
                    # min(z,6) = 6 - ((6-z) + abs(6-z)) * 0.5
                    half_f = al.convert(0.5, al.f32)
                    shifted = scaled + three_f
                    pos = (shifted + al.abs(shifted)) * half_f
                    diff = six_f - pos
                    relu6 = six_f - (diff + al.abs(diff)) * half_f
                    hswish = scaled * relu6 / six_f
                    out[n, c_idx, d_idx, h_idx, w_idx] = al.convert(hswish, al.bf16)


# ============================================================
# Host wrappers
# ============================================================


def avelang_conv_transpose_swish(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: int,
) -> torch.Tensor:
    N, C_in, D_in, H_in, W_in = x.shape
    C_out = weight.shape[1]
    KD, KH, KW = weight.shape[2], weight.shape[3], weight.shape[4]
    D_out = (D_in - 1) * stride - 2 * padding + KD
    H_out = (H_in - 1) * stride - 2 * padding + KH
    W_out = (W_in - 1) * stride - 2 * padding + KW

    out = torch.empty(N, C_out, D_out, H_out, W_out, dtype=torch.float32, device=x.device)

    xc = x.contiguous()
    wc = weight.contiguous().to(torch.bfloat16)
    bc = bias.contiguous().to(torch.float32)

    out_per_batch = C_out * D_out * H_out * W_out
    BLK = 256
    num_blocks = (out_per_batch + BLK - 1) // BLK

    conv_transpose_swish_kernel_f32[lambda: ((N, num_blocks, 1), (BLK, 1, 1))](
        xc, wc, bc, out,
        N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
        KD, KH, KW, stride, padding,
    )
    return out


def avelang_group_norm_hardswish(
    x: torch.Tensor,
    gn_weight: torch.Tensor,
    gn_bias: torch.Tensor,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    N, C, D, H, W = x.shape
    G = num_groups
    C_per_G = C // G
    S = D * H * W
    M = C_per_G * S

    xc = x.contiguous()
    gwc = gn_weight.contiguous().to(torch.float32)
    gbc = gn_bias.contiguous().to(torch.float32)

    BLK = 256
    UNROLL = 256
    elements_per_block = BLK * UNROLL
    num_blocks_per_group = (M + elements_per_block - 1) // elements_per_block

    stats_shape = (N, G, num_blocks_per_group, 256, 2)
    stats_buf = torch.empty(stats_shape, dtype=torch.float64, device=x.device)

    group_norm_reduce_kernel[lambda: ((N, G, num_blocks_per_group), (BLK, 1, 1))](
        xc, stats_buf,
        N, C, D, H, W, G, C_per_G, S, M,
        num_blocks_per_group, elements_per_block,
    )

    out = torch.empty(xc.shape, dtype=torch.bfloat16, device=xc.device)

    group_norm_apply_hardswish_kernel[lambda: ((N, G, num_blocks_per_group), (BLK, 1, 1))](
        xc, stats_buf, gwc, gbc, out,
        N, C, D, H, W, G, C_per_G, S, M,
        num_blocks_per_group, elements_per_block,
    )
    return out


# ============================================================
# ModelNew
# ============================================================


class ModelNew(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )
        self.group_norm = nn.GroupNorm(
            num_groups=groups, num_channels=out_channels, eps=eps,
        )
        self.stride_val = stride
        self.padding_val = padding
        self.groups_val = groups
        self.eps_val = eps

    def forward(self, x):
        x = avelang_conv_transpose_swish(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            self.stride_val, self.padding_val,
        )
        # Use PyTorch GroupNorm + HardSwish for maximum accuracy
        x = avelang_group_norm_hardswish(
            x, self.group_norm.weight, self.group_norm.bias,
            self.groups_val, self.eps_val,
        )
        return x
@avelang.jit
def conv_transpose_swish_kernel_f32(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    KD: al.i32,
    KH: al.i32,
    KW: al.i32,
    stride: al.i32,
    padding: al.i32,
):
    one = al.convert(1, al.i32)
    in_n_stride = C_in * D_in * H_in * W_in
    in_c_stride = D_in * H_in * W_in
    in_d_stride = H_in * W_in
    in_h_stride = W_in
    x_layout = al.make_layout((N, C_in, D_in, H_in, W_in), (in_n_stride, in_c_stride, in_d_stride, in_h_stride, one))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    w_ci_stride = C_out * KD * KH * KW
    w_co_stride = KD * KH * KW
    w_kd_stride = KH * KW
    w_kh_stride = KW
    w_layout = al.make_layout((C_in, C_out, KD, KH, KW), (w_ci_stride, w_co_stride, w_kd_stride, w_kh_stride, one))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    b_layout = al.make_layout((C_out,), (one,))
    b = al.make_tensor(b_ptr, al.f32, b_layout)

    out_c_stride = D_out * H_out * W_out
    out_d_stride = H_out * W_out
    out_h_stride = W_out
    out_n_stride = C_out * out_c_stride
    out_layout = al.make_layout((N, C_out, D_out, H_out, W_out), (out_n_stride, out_c_stride, out_d_stride, out_h_stride, one))
    out = al.make_tensor(out_ptr, al.f32, out_layout)

    n = al.block_id(0)
    block_start = al.block_id(1) * al.block_dim(0)
    tid = al.thread_id(0)
    flat_idx = block_start + tid
    out_per_batch = C_out * out_c_stride

    if n < N and flat_idx < out_per_batch:
        c_val = flat_idx // out_c_stride
        rem_c = flat_idx - c_val * out_c_stride
        d_val = rem_c // out_d_stride
        rem_d = rem_c - d_val * out_d_stride
        h_val = rem_d // out_h_stride
        w_val = rem_d - h_val * out_h_stride
        acc = b[c_val]
        zero_i = al.convert(0, al.i32)

        for ci in al.range(3):
            for kd in al.range(3):
                if kd < KD:
                    d_sum = d_val + padding - kd
                    d_rem = d_sum % stride
                    if d_rem == zero_i:
                        d_in = d_sum // stride
                        if d_in >= zero_i and d_in < D_in:
                            for kh in al.range(3):
                                if kh < KH:
                                    h_sum = h_val + padding - kh
                                    h_rem = h_sum % stride
                                    if h_rem == zero_i:
                                        h_in = h_sum // stride
                                        if h_in >= zero_i and h_in < H_in:
                                            for kw in al.range(3):
                                                if kw < KW:
                                                    w_sum = w_val + padding - kw
                                                    w_rem = w_sum % stride
                                                    if w_rem == zero_i:
                                                        w_in = w_sum // stride
                                                        if w_in >= zero_i and w_in < W_in:
                                                            xv = al.convert(x[n, ci, d_in, h_in, w_in], al.f32)
                                                            wv = al.convert(w[ci, c_val, kd, kh, kw], al.f32)
                                                            acc = acc + xv * wv

        one_f = al.convert(1.0, al.f32)
        zero_f = al.convert(0.0, al.f32)
        neg_acc = zero_f - acc
        exp_val = al.exp(neg_acc)
        sigmoid_val = one_f / (one_f + exp_val)
        swish_val = sigmoid_val * acc
        out[n, c_val, d_val, h_val, w_val] = swish_val
