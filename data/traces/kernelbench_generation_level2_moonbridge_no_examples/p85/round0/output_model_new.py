import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ============================================================
# Kernel 1: 2D Convolution (BF16 compute, FP32 accumulate)
# Optimized: multi-output-channel blocking for input reuse
# ============================================================

@avelang.jit
def conv2d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H: al.i32,
    W: al.i32,
    KH: al.i32,
    KW: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    in_stride_N: al.i32,
    in_stride_C: al.i32,
    in_stride_H: al.i32,
    w_stride_CO: al.i32,
    w_stride_CI: al.i32,
    w_stride_KH: al.i32,
    out_stride_N: al.i32,
    out_stride_C: al.i32,
    out_stride_H: al.i32,
    BLOCK_C: al.constexpr,
):
    in_layout = al.make_layout((N, C_in, H, W), (in_stride_N, in_stride_C, in_stride_H, 1))
    inp = al.make_tensor(input_ptr, al.bf16, in_layout)

    w_layout = al.make_layout((C_out, C_in, KH, KW), (w_stride_CO, w_stride_CI, w_stride_KH, 1))
    weight = al.make_tensor(weight_ptr, al.bf16, w_layout)

    out_layout = al.make_layout((N, C_out, H_out, W_out), (out_stride_N, out_stride_C, out_stride_H, 1))
    out = al.make_tensor(output_ptr, al.bf16, out_layout)

    bias_layout = al.make_layout((C_out,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    n = al.block_id(0)
    c_block = al.block_id(1)
    h = al.block_id(2)
    w = al.thread_id(0)

    c_start = c_block * BLOCK_C

    if w < W_out:
        acc = al.make_local((BLOCK_C,), al.f32)
        zero = al.convert(0, al.f32)

        for c_off in al.range(BLOCK_C):
            c_out_cur = c_start + c_off
            if c_out_cur < C_out:
                acc[c_off] = al.convert(bias[c_out_cur], al.f32)
            else:
                acc[c_off] = zero

        for c_in in al.range(C_in):
            for kh in al.range(KH):
                for kw in al.range(KW):
                    in_val = al.convert(inp[n, c_in, h + kh, w + kw], al.f32)
                    for c_off in al.range(BLOCK_C):
                        c_out_cur = c_start + c_off
                        if c_out_cur < C_out:
                            w_val = al.convert(weight[c_out_cur, c_in, kh, kw], al.f32)
                            acc[c_off] = acc[c_off] + in_val * w_val

        for c_off in al.range(BLOCK_C):
            c_out_cur = c_start + c_off
            if c_out_cur < C_out:
                out[n, c_out_cur, h, w] = al.convert(acc[c_off], al.bf16)


# ============================================================
# Kernel 2: Group Normalization + Scale (fused)
# Training mode: compute stats from input batch, normalize,
# apply GroupNorm affine, then multiply by channel-wise scale.
# ============================================================

@avelang.jit
def group_norm_scale_kernel(
    input_ptr: al.Pointer(al.bf16),
    gn_weight_ptr: al.Pointer(al.bf16),
    gn_bias_ptr: al.Pointer(al.bf16),
    scale_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_out: al.i32,
    C_per_group: al.i32,
    H_conv: al.i32,
    W_conv: al.i32,
    in_stride_N: al.i32,
    in_stride_C: al.i32,
    in_stride_H: al.i32,
    out_stride_N: al.i32,
    out_stride_C: al.i32,
    out_stride_H: al.i32,
    BLOCK_SIZE: al.constexpr,
):
    in_layout = al.make_layout((N, C_out, H_conv, W_conv), (in_stride_N, in_stride_C, in_stride_H, 1))
    inp = al.make_tensor(input_ptr, al.bf16, in_layout)

    w_layout = al.make_layout((C_out,), (1,))
    gn_w = al.make_tensor(gn_weight_ptr, al.bf16, w_layout)
    gn_b = al.make_tensor(gn_bias_ptr, al.bf16, w_layout)
    scale = al.make_tensor(scale_ptr, al.bf16, w_layout)

    out_layout = al.make_layout((N, C_out, H_conv, W_conv), (out_stride_N, out_stride_C, out_stride_H, 1))
    out = al.make_tensor(output_ptr, al.bf16, out_layout)

    n = al.block_id(0)
    group = al.block_id(1)
    tid = al.thread_id(0)

    group_start_ch = group * C_per_group
    total_elements = C_per_group * H_conv * W_conv
    hw_elems = H_conv * W_conv

    eps = al.convert(1e-5, al.f32)
    one = al.convert(1, al.f32)
    zero = al.convert(0, al.f32)

    s_sum = al.make_shared((BLOCK_SIZE,), al.f32)
    s_sum_sq = al.make_shared((BLOCK_SIZE,), al.f32)

    # Phase 1: accumulate partial sum and sum_sq
    local_sum = zero
    local_sum_sq = zero

    for idx in al.range(tid, total_elements, BLOCK_SIZE):
        ch_off = idx // hw_elems
        spatial = idx % hw_elems
        h = spatial // W_conv
        w = spatial % W_conv
        c_out_idx = group_start_ch + ch_off
        val = al.convert(inp[n, c_out_idx, h, w], al.f32)
        local_sum = local_sum + val
        local_sum_sq = local_sum_sq + val * val

    s_sum[tid] = local_sum
    s_sum_sq[tid] = local_sum_sq
    al.syncthreads()

    # Phase 2: thread 0 reduces
    if tid == 0:
        total_sum = s_sum[0]
        total_sum_sq = s_sum_sq[0]
        for i in al.range(1, BLOCK_SIZE):
            total_sum = total_sum + s_sum[i]
            total_sum_sq = total_sum_sq + s_sum_sq[i]

        count = al.convert(total_elements, al.f32)
        mean = total_sum / count
        var = total_sum_sq / count - mean * mean
        inv_std = one / al.sqrt(var + eps)

        s_sum[0] = mean
        s_sum[1] = inv_std
    al.syncthreads()

    mean = s_sum[0]
    inv_std = s_sum[1]

    # Phase 3: normalize, apply affine, and multiply by scale (fused)
    for idx in al.range(tid, total_elements, BLOCK_SIZE):
        ch_off = idx // hw_elems
        spatial = idx % hw_elems
        h = spatial // W_conv
        w = spatial % W_conv
        c_out_idx = group_start_ch + ch_off

        val = al.convert(inp[n, c_out_idx, h, w], al.f32)
        norm_val = (val - mean) * inv_std

        w_val = al.convert(gn_w[c_out_idx], al.f32)
        b_val = al.convert(gn_b[c_out_idx], al.f32)
        s_val = al.convert(scale[c_out_idx], al.f32)

        result = (norm_val * w_val + b_val) * s_val
        out[n, c_out_idx, h, w] = al.convert(result, al.bf16)


# ============================================================
# Kernel 3: MaxPool 4x4 + Clamp (fused)
# ============================================================

@avelang.jit
def maxpool_clamp_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_out: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    pool_size: al.i32,
    H_pool: al.i32,
    W_pool: al.i32,
    in_stride_N: al.i32,
    in_stride_C: al.i32,
    in_stride_H: al.i32,
    out_stride_N: al.i32,
    out_stride_C: al.i32,
    out_stride_H: al.i32,
):
    in_layout = al.make_layout((N, C_out, H_in, W_in), (in_stride_N, in_stride_C, in_stride_H, 1))
    inp = al.make_tensor(input_ptr, al.bf16, in_layout)

    out_layout = al.make_layout((N, C_out, H_pool, W_pool), (out_stride_N, out_stride_C, out_stride_H, 1))
    out = al.make_tensor(output_ptr, al.bf16, out_layout)

    n = al.block_id(0)
    c_out = al.block_id(1)
    h_out = al.block_id(2)
    w_out = al.thread_id(0)

    if w_out < W_pool:
        h_in_start = h_out * pool_size
        w_in_start = w_out * pool_size

        max_val = al.convert(-10000000, al.f32)
        for ph in al.range(pool_size):
            for pw in al.range(pool_size):
                val = al.convert(inp[n, c_out, h_in_start + ph, w_in_start + pw], al.f32)
                if val > max_val:
                    max_val = val

        zero = al.convert(0, al.f32)
        one = al.convert(1, al.f32)
        if max_val < zero:
            max_val = zero
        if max_val > one:
            max_val = one

        out[n, c_out, h_out, w_out] = al.convert(max_val, al.bf16)


# ============================================================
# Host wrappers
# ============================================================

def avelang_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """BF16 convolution with FP32 accumulation. Multi-channel blocking."""
    N, C_in, H, W = x.shape
    C_out, _, KH, KW = weight.shape
    H_out = H - KH + 1
    W_out = W - KW + 1

    x_bf16 = x.contiguous().to(torch.bfloat16)
    w_bf16 = weight.contiguous().to(torch.bfloat16)
    b_bf16 = bias.contiguous().to(torch.bfloat16)
    out = torch.empty(N, C_out, H_out, W_out, dtype=torch.bfloat16, device=x.device)

    in_stride_N = C_in * H * W
    in_stride_C = H * W
    in_stride_H = W
    w_stride_CO = C_in * KH * KW
    w_stride_CI = KH * KW
    w_stride_KH = KW
    out_stride_N = C_out * H_out * W_out
    out_stride_C = H_out * W_out
    out_stride_H = W_out

    BLOCK_C = 16
    grid = (N, (C_out + BLOCK_C - 1) // BLOCK_C, H_out)
    block = (W_out, 1, 1)

    conv2d_kernel[lambda: (grid, block)](
        x_bf16, w_bf16, b_bf16, out,
        N, C_in, C_out, H, W, KH, KW, H_out, W_out,
        in_stride_N, in_stride_C, in_stride_H,
        w_stride_CO, w_stride_CI, w_stride_KH,
        out_stride_N, out_stride_C, out_stride_H,
        BLOCK_C,
    )
    return out


def avelang_group_norm_scale(
    x: torch.Tensor,
    gn_weight: torch.Tensor,
    gn_bias: torch.Tensor,
    scale: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """Fused GroupNorm (training mode) + channel-wise scale multiply."""
    N, C_out, H_conv, W_conv = x.shape
    C_per_group = C_out // num_groups

    x_bf16 = x.contiguous().to(torch.bfloat16)
    gn_w_bf16 = gn_weight.contiguous().to(torch.bfloat16)
    gn_b_bf16 = gn_bias.contiguous().to(torch.bfloat16)
    scale_flat = scale.contiguous().view(-1).to(torch.bfloat16)
    out = torch.empty_like(x_bf16)

    in_stride_N = C_out * H_conv * W_conv
    in_stride_C = H_conv * W_conv
    in_stride_H = W_conv
    out_stride_N = in_stride_N
    out_stride_C = in_stride_C
    out_stride_H = in_stride_H

    BLOCK_SIZE = 256
    grid = (N, num_groups, 1)
    block = (BLOCK_SIZE, 1, 1)

    group_norm_scale_kernel[lambda: (grid, block)](
        x_bf16, gn_w_bf16, gn_b_bf16, scale_flat, out,
        N, C_out, C_per_group, H_conv, W_conv,
        in_stride_N, in_stride_C, in_stride_H,
        out_stride_N, out_stride_C, out_stride_H,
        BLOCK_SIZE,
    )
    return out


def avelang_maxpool_clamp(
    x: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """MaxPool + Clamp fused into one kernel. Clamp fixed at [0, 1]."""
    N, C_out, H_in, W_in = x.shape
    H_pool = (H_in - pool_size) // pool_size + 1
    W_pool = (W_in - pool_size) // pool_size + 1

    x_bf16 = x.contiguous().to(torch.bfloat16)
    out = torch.empty(N, C_out, H_pool, W_pool, dtype=torch.bfloat16, device=x.device)

    in_stride_N = C_out * H_in * W_in
    in_stride_C = H_in * W_in
    in_stride_H = W_in
    out_stride_N = C_out * H_pool * W_pool
    out_stride_C = H_pool * W_pool
    out_stride_H = W_pool

    grid = (N, C_out, H_pool)
    block = (W_pool, 1, 1)

    maxpool_clamp_kernel[lambda: (grid, block)](
        x_bf16, out,
        N, C_out, H_in, W_in, pool_size, H_pool, W_pool,
        in_stride_N, in_stride_C, in_stride_H,
        out_stride_N, out_stride_C, out_stride_H,
    )
    return out


# ============================================================
# ModelNew entrypoint
# ============================================================

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups,
                 scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.num_groups = num_groups

    def forward(self, x):
        x = x.contiguous()

        x = avelang_conv2d(x, self.conv.weight, self.conv.bias)
        x = avelang_group_norm_scale(x, self.group_norm.weight, self.group_norm.bias,
                                     self.scale, self.num_groups)
        x = avelang_maxpool_clamp(x, self.maxpool_kernel_size)

        return x
