import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ============================================================
# Helper: HardSwish on f32 scalar
# ============================================================

@avelang.jit
def hardswish_f32(x: al.f32) -> al.f32:
    zero = al.convert(0.0, al.f32)
    six = al.convert(6.0, al.f32)
    three = al.convert(3.0, al.f32)
    val = x + three
    if val < zero:
        val = zero
    if val > six:
        val = six
    return x * val / six


# ============================================================
# Kernel 1: Conv2d 3x3 direct convolution (BF16 in, BF16 out, FP32 acc)
# ============================================================

@avelang.jit
def conv2d_3x3_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    C_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.i32,
    BLOCK_SIZE: al.constexpr,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    idx = bid * BLOCK_SIZE + tid
    total = N * C_out * H_out * W_out

    x_flat = N * C_in * H_in * W_in
    x_layout = al.make_layout((x_flat,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    w_flat = C_out * C_in * K * K
    w_layout = al.make_layout((w_flat,), (1,))
    w_ten = al.make_tensor(w_ptr, al.bf16, w_layout)

    b_layout = al.make_layout((C_out,), (1,))
    b_ten = al.make_tensor(b_ptr, al.bf16, b_layout)

    out_layout = al.make_layout((total,), (1,))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    if idx < total:
        hw_out = H_out * W_out
        c_hw_out = C_out * hw_out

        n = idx // c_hw_out
        rem = idx % c_hw_out
        c_out = rem // hw_out
        rem = rem % hw_out
        h = rem // W_out
        w = rem % W_out

        c_in_hw = C_in * H_in * W_in
        c_in_kk = C_in * K * K
        n_c_in_hw = n * c_in_hw
        c_out_c_in_kk = c_out * c_in_kk

        acc = al.convert(0.0, al.f32)
        for ki in al.range(K):
            h_in = h + ki
            h_in_w = h_in * W_in
            ki_k = ki * K
            for kj in al.range(K):
                w_in_flat = h_in_w + w + kj
                w_base = c_out_c_in_kk + ki_k + kj
                for ci in al.range(C_in):
                    x_flat_idx = n_c_in_hw + ci * (H_in * W_in) + w_in_flat
                    w_flat_idx = w_base + ci * (K * K)
                    x_val = al.convert(x[x_flat_idx], al.f32)
                    w_val = al.convert(w_ten[w_flat_idx], al.f32)
                    acc = acc + x_val * w_val

        acc = acc + al.convert(b_ten[c_out], al.f32)
        out[idx] = al.convert(acc, al.bf16)


# ============================================================
# Kernel 2: GroupNorm stats reduction (sum and sum of squares)
#           One block per (N, group) pair
# ============================================================

@avelang.jit
def group_norm_stats_kernel(
    x_ptr: al.Pointer(al.bf16),
    sum_ptr: al.Pointer(al.f32),
    sum_sq_ptr: al.Pointer(al.f32),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    groups: al.i32,
    BLOCK_SIZE: al.constexpr,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    n = bid // groups
    g = bid % groups

    c_per_group = C // groups
    c_start = g * c_per_group
    num_elements = c_per_group * H * W

    x_flat = N * C * H * W
    x_layout = al.make_layout((x_flat,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    hw = H * W
    chw = C * hw
    n_chw = n * chw

    local_sum = al.convert(0.0, al.f32)
    local_sum_sq = al.convert(0.0, al.f32)

    for i in al.range(tid, num_elements, BLOCK_SIZE):
        c_local = i // hw
        hw_idx = i % hw
        h_idx = hw_idx // W
        w_idx = hw_idx % W
        c_idx = c_start + c_local
        flat_idx = n_chw + c_idx * hw + h_idx * W + w_idx
        val = al.convert(x[flat_idx], al.f32)
        local_sum = local_sum + val
        local_sum_sq = local_sum_sq + val * val

    smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
    smem_sum_sq = al.make_shared((BLOCK_SIZE,), al.f32)
    smem_sum[tid] = local_sum
    smem_sum_sq[tid] = local_sum_sq
    al.syncthreads()

    if tid == 0:
        total_sum = al.convert(0.0, al.f32)
        total_sum_sq = al.convert(0.0, al.f32)
        for i in al.range(BLOCK_SIZE):
            total_sum = total_sum + smem_sum[i]
            total_sum_sq = total_sum_sq + smem_sum_sq[i]

        stats_flat = N * groups
        sum_layout = al.make_layout((stats_flat,), (1,))
        sums = al.make_tensor(sum_ptr, al.f32, sum_layout)
        sums[bid] = total_sum

        sum_sq_layout = al.make_layout((stats_flat,), (1,))
        sum_sqs = al.make_tensor(sum_sq_ptr, al.f32, sum_sq_layout)
        sum_sqs[bid] = total_sum_sq


# ============================================================
# Kernel 3: GroupNorm apply + Tanh + HardSwish
# ============================================================

@avelang.jit
def group_norm_apply_tanh_hardswish_kernel(
    x_ptr: al.Pointer(al.bf16),
    sum_ptr: al.Pointer(al.f32),
    sum_sq_ptr: al.Pointer(al.f32),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    groups: al.i32,
    BLOCK_SIZE: al.constexpr,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    idx = bid * BLOCK_SIZE + tid
    total = N * C * H * W

    x_layout = al.make_layout((total,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    stats_flat = N * groups
    sum_layout = al.make_layout((stats_flat,), (1,))
    sums = al.make_tensor(sum_ptr, al.f32, sum_layout)
    sum_sqs = al.make_tensor(sum_sq_ptr, al.f32, sum_layout)

    gamma_layout = al.make_layout((C,), (1,))
    gamma = al.make_tensor(gamma_ptr, al.bf16, gamma_layout)
    beta_layout = al.make_layout((C,), (1,))
    beta = al.make_tensor(beta_ptr, al.bf16, beta_layout)

    out_layout = al.make_layout((total,), (1,))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    if idx < total:
        c_per_group = C // groups
        hw = H * W
        chw = C * hw

        n = idx // chw
        rem = idx % chw
        c = rem // hw

        g = c // c_per_group
        num_el = al.convert(c_per_group * hw, al.f32)

        stat_idx = n * groups + g
        mean = sums[stat_idx] / num_el
        mean_sq = sum_sqs[stat_idx] / num_el
        var = mean_sq - mean * mean
        inv_std = al.convert(1.0, al.f32) / al.sqrt(var + al.convert(1e-5, al.f32))

        x_val = al.convert(x[idx], al.f32)
        x_hat = (x_val - mean) * inv_std

        g_val = al.convert(gamma[c], al.f32)
        b_val = al.convert(beta[c], al.f32)
        normed = g_val * x_hat + b_val

        t = al.tanh(normed)
        hs = hardswish_f32(t)
        out[idx] = al.convert(hs, al.bf16)


# ============================================================
# Kernel 4: Residual add + LogSumExp over dimension 1 (channels)
# ============================================================

@avelang.jit
def residual_logsumexp_kernel(
    conv_ptr: al.Pointer(al.bf16),
    processed_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    BLOCK_SIZE: al.constexpr,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    idx = bid * BLOCK_SIZE + tid
    total = N * H * W

    conv_flat = N * C * H * W
    conv_layout = al.make_layout((conv_flat,), (1,))
    conv = al.make_tensor(conv_ptr, al.bf16, conv_layout)

    proc_layout = al.make_layout((conv_flat,), (1,))
    proc = al.make_tensor(processed_ptr, al.bf16, proc_layout)

    out_layout = al.make_layout((total,), (1,))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    if idx < total:
        hw = H * W
        chw = C * hw

        n = idx // hw
        hw_idx = idx % hw
        h = hw_idx // W
        w = hw_idx % W

        n_chw = n * chw
        base = n_chw + h * W + w

        # First pass: find max for numerical stability
        flat_idx = base
        res_val = al.convert(conv[flat_idx], al.f32) + al.convert(proc[flat_idx], al.f32)
        max_val = res_val

        for c in al.range(1, C):
            flat_idx = base + c * hw
            res_val = al.convert(conv[flat_idx], al.f32) + al.convert(proc[flat_idx], al.f32)
            if res_val > max_val:
                max_val = res_val

        # Second pass: sum exp(x - max)
        sum_exp = al.convert(0.0, al.f32)
        for c in al.range(C):
            flat_idx = base + c * hw
            res_val = al.convert(conv[flat_idx], al.f32) + al.convert(proc[flat_idx], al.f32)
            sum_exp = sum_exp + al.exp(res_val - max_val)

        lse = max_val + al.log(sum_exp)
        out[idx] = al.convert(lse, al.bf16)


# ============================================================
# ModelNew: host wrapper that launches AveLang kernels
# ============================================================

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.eps = eps
        # Instantiate PyTorch layers to obtain learnable weights
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)

    def forward(self, x):
        N, C_in, H_in, W_in = x.shape
        C_out = self.out_channels
        K = self.kernel_size
        H_out = H_in - K + 1
        W_out = W_in - K + 1
        groups = self.groups

        # Convert weights to BF16 contiguous
        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = self.conv.weight.data.to(torch.bfloat16).contiguous()
        b_bf16 = self.conv.bias.data.to(torch.bfloat16).contiguous()
        gn_w = self.group_norm.weight.data.to(torch.bfloat16).contiguous()
        gn_b = self.group_norm.bias.data.to(torch.bfloat16).contiguous()

        # Intermediate buffers
        conv_out = torch.empty(N, C_out, H_out, W_out, dtype=torch.bfloat16, device=x.device)
        processed = torch.empty(N, C_out, H_out, W_out, dtype=torch.bfloat16, device=x.device)
        gn_sum = torch.empty(N * groups, dtype=torch.float32, device=x.device)
        gn_sum_sq = torch.empty(N * groups, dtype=torch.float32, device=x.device)
        output = torch.empty(N, 1, H_out, W_out, dtype=torch.bfloat16, device=x.device)

        BLOCK = 256

        # --- Launch Conv2d ---
        total_conv = N * C_out * H_out * W_out
        grid_conv = (total_conv + BLOCK - 1) // BLOCK
        conv2d_3x3_kernel[lambda: ((grid_conv, 1, 1), (BLOCK, 1, 1))](
            x_bf16, w_bf16, b_bf16, conv_out,
            N, C_in, H_in, W_in, C_out, H_out, W_out, K,
            BLOCK,
        )

        # --- Launch GroupNorm stats ---
        grid_stats = N * groups
        group_norm_stats_kernel[lambda: ((grid_stats, 1, 1), (BLOCK, 1, 1))](
            conv_out, gn_sum, gn_sum_sq,
            N, C_out, H_out, W_out, groups,
            BLOCK,
        )

        # --- Launch GroupNorm apply + Tanh + HardSwish ---
        total_gn = N * C_out * H_out * W_out
        grid_gn = (total_gn + BLOCK - 1) // BLOCK
        group_norm_apply_tanh_hardswish_kernel[lambda: ((grid_gn, 1, 1), (BLOCK, 1, 1))](
            conv_out, gn_sum, gn_sum_sq, gn_w, gn_b, processed, N, C_out, H_out, W_out, groups,
            BLOCK,
        )

        # --- Launch Residual + LogSumExp ---
        total_lse = N * H_out * W_out
        grid_lse = (total_lse + BLOCK - 1) // BLOCK
        residual_logsumexp_kernel[lambda: ((grid_lse, 1, 1), (BLOCK, 1, 1))](
            conv_out, processed, output,
            N, C_out, H_out, W_out,
            BLOCK,
        )

        return output
