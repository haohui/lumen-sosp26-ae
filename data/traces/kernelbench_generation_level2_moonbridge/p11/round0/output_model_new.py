import torch
import torch.nn as nn
import torch.nn.functional as F
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256

EPSILON: al.constexpr = 1e-5


# ---------------------------------------------------------------------------
# Kernel 1: BatchNorm eval + Tanh + MaxPool (fused, using running stats)
# ---------------------------------------------------------------------------
@avelang.jit
def batchnorm_tanh_maxpool_kernel(
    x_ptr: al.Pointer(al.bf16),
    running_mean_ptr: al.Pointer(al.bf16),
    running_var_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H_IN: al.i32,
    W_IN: al.i32,
    H_OUT: al.i32,
    W_OUT: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    gid = bid * BLOCK_SIZE + tid
    total_outputs = B * C * H_OUT * W_OUT

    if gid < total_outputs:
        wp = gid % W_OUT
        rem = gid // W_OUT
        hp = rem % H_OUT
        rem = rem // H_OUT
        c = rem % C
        b = rem // C

        layout_x = al.make_layout(
            (B, C, H_IN, W_IN),
            (C * H_IN * W_IN, H_IN * W_IN, W_IN, 1),
        )
        x = al.make_tensor(x_ptr, al.bf16, layout_x)

        layout_s = al.make_layout((C,), (1,))
        rm_t = al.make_tensor(running_mean_ptr, al.bf16, layout_s)
        rv_t = al.make_tensor(running_var_ptr, al.bf16, layout_s)

        layout_w = al.make_layout((C,), (1,))
        w_t = al.make_tensor(weight_ptr, al.bf16, layout_w)
        b_t = al.make_tensor(bias_ptr, al.bf16, layout_w)

        mean_c = al.convert(rm_t[c], al.f32)
        var_c = al.convert(rv_t[c], al.f32)
        eps = al.convert(EPSILON, al.f32)
        inv_std = al.convert(1.0, al.f32) / al.sqrt(var_c + eps)

        w_scale = al.convert(w_t[c], al.f32)
        b_shift = al.convert(b_t[c], al.f32)

        h0 = hp * 2
        h1 = hp * 2 + 1
        w0 = wp * 2
        w1 = wp * 2 + 1

        v00 = al.convert(x[b, c, h0, w0], al.f32)
        n00 = (v00 - mean_c) * inv_std
        n00 = n00 * w_scale + b_shift
        t00 = al.tanh(n00)
        best = t00

        v01 = al.convert(x[b, c, h0, w1], al.f32)
        n01 = (v01 - mean_c) * inv_std
        n01 = n01 * w_scale + b_shift
        t01 = al.tanh(n01)
        if t01 > best:
            best = t01

        v10 = al.convert(x[b, c, h1, w0], al.f32)
        n10 = (v10 - mean_c) * inv_std
        n10 = n10 * w_scale + b_shift
        t10 = al.tanh(n10)
        if t10 > best:
            best = t10

        v11 = al.convert(x[b, c, h1, w1], al.f32)
        n11 = (v11 - mean_c) * inv_std
        n11 = n11 * w_scale + b_shift
        t11 = al.tanh(n11)
        if t11 > best:
            best = t11

        layout_out = al.make_layout(
            (B, C, H_OUT, W_OUT),
            (C * H_OUT * W_OUT, H_OUT * W_OUT, W_OUT, 1),
        )
        o = al.make_tensor(out_ptr, al.bf16, layout_out)
        o[b, c, hp, wp] = al.convert(best, al.bf16)


# ---------------------------------------------------------------------------
# Kernel 2: GroupNorm reduce (per batch-item, per group)
# ---------------------------------------------------------------------------
@avelang.jit
def groupnorm_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    var_ptr: al.Pointer(al.f32),
    B: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    NUM_GROUPS: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    total_norms = B * NUM_GROUPS
    if bid < total_norms:
        b = bid // NUM_GROUPS
        g = bid - b * NUM_GROUPS

        C_per_group = C // NUM_GROUPS
        N = C_per_group * H * W

        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        layout_x = al.make_layout(
            (B, C, H, W),
            (C * H * W, H * W, W, 1),
        )
        x = al.make_tensor(x_ptr, al.bf16, layout_x)

        local_sum = al.convert(0.0, al.f32)
        local_sq = al.convert(0.0, al.f32)

        c_start = g * C_per_group
        stride = BLOCK_SIZE

        for idx in al.range(tid, N, stride):
            c_off = idx // (H * W)
            rem = idx - c_off * (H * W)
            h_idx = rem // W
            w_idx = rem - h_idx * W

            c_idx = c_start + c_off
            val = al.convert(x[b, c_idx, h_idx, w_idx], al.f32)
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
            N_f32 = al.convert(N, al.f32)
            layout_out = al.make_layout((total_norms,), (1,))
            m = al.make_tensor(mean_ptr, al.f32, layout_out)
            v = al.make_tensor(var_ptr, al.f32, layout_out)

            total_sum = smem_sum[0]
            total_sq = smem_sq[0]
            mean_val = total_sum / N_f32
            var_val = total_sq / N_f32 - mean_val * mean_val

            m[bid] = mean_val
            v[bid] = var_val


# ---------------------------------------------------------------------------
# Kernel 3: GroupNorm apply
# ---------------------------------------------------------------------------
@avelang.jit
def groupnorm_apply_kernel(
    x_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    var_ptr: al.Pointer(al.f32),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    NUM_GROUPS: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    gid = bid * BLOCK_SIZE + tid
    total_outputs = B * C * H * W

    if gid < total_outputs:
        w_idx = gid % W
        rem = gid // W
        h_idx = rem % H
        rem = rem // H
        c = rem % C
        b = rem // C

        C_per_group = C // NUM_GROUPS
        g = c // C_per_group

        stat_idx = b * NUM_GROUPS + g

        layout_x = al.make_layout(
            (B, C, H, W),
            (C * H * W, H * W, W, 1),
        )
        x = al.make_tensor(x_ptr, al.bf16, layout_x)

        total_norms = B * NUM_GROUPS
        layout_s = al.make_layout((total_norms,), (1,))
        m_t = al.make_tensor(mean_ptr, al.f32, layout_s)
        v_t = al.make_tensor(var_ptr, al.f32, layout_s)

        layout_w = al.make_layout((C,), (1,))
        w_t = al.make_tensor(weight_ptr, al.bf16, layout_w)
        b_t = al.make_tensor(bias_ptr, al.bf16, layout_w)

        mean_val = m_t[stat_idx]
        var_val = v_t[stat_idx]
        eps = al.convert(EPSILON, al.f32)
        inv_std = al.convert(1.0, al.f32) / al.sqrt(var_val + eps)

        w_scale = al.convert(w_t[c], al.f32)
        b_shift = al.convert(b_t[c], al.f32)

        x_val = al.convert(x[b, c, h_idx, w_idx], al.f32)
        norm_val = (x_val - mean_val) * inv_std
        result = norm_val * w_scale + b_shift

        layout_out = al.make_layout(
            (B, C, H, W),
            (C * H * W, H * W, W, 1),
        )
        o = al.make_tensor(out_ptr, al.bf16, layout_out)
        o[b, c, h_idx, w_idx] = al.convert(result, al.bf16)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    return t.contiguous().to(dtype=torch.bfloat16)


def _run_model(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
    gn_weight: torch.Tensor,
    gn_bias: torch.Tensor,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    num_groups: int,
) -> torch.Tensor:
    B = x.shape[0]
    IC = x.shape[1]
    H_IN = x.shape[2]
    W_IN = x.shape[3]
    OC = out_channels
    K = kernel_size
    STRIDE = stride
    PAD = padding

    H_OUT = (H_IN - 1) * STRIDE - 2 * PAD + K
    W_OUT = (W_IN - 1) * STRIDE - 2 * PAD + K
    H_POOL = H_OUT // 2
    W_POOL = W_OUT // 2

    x_bf16 = _to_bf16_contiguous(x)
    cw_bf16 = _to_bf16_contiguous(conv_weight)
    cb_bf16 = _to_bf16_contiguous(conv_bias)
    bnw_bf16 = _to_bf16_contiguous(bn_weight)
    bnb_bf16 = _to_bf16_contiguous(bn_bias)
    bnrm_bf16 = _to_bf16_contiguous(bn_running_mean)
    bnrv_bf16 = _to_bf16_contiguous(bn_running_var)
    gnw_bf16 = _to_bf16_contiguous(gn_weight)
    gnb_bf16 = _to_bf16_contiguous(gn_bias)

    # Phase 1: ConvTranspose2d via PyTorch functional API
    conv_out = F.conv_transpose2d(
        x_bf16, cw_bf16, cb_bf16,
        stride=STRIDE, padding=PAD, groups=1,
    )

    # Phase 2: BatchNorm (eval mode) + Tanh + MaxPool (fused)
    pooled = torch.empty(
        (B, OC, H_POOL, W_POOL), dtype=torch.bfloat16, device=x.device
    )
    total_pool = B * OC * H_POOL * W_POOL
    grid_pool = ((total_pool + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
    batchnorm_tanh_maxpool_kernel[lambda: (grid_pool, (BLOCK_SIZE, 1, 1))](
        conv_out,
        bnrm_bf16,
        bnrv_bf16,
        bnw_bf16,
        bnb_bf16,
        pooled,
        B,
        OC,
        H_OUT,
        W_OUT,
        H_POOL,
        W_POOL,
    )

    # Phase 3: GroupNorm reduce
    total_gn_norms = B * num_groups
    gn_mean = torch.empty((total_gn_norms,), dtype=torch.float32, device=x.device)
    gn_var = torch.empty((total_gn_norms,), dtype=torch.float32, device=x.device)
    groupnorm_reduce_kernel[lambda: ((total_gn_norms, 1, 1), (BLOCK_SIZE, 1, 1))](
        pooled, gn_mean, gn_var, B, OC, H_POOL, W_POOL, num_groups
    )

    # Phase 4: GroupNorm apply
    out = torch.empty(
        (B, OC, H_POOL, W_POOL), dtype=torch.bfloat16, device=x.device
    )
    total_gn_out = B * OC * H_POOL * W_POOL
    grid_gn_apply = ((total_gn_out + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
    groupnorm_apply_kernel[lambda: (grid_gn_apply, (BLOCK_SIZE, 1, 1))](
        pooled,
        gn_mean,
        gn_var,
        gnw_bf16,
        gnb_bf16,
        out,
        B,
        OC,
        H_POOL,
        W_POOL,
        num_groups,
    )

    return out


class ModelNew(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.group_norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=out_channels,
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_groups = num_groups
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return _run_model(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.group_norm.weight,
            self.group_norm.bias,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.num_groups,
        )


batch_size = 512
in_channels = 64
out_channels = 128
height_in = 32
width_in = 32
kernel_size = 5
stride = 1
padding = 1
groups = 8
num_groups = 8


def get_inputs():
    return [torch.rand(batch_size, in_channels, height_in, width_in)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, num_groups]
