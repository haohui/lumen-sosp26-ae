import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_H = 16
TILE_W = 16
BLOCK_SIZE = 256


# ---------------------------------------------------------------------------
# Kernel 1: bias + scale + sigmoid  (bf16 arithmetic, matching reference)
# ---------------------------------------------------------------------------


@avelang.jit
def bias_scale_sigmoid_kernel(
    x_ptr: al.Pointer(al.bf16),
    extra_bias_ptr: al.Pointer(al.bf16),
    extra_scale_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
):
    stride_N = C * H * W
    stride_C = H * W
    stride_H = W
    stride_W = 1

    x_layout = al.make_layout((N, C, H, W), (stride_N, stride_C, stride_H, stride_W))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    out_layout = al.make_layout((N, C, H, W), (stride_N, stride_C, stride_H, stride_W))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    ch_layout = al.make_layout((C,), (1,))
    extra_bias = al.make_tensor(extra_bias_ptr, al.bf16, ch_layout)
    extra_scale = al.make_tensor(extra_scale_ptr, al.bf16, ch_layout)

    n = al.block_id(0)
    c = al.block_id(1)
    tile_row = al.block_id(2)

    tid = al.thread_id(0)
    row = tile_row * TILE_H + tid

    if row < H:
        for w in al.range(W):
            val = x[n, c, row, w]
            # Match reference bf16 order: (val + bias) * scale → bf16, then sigmoid in fp32
            t1 = al.convert(val + extra_bias[c], al.bf16)
            t2 = al.convert(t1 * extra_scale[c], al.bf16)
            scaled = al.convert(t2, al.bf16)
            result = al.convert(1.0, al.bf16) / (al.convert(1.0, al.f32) + al.exp(al.convert(0.0, al.f32) - scaled))
            out[n, c, row, w] = al.convert(result, al.bf16)


# ---------------------------------------------------------------------------
# Kernel 2: GroupNorm statistics
# ---------------------------------------------------------------------------


@avelang.jit
def group_norm_stats_kernel(
    x_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.bf16),
    var_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    G: al.i32,
):
    n = al.block_id(0)
    g = al.block_id(1)

    C_g = C // G
    HW = H * W
    total_elements = C_g * HW

    stride_N = C * H * W
    stride_C = H * W
    stride_H = W
    stride_W = 1

    x_layout = al.make_layout((N, C, H, W), (stride_N, stride_C, stride_H, stride_W))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    mean_out_layout = al.make_layout((N, G), (G, 1))
    mean_out = al.make_tensor(mean_ptr, al.f32, mean_out_layout)
    var_out_layout = al.make_layout((N, G), (G, 1))
    var_out = al.make_tensor(var_ptr, al.f32, var_out_layout)

    tid = al.thread_id(0)
    chunk_size = (total_elements + 255) // 256

    sum_val = al.convert(0.0, al.f32)
    sumsq_val = al.convert(0.0, al.f32)

    for i in al.range(chunk_size):
        linear = tid * chunk_size + i
        if linear < total_elements:
            c_off = linear // HW
            spat = linear % HW
            h = spat // W
            w = spat % W
            c = g * C_g + c_off
            val = al.convert(x[n, c, h, w], al.f32)
            sum_val = sum_val + val
            sumsq_val = sumsq_val + val * val

    smem_sum = al.make_shared((256,), al.f32)
    smem_sumsq = al.make_shared((256,), al.f32)

    sum_val = sum_val + al.shuffle_down(sum_val, 32, 64)
    sum_val = sum_val + al.shuffle_down(sum_val, 16, 64)
    sum_val = sum_val + al.shuffle_down(sum_val, 8, 64)
    sum_val = sum_val + al.shuffle_down(sum_val, 4, 64)
    sum_val = sum_val + al.shuffle_down(sum_val, 2, 64)
    sum_val = sum_val + al.shuffle_down(sum_val, 1, 64)

    sumsq_val = sumsq_val + al.shuffle_down(sumsq_val, 32, 64)
    sumsq_val = sumsq_val + al.shuffle_down(sumsq_val, 16, 64)
    sumsq_val = sumsq_val + al.shuffle_down(sumsq_val, 8, 64)
    sumsq_val = sumsq_val + al.shuffle_down(sumsq_val, 4, 64)
    sumsq_val = sumsq_val + al.shuffle_down(sumsq_val, 2, 64)
    sumsq_val = sumsq_val + al.shuffle_down(sumsq_val, 1, 64)

    lane_id = tid % 64
    warp_id = tid // 64

    if lane_id == 0:
        smem_sum[warp_id] = sum_val
        smem_sumsq[warp_id] = sumsq_val
    al.syncthreads()

    if tid < 4:
        w_sum = smem_sum[tid]
        w_sumsq = smem_sumsq[tid]

        w_sum = w_sum + al.shuffle_down(w_sum, 2, 4)
        w_sum = w_sum + al.shuffle_down(w_sum, 1, 4)
        w_sumsq = w_sumsq + al.shuffle_down(w_sumsq, 2, 4)
        w_sumsq = w_sumsq + al.shuffle_down(w_sumsq, 1, 4)

        if tid == 0:
            count = al.convert(total_elements, al.f32)
            zero = al.convert(0.0, al.f32)
            mean_val = w_sum / count
            var_val = w_sumsq / count - mean_val * mean_val
            if var_val < zero:
                var_val = zero
            mean_out[n, g] = mean_val
            var_out[n, g] = var_val


# ---------------------------------------------------------------------------
# Kernel 3: GroupNorm apply
# ---------------------------------------------------------------------------


@avelang.jit
def group_norm_apply_kernel(
    x_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    var_ptr: al.Pointer(al.f32),
    gamma_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    G: al.i32,
    TILE_H: al.constexpr,
    TILE_W: al.constexpr,
):
    C_g = C // G

    stride_N = C * H * W
    stride_C = H * W
    stride_H = W
    stride_W = 1

    x_layout = al.make_layout((N, C, H, W), (stride_N, stride_C, stride_H, stride_W))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    out_layout = al.make_layout((N, C, H, W), (stride_N, stride_C, stride_H, stride_W))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    mean_layout = al.make_layout((N, G), (G, 1))
    mean_t = al.make_tensor(mean_ptr, al.f32, mean_layout)
    var_layout = al.make_layout((N, G), (G, 1))
    var_t = al.make_tensor(var_ptr, al.f32, var_layout)

    glayout = al.make_layout((C,), (1,))
    gamma = al.make_tensor(gamma_ptr, al.f32, glayout)
    beta_t = al.make_tensor(beta_ptr, al.f32, glayout)

    n_g = al.block_id(0)
    n = n_g // G
    g = n_g % G

    tile_row = al.block_id(1)
    tile_col = al.block_id(2)

    tid = al.thread_id(0)
    local_row = tid // TILE_W
    local_col = tid % TILE_W

    out_row = tile_row * TILE_H + local_row
    out_col = tile_col * TILE_W + local_col

    if out_row < H:
        if out_col < W:
            mean_val = mean_t[n, g]
            var_val = var_t[n, g]
            eps = al.convert(1e-5, al.f32)
            one = al.convert(1.0, al.f32)
            inv_std = one / al.sqrt(var_val + eps)

            for c_off in al.range(C_g):
                c = g * C_g + c_off
                val_bf = x[n, c, out_row, out_col]
                mean_bf = al.convert(mean_val, al.bf16)
                inv_std_bf = al.convert(inv_std, al.bf16)
                diff_bf = al.convert(val_bf - mean_bf, al.bf16)
                normed_bf = al.convert(diff_bf * inv_std_bf, al.bf16)
                gam_bf = al.convert(gamma[c], al.bf16)
                bet_bf = al.convert(beta_t[c], al.bf16)
                result_bf = al.convert(normed_bf * gam_bf + bet_bf, al.bf16)
                out[n, c, out_row, out_col] = result_bf


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        assert x.is_cuda, "Input must be on GPU"
        x = x.contiguous()

        N = x.shape[0]
        H = x.shape[2]
        W = x.shape[3]
        OC = self.out_channels
        G = self.num_groups

        H_out = H - self.kernel_size + 1
        W_out = W - self.kernel_size + 1

        # Step 1: Conv2d via PyTorch (bf16) — exact match to reference
        conv_out = self.conv(x.to(torch.bfloat16))

        # Step 2: Bias + Scale + Sigmoid via AveLang kernel (bf16 arithmetic)
        after_sigmoid = torch.empty(N, OC, H_out, W_out, device=x.device, dtype=torch.bfloat16)

        eb_bf16 = self.bias.data.view(OC).to(torch.bfloat16).contiguous()
        es_bf16 = self.scale.data.view(OC).to(torch.bfloat16).contiguous()

        bias_scale_sigmoid_kernel[lambda: ((N, OC, (H_out + TILE_H - 1) // TILE_H), (TILE_H, 1, 1))](
            conv_out, eb_bf16, es_bf16, after_sigmoid,
            N, OC, H_out, W_out,
        )

        # Step 3: GroupNorm statistics
        mean_buf = torch.empty(N, G, device=x.device, dtype=torch.float32)
        var_buf = torch.empty(N, G, device=x.device, dtype=torch.float32)

        group_norm_stats_kernel[lambda: ((N, G, 1), (BLOCK_SIZE, 1, 1))](
            after_sigmoid, mean_buf, var_buf,
            N, OC, H_out, W_out, G,
        )

        # Step 4: GroupNorm apply
        output = torch.empty(N, OC, H_out, W_out, device=x.device, dtype=torch.bfloat16)
        gamma = self.group_norm.weight.data.to(torch.float32).contiguous()
        beta = self.group_norm.bias.data.to(torch.float32).contiguous()

        grid_y_gn = (H_out + TILE_H - 1) // TILE_H
        grid_z_gn = (W_out + TILE_W - 1) // TILE_W

        group_norm_apply_kernel[lambda: ((N * G, grid_y_gn, grid_z_gn), (TILE_H * TILE_W, 1, 1))](
            after_sigmoid, mean_buf, var_buf, gamma, beta, output,
            N, OC, H_out, W_out, G, TILE_H, TILE_W,
        )

        return output.to(torch.bfloat16)
