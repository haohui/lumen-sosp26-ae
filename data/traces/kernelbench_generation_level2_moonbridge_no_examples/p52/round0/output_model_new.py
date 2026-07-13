import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_activation_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    layout_x = al.make_layout(
        (N, C_in, H, W),
        (C_in * H * W, H * W, W, al.convert(1, al.i32)),
    )
    layout_w = al.make_layout(
        (C_out, C_in, al.convert(3, al.i32), al.convert(3, al.i32)),
        (C_in * al.convert(9, al.i32), al.convert(9, al.i32),
         al.convert(3, al.i32), al.convert(1, al.i32)),
    )
    layout_b = al.make_layout((C_out,), (al.convert(1, al.i32),))
    layout_out = al.make_layout(
        (N, C_out, H_out, W_out),
        (C_out * H_out * W_out, H_out * W_out, W_out, al.convert(1, al.i32)),
    )

    x = al.make_tensor(x_ptr, al.bf16, layout_x)
    w = al.make_tensor(w_ptr, al.bf16, layout_w)
    b = al.make_tensor(b_ptr, al.bf16, layout_b)
    out = al.make_tensor(out_ptr, al.bf16, layout_out)

    tid = al.thread_id(0)
    bid = al.block_id(0)
    bdim = al.block_dim(0)

    gid = bid * bdim + tid
    total = N * C_out * H_out * W_out

    if gid < total:
        stride_n_idx = C_out * H_out * W_out
        stride_oc_idx = H_out * W_out

        n = gid // stride_n_idx
        rem_n = gid - n * stride_n_idx
        oc = rem_n // stride_oc_idx
        rem_oc = rem_n - oc * stride_oc_idx
        oh = rem_oc // W_out
        ow = rem_oc - oh * W_out

        acc = al.convert(b[oc], al.f32)

        three = al.convert(3, al.i32)
        for ic in al.range(C_in):
            for kh in al.range(three):
                for kw in al.range(three):
                    vx = al.convert(x[n, ic, oh + kh, ow + kw], al.f32)
                    vw = al.convert(w[oc, ic, kh, kw], al.f32)
                    acc = acc + vx * vw

        one = al.convert(1.0, al.f32)
        exp_val = al.exp(acc)
        sp_val = al.log(one + exp_val)
        th_val = al.tanh(sp_val)
        result = acc * th_val

        out[n, oc, oh, ow] = al.convert(result, al.bf16)


@avelang.jit
def bn_compute_stats_kernel(
    x_ptr: al.Pointer(al.bf16),
    sum_ptr: al.Pointer(al.f32),
    sum_sq_ptr: al.Pointer(al.f32),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    BLOCK_SIZE: al.constexpr,
):
    tid = al.thread_id(0)
    c = al.block_id(0)

    if c < C:
        spatial_total = N * H * W

        layout_x = al.make_layout(
            (N, C, H, W),
            (C * H * W, H * W, W, al.convert(1, al.i32)),
        )
        layout_s = al.make_layout((C,), (al.convert(1, al.i32),))

        x = al.make_tensor(x_ptr, al.bf16, layout_x)
        sum_t = al.make_tensor(sum_ptr, al.f32, layout_s)
        sum_sq_t = al.make_tensor(sum_sq_ptr, al.f32, layout_s)

        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        local_sum = al.convert(0.0, al.f32)
        local_sq = al.convert(0.0, al.f32)

        hw = H * W
        step = al.convert(256, al.i32)
        for idx in al.range(tid, spatial_total, step):
            n = idx // hw
            rem_n = idx - n * hw
            h = rem_n // W
            w = rem_n - h * W

            val = al.convert(x[n, c, h, w], al.f32)
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
        al.syncthreads()

        if tid == 0:
            sum_t[c] = smem_sum[0]
            sum_sq_t[c] = smem_sq[0]


@avelang.jit
def bn_normalize_kernel(
    x_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    var_ptr: al.Pointer(al.f32),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
):
    layout_x = al.make_layout(
        (N, C, H, W),
        (C * H * W, H * W, W, al.convert(1, al.i32)),
    )
    layout_1d = al.make_layout((C,), (al.convert(1, al.i32),))
    layout_out = al.make_layout(
        (N, C, H, W),
        (C * H * W, H * W, W, al.convert(1, al.i32)),
    )

    x = al.make_tensor(x_ptr, al.bf16, layout_x)
    mean = al.make_tensor(mean_ptr, al.f32, layout_1d)
    var = al.make_tensor(var_ptr, al.f32, layout_1d)
    gamma = al.make_tensor(gamma_ptr, al.bf16, layout_1d)
    beta = al.make_tensor(beta_ptr, al.bf16, layout_1d)
    out = al.make_tensor(out_ptr, al.bf16, layout_out)

    tid = al.thread_id(0)
    bid = al.block_id(0)
    bdim = al.block_dim(0)

    gid = bid * bdim + tid
    total = N * C * H * W

    if gid < total:
        stride_c = H * W
        stride_n = C * stride_c

        n = gid // stride_n
        rem_n = gid - n * stride_n
        c = rem_n // stride_c
        rem_c = rem_n - c * stride_c
        h = rem_c // W
        w = rem_c - h * W

        x_val = al.convert(x[n, c, h, w], al.f32)
        m = mean[c]
        v = var[c]
        g = al.convert(gamma[c], al.f32)
        b = al.convert(beta[c], al.f32)

        one = al.convert(1.0, al.f32)
        eps = al.convert(1e-5, al.f32)
        denom = v + eps
        if denom < al.convert(0.0, al.f32):
            denom = eps
        inv_std = one / al.sqrt(denom)
        y = (x_val - m) * inv_std * g + b

        out[n, c, h, w] = al.convert(y, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.momentum = momentum

        self.conv_weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.conv_bias = nn.Parameter(torch.empty(out_channels))
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))

        self.register_buffer("running_mean", torch.zeros(out_channels))
        self.register_buffer("running_var", torch.ones(out_channels))
        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.long)
        )

    def forward(self, x):
        N, C_in, H, W = x.shape
        C_out = self.out_channels
        kH = self.kernel_size
        kW = self.kernel_size
        H_out = H - kH + 1
        W_out = W - kW + 1

        x = x.contiguous()
        w = self.conv_weight.contiguous()
        b = self.conv_bias.contiguous()

        conv_out = torch.empty(
            N, C_out, H_out, W_out, dtype=torch.bfloat16, device=x.device
        )

        total_elems = N * C_out * H_out * W_out
        block_size = 256
        grid = (total_elems + block_size - 1) // block_size

        conv_activation_kernel[lambda: ((grid, 1, 1), (block_size, 1, 1))](
            x, w, b, conv_out,
            N, C_in, C_out, H, W, H_out, W_out,
        )

        C = C_out
        H, W = H_out, W_out

        sum_buf = torch.empty(C, dtype=torch.float32, device=conv_out.device)
        sum_sq_buf = torch.empty(C, dtype=torch.float32, device=conv_out.device)

        bn_compute_stats_kernel[lambda: ((C, 1, 1), (256, 1, 1))](
            conv_out, sum_buf, sum_sq_buf,
            N, C, H, W,
            BLOCK_SIZE=256,
        )

        count = float(N * H * W)
        mean_f32 = sum_buf / count
        var_f32 = torch.clamp(sum_sq_buf / count - mean_f32 * mean_f32, min=0.0)

        bn_out = torch.empty(N, C, H, W, dtype=torch.bfloat16, device=conv_out.device)

        bn_total = N * C * H * W
        bn_grid = (bn_total + 255) // 256

        bn_gamma = self.bn_weight.contiguous()
        bn_beta = self.bn_bias.contiguous()

        bn_normalize_kernel[lambda: ((bn_grid, 1, 1), (256, 1, 1))](
            conv_out, mean_f32, var_f32,
            bn_gamma, bn_beta, bn_out,
            N, C, H, W,
        )

        return bn_out
