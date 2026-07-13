import torch
import torch.nn as nn
import avelang
import avelang.language as al

# =============================================================================
# Kernel 1: Spatial reduction – sum input tensor over D, H, W dimensions.
# =============================================================================

@avelang.jit
def spatial_reduce_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.f32),
    B: al.i32,
    IC: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    idx = bid * al.block_dim(0) + tid
    b = idx // IC
    ic = idx % IC

    if b < B:
        in_layout = al.make_layout(
            (B, IC, D, H, W),
            (IC * D * H * W, D * H * W, H * W, W, 1),
        )
        in_tensor = al.make_tensor(input_ptr, al.bf16, in_layout)

        out_layout = al.make_layout((B, IC), (IC, 1))
        out_tensor = al.make_tensor(output_ptr, al.f32, out_layout)

        acc = al.convert(0.0, al.f32)
        for d in al.range(D):
            for h in al.range(H):
                for w in al.range(W):
                    val = al.convert(in_tensor[b, ic, d, h, w], al.f32)
                    acc = acc + val

        out_tensor[b, ic] = acc


# =============================================================================
# Kernel 2: Full 3D transposed convolution (with bias).
# Used only in training mode to compute full conv output for BN stats.
# =============================================================================

@avelang.jit
def conv_transpose3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    K: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    BLK = al.block_dim(0)
    tid = al.thread_id(0)
    bid = al.block_id(0)

    flat_idx = bid * BLK + tid

    total_elems = B * OC * D_out * H_out * W_out
    if flat_idx < total_elems:
        w_pos = flat_idx % W_out
        rem = flat_idx // W_out
        h_pos = rem % H_out
        rem = rem // H_out
        d_pos = rem % D_out
        rem = rem // D_out
        oc = rem % OC
        b = rem // OC

        in_layout = al.make_layout(
            (B, IC, D, H, W),
            (IC * D * H * W, D * H * W, H * W, W, 1),
        )
        in_tensor = al.make_tensor(input_ptr, al.bf16, in_layout)

        K2 = K * K
        K3 = K * K * K
        w_layout = al.make_layout(
            (IC, OC, K, K, K),
            (OC * K3, K3, K2, K, 1),
        )
        w_tensor = al.make_tensor(weight_ptr, al.bf16, w_layout)

        bias_layout = al.make_layout((OC,), (1,))
        b_tensor = al.make_tensor(bias_ptr, al.f32, bias_layout)

        out_layout = al.make_layout(
            (B, OC, D_out, H_out, W_out),
            (OC * D_out * H_out * W_out, D_out * H_out * W_out, H_out * W_out, W_out, 1),
        )
        out_tensor = al.make_tensor(output_ptr, al.bf16, out_layout)

        acc = b_tensor[oc]

        for ic in al.range(IC):
            for kd in al.range(K):
                d_in = d_pos - kd
                if d_in >= 0:
                    if d_in < D:
                        for kh in al.range(K):
                            h_in = h_pos - kh
                            if h_in >= 0:
                                if h_in < H:
                                    for kw in al.range(K):
                                        w_in = w_pos - kw
                                        if w_in >= 0:
                                            if w_in < W:
                                                in_val = al.convert(
                                                    in_tensor[b, ic, d_in, h_in, w_in], al.f32
                                                )
                                                w_val = al.convert(
                                                    w_tensor[ic, oc, kd, kh, kw], al.f32
                                                )
                                                acc = acc + in_val * w_val

        out_tensor[b, oc, d_pos, h_pos, w_pos] = al.convert(acc, al.bf16)


# =============================================================================
# Kernel 3: Per-channel sum and sum-of-squares reduction (training mode only).
# =============================================================================

@avelang.jit
def bn_reduce_kernel(
    conv_out_ptr: al.Pointer(al.bf16),
    sum_out_ptr: al.Pointer(al.f32),
    ssq_out_ptr: al.Pointer(al.f32),
    OC: al.i32,
    N: al.i32,
    stride_oc: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    oc = bid

    conv_layout = al.make_layout((OC, N), (stride_oc, 1))
    conv_tensor = al.make_tensor(conv_out_ptr, al.bf16, conv_layout)

    shared_sum = al.make_shared((256,), al.f32)
    shared_ssq = al.make_shared((256,), al.f32)

    sf = al.convert(2.0, al.f32)

    partial_sum = al.convert(0.0, al.f32)
    partial_ssq = al.convert(0.0, al.f32)

    for idx in al.range(tid, N, 256):
        val = sf * al.convert(conv_tensor[oc, idx], al.f32)
        partial_sum = partial_sum + val
        partial_ssq = partial_ssq + val * val

    shared_sum[tid] = partial_sum
    shared_ssq[tid] = partial_ssq
    al.syncthreads()

    if tid == 0:
        final_sum = shared_sum[0]
        final_ssq = shared_ssq[0]
        for i in al.range(1, 256):
            final_sum = final_sum + shared_sum[i]
            final_ssq = final_ssq + shared_ssq[i]

        sum_layout = al.make_layout((OC,), (1,))
        ssq_layout = al.make_layout((OC,), (1,))
        sum_tensor = al.make_tensor(sum_out_ptr, al.f32, sum_layout)
        ssq_tensor = al.make_tensor(ssq_out_ptr, al.f32, ssq_layout)
        sum_tensor[oc] = final_sum
        ssq_tensor[oc] = final_ssq


# =============================================================================
# Kernel 4a: BN apply + pool – TRAINING mode (uses precomputed batch stats).
# =============================================================================

@avelang.jit
def bn_apply_pool_train_kernel(
    input_spatial_sum_ptr: al.Pointer(al.f32),
    weight_sum_ptr: al.Pointer(al.f32),
    bias_ptr: al.Pointer(al.f32),
    bn_sum_ptr: al.Pointer(al.f32),
    bn_ssq_ptr: al.Pointer(al.f32),
    bn_weight_ptr: al.Pointer(al.f32),
    bn_bias_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    spatial_size: al.i32,
    N: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    flat_idx = bid * al.block_dim(0) + tid
    b = flat_idx // OC
    oc = flat_idx % OC

    if b < B:
        sp_layout = al.make_layout((B, IC), (IC, 1))
        in_sp = al.make_tensor(input_spatial_sum_ptr, al.f32, sp_layout)

        ws_layout = al.make_layout((IC, OC), (OC, 1))
        w_sum = al.make_tensor(weight_sum_ptr, al.f32, ws_layout)

        bias_layout = al.make_layout((OC,), (1,))
        bias = al.make_tensor(bias_ptr, al.f32, bias_layout)

        bn_sum_layout = al.make_layout((OC,), (1,))
        bn_s = al.make_tensor(bn_sum_ptr, al.f32, bn_sum_layout)

        bn_ssq_layout = al.make_layout((OC,), (1,))
        bn_sq = al.make_tensor(bn_ssq_ptr, al.f32, bn_ssq_layout)

        bn_w_layout = al.make_layout((OC,), (1,))
        bn_w = al.make_tensor(bn_weight_ptr, al.f32, bn_w_layout)

        bn_b_layout = al.make_layout((OC,), (1,))
        bn_b = al.make_tensor(bn_bias_ptr, al.f32, bn_b_layout)

        out_layout = al.make_layout((B, OC), (OC, 1))
        out_tensor = al.make_tensor(output_ptr, al.bf16, out_layout)

        sf = al.convert(2.0, al.f32)
        eps_val = al.convert(1e-5, al.f32)

        acc = al.convert(0.0, al.f32)
        for ic in al.range(IC):
            acc = acc + in_sp[b, ic] * w_sum[ic, oc]
        acc = acc + al.convert(spatial_size, al.f32) * bias[oc]

        spatial_mean = sf * acc / al.convert(spatial_size, al.f32)

        n_f32 = al.convert(N, al.f32)
        mean_all = bn_s[oc] / n_f32
        var_all = bn_sq[oc] / n_f32 - mean_all * mean_all

        denom = al.sqrt(var_all + eps_val)
        normalized = (spatial_mean - mean_all) / denom
        result = bn_w[oc] * normalized + bn_b[oc]

        out_tensor[b, oc] = al.convert(result, al.bf16)


# =============================================================================
# Kernel 4b: BN apply + pool – EVAL mode (uses running mean/var).
# =============================================================================

@avelang.jit
def bn_apply_pool_eval_kernel(
    input_spatial_sum_ptr: al.Pointer(al.f32),
    weight_sum_ptr: al.Pointer(al.f32),
    bias_ptr: al.Pointer(al.f32),
    bn_mean_ptr: al.Pointer(al.f32),
    bn_var_ptr: al.Pointer(al.f32),
    bn_weight_ptr: al.Pointer(al.f32),
    bn_bias_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    spatial_size: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    flat_idx = bid * al.block_dim(0) + tid
    b = flat_idx // OC
    oc = flat_idx % OC

    if b < B:
        sp_layout = al.make_layout((B, IC), (IC, 1))
        in_sp = al.make_tensor(input_spatial_sum_ptr, al.f32, sp_layout)

        ws_layout = al.make_layout((IC, OC), (OC, 1))
        w_sum = al.make_tensor(weight_sum_ptr, al.f32, ws_layout)

        bias_layout = al.make_layout((OC,), (1,))
        bias = al.make_tensor(bias_ptr, al.f32, bias_layout)

        rm_layout = al.make_layout((OC,), (1,))
        rm = al.make_tensor(bn_mean_ptr, al.f32, rm_layout)

        rv_layout = al.make_layout((OC,), (1,))
        rv = al.make_tensor(bn_var_ptr, al.f32, rv_layout)

        bn_w_layout = al.make_layout((OC,), (1,))
        bn_w = al.make_tensor(bn_weight_ptr, al.f32, bn_w_layout)

        bn_b_layout = al.make_layout((OC,), (1,))
        bn_b = al.make_tensor(bn_bias_ptr, al.f32, bn_b_layout)

        out_layout = al.make_layout((B, OC), (OC, 1))
        out_tensor = al.make_tensor(output_ptr, al.bf16, out_layout)

        sf = al.convert(2.0, al.f32)
        eps_val = al.convert(1e-5, al.f32)

        acc = al.convert(0.0, al.f32)
        for ic in al.range(IC):
            acc = acc + in_sp[b, ic] * w_sum[ic, oc]
        acc = acc + al.convert(spatial_size, al.f32) * bias[oc]

        spatial_mean = sf * acc / al.convert(spatial_size, al.f32)

        denom = al.sqrt(rv[oc] + eps_val)
        normalized = (spatial_mean - rm[oc]) / denom
        result = bn_w[oc] * normalized + bn_b[oc]

        out_tensor[b, oc] = al.convert(result, al.bf16)


# =============================================================================
# ModelNew – host wrapper with training/eval mode dispatch.
# =============================================================================

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps

        self.weight_sum = None

    def _ensure_weight_sum(self, device):
        if self.weight_sum is None or self.weight_sum.device != device:
            w = self.conv_transpose.weight.data
            self.weight_sum = w.sum(dim=(2, 3, 4)).to(device=device, dtype=torch.float32)

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()

        B, IC, D, H, W = x.shape
        OC = self.out_channels
        K = self.kernel_size

        D_out = D + K - 1
        H_out = H + K - 1
        W_out = W + K - 1
        spatial_size = D_out * H_out * W_out
        N = B * spatial_size

        device = x.device
        self._ensure_weight_sum(device)

        bias_f32 = self.conv_transpose.bias.data.to(device=device, dtype=torch.float32).contiguous()
        bn_weight = self.batch_norm.weight.data.to(device=device, dtype=torch.float32)
        bn_bias = self.batch_norm.bias.data.to(device=device, dtype=torch.float32)

        # ---- Step 1: Spatial reduction (always needed) ----
        input_spatial_sum = torch.empty(B, IC, device=device, dtype=torch.float32)
        BS = 256
        grid1 = ((B * IC + BS - 1) // BS, 1, 1)
        spatial_reduce_kernel[lambda: (grid1, (BS, 1, 1))](
            x, input_spatial_sum, B, IC, D, H, W,
        )

        if self.training:
            # ---- Training mode: need full conv output for BN batch stats ----
            weight_bf16 = self.conv_transpose.weight.data.to(device=device, dtype=torch.bfloat16).contiguous()

            # Step 2: Full conv transpose
            total_conv = B * OC * spatial_size
            conv_out = torch.empty(B, OC, D_out, H_out, W_out, device=device, dtype=torch.bfloat16)
            grid2 = ((total_conv + BS - 1) // BS, 1, 1)
            conv_transpose3d_kernel[lambda: (grid2, (BS, 1, 1))](
                x, weight_bf16, bias_f32, conv_out,
                B, IC, OC, D, H, W, K, D_out, H_out, W_out,
            )

            # Step 3: BN stats reduction
            bn_sum = torch.empty(OC, device=device, dtype=torch.float32)
            bn_ssq = torch.empty(OC, device=device, dtype=torch.float32)
            grid3 = (OC, 1, 1)
            bn_reduce_kernel[lambda: (grid3, (BS, 1, 1))](
                conv_out, bn_sum, bn_ssq, OC, N, spatial_size,
            )

            # Step 4: BN apply + pool (training)
            output_2d = torch.empty(B, OC, device=device, dtype=torch.bfloat16)
            grid4 = ((B * OC + BS - 1) // BS, 1, 1)
            bn_apply_pool_train_kernel[lambda: (grid4, (BS, 1, 1))](
                input_spatial_sum, self.weight_sum, bias_f32,
                bn_sum, bn_ssq, bn_weight, bn_bias,
                output_2d,
                B, IC, OC, spatial_size, N,
            )
        else:
            # ---- Eval mode: use running stats, no conv output needed ----
            bn_running_mean = self.batch_norm.running_mean.data.to(device=device, dtype=torch.float32)
            bn_running_var = self.batch_norm.running_var.data.to(device=device, dtype=torch.float32)

            output_2d = torch.empty(B, OC, device=device, dtype=torch.bfloat16)
            grid = ((B * OC + BS - 1) // BS, 1, 1)
            bn_apply_pool_eval_kernel[lambda: (grid, (BS, 1, 1))](
                input_spatial_sum, self.weight_sum, bias_f32,
                bn_running_mean, bn_running_var, bn_weight, bn_bias,
                output_2d,
                B, IC, OC, spatial_size,
            )

        return output_2d.view(B, OC, 1, 1, 1)
