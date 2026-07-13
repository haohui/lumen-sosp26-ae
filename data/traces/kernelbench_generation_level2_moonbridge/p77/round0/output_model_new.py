import torch
import torch.nn as nn
import avelang
import avelang.language as al


BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def input_spatial_sum_kernel(
    x_ptr: al.Pointer(al.bf16),
    sum_out_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    in_channels: al.i32,
    in_d: al.i32,
    in_h: al.i32,
    in_w: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    b = bid // in_channels
    ic = bid - b * in_channels

    if b < batch_size:
        smem = al.make_shared((BLOCK_SIZE,), al.f32)
        one = al.convert(1, al.i32)
        spatial_total = in_d * in_h * in_w

        x = al.make_tensor(
            x_ptr, al.bf16,
            al.make_layout((batch_size, in_channels, in_d, in_h, in_w),
                           (in_channels * in_d * in_h * in_w,
                            in_d * in_h * in_w,
                            in_h * in_w,
                            in_w, one)),
        )

        local_sum = al.convert(0.0, al.f32)
        for idx in al.range(tid, spatial_total, BLOCK_SIZE):
            d = idx // (in_h * in_w)
            rem = idx - d * (in_h * in_w)
            h = rem // in_w
            w = rem - h * in_w
            local_sum = local_sum + al.convert(x[b, ic, d, h, w], al.f32)

        smem[tid] = local_sum
        al.syncthreads()

        if tid < 128:
            smem[tid] = smem[tid] + smem[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem[tid] = smem[tid] + smem[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem[tid] = smem[tid] + smem[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem[tid] = smem[tid] + smem[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem[tid] = smem[tid] + smem[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem[tid] = smem[tid] + smem[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem[tid] = smem[tid] + smem[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem[tid] = smem[tid] + smem[tid + 1]

        if tid == 0:
            o = al.make_tensor(sum_out_ptr, al.f32,
                               al.make_layout((batch_size * in_channels,), (one,)))
            o[bid] = smem[0]


@avelang.jit
def final_matmul_epilogue_kernel(
    input_sum_ptr: al.Pointer(al.f32),
    weight_sum_ptr: al.Pointer(al.f32),
    bias_ptr: al.Pointer(al.bf16),
    running_mean_ptr: al.Pointer(al.bf16),
    running_var_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    in_channels: al.i32,
    out_channels: al.i32,
    n_spatial: al.i32,
    scale: al.f32,
    eps: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    b = bid // out_channels
    oc = bid - b * out_channels

    if b < batch_size and oc < out_channels:
        one = al.convert(1, al.i32)

        input_sum = al.make_tensor(input_sum_ptr, al.f32,
                                   al.make_layout((batch_size, in_channels),
                                                  (in_channels, one)))

        weight_sum = al.make_tensor(weight_sum_ptr, al.f32,
                                    al.make_layout((in_channels, out_channels),
                                                   (out_channels, one)))

        layout_c = al.make_layout((out_channels,), (one,))
        bias_t = al.make_tensor(bias_ptr, al.bf16, layout_c)
        rm = al.make_tensor(running_mean_ptr, al.bf16, layout_c)
        rv = al.make_tensor(running_var_ptr, al.bf16, layout_c)
        gm = al.make_tensor(gamma_ptr, al.bf16, layout_c)
        bt = al.make_tensor(beta_ptr, al.bf16, layout_c)

        # Accumulate matmul: sum over ic
        acc = al.convert(0.0, al.f32)
        for ic in al.range(in_channels):
            acc = acc + input_sum[b, ic] * weight_sum[ic, oc]

        # Apply bias
        acc = acc + al.convert(n_spatial, al.f32) * al.convert(bias_t[oc], al.f32)
        # Apply scale
        acc = acc * scale
        # Convert to per-element average
        acc = acc / al.convert(n_spatial, al.f32)

        # BN eval: gamma * (x - running_mean) / sqrt(running_var + eps) + beta
        mean_val = al.convert(rm[oc], al.f32)
        var_val = al.convert(rv[oc], al.f32)
        g_val = al.convert(gm[oc], al.f32)
        b_val = al.convert(bt[oc], al.f32)

        inv_std = al.convert(1.0, al.f32) / al.sqrt(var_val + eps)
        result = g_val * (acc - mean_val) * inv_std + b_val

        o = al.make_tensor(out_ptr, al.bf16,
                           al.make_layout((batch_size * out_channels,), (one,)))
        o[bid] = al.convert(result, al.bf16)


def avelang_forward(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
    scale_factor: float,
    eps: float,
) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA/HIP device."

    x_bf16 = x.contiguous().to(torch.bfloat16)
    B, IC, D_in, H_in, W_in = x_bf16.shape
    OC = conv_weight.shape[1]
    KD = conv_weight.shape[2]
    KH = conv_weight.shape[3]
    KW = conv_weight.shape[4]

    OD = D_in + KD - 1
    OH = H_in + KH - 1
    OW = W_in + KW - 1
    N_spatial = OD * OH * OW

    # Kernel 1: reduce input over spatial dims → (B, IC) in FP32
    input_sum = torch.empty((B * IC,), dtype=torch.float32, device=x.device)
    input_spatial_sum_kernel[lambda: ((B * IC, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, input_sum, B, IC, D_in, H_in, W_in,
    )

    # Host: compute weight_sum[ic, oc] = sum over KD, KH, KW in FP32
    w_bf16 = conv_weight.contiguous().to(torch.bfloat16)
    weight_sum = w_bf16.float().sum(dim=(2, 3, 4)).contiguous()  # (IC, OC) in FP32

    # Kernel 2: matmul input_sum @ weight_sum + bias + scale + BN eval
    b_bf16 = conv_bias.contiguous().to(torch.bfloat16)
    bn_w_bf16 = bn_weight.contiguous().to(torch.bfloat16)
    bn_b_bf16 = bn_bias.contiguous().to(torch.bfloat16)
    bn_rm_bf16 = bn_running_mean.contiguous().to(torch.bfloat16)
    bn_rv_bf16 = bn_running_var.contiguous().to(torch.bfloat16)

    out_flat = torch.empty((B * OC,), dtype=torch.bfloat16, device=x.device)

    final_matmul_epilogue_kernel[lambda: ((B * OC, 1, 1), (1, 1, 1))](
        input_sum, weight_sum, b_bf16, bn_rm_bf16, bn_rv_bf16,
        bn_w_bf16, bn_b_bf16, out_flat,
        B, IC, OC, N_spatial, scale_factor, eps,
    )

    return out_flat.view(B, OC, 1, 1, 1)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        conv_weight = self.conv_transpose.weight.data
        conv_bias = self.conv_transpose.bias.data
        bn_weight = self.batch_norm.weight.data
        bn_bias = self.batch_norm.bias.data
        bn_running_mean = self.batch_norm.running_mean.data
        bn_running_var = self.batch_norm.running_var.data

        return avelang_forward(
            x, conv_weight, conv_bias, bn_weight, bn_bias,
            bn_running_mean, bn_running_var,
            self.scale_factor, self.batch_norm.eps,
        )
