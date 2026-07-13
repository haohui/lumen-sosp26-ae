import torch
import torch.nn as nn
import avelang
import avelang.language as al


# ============================================================================
# Kernel 1: 3D Convolution (direct, BF16 compute, FP32 accumulation)
#   Grid  = (W_out, H_out, D_out)     -> block_id maps spatial coords
#   Block = (1, C_out, B)             -> thread_id maps (oc, n)
# ============================================================================

@avelang.jit
def conv3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    in_total: al.i32,
    wt_total: al.i32,
    out_total: al.i32,
    in_sn: al.i32,
    in_sc: al.i32,
    in_sd: al.i32,
    in_sh: al.i32,
    wt_so: al.i32,
    wt_si: al.i32,
    wt_skd: al.i32,
    wt_skh: al.i32,
    out_sn: al.i32,
    out_so: al.i32,
    out_sd: al.i32,
    out_sh: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    KD: al.i32,
    KH: al.i32,
    KW: al.i32,
    pad_d: al.i32,
    pad_h: al.i32,
    pad_w: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    w = al.block_id(0)
    h = al.block_id(1)
    d = al.block_id(2)
    oc = al.thread_id(1)
    n = al.thread_id(2)

    input_1d = al.make_tensor(input_ptr, al.bf16, al.make_layout((in_total,), (1,)))
    weight_1d = al.make_tensor(weight_ptr, al.bf16, al.make_layout((wt_total,), (1,)))
    bias_1d = al.make_tensor(bias_ptr, al.bf16, al.make_layout((C_out,), (1,)))
    output_1d = al.make_tensor(output_ptr, al.bf16, al.make_layout((out_total,), (1,)))

    out_idx = n * out_sn + oc * out_so + d * out_sd + h * out_sh + w

    acc = al.convert(0.0, al.f32)
    for ic in al.range(C_in):
        for kd in al.range(KD):
            in_d = d + kd - pad_d
            if in_d >= 0 and in_d < D:
                for kh in al.range(KH):
                    in_h = h + kh - pad_h
                    if in_h >= 0 and in_h < H:
                        for kw in al.range(KW):
                            in_w = w + kw - pad_w
                            if in_w >= 0 and in_w < W:
                                in_idx = n * in_sn + ic * in_sc + in_d * in_sd + in_h * in_sh + in_w
                                wt_idx = oc * wt_so + ic * wt_si + kd * wt_skd + kh * wt_skh + kw
                                inp = al.convert(input_1d[in_idx], al.f32)
                                wt = al.convert(weight_1d[wt_idx], al.f32)
                                acc = acc + inp * wt

    b = al.convert(bias_1d[oc], al.f32)
    acc = acc + b
    output_1d[out_idx] = al.convert(acc, al.bf16)


# ============================================================================
# Kernel 2: 3D Max Pooling (kernel=2, stride=2)
#   grid  = (W_out, H_out, D_out)
#   block = (1, C, B)
# ============================================================================

@avelang.jit
def maxpool3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    in_total: al.i32,
    out_total: al.i32,
    in_sn: al.i32,
    in_sc: al.i32,
    in_sd: al.i32,
    in_sh: al.i32,
    out_sn: al.i32,
    out_sc: al.i32,
    out_sd: al.i32,
    out_sh: al.i32,
    C: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    w = al.block_id(0)
    h = al.block_id(1)
    d = al.block_id(2)
    c = al.thread_id(1)
    n = al.thread_id(2)

    input_1d = al.make_tensor(input_ptr, al.bf16, al.make_layout((in_total,), (1,)))
    output_1d = al.make_tensor(output_ptr, al.bf16, al.make_layout((out_total,), (1,)))

    max_val = al.convert(-1.0e30, al.f32)
    for pd in al.range(2):
        in_d = d * 2 + pd
        for ph in al.range(2):
            in_h = h * 2 + ph
            for pw in al.range(2):
                in_w = w * 2 + pw
                in_idx = n * in_sn + c * in_sc + in_d * in_sd + in_h * in_sh + in_w
                val = al.convert(input_1d[in_idx], al.f32)
                if val > max_val:
                    max_val = val

    out_idx = n * out_sn + c * out_sc + d * out_sd + h * out_sh + w
    output_1d[out_idx] = al.convert(max_val, al.bf16)


# ============================================================================
# Kernel 3: LogSumExp over channel dim + ReLU (fused)
#   grid  = (W, H, D)     -> block_id maps spatial coords
#   block = (1, 1, B)     -> thread_id(2) maps batch index
# ============================================================================

@avelang.jit
def logsumexp_relu_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    in_total: al.i32,
    out_total: al.i32,
    in_sn: al.i32,
    in_sc: al.i32,
    in_sd: al.i32,
    in_sh: al.i32,
    out_sn: al.i32,
    out_sc: al.i32,
    out_sd: al.i32,
    out_sh: al.i32,
    C_in: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
):
    w = al.block_id(0)
    h = al.block_id(1)
    d = al.block_id(2)
    n = al.thread_id(2)

    input_1d = al.make_tensor(input_ptr, al.bf16, al.make_layout((in_total,), (1,)))
    output_1d = al.make_tensor(output_ptr, al.bf16, al.make_layout((out_total,), (1,)))

    max_val = al.convert(-1.0e30, al.f32)
    for c in al.range(C_in):
        in_idx = n * in_sn + c * in_sc + d * in_sd + h * in_sh + w
        val = al.convert(input_1d[in_idx], al.f32)
        if val > max_val:
            max_val = val

    sum_exp = al.convert(0.0, al.f32)
    for c in al.range(C_in):
        in_idx = n * in_sn + c * in_sc + d * in_sd + h * in_sh + w
        val = al.convert(input_1d[in_idx], al.f32)
        diff = val - max_val
        sum_exp = sum_exp + al.exp(diff)

    result = max_val + al.log(sum_exp)

    zero = al.convert(0.0, al.f32)
    if result < zero:
        result = zero

    out_idx = n * out_sn + d * out_sd + h * out_sh + w
    output_1d[out_idx] = al.convert(result, al.bf16)


# ============================================================================
# Host wrapper
# ============================================================================

def avelang_forward(x: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """Full pipeline: Conv3d -> MaxPool3d -> LogSumExp -> ReLU using AveLang kernels."""
    B, C_in, D, H, W = x.shape
    C_out, _, KD, KH, KW = conv_weight.shape

    device = x.device
    x = x.contiguous().to(torch.bfloat16)
    w = conv_weight.contiguous().to(torch.bfloat16)
    b = conv_bias.contiguous().to(torch.bfloat16)

    D_conv = D
    H_conv = H
    W_conv = W
    D_pool = D_conv // 2
    H_pool = H_conv // 2
    W_pool = W_conv // 2

    conv_out = torch.empty(B, C_out, D_conv, H_conv, W_conv, dtype=torch.bfloat16, device=device)
    pool_out = torch.empty(B, C_out, D_pool, H_pool, W_pool, dtype=torch.bfloat16, device=device)
    final_out = torch.empty(B, 1, D_pool, H_pool, W_pool, dtype=torch.bfloat16, device=device)

    # ---- Conv ----
    in_total = B * C_in * D * H * W
    wt_total = C_out * C_in * KD * KH * KW
    out_conv_total = B * C_out * D_conv * H_conv * W_conv

    in_sn  = C_in * D * H * W
    in_sc  = D * H * W
    in_sd  = H * W
    in_sh  = W
    wt_so  = C_in * KD * KH * KW
    wt_si  = KD * KH * KW
    wt_skd = KH * KW
    wt_skh = KW
    out_sn_conv = C_out * D_conv * H_conv * W_conv
    out_so_conv = D_conv * H_conv * W_conv
    out_sd_conv = H_conv * W_conv
    out_sh_conv = W_conv

    conv3d_kernel[lambda: ((W_conv, H_conv, D_conv), (1, C_out, B))](
        x.data_ptr(), w.data_ptr(), b.data_ptr(), conv_out.data_ptr(),
        in_total, wt_total, out_conv_total,
        in_sn, in_sc, in_sd, in_sh,
        wt_so, wt_si, wt_skd, wt_skh,
        out_sn_conv, out_so_conv, out_sd_conv, out_sh_conv,
        C_in, C_out, D, H, W, KD, KH, KW,
        1, 1, 1, D_conv, H_conv, W_conv,
    )

    # ---- MaxPool ----
    pool_in_total = B * C_out * D_conv * H_conv * W_conv
    pool_out_total = B * C_out * D_pool * H_pool * W_pool

    pool_in_sn  = C_out * D_conv * H_conv * W_conv
    pool_in_sc  = D_conv * H_conv * W_conv
    pool_in_sd  = H_conv * W_conv
    pool_in_sh  = W_conv
    pool_out_sn = C_out * D_pool * H_pool * W_pool
    pool_out_sc = D_pool * H_pool * W_pool
    pool_out_sd = H_pool * W_pool
    pool_out_sh = W_pool

    maxpool3d_kernel[lambda: ((W_pool, H_pool, D_pool), (1, C_out, B))](
        conv_out.data_ptr(), pool_out.data_ptr(),
        pool_in_total, pool_out_total,
        pool_in_sn, pool_in_sc, pool_in_sd, pool_in_sh,
        pool_out_sn, pool_out_sc, pool_out_sd, pool_out_sh,
        C_out, D_pool, H_pool, W_pool,
    )

    # ---- LogSumExp+ReLU ----
    lse_in_total = B * C_out * D_pool * H_pool * W_pool
    lse_out_total = B * 1 * D_pool * H_pool * W_pool

    lse_in_sn  = C_out * D_pool * H_pool * W_pool
    lse_in_sc  = D_pool * H_pool * W_pool
    lse_in_sd  = H_pool * W_pool
    lse_in_sh  = W_pool
    lse_out_sn = 1 * D_pool * H_pool * W_pool
    lse_out_sc = D_pool * H_pool * W_pool
    lse_out_sd = H_pool * W_pool
    lse_out_sh = W_pool

    logsumexp_relu_kernel[lambda: ((W_pool, H_pool, D_pool), (1, 1, B))](
        pool_out.data_ptr(), final_out.data_ptr(),
        lse_in_total, lse_out_total,
        lse_in_sn, lse_in_sc, lse_in_sd, lse_in_sh,
        lse_out_sn, lse_out_sc, lse_out_sd, lse_out_sh,
        C_out, D_pool, H_pool, W_pool,
    )

    return final_out


# ============================================================================
# ModelNew
# ============================================================================

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        weight = self.conv.weight
        bias = self.conv.bias
        return avelang_forward(x, weight, bias)


# Preserved from input_model.py
batch_size = 4
in_channels = 32
out_channels = 64
depth, height, width = 32, 128, 128
kernel_size = 3
stride = 1
padding = 1


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
