import math
import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# Fused kernel: ConvTranspose3d + LogSumExp + HardSwish + Bias + Clamp
#
# For each (batch, spatial-out-position):
#   1. Compute all C_out channel values via transposed convolution
#   2. LogSumExp across the C_out channels -> single scalar
#   3. HardSwish: x * sigmoid(x+3) / 6
#   4. Subtract external bias
#   5. Clamp to [-1, 1]
#
# This eliminates the intermediate (N, C_out, D_out, H_out, W_out) tensor
# and fuses three operations into one kernel.
# ---------------------------------------------------------------------------

@avelang.jit
def fused_conv_lse_hardswish_kernel(
    in_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    conv_bias_ptr: al.Pointer(al.bf16),
    ext_bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.constexpr,
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
    BLOCK_SIZE: al.constexpr,
):
    n = al.block_id(0)
    block_idx = al.block_id(1)
    tid = al.thread_id(0)

    # Input layout: (N, C_in, D_in, H_in, W_in)
    in_stride_n = C_in * D_in * H_in * W_in
    in_stride_c = D_in * H_in * W_in
    in_stride_d = H_in * W_in
    in_stride_h = W_in
    in_stride_w = 1
    in_layout = al.make_layout(
        (N, C_in, D_in, H_in, W_in),
        (in_stride_n, in_stride_c, in_stride_d, in_stride_h, in_stride_w),
    )
    inp = al.make_tensor(in_ptr, al.bf16, in_layout)

    # Weight layout: (C_in, C_out, KD, KH, KW)
    w_stride_ic = C_out * KD * KH * KW
    w_stride_oc = KD * KH * KW
    w_stride_kd = KH * KW
    w_stride_kh = KW
    w_stride_kw = 1
    w_layout = al.make_layout(
        (C_in, C_out, KD, KH, KW),
        (w_stride_ic, w_stride_oc, w_stride_kd, w_stride_kh, w_stride_kw),
    )
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    # Conv bias layout: (C_out,)
    cb_layout = al.make_layout((C_out,), (1,))
    conv_bias = al.make_tensor(conv_bias_ptr, al.bf16, cb_layout)

    # External bias layout: (1, 1, 1, 1)
    eb_layout = al.make_layout((1, 1, 1, 1), (1, 1, 1, 1))
    ext_bias = al.make_tensor(ext_bias_ptr, al.bf16, eb_layout)

    # Output layout: (N, 1, D_out, H_out, W_out)
    out_stride_n = D_out * H_out * W_out
    out_stride_1 = D_out * H_out * W_out
    out_stride_d = H_out * W_out
    out_stride_h = W_out
    out_stride_w = 1
    out_layout = al.make_layout(
        (N, 1, D_out, H_out, W_out),
        (out_stride_n, out_stride_1, out_stride_d, out_stride_h, out_stride_w),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    total_spatial = D_out * H_out * W_out
    idx = block_idx * BLOCK_SIZE + tid

    if idx < total_spatial:
        d_out = idx // (H_out * W_out)
        rem_hw = idx % (H_out * W_out)
        h_out = rem_hw // W_out
        w_out = rem_hw % W_out

        # Local storage for all C_out channel values (computed once)
        vals = al.make_local((C_out,), al.f32)

        # Constants for post-ops
        zero_f32 = al.convert(0.0, al.f32)
        one_f32 = al.convert(1.0, al.f32)
        three_f32 = al.convert(3.0, al.f32)
        six_f32 = al.convert(6.0, al.f32)
        neg_one_f32 = al.convert(-1.0, al.f32)

        # --- Pass 1: compute all C_out channel values, find max for LogSumExp ---
        max_val = al.convert(-1.0e30, al.f32)

        for oc in al.range(C_out):
            acc = al.convert(0.0, al.f32)
            for ic in al.range(C_in):
                for kd in al.range(KD):
                    d_candidate = (d_out + padding - kd) // stride
                    d_check = d_candidate * stride
                    if d_check == (d_out + padding - kd):
                        if d_candidate >= 0:
                            if d_candidate < D_in:
                                for kh in al.range(KH):
                                    h_candidate = (h_out + padding - kh) // stride
                                    h_check = h_candidate * stride
                                    if h_check == (h_out + padding - kh):
                                        if h_candidate >= 0:
                                            if h_candidate < H_in:
                                                for kw in al.range(KW):
                                                    w_candidate = (w_out + padding - kw) // stride
                                                    w_check = w_candidate * stride
                                                    if w_check == (w_out + padding - kw):
                                                        if w_candidate >= 0:
                                                            if w_candidate < W_in:
                                                                in_val = al.convert(inp[n, ic, d_candidate, h_candidate, w_candidate], al.f32)
                                                                w_val = al.convert(w[ic, oc, kd, kh, kw], al.f32)
                                                                acc = acc + in_val * w_val
            b_val = al.convert(conv_bias[oc], al.f32)
            channel_val = acc + b_val
            vals[oc] = channel_val
            if channel_val > max_val:
                max_val = channel_val

        # --- Pass 2: sum exp(x - max) using stored channel values ---
        sum_exp = zero_f32

        for oc in al.range(C_out):
            channel_val = vals[oc]
            diff = channel_val - max_val
            sum_exp = sum_exp + al.exp(diff)

        lse_result = al.log(sum_exp) + max_val

        # --- Step 3: HardSwish: x * sigmoid(x+3) / 6 ---
        sigmoid_input = lse_result + three_f32
        neg_sigmoid_input = zero_f32 - sigmoid_input
        sigmoid_val = one_f32 / (one_f32 + al.exp(neg_sigmoid_input))
        y = lse_result * sigmoid_val / six_f32

        # --- Step 4: subtract external bias ---
        eb_val = al.convert(ext_bias[0, 0, 0, 0], al.f32)
        y = y - eb_val

        # --- Step 5: clamp to [-1, 1] ---
        if y > one_f32:
            y = one_f32
        if y < neg_one_f32:
            y = neg_one_f32

        out[n, 0, d_out, h_out, w_out] = al.convert(y, al.bf16)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------

def run_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_bias: torch.Tensor,
    ext_bias: torch.Tensor,
    stride: int,
    padding: int,
) -> torch.Tensor:
    N, C_in, D_in, H_in, W_in = x.shape
    C_in_w, C_out, KD, KH, KW = weight.shape
    assert C_in_w == C_in

    D_out = (D_in - 1) * stride - 2 * padding + KD
    H_out = (H_in - 1) * stride - 2 * padding + KH
    W_out = (W_in - 1) * stride - 2 * padding + KW

    out = torch.empty(N, 1, D_out, H_out, W_out, dtype=torch.bfloat16, device=x.device)

    BLOCK_SIZE = 256
    total_spatial = D_out * H_out * W_out
    num_blocks = (total_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (N, num_blocks, 1)
    block = (BLOCK_SIZE, 1, 1)

    fused_conv_lse_hardswish_kernel[lambda: (grid, block)](
        x.data_ptr(),
        weight.data_ptr(),
        conv_bias.data_ptr(),
        ext_bias.contiguous().data_ptr(),
        out.data_ptr(),
        N,
        C_in,
        C_out,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        KD,
        KH,
        KW,
        stride,
        padding,
        BLOCK_SIZE,
    )
    return out


# ---------------------------------------------------------------------------
# ModelNew
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.stride = stride
        self.padding = padding

        if isinstance(kernel_size, int):
            KD = KH = KW = kernel_size
        else:
            KD, KH, KW = kernel_size

        self.conv_weight = nn.Parameter(
            torch.empty(in_channels, out_channels, KD, KH, KW)
        )
        self.conv_bias = nn.Parameter(torch.empty(out_channels))

        # Match nn.ConvTranspose3d.reset_parameters() exactly
        torch.nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
        fan_in = in_channels * KD * KH * KW
        bound = 1.0 / math.sqrt(fan_in) if fan_in != 0 else 0.0
        torch.nn.init.uniform_(self.conv_bias, -bound, bound)

        # External bias - 3rd RNG call matching input model order
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()
        return run_fused(x, self.conv_weight, self.conv_bias, self.bias, self.stride, self.padding)
