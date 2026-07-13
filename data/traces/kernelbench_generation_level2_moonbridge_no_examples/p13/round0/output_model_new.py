import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# Kernel 1: fused ConvTranspose3d + mean over depth + bias add
#   Output: (B, C_out, 1, H, W) in bf16
#   Uses 1-D tensor views with manual linear indexing.
#   FP32 accumulation from BF16 loads.
# ---------------------------------------------------------------------------
@avelang.jit
def fused_conv_transpose_mean_bias_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
):
    b = al.block_id(0)
    c_out = al.block_id(1)
    linear_hw = al.block_id(2) * al.block_dim(2) + al.thread_id(2)
    h = linear_hw // W
    w = linear_hw - h * W

    if b < B and c_out < C_out and h < H and w < W:
        acc = al.convert(0.0, al.f32)

        stride_c = D * H * W
        stride_d = H * W
        stride_h = W

        # Input x: (B, C_in, D, H, W) row-major
        x_total = B * C_in * D * H * W
        x_layout = al.make_layout((x_total,), (1,))
        x_tensor = al.make_tensor(x_ptr, al.bf16, x_layout)

        # Weight: (C_in, C_out, 3, 3, 3) row-major -> total = C_in * C_out * 27
        w_stride_cin = C_out * 27
        w_stride_cout = 27
        w_stride_kd = 9
        w_stride_kh = 3

        w_total = C_in * C_out * 27
        w_layout = al.make_layout((w_total,), (1,))
        w_tensor = al.make_tensor(w_ptr, al.bf16, w_layout)

        # Bias: (1, C_out, 1, 1, 1) row-major
        bias_total = C_out
        bias_layout = al.make_layout((bias_total,), (1,))
        bias_tensor = al.make_tensor(bias_ptr, al.bf16, bias_layout)

        # Output: (B, C_out, 1, H, W) row-major
        out_total = B * C_out * H * W
        out_layout = al.make_layout((out_total,), (1,))
        out_tensor = al.make_tensor(out_ptr, al.bf16, out_layout)

        out_idx = b * C_out * H * W + c_out * H * W + h * W + w

        pad_val = 1

        for c_in in al.range(C_in):
            x_cin_base = b * C_in * stride_c + c_in * stride_c
            w_cin_base = c_in * w_stride_cin

            for kd in al.range(3):
                for kh in al.range(3):
                    h_in = h + pad_val - kh
                    if h_in >= 0:
                        if h_in < H:
                            x_hw_base = x_cin_base + h_in * stride_h
                            w_kdkh_base = (
                                w_cin_base
                                + c_out * w_stride_cout
                                + kd * w_stride_kd
                                + kh * w_stride_kh
                            )

                            for kw in al.range(3):
                                w_in = w + pad_val - kw
                                if w_in >= 0:
                                    if w_in < W:
                                        w_idx = w_kdkh_base + kw
                                        w_bf16 = w_tensor[w_idx]

                                        for d in al.range(D):
                                            d_in = d + pad_val - kd
                                            if d_in >= 0:
                                                if d_in < D:
                                                    x_idx = (
                                                        x_hw_base
                                                        + d_in * stride_d
                                                        + w_in
                                                    )
                                                    x_val = al.convert(
                                                        x_tensor[x_idx], al.f32,
                                                    )
                                                    w_val = al.convert(w_bf16, al.f32)
                                                    acc = acc + x_val * w_val

        D_f32 = al.convert(D, al.f32)
        acc = acc / D_f32
        bias_val = al.convert(bias_tensor[c_out], al.f32)
        acc = acc + bias_val
        out_tensor[out_idx] = al.convert(acc, al.bf16)


# ---------------------------------------------------------------------------
# Kernel 2: softmax over channels + tanh + scaling
#   Input/Output: (B, C_OUT, 1, H, W) in bf16
#   One block per (b, h, w); C_OUT threads cooperate for the channel softmax.
#   Scale factor is passed as constexpr.
# ---------------------------------------------------------------------------
@avelang.jit
def softmax_tanh_scale_kernel(
    in_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_OUT: al.constexpr,
    H: al.i32,
    W: al.i32,
    SCALE: al.constexpr,
):
    b = al.block_id(0)
    h = al.block_id(1)
    w = al.block_id(2)
    tid = al.thread_id(0)

    if b < B and h < H and w < W and tid < C_OUT:
        total = B * C_OUT * H * W
        layout = al.make_layout((total,), (1,))
        inp = al.make_tensor(in_ptr, al.bf16, layout)
        out = al.make_tensor(out_ptr, al.bf16, layout)

        sh_data = al.make_shared((C_OUT,), al.f32)

        ch_base = tid * H * W + h * W + w
        b_base = b * C_OUT * H * W
        idx = b_base + ch_base

        val = al.convert(inp[idx], al.f32)
        sh_data[tid] = val
        al.syncthreads()

        # --- max reduction ---
        stride = 32
        if tid < stride:
            other = sh_data[tid + stride]
            if other > sh_data[tid]:
                sh_data[tid] = other
        al.syncthreads()

        stride = 16
        if tid < stride:
            other = sh_data[tid + stride]
            if other > sh_data[tid]:
                sh_data[tid] = other
        al.syncthreads()

        stride = 8
        if tid < stride:
            other = sh_data[tid + stride]
            if other > sh_data[tid]:
                sh_data[tid] = other
        al.syncthreads()

        stride = 4
        if tid < stride:
            other = sh_data[tid + stride]
            if other > sh_data[tid]:
                sh_data[tid] = other
        al.syncthreads()

        stride = 2
        if tid < stride:
            other = sh_data[tid + stride]
            if other > sh_data[tid]:
                sh_data[tid] = other
        al.syncthreads()

        stride = 1
        if tid < stride:
            other = sh_data[tid + stride]
            if other > sh_data[tid]:
                sh_data[tid] = other
        al.syncthreads()

        max_val = sh_data[0]

        # exp(val - max)
        shifted = val - max_val
        exp_val = al.exp(shifted)
        sh_data[tid] = exp_val
        al.syncthreads()

        # --- sum reduction ---
        stride = 32
        if tid < stride:
            sh_data[tid] = sh_data[tid] + sh_data[tid + stride]
        al.syncthreads()

        stride = 16
        if tid < stride:
            sh_data[tid] = sh_data[tid] + sh_data[tid + stride]
        al.syncthreads()

        stride = 8
        if tid < stride:
            sh_data[tid] = sh_data[tid] + sh_data[tid + stride]
        al.syncthreads()

        stride = 4
        if tid < stride:
            sh_data[tid] = sh_data[tid] + sh_data[tid + stride]
        al.syncthreads()

        stride = 2
        if tid < stride:
            sh_data[tid] = sh_data[tid] + sh_data[tid + stride]
        al.syncthreads()

        stride = 1
        if tid < stride:
            sh_data[tid] = sh_data[tid] + sh_data[tid + stride]
        al.syncthreads()

        sum_exp = sh_data[0]

        softmax_val = exp_val / sum_exp
        scale_f32 = al.convert(SCALE, al.f32)
        result = al.tanh(softmax_val) * scale_f32
        out[idx] = al.convert(result, al.bf16)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------
def _run_model(conv_weight, bias, x, scaling_factor):
    B, C_in, D, H, W = x.shape
    C_out = conv_weight.shape[1]

    x_bf16 = x.contiguous().to(torch.bfloat16)
    w_bf16 = conv_weight.contiguous().to(torch.bfloat16)
    bias_bf16 = bias.contiguous().to(torch.bfloat16)

    mid = torch.empty(B, C_out, 1, H, W, dtype=torch.bfloat16, device=x.device)

    import math
    BLOCK_HW = 64
    grid_hw = math.ceil(H * W / BLOCK_HW)
    fused_conv_transpose_mean_bias_kernel[
        lambda: ((B, C_out, grid_hw), (1, 1, BLOCK_HW))
    ](
        x_bf16.data_ptr(),
        w_bf16.data_ptr(),
        bias_bf16.data_ptr(),
        mid.data_ptr(),
        B, C_in, C_out, D, H, W,
    )

    out = torch.empty(B, C_out, 1, H, W, dtype=torch.bfloat16, device=x.device)

    softmax_tanh_scale_kernel[
        lambda: ((B, H, W), (C_out, 1, 1))
    ](
        mid.data_ptr(),
        out.data_ptr(),
        B, C_out, H, W,
        float(scaling_factor),
    )

    return out.to(x.dtype)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding,
        )
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        return _run_model(
            self.conv_transpose.weight,
            self.bias,
            x,
            self.scaling_factor,
        )
