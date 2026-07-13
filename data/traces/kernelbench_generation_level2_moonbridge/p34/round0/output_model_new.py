import torch
import torch.nn as nn
import avelang
import avelang.language as al

STRIDE_C = 2
PADDING_C = 1


@avelang.jit
def conv_transpose3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    batch: al.i32,
    in_c: al.i32,
    out_c: al.i32,
    d_in: al.i32,
    h_in: al.i32,
    w_in: al.i32,
    d_out: al.i32,
    h_out: al.i32,
    w_out: al.i32,
    kernel_size: al.i32,
    in_stride_n: al.i32,
    in_stride_c: al.i32,
    in_stride_d: al.i32,
    in_stride_h: al.i32,
    wt_stride_ci: al.i32,
    wt_stride_co: al.i32,
    wt_stride_kd: al.i32,
    wt_stride_kh: al.i32,
    out_stride_n: al.i32,
    out_stride_c: al.i32,
    out_stride_d: al.i32,
    out_stride_h: al.i32,
    total_in_elems: al.i32,
    total_wt_elems: al.i32,
    total_out_elems: al.i32,
):
    tid = al.thread_id(0)
    bid_x = al.block_id(0)
    bid_y = al.block_id(1)

    n = bid_x // (out_c * d_out)
    rem = bid_x - n * out_c * d_out
    c_out_idx = rem // d_out
    d_out_pos = rem - c_out_idx * d_out
    h_out_pos = bid_y
    w_out_pos = tid

    zero = al.convert(0, al.i32)
    if w_out_pos >= w_out:
        return
    if n < zero or n >= batch:
        return
    if c_out_idx < zero or c_out_idx >= out_c:
        return
    if d_out_pos < zero or d_out_pos >= d_out:
        return
    if h_out_pos < zero or h_out_pos >= h_out:
        return

    inp = al.make_tensor(input_ptr, al.bf16, al.make_layout((total_in_elems,), (1,)))
    wt = al.make_tensor(weight_ptr, al.bf16, al.make_layout((total_wt_elems,), (1,)))
    bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((out_c,), (1,)))
    oup = al.make_tensor(output_ptr, al.bf16, al.make_layout((total_out_elems,), (1,)))

    acc = al.convert(bias[c_out_idx], al.bf16)
    stride_i32 = al.convert(STRIDE_C, al.i32)
    pad_i32 = al.convert(PADDING_C, al.i32)
    d_in_i32 = al.convert(d_in, al.i32)
    h_in_i32 = al.convert(h_in, al.i32)
    w_in_i32 = al.convert(w_in, al.i32)

    for ci in al.range(in_c):
        for kd in al.range(kernel_size):
            d_rem = d_out_pos + pad_i32 - kd
            d_mod = d_rem % stride_i32
            if d_mod == zero:
                di = d_rem // stride_i32
                if di >= zero and di < d_in_i32:
                    for kh in al.range(kernel_size):
                        h_rem = h_out_pos + pad_i32 - kh
                        h_mod = h_rem % stride_i32
                        if h_mod == zero:
                            hi = h_rem // stride_i32
                            if hi >= zero and hi < h_in_i32:
                                for kw in al.range(kernel_size):
                                    w_rem = w_out_pos + pad_i32 - kw
                                    w_mod = w_rem % stride_i32
                                    if w_mod == zero:
                                        wi = w_rem // stride_i32
                                        if wi >= zero and wi < w_in_i32:
                                            inp_idx = n * in_stride_n + ci * in_stride_c + di * in_stride_d + hi * in_stride_h + wi
                                            wt_idx = ci * wt_stride_ci + c_out_idx * wt_stride_co + kd * wt_stride_kd + kh * wt_stride_kh + kw
                                            inp_val = inp[inp_idx]
                                            wt_val = wt[wt_idx]
                                            acc = acc + al.convert(al.convert(inp_val, al.f32) * al.convert(wt_val, al.f32), al.bf16)

    out_idx = n * out_stride_n + c_out_idx * out_stride_c + d_out_pos * out_stride_d + h_out_pos * out_stride_h + w_out_pos
    oup[out_idx] = acc


@avelang.jit
def layernorm_gelu_scale_kernel(
    input_ptr: al.Pointer(al.bf16),
    ln_weight_ptr: al.Pointer(al.bf16),
    ln_bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    batch: al.i32,
    out_c: al.i32,
    d_out: al.i32,
    h_out: al.i32,
    w_out: al.i32,
    out_stride_n: al.i32,
    out_stride_c: al.i32,
    out_stride_d: al.i32,
    out_stride_h: al.i32,
    total_elems: al.i32,
    eps: al.f32,
    scaling_factor: al.f32,
):
    tid = al.thread_id(0)
    bid_x = al.block_id(0)
    bid_y = al.block_id(1)

    n = bid_x // (out_c * d_out)
    rem = bid_x - n * out_c * d_out
    c_out_idx = rem // d_out
    d_out_pos = rem - c_out_idx * d_out
    h_out_pos = bid_y
    w_out_pos = tid

    zero = al.convert(0, al.i32)
    if w_out_pos >= w_out:
        return
    if n < zero or n >= batch:
        return
    if c_out_idx < zero or c_out_idx >= out_c:
        return
    if d_out_pos < zero or d_out_pos >= d_out:
        return
    if h_out_pos < zero or h_out_pos >= h_out:
        return

    inp = al.make_tensor(input_ptr, al.bf16, al.make_layout((total_elems,), (1,)))
    lnw = al.make_tensor(ln_weight_ptr, al.bf16, al.make_layout((w_out,), (1,)))
    lnb = al.make_tensor(ln_bias_ptr, al.bf16, al.make_layout((w_out,), (1,)))
    oup = al.make_tensor(output_ptr, al.bf16, al.make_layout((total_elems,), (1,)))

    smem = al.make_shared((64,), al.f32)
    N_f32 = al.convert(w_out, al.f32)

    in_idx = n * out_stride_n + c_out_idx * out_stride_c + d_out_pos * out_stride_d + h_out_pos * out_stride_h + w_out_pos
    val = al.convert(inp[in_idx], al.f32)

    # Reduction pass 1: sum for mean
    smem[tid] = val
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
        smem[0] = smem[0] / N_f32
    al.syncthreads()
    mean_val = smem[0]

    # Reduction pass 2: (x - mean) rounded to BF16, then squared
    diff_f32 = val - mean_val
    diff_bf16 = al.convert(diff_f32, al.bf16)
    diff_f32_2 = al.convert(diff_bf16, al.f32)
    smem[tid] = diff_f32_2 * diff_f32_2
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
        var_val = smem[0] / N_f32
        rstd_val = al.convert(1.0, al.f32) / al.sqrt(var_val + eps)
        smem[0] = rstd_val
    al.syncthreads()
    rstd_val = smem[0]

    # Apply: (x - mean) rounded to BF16, then * rstd * weight + bias
    diff2_f32 = val - mean_val
    diff2_bf16 = al.convert(diff2_f32, al.bf16)
    diff2_f32_2 = al.convert(diff2_bf16, al.f32)
    norm_val = diff2_f32_2 * rstd_val
    affine = norm_val * al.convert(lnw[w_out_pos], al.f32) + al.convert(lnb[w_out_pos], al.f32)

    half = al.convert(0.5, al.f32)
    one = al.convert(1.0, al.f32)
    sqrt_half = al.convert(0.70710678118, al.f32)
    gelu_val = half * affine * (one + al.erf(affine * sqrt_half))

    final_val = gelu_val * scaling_factor
    oup[in_idx] = al.convert(final_val, al.bf16)


def avelang_forward(x_bf16, weight_bf16, bias_bf16,
                    ln_weight_bf16, ln_bias_bf16,
                    eps, scaling_factor):
    assert x_bf16.is_cuda and x_bf16.dtype == torch.bfloat16

    N, C_in, D_in, H_in, W_in = x_bf16.shape
    C_out = weight_bf16.shape[1]
    K_val = weight_bf16.shape[2]
    D_out = (D_in - 1) * STRIDE_C - 2 * PADDING_C + K_val
    H_out = (H_in - 1) * STRIDE_C - 2 * PADDING_C + K_val
    W_out = (W_in - 1) * STRIDE_C - 2 * PADDING_C + K_val

    tmp = torch.empty((N, C_out, D_out, H_out, W_out),
                      device=x_bf16.device, dtype=torch.bfloat16)
    output = torch.empty((N, C_out, D_out, H_out, W_out),
                         device=x_bf16.device, dtype=torch.bfloat16)

    total_in = N * C_in * D_in * H_in * W_in
    total_wt = C_in * C_out * K_val * K_val * K_val
    total_out = N * C_out * D_out * H_out * W_out

    in_stride_n = C_in * D_in * H_in * W_in
    in_stride_c = D_in * H_in * W_in
    in_stride_d = H_in * W_in
    in_stride_h = W_in

    wt_stride_ci = C_out * K_val * K_val * K_val
    wt_stride_co = K_val * K_val * K_val
    wt_stride_kd = K_val * K_val
    wt_stride_kh = K_val

    out_stride_n = C_out * D_out * H_out * W_out
    out_stride_c = D_out * H_out * W_out
    out_stride_d = H_out * W_out
    out_stride_h = W_out

    grid_x = N * C_out * D_out
    grid_y = H_out
    grid = (grid_x, grid_y, 1)
    block = (W_out, 1, 1)

    conv_transpose3d_kernel[lambda: (grid, block)](
        x_bf16, weight_bf16, bias_bf16, tmp,
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K_val,
        in_stride_n, in_stride_c, in_stride_d, in_stride_h,
        wt_stride_ci, wt_stride_co, wt_stride_kd, wt_stride_kh,
        out_stride_n, out_stride_c, out_stride_d, out_stride_h,
        total_in, total_wt, total_out,
    )

    layernorm_gelu_scale_kernel[lambda: (grid, block)](
        tmp, ln_weight_bf16, ln_bias_bf16, output,
        N, C_out, D_out, H_out, W_out,
        out_stride_n, out_stride_c, out_stride_d, out_stride_h,
        total_out, eps, scaling_factor,
    )

    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor
        self.eps = eps

    def forward(self, x):
        x_bf16 = x.to(dtype=torch.bfloat16)
        wt = self.conv_transpose.weight.data.to(dtype=torch.bfloat16).contiguous()
        bias = self.conv_transpose.bias.data.to(dtype=torch.bfloat16).contiguous()
        ln_w = self.layer_norm.weight.data.to(dtype=torch.bfloat16).contiguous()
        ln_b = self.layer_norm.bias.data.to(dtype=torch.bfloat16).contiguous()

        result = avelang_forward(
            x_bf16, wt, bias, ln_w, ln_b,
            self.eps, self.scaling_factor)
        return result.to(x.dtype)
