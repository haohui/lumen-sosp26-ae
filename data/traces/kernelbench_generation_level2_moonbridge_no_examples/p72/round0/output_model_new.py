import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ===========================================================================
# Kernel 1: 3D Transposed Convolution
#
# For each output position, iterates over the 27 kernel positions. For each
# kernel position, checks if the corresponding input position is valid using:
#     candidate = output_pos + padding - kernel_pos
#     Valid iff candidate == (candidate / stride) * stride  (i.e., divisible)
# Hardcodes stride=2, padding=1, kernel_size=3.
# ===========================================================================
@avelang.jit
def conv_transpose3d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    H_TILES_PER_THREAD: al.i32,
    W_TILES_PER_THREAD: al.i32,
):
    one = al.convert(1, al.i32)
    zero = al.convert(0, al.i32)
    two = al.convert(2, al.i32)
    three = al.convert(3, al.i32)

    x_stride_n = C_in * D_in * H_in * W_in
    x_stride_c = D_in * H_in * W_in
    x_stride_d = H_in * W_in
    x_stride_h = W_in

    w_stride_ci = C_out * three * three * three
    w_stride_co = three * three * three
    w_stride_kd = three * three
    w_stride_kh = three

    out_stride_n = C_out * D_out * H_out * W_out
    out_stride_c = D_out * H_out * W_out
    out_stride_d = H_out * W_out
    out_stride_h = W_out

    x_layout = al.make_layout(
        (N, C_in, D_in, H_in, W_in),
        (x_stride_n, x_stride_c, x_stride_d, x_stride_h, one),
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    w_layout = al.make_layout(
        (C_in, C_out, three, three, three),
        (w_stride_ci, w_stride_co, w_stride_kd, w_stride_kh, one),
    )
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    b_layout = al.make_layout((C_out,), (one,))
    bias = al.make_tensor(b_ptr, al.f32, b_layout)

    out_layout = al.make_layout(
        (N, C_out, D_out, H_out, W_out),
        (out_stride_n, out_stride_c, out_stride_d, out_stride_h, one),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    n = al.block_id(0)
    c_o = al.block_id(1)
    d_o = al.block_id(2)
    h_tid = al.thread_id(0)
    w_tid = al.thread_id(1)

    if n < N and c_o < C_out and d_o < D_out:
        h_base = h_tid * H_TILES_PER_THREAD
        w_base = w_tid * W_TILES_PER_THREAD

        for hh in al.range(H_TILES_PER_THREAD):
            h_o = h_base + hh
            if h_o < H_out:
                for ww in al.range(W_TILES_PER_THREAD):
                    w_o = w_base + ww
                    if w_o < W_out:
                        acc = al.convert(0.0, al.f32)

                        for k_d in al.range(three):
                            d_cand = d_o + one - k_d
                            d_div = d_cand / two
                            if d_cand == d_div * two:
                                d_in = d_div
                                if d_in >= zero and d_in < D_in:
                                    for k_h in al.range(three):
                                        h_cand = h_o + one - k_h
                                        h_div = h_cand / two
                                        if h_cand == h_div * two:
                                            h_in = h_div
                                            if h_in >= zero and h_in < H_in:
                                                for k_w in al.range(three):
                                                    w_cand = w_o + one - k_w
                                                    w_div = w_cand / two
                                                    if w_cand == w_div * two:
                                                        w_in = w_div
                                                        if w_in >= zero and w_in < W_in:
                                                            for c_in in al.range(C_in):
                                                                xv = al.convert(x[n, c_in, d_in, h_in, w_in], al.f32)
                                                                wv = al.convert(w[c_in, c_o, k_d, k_h, k_w], al.f32)
                                                                acc = acc + xv * wv

                        acc = acc + bias[c_o]
                        out[n, c_o, d_o, h_o, w_o] = al.convert(acc, al.bf16)
    return


# ===========================================================================
# Kernel 2: BatchNorm apply + 4x4x4 average pool (fused two 2x2x2 pools)
# ===========================================================================
@avelang.jit
def bn_apply_pool_kernel(
    data_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    mean_ptr: al.Pointer(al.f32),
    var_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_out: al.i32,
    D_int: al.i32,
    H_int: al.i32,
    W_int: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    stride_n: al.i32,
    stride_c: al.i32,
    stride_d: al.i32,
    stride_h: al.i32,
    out_stride_n: al.i32,
    out_stride_c: al.i32,
    out_stride_d: al.i32,
    out_stride_h: al.i32,
):
    one = al.convert(1, al.i32)
    four = al.convert(4, al.i32)
    pool_size_f = al.convert(64.0, al.f32)

    data_layout = al.make_layout(
        (N, C_out, D_int, H_int, W_int),
        (stride_n, stride_c, stride_d, stride_h, one),
    )
    data = al.make_tensor(data_ptr, al.bf16, data_layout)

    out_layout = al.make_layout(
        (N, C_out, D_out, H_out, W_out),
        (out_stride_n, out_stride_c, out_stride_d, out_stride_h, one),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    ch_layout = al.make_layout((C_out,), (one,))
    gamma = al.make_tensor(gamma_ptr, al.f32, ch_layout)
    betav = al.make_tensor(beta_ptr, al.f32, ch_layout)
    meanv = al.make_tensor(mean_ptr, al.f32, ch_layout)
    varv = al.make_tensor(var_ptr, al.f32, ch_layout)

    n = al.block_id(0)
    c = al.block_id(1)
    d_o = al.block_id(2)
    h_o = al.thread_id(0)
    w_o = al.thread_id(1)

    if n < N and c < C_out and d_o < D_out and h_o < H_out and w_o < W_out:
        gm = gamma[c]
        bt = betav[c]
        mn = meanv[c]
        vr = varv[c]
        inv_std = al.convert(1.0, al.f32) / al.sqrt(vr)

        acc = al.convert(0.0, al.f32)
        d_base = d_o * four
        h_base = h_o * four
        w_base = w_o * four

        for dd in al.range(four):
            d_int = d_base + dd
            for hh in al.range(four):
                h_int = h_base + hh
                for ww in al.range(four):
                    w_int = w_base + ww
                    val = al.convert(data[n, c, d_int, h_int, w_int], al.f32)
                    normed = (val - mn) * inv_std * gm + bt
                    acc = acc + normed

        result = acc / pool_size_f
        out[n, c, d_o, h_o, w_o] = al.convert(result, al.bf16)
    return


# ===========================================================================
# Host launchers
# ===========================================================================

def _launch_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    N: int, C_in: int, C_out: int,
    D_in: int, H_in: int, W_in: int,
    D_out: int, H_out: int, W_out: int,
    stride: int, padding: int, kernel_sz: int,
) -> torch.Tensor:
    out = torch.empty(N, C_out, D_out, H_out, W_out, dtype=torch.bfloat16, device=x.device)
    BLOCK_H = 7
    BLOCK_W = 9
    H_TILES_PER_THREAD = max(H_out // BLOCK_H, 1)
    W_TILES_PER_THREAD = max(W_out // BLOCK_W, 1)
    conv_transpose3d_kernel[lambda: ((N, C_out, D_out), (BLOCK_H, BLOCK_W, 1))](
        x.data_ptr(), weight.data_ptr(), bias.data_ptr(), out.data_ptr(),
        N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
        H_TILES_PER_THREAD, W_TILES_PER_THREAD,
    )
    return out


def _launch_bn_compute_stats(
    data: torch.Tensor,
    C_out: int,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    N, C, D, H, W = data.shape
    data_f32 = data.to(torch.float32)
    data_flat = data_f32.permute(1, 0, 2, 3, 4).reshape(C, -1)
    mean = data_flat.mean(dim=1)
    var = data_flat.var(dim=1, unbiased=False) + eps
    return mean, var


def _launch_bn_apply_pool(
    data: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    N: int, C_out: int,
    D_int: int, H_int: int, W_int: int,
    D_out: int, H_out: int, W_out: int,
) -> torch.Tensor:
    out = torch.empty(N, C_out, D_out, H_out, W_out, dtype=torch.bfloat16, device=data.device)
    bn_apply_pool_kernel[lambda: ((N, C_out, D_out), (H_out, W_out, 1))](
        data.data_ptr(), gamma.data_ptr(), beta.data_ptr(),
        mean.data_ptr(), var.data_ptr(), out.data_ptr(),
        N, C_out, D_int, H_int, W_int, D_out, H_out, W_out,
        C_out * D_int * H_int * W_int,
        D_int * H_int * W_int,
        H_int * W_int,
        W_int,
        C_out * D_out * H_out * W_out,
        D_out * H_out * W_out,
        H_out * W_out,
        W_out,
    )
    return out


# ===========================================================================
# ModelNew
# ===========================================================================

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = x.contiguous()
        N, C_in, D_in, H_in, W_in = x.shape

        w_conv = self.conv_transpose.weight.detach()
        b_conv = self.conv_transpose.bias.detach()
        gamma = self.batch_norm.weight.detach()
        beta_bn = self.batch_norm.bias.detach()

        out_channels = self.out_channels
        stride = self.stride
        padding = self.padding
        K = self.kernel_size

        x_bf16 = x.to(torch.bfloat16)
        w_bf16 = w_conv.to(torch.bfloat16)
        b_f32 = b_conv.to(torch.float32)
        gamma_f32 = gamma.to(torch.float32)
        beta_f32 = beta_bn.to(torch.float32)

        D_out = (D_in - 1) * stride - 2 * padding + K
        H_out = (H_in - 1) * stride - 2 * padding + K
        W_out = (W_in - 1) * stride - 2 * padding + K

        intermediate = _launch_conv_transpose3d(
            x_bf16, w_bf16, b_f32,
            N, C_in, out_channels, D_in, H_in, W_in,
            D_out, H_out, W_out, stride, padding, K,
        )

        # BatchNorm: use running stats in eval mode, batch stats in training mode
        if self.batch_norm.training:
            mean, var = _launch_bn_compute_stats(intermediate, out_channels, eps=1e-5)
        else:
            mean = self.batch_norm.running_mean.detach().to(torch.float32)
            var = self.batch_norm.running_var.detach().to(torch.float32) + 1e-5

        D_final = D_out // 4
        H_final = H_out // 4
        W_final = W_out // 4

        output = _launch_bn_apply_pool(
            intermediate, gamma_f32, beta_f32, mean, var,
            N, out_channels, D_out, H_out, W_out, D_final, H_final, W_final,
        )

        return output
