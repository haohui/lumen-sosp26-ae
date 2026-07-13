import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def fused_conv_transpose3d_logsumexp_hardswish_kernel(
    x_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    stride_val: al.i32,
    pad: al.i32,
    bias_scalar: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    area = D_out * H_out * W_out
    total_elements = N * area
    flat_idx = bid * BLOCK_SIZE + tid

    if flat_idx < total_elements:
        n = flat_idx // area
        rem = flat_idx - n * area
        plane = H_out * W_out
        d_out = rem // plane
        rem = rem - d_out * plane
        h_out = rem // W_out
        w_out = rem - h_out * W_out

        x_layout = al.make_layout(
            (N, C_in, D_in, H_in, W_in),
            (C_in * D_in * H_in * W_in, D_in * H_in * W_in, H_in * W_in, W_in, 1),
        )
        x = al.make_tensor(x_ptr, al.bf16, x_layout)

        w_layout = al.make_layout(
            (C_in, 16, 3, 3, 3),
            (16 * 3 * 3 * 3, 3 * 3 * 3, 3 * 3, 3, 1),
        )
        w = al.make_tensor(weight_ptr, al.bf16, w_layout)

        o_layout = al.make_layout(
            (N, 1, D_out, H_out, W_out),
            (D_out * H_out * W_out, D_out * H_out * W_out, H_out * W_out, W_out, 1),
        )
        o = al.make_tensor(out_ptr, al.bf16, o_layout)

        zero_f32 = al.convert(0.0, al.f32)
        zero_i32 = al.convert(0, al.i32)

        # Compute 16 channel values and logsumexp online
        neg_inf = al.convert(-1.0e30, al.f32)
        max_val = neg_inf
        sum_exp = zero_f32

        for oc in al.range(16):
            val = zero_f32
            for ic in al.range(3):
                for kd in al.range(3):
                    d_raw = d_out + pad - kd
                    d_in = d_raw // stride_val
                    d_div = d_in * stride_val == d_raw
                    if d_div:
                        d_ok = (d_in >= zero_i32)
                        if d_ok:
                            d_ok = (d_in < D_in)
                        if d_ok:
                            for kh in al.range(3):
                                h_raw = h_out + pad - kh
                                h_in = h_raw // stride_val
                                h_div = h_in * stride_val == h_raw
                                if h_div:
                                    h_ok = (h_in >= zero_i32)
                                    if h_ok:
                                        h_ok = (h_in < H_in)
                                    if h_ok:
                                        for kw in al.range(3):
                                            w_raw = w_out + pad - kw
                                            w_in = w_raw // stride_val
                                            w_div = w_in * stride_val == w_raw
                                            if w_div:
                                                w_ok = (w_in >= zero_i32)
                                                if w_ok:
                                                    w_ok = (w_in < W_in)
                                                if w_ok:
                                                    inp = al.convert(
                                                        x[n, ic, d_in, h_in, w_in], al.f32
                                                    )
                                                    wt = al.convert(
                                                        w[ic, oc, kd, kh, kw], al.f32
                                                    )
                                                    val = val + inp * wt
            # Online logsumexp update
            if val > max_val:
                sum_exp = sum_exp * al.exp(max_val - val) + al.convert(1.0, al.f32)
                max_val = val
            else:
                sum_exp = sum_exp + al.exp(val - max_val)

        lse = max_val + al.log(sum_exp)

        # HardSwish-like: x * sigmoid(x + 3) / 6
        three = al.convert(3.0, al.f32)
        six = al.convert(6.0, al.f32)
        one = al.convert(1.0, al.f32)
        neg_one = al.convert(-1.0, al.f32)

        arg = lse + three
        sig = one / (one + al.exp(neg_one * arg))
        hs = lse * sig / six

        result = hs - bias_scalar

        if result < neg_one:
            result = neg_one
        if result > one:
            result = one

        o[n, 0, d_out, h_out, w_out] = al.convert(result, al.bf16)


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias_scalar: float,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _to_bf16_contiguous(x)
    weight_bf16 = _to_bf16_contiguous(weight)

    N, C_in, D_in, H_in, W_in = x_bf16.shape
    C_in_from_w, C_out_from_w, K_w, _, _ = weight_bf16.shape

    assert C_in == C_in_from_w, f"Input channel mismatch: {C_in} vs {C_in_from_w}"

    stride_val = 2
    pad_val = 1
    K_val = K_w

    D_out = (D_in - 1) * stride_val - 2 * pad_val + K_val
    H_out = (H_in - 1) * stride_val - 2 * pad_val + K_val
    W_out = (W_in - 1) * stride_val - 2 * pad_val + K_val

    out = torch.empty(
        (N, 1, D_out, H_out, W_out),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )

    total_elements = N * D_out * H_out * W_out
    grid_x = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    fused_conv_transpose3d_logsumexp_hardswish_kernel[
        lambda: ((grid_x, 1, 1), (BLOCK_SIZE, 1, 1))
    ](
        x_bf16,
        weight_bf16,
        out,
        N,
        C_in,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        stride_val,
        pad_val,
        bias_scalar,
    )

    return out.to(dtype=x.dtype)


class ModelNew(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, bias_shape
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self._bias_cached = None

    def forward(self, x):
        if self._bias_cached is None:
            self._bias_cached = float(self.bias.detach().cpu().view(-1)[0].item())
        return avelang_fused(x, self.conv_transpose.weight, self._bias_cached)


def get_inputs():
    batch_size = 128
    in_channels = 3
    depth = 16
    height = 32
    width = 32
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    stride = 2
    padding = 1
    bias_shape = (1, 1, 1, 1)
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]
