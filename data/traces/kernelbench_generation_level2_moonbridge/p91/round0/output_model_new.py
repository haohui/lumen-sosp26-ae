import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Problem constants
BATCH_SIZE = 128
IN_CHANNELS = 64
OUT_CHANNELS = 128
H_IN = 64
W_IN = 64
KERNEL_H = 4
KERNEL_W = 4
STRIDE = 2
PAD = 1
OUTPUT_PAD = 1
H_OUT = 129
W_OUT = 129
SCALE = 2.0

BLOCK_C: al.constexpr = 128


@avelang.jit
def conv_transpose_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    conv_bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    in_c: al.i32,
    out_c: al.i32,
    h_in: al.i32,
    w_in: al.i32,
    h_out: al.i32,
    w_out: al.i32,
    k_h: al.i32,
    k_w: al.i32,
    s_h: al.i32,
    s_w: al.i32,
    p_h: al.i32,
    p_w: al.i32,
):
    tid = al.thread_id(0)
    block_id = al.block_id(0)
    batch_id = al.block_id(1)

    h_idx = block_id // w_out
    w_idx = block_id - h_idx * w_out
    c_out = tid

    if batch_id < batch_size:
        if c_out < out_c:
            if h_idx < h_out:
                if w_idx < w_out:
                    acc = al.convert(0.0, al.f32)

                    layout_in = al.make_layout(
                        (batch_size, in_c, h_in, w_in),
                        (in_c * h_in * w_in, h_in * w_in, w_in, 1),
                    )
                    in_t = al.make_tensor(input_ptr, al.bf16, layout_in)

                    layout_wt = al.make_layout(
                        (in_c, out_c, k_h, k_w),
                        (out_c * k_h * k_w, k_h * k_w, k_w, 1),
                    )
                    wt_t = al.make_tensor(weight_ptr, al.bf16, layout_wt)

                    for kh in al.range(k_h):
                        h_in_val = (h_idx + p_h - kh) // s_h
                        h_check = s_h * h_in_val + kh - p_h
                        if h_check == h_idx:
                            h_in_lo = h_in_val >= 0
                            h_in_hi = h_in_val < h_in
                            if h_in_lo:
                                if h_in_hi:
                                    for kw in al.range(k_w):
                                        w_in_val = (w_idx + p_w - kw) // s_w
                                        w_check = s_w * w_in_val + kw - p_w
                                        if w_check == w_idx:
                                            w_in_lo = w_in_val >= 0
                                            w_in_hi = w_in_val < w_in
                                            if w_in_lo:
                                                if w_in_hi:
                                                    for ci in al.range(in_c):
                                                        in_val = al.convert(in_t[batch_id, ci, h_in_val, w_in_val], al.f32)
                                                        wt_val = al.convert(wt_t[ci, c_out, kh, kw], al.f32)
                                                        acc = acc + in_val * wt_val

                    layout_cb = al.make_layout((out_c,), (1,))
                    cb_t = al.make_tensor(conv_bias_ptr, al.bf16, layout_cb)
                    acc = acc + al.convert(cb_t[c_out], al.f32)

                    layout_out = al.make_layout(
                        (batch_size, out_c, h_out, w_out),
                        (out_c * h_out * w_out, h_out * w_out, w_out, 1),
                    )
                    out_t = al.make_tensor(output_ptr, al.bf16, layout_out)
                    out_t[batch_id, c_out, h_idx, w_idx] = al.convert(acc, al.bf16)


@avelang.jit
def channel_softmax_fused_kernel(
    input_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    channels: al.i32,
    h_out: al.i32,
    w_out: al.i32,
    scale: al.f32,
):
    tid = al.thread_id(0)
    block_id = al.block_id(0)
    batch_id = al.block_id(1)

    h_idx = block_id // w_out
    w_idx = block_id - h_idx * w_out
    c = tid

    if batch_id < batch_size:
        if c < channels:
            if h_idx < h_out:
                if w_idx < w_out:
                    layout_in = al.make_layout(
                        (batch_size, channels, h_out, w_out),
                        (channels * h_out * w_out, h_out * w_out, w_out, 1),
                    )
                    in_t = al.make_tensor(input_ptr, al.bf16, layout_in)

                    layout_bias = al.make_layout((channels, 1, 1), (1, 1, 1))
                    bias_t = al.make_tensor(bias_ptr, al.bf16, layout_bias)

                    layout_out = al.make_layout(
                        (batch_size, channels, h_out, w_out),
                        (channels * h_out * w_out, h_out * w_out, w_out, 1),
                    )
                    out_t = al.make_tensor(output_ptr, al.bf16, layout_out)

                    smem = al.make_shared((BLOCK_C,), al.f32)

                    val = al.convert(in_t[batch_id, c, h_idx, w_idx], al.f32)
                    smem[c] = val
                    al.syncthreads()

                    # Reduce to find max
                    if c < 64:
                        a = smem[c]
                        b = smem[c + 64]
                        smem[c] = a if a > b else b
                    al.syncthreads()
                    if c < 32:
                        a = smem[c]
                        b = smem[c + 32]
                        smem[c] = a if a > b else b
                    al.syncthreads()
                    if c < 16:
                        a = smem[c]
                        b = smem[c + 16]
                        smem[c] = a if a > b else b
                    al.syncthreads()
                    if c < 8:
                        a = smem[c]
                        b = smem[c + 8]
                        smem[c] = a if a > b else b
                    al.syncthreads()
                    if c < 4:
                        a = smem[c]
                        b = smem[c + 4]
                        smem[c] = a if a > b else b
                    al.syncthreads()
                    if c < 2:
                        a = smem[c]
                        b = smem[c + 2]
                        smem[c] = a if a > b else b
                    al.syncthreads()
                    if c < 1:
                        a = smem[c]
                        b = smem[c + 1]
                        smem[c] = a if a > b else b
                    al.syncthreads()

                    max_val = smem[0]

                    # Compute exp(x - max) and store in shared memory
                    shifted = val - max_val
                    exp_val = al.exp(shifted)
                    smem[c] = exp_val
                    al.syncthreads()

                    # Reduce to find sum
                    if c < 64:
                        smem[c] = smem[c] + smem[c + 64]
                    al.syncthreads()
                    if c < 32:
                        smem[c] = smem[c] + smem[c + 32]
                    al.syncthreads()
                    if c < 16:
                        smem[c] = smem[c] + smem[c + 16]
                    al.syncthreads()
                    if c < 8:
                        smem[c] = smem[c] + smem[c + 8]
                    al.syncthreads()
                    if c < 4:
                        smem[c] = smem[c] + smem[c + 4]
                    al.syncthreads()
                    if c < 2:
                        smem[c] = smem[c] + smem[c + 2]
                    al.syncthreads()
                    if c < 1:
                        smem[c] = smem[c] + smem[c + 1]
                    al.syncthreads()

                    sum_val = smem[0]

                    # Softmax normalized value
                    softmax_val = exp_val / sum_val

                    # Add bias, scale, sigmoid
                    bias_val = al.convert(bias_t[c, 0, 0], al.f32)
                    result = softmax_val + bias_val
                    result = result * scale

                    # Sigmoid: 1 / (1 + exp(-x))
                    result_neg = al.convert(0.0, al.f32) - result
                    one_f = al.convert(1.0, al.f32)
                    sig_val = one_f / (one_f + al.exp(result_neg))

                    out_t[batch_id, c, h_idx, w_idx] = al.convert(sig_val, al.bf16)


def _prepare_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_bias: torch.Tensor,
    sep_bias: torch.Tensor,
    scaling_factor: float,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_contiguous(x)
    weight_bf16 = _prepare_bf16_contiguous(weight)
    conv_bias_bf16 = _prepare_bf16_contiguous(conv_bias)
    sep_bias_bf16 = _prepare_bf16_contiguous(sep_bias)

    batch_size = x_bf16.shape[0]
    in_c = x_bf16.shape[1]
    h_in = x_bf16.shape[2]
    w_in = x_bf16.shape[3]
    out_c = weight_bf16.shape[1]
    k_h = weight_bf16.shape[2]
    k_w = weight_bf16.shape[3]

    h_out = (h_in - 1) * STRIDE - 2 * PAD + k_h + OUTPUT_PAD
    w_out = (w_in - 1) * STRIDE - 2 * PAD + k_w + OUTPUT_PAD

    num_spatial = h_out * w_out
    scale_f32 = float(scaling_factor)

    # Intermediate buffer for conv_transpose output
    conv_out = torch.empty(
        (batch_size, out_c, h_out, w_out),
        dtype=torch.bfloat16,
        device=x_bf16.device,
    )

    conv_transpose_kernel[lambda: ((num_spatial, batch_size, 1), (BLOCK_C, 1, 1))](
        x_bf16, weight_bf16, conv_bias_bf16, conv_out,
        batch_size, in_c, out_c, h_in, w_in, h_out, w_out,
        k_h, k_w, STRIDE, STRIDE, PAD, PAD,
    )

    final_out = torch.empty(
        (batch_size, out_c, h_out, w_out),
        dtype=torch.bfloat16,
        device=x_bf16.device,
    )

    channel_softmax_fused_kernel[lambda: ((num_spatial, batch_size, 1), (BLOCK_C, 1, 1))](
        conv_out, sep_bias_bf16, final_out,
        batch_size, out_c, h_out, w_out, scale_f32,
    )

    return final_out


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, applies softmax, adds a bias term,
    scales the result, and applies sigmoid — implemented with AveLang DSL kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        weight = self.conv_transpose.weight.data
        conv_bias = self.conv_transpose.bias.data
        bias = self.bias.data
        result_bf16 = avelang_forward(
            x, weight, conv_bias, bias, self.scaling_factor,
        )
        return result_bf16.to(x.dtype)


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_CHANNELS, H_IN, W_IN)]


def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_H, STRIDE, PAD, OUTPUT_PAD, (OUT_CHANNELS, 1, 1), SCALE]
