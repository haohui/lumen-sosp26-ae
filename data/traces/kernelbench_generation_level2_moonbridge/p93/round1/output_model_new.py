import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── Problem dimensions ──────────────────────────────────────────────────
N_BATCH = 128
C_IN = 64
C_OUT = 128
H_IN = 64
W_IN = 64
K_SIZE = 4
STRIDE = 2
H_OUT = (H_IN - 1) * STRIDE + K_SIZE  # 130
W_OUT = (W_IN - 1) * STRIDE + K_SIZE  # 130
ADD_VALUE = 0.5
MULTIPLY_VALUE = 2.0


# ── GELU approximation using tanh ────────────────────────────────────────
@avelang.jit
def _gelu_approx(x: al.f32) -> al.f32:
    sqrt2_over_pi = al.convert(0.7978845608028654, al.f32)
    coeff = al.convert(0.044715, al.f32)
    one = al.convert(1.0, al.f32)
    half = al.convert(0.5, al.f32)
    x3 = x * x * x
    inner = sqrt2_over_pi * (x + coeff * x3)
    tanh_inner = al.tanh(inner)
    return half * x * (one + tanh_inner)


# ── Main fused ConvTranspose2d + epilogue kernel ─────────────────────────
@avelang.jit
def conv_transpose_fused_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
):
    """
    Each block handles one (batch, oh, ow) output position.
    Each thread handles one output channel.
    Grid: (H_OUT, W_OUT, N_BATCH)  Block: (C_OUT, 1, 1)
    """
    oh_u32 = al.block_id(0)
    ow_u32 = al.block_id(1)
    batch_u32 = al.block_id(2)
    oc = al.thread_id(0)

    oh = al.convert(oh_u32, al.i32)
    ow = al.convert(ow_u32, al.i32)
    batch = al.convert(batch_u32, al.i32)
    oc_i32 = al.convert(oc, al.i32)

    # Bounds check
    h_out_i32 = al.convert(H_OUT, al.i32)
    w_out_i32 = al.convert(W_OUT, al.i32)
    n_batch_i32 = al.convert(N_BATCH, al.i32)
    c_out_i32 = al.convert(C_OUT, al.i32)
    if oh >= h_out_i32 or ow >= w_out_i32 or batch >= n_batch_i32 or oc_i32 >= c_out_i32:
        return

    # Flat tensor views
    x_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((N_BATCH * C_IN * H_IN * W_IN,), (1,)))
    w_flat = al.make_tensor(w_ptr, al.bf16, al.make_layout((C_IN * C_OUT * K_SIZE * K_SIZE,), (1,)))
    out_flat = al.make_tensor(out_ptr, al.bf16, al.make_layout((N_BATCH * C_OUT * H_OUT * W_OUT,), (1,)))

    # Strides (in elements)
    x_sb = al.convert(C_IN * H_IN * W_IN, al.i32)   # 262144
    x_sic = al.convert(H_IN * W_IN, al.i32)          # 4096
    x_sh = al.convert(W_IN, al.i32)                  # 64
    w_sic = al.convert(C_OUT * K_SIZE * K_SIZE, al.i32)  # 2048
    w_soc = al.convert(K_SIZE * K_SIZE, al.i32)          # 16
    out_sb = al.convert(C_OUT * H_OUT * W_OUT, al.i32)   # 2163200
    out_soc = al.convert(H_OUT * W_OUT, al.i32)          # 16900
    out_soh = al.convert(W_OUT, al.i32)                  # 130

    # Constants
    stride_i32 = al.convert(STRIDE, al.i32)
    h_in_i32 = al.convert(H_IN, al.i32)
    w_in_i32 = al.convert(W_IN, al.i32)
    zero_i32 = al.convert(0, al.i32)
    one_i32 = al.convert(1, al.i32)
    three_i32 = al.convert(3, al.i32)
    zero_f32 = al.convert(0.0, al.f32)
    add_val_f32 = al.convert(ADD_VALUE, al.f32)
    mul_val_f32 = al.convert(MULTIPLY_VALUE, al.f32)

    # Accumulate in f64 for maximum precision
    acc = al.convert(0.0, al.f64)

    for ic in al.range(C_IN):
        ic_i32 = al.convert(ic, al.i32)
        w_base = ic_i32 * w_sic + oc_i32 * w_soc
        x_base = batch * x_sb + ic_i32 * x_sic

        # (0,0): oh even, ow even
        if oh % stride_i32 == zero_i32 and ow % stride_i32 == zero_i32:
            h_in = oh // stride_i32
            w_in = ow // stride_i32
            if h_in < h_in_i32 and w_in < w_in_i32:
                xv = al.convert(x_flat[x_base + h_in * x_sh + w_in], al.f64)
                wv = al.convert(w_flat[w_base], al.f64)
                acc = acc + xv * wv

        # (0,1): oh even, ow odd
        if oh % stride_i32 == zero_i32 and ow >= one_i32:
            ow_m1 = ow - one_i32
            if ow_m1 % stride_i32 == zero_i32:
                h_in = oh // stride_i32
                w_in = ow_m1 // stride_i32
                if h_in < h_in_i32 and w_in < w_in_i32:
                    xv = al.convert(x_flat[x_base + h_in * x_sh + w_in], al.f64)
                    wv = al.convert(w_flat[w_base + 1], al.f64)
                    acc = acc + xv * wv

        # (0,2): oh even, ow even (2 back)
        if oh % stride_i32 == zero_i32 and ow >= stride_i32:
            ow_m2 = ow - stride_i32
            if ow_m2 % stride_i32 == zero_i32:
                h_in = oh // stride_i32
                w_in = ow_m2 // stride_i32
                if h_in < h_in_i32 and w_in < w_in_i32:
                    xv = al.convert(x_flat[x_base + h_in * x_sh + w_in], al.f64)
                    wv = al.convert(w_flat[w_base + 2], al.f64)
                    acc = acc + xv * wv

        # (0,3): oh even, ow odd (3 back)
        if oh % stride_i32 == zero_i32 and ow >= three_i32:
            ow_m3 = ow - three_i32
            if ow_m3 % stride_i32 == zero_i32:
                h_in = oh // stride_i32
                w_in = ow_m3 // stride_i32
                if h_in < h_in_i32 and w_in < w_in_i32:
                    xv = al.convert(x_flat[x_base + h_in * x_sh + w_in], al.f64)
                    wv = al.convert(w_flat[w_base + 3], al.f64)
                    acc = acc + xv * wv

        # (1,0): oh odd, ow even
        if oh >= one_i32:
            oh_m1 = oh - one_i32
            if oh_m1 % stride_i32 == zero_i32 and ow % stride_i32 == zero_i32:
                h_in = oh_m1 // stride_i32
                w_in = ow // stride_i32
                if h_in < h_in_i32 and w_in < w_in_i32:
                    xv = al.convert(x_flat[x_base + h_in * x_sh + w_in], al.f64)
                    wv = al.convert(w_flat[w_base + 4], al.f64)
                    acc = acc + xv * wv

        # (1,1): oh odd, ow odd
        if oh >= one_i32 and ow >= one_i32:
            oh_m1 = oh - one_i32
            ow_m1 = ow - one_i32
            if oh_m1 % stride_i32 == zero_i32 and ow_m1 % stride_i32 == zero_i32:
                h_in = oh_m1 // stride_i32
                w_in = ow_m1 // stride_i32
                if h_in < h_in_i32 and w_in < w_in_i32:
                    xv = al.convert(x_flat[x_base + h_in * x_sh + w_in], al.f64)
                    wv = al.convert(w_flat[w_base + 5], al.f64)
                    acc = acc + xv * wv

        # (1,2): oh odd, ow even
        if oh >= one_i32 and ow >= stride_i32:
            oh_m1 = oh - one_i32
            ow_m2 = ow - stride_i32
            if oh_m1 % stride_i32 == zero_i32 and ow_m2 % stride_i32 == zero_i32:
                h_in = oh_m1 // stride_i32
                w_in = ow_m2 // stride_i32
                if h_in < h_in_i32 and w_in < w_in_i32:
                    xv = al.convert(x_flat[x_base + h_in * x_sh + w_in], al.f64)
                    wv = al.convert(w_flat[w_base + 6], al.f64)
                    acc = acc + xv * wv

        # (1,3): oh odd, ow odd
        if oh >= one_i32 and ow >= three_i32:
            oh_m1 = oh - one_i32
            ow_m3 = ow - three_i32
            if oh_m1 % stride_i32 == zero_i32 and ow_m3 % stride_i32 == zero_i32:
                h_in = oh_m1 // stride_i32
                w_in = ow_m3 // stride_i32
                if h_in < h_in_i32 and w_in < w_in_i32:
                    xv = al.convert(x_flat[x_base + h_in * x_sh + w_in], al.f64)
                    wv = al.convert(w_flat[w_base + 7], al.f64)
                    acc = acc + xv * wv

        # (2,0): oh even (2 back), ow even
        if oh >= stride_i32:
            oh_m2 = oh - stride_i32
            if oh_m2 % stride_i32 == zero_i32 and ow % stride_i32 == zero_i32:
                h_in = oh_m2 // stride_i32
                w_in = ow // stride_i32
                if h_in < h_in_i32 and w_in < w_in_i32:
                    xv = al.convert(x_flat[x_base + h_in * x_sh + w_in], al.f64)
                    wv = al.convert(w_flat[w_base + 8], al.f64)
                    acc = acc + xv * wv

        # (2,1): oh even, ow odd
        if oh >= stride_i32 and ow >= one_i32:
            oh_m2 = oh - stride_i32
            ow_m1 = ow - one_i32
            if oh_m2 % stride_i32 == zero_i32 and ow_m1 % stride_i32 == zero_i32:
                h_in = oh_m2 // stride_i32
                w_in = ow_m1 // stride_i32
                if h_in < h_in_i32 and w_in < w_in_i32:
                    xv = al.convert(x_flat[x_base + h_in * x_sh + w_in], al.f64)
                    wv = al.convert(w_flat[w_base + 9], al.f64)
                    acc = acc + xv * wv

        # (2,2): oh even, ow even
        if oh >= stride_i32 and ow >= stride_i32:
            oh_m2 = oh - stride_i32
            ow_m2 = ow - stride_i32
            if oh_m2 % stride_i32 == zero_i32 and ow_m2 % stride_i32 == zero_i32:
                h_in = oh_m2 // stride_i32
                w_in = ow_m2 // stride_i32
                if h_in < h_in_i32 and w_in < w_in_i32:
                    xv = al.convert(x_flat[x_base + h_in * x_sh + w_in], al.f64)
                    wv = al.convert(w_flat[w_base + 10], al.f64)
                    acc = acc + xv * wv

        # (2,3): oh even, ow odd
        if oh >= stride_i32 and ow >= three_i32:
            oh_m2 = oh - stride_i32
            ow_m3 = ow - three_i32
            if oh_m2 % stride_i32 == zero_i32 and ow_m3 % stride_i32 == zero_i32:
                h_in = oh_m2 // stride_i32
                w_in = ow_m3 // stride_i32
                if h_in < h_in_i32 and w_in < w_in_i32:
                    xv = al.convert(x_flat[x_base + h_in * x_sh + w_in], al.f64)
                    wv = al.convert(w_flat[w_base + 11], al.f64)
                    acc = acc + xv * wv

        # (3,0): oh odd (3 back), ow even
        if oh >= three_i32:
            oh_m3 = oh - three_i32
            if oh_m3 % stride_i32 == zero_i32 and ow % stride_i32 == zero_i32:
                h_in = oh_m3 // stride_i32
                w_in = ow // stride_i32
                if h_in < h_in_i32 and w_in < w_in_i32:
                    xv = al.convert(x_flat[x_base + h_in * x_sh + w_in], al.f64)
                    wv = al.convert(w_flat[w_base + 12], al.f64)
                    acc = acc + xv * wv

        # (3,1): oh odd, ow odd
        if oh >= three_i32 and ow >= one_i32:
            oh_m3 = oh - three_i32
            ow_m1 = ow - one_i32
            if oh_m3 % stride_i32 == zero_i32 and ow_m1 % stride_i32 == zero_i32:
                h_in = oh_m3 // stride_i32
                w_in = ow_m1 // stride_i32
                if h_in < h_in_i32 and w_in < w_in_i32:
                    xv = al.convert(x_flat[x_base + h_in * x_sh + w_in], al.f64)
                    wv = al.convert(w_flat[w_base + 13], al.f64)
                    acc = acc + xv * wv

        # (3,2): oh odd, ow even
        if oh >= three_i32 and ow >= stride_i32:
            oh_m3 = oh - three_i32
            ow_m2 = ow - stride_i32
            if oh_m3 % stride_i32 == zero_i32 and ow_m2 % stride_i32 == zero_i32:
                h_in = oh_m3 // stride_i32
                w_in = ow_m2 // stride_i32
                if h_in < h_in_i32 and w_in < w_in_i32:
                    xv = al.convert(x_flat[x_base + h_in * x_sh + w_in], al.f64)
                    wv = al.convert(w_flat[w_base + 14], al.f64)
                    acc = acc + xv * wv

        # (3,3): oh odd, ow odd
        if oh >= three_i32 and ow >= three_i32:
            oh_m3 = oh - three_i32
            ow_m3 = ow - three_i32
            if oh_m3 % stride_i32 == zero_i32 and ow_m3 % stride_i32 == zero_i32:
                h_in = oh_m3 // stride_i32
                w_in = ow_m3 // stride_i32
                if h_in < h_in_i32 and w_in < w_in_i32:
                    xv = al.convert(x_flat[x_base + h_in * x_sh + w_in], al.f64)
                    wv = al.convert(w_flat[w_base + 15], al.f64)
                    acc = acc + xv * wv

    # Epilogue: convert f64→f32, add, min(0), gelu, multiply
    result = al.convert(acc, al.f32)
    result = result + add_val_f32
    if result > zero_f32:
        result = zero_f32
    result = _gelu_approx(result)
    result = result * mul_val_f32

    # Write output
    out_off = batch * out_sb + oc_i32 * out_soc + oh * out_soh + ow
    out_flat[out_off] = al.convert(result, al.bf16)


# ── Host wrapper ─────────────────────────────────────────────────────────
def avelang_conv_transpose_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x = x.cuda() if not x.is_cuda else x
    x_bf16 = x.contiguous().to(dtype=torch.bfloat16)
    w = weight.cuda() if not weight.is_cuda else weight
    w_bf16 = w.contiguous().to(dtype=torch.bfloat16)

    out = torch.empty(
        (N_BATCH, C_OUT, H_OUT, W_OUT),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )

    conv_transpose_fused_kernel[lambda: (
        (H_OUT, W_OUT, N_BATCH),
        (C_OUT, 1, 1),
    )](
        x_bf16, w_bf16, out,
    )

    torch.cuda.synchronize()
    return out


class ModelNew(nn.Module):
    """
    BF16 ConvTranspose2d with fused add, min, GELU, multiply epilogue
    implemented as a single AveLang kernel with f64 accumulation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        weight = self.conv_transpose.weight.data
        return avelang_conv_transpose_fused(x, weight)


# Keep original globals for compatibility with get_inputs / get_init_inputs
batch_size = N_BATCH
in_channels = C_IN
out_channels = C_OUT
height, width = H_IN, W_IN
kernel_size = K_SIZE
stride = STRIDE
add_value = ADD_VALUE
multiply_value = MULTIPLY_VALUE


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]
