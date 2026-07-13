import torch
import torch.nn as nn
import math
import struct
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# Kernel 1: Tiled 3D Transposed Convolution + bias + scale1
#
# Strategy: each block loads the tiny weight into shared memory, then each
# thread strides over a portion of the output elements. Scale1 is fused in.
# Kernel dimensions (KD, KH, KW, stride, padding) are constexpr to enable
# compiler loop unrolling for this fixed-problem-size benchmark.
# ---------------------------------------------------------------------------
@avelang.jit
def conv_transpose3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    scale1_bits: al.i32,
    output_ptr: al.Pointer(al.bf16),
    total_in: al.i32,
    total_wt: al.i32,
    total_bias: al.i32,
    total_out: al.i32,
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    KD: al.constexpr,
    KH: al.constexpr,
    KW: al.constexpr,
    stride_val: al.constexpr,
    padding_val: al.constexpr,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    BLOCK_SIZE: al.constexpr,
    TOTAL_WT: al.constexpr,
    GRID_STRIDE: al.constexpr,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    # Load entire weight into shared memory
    wt_shared = al.make_shared((TOTAL_WT,), al.bf16)
    wt_global = al.make_tensor(weight_ptr, al.bf16, al.make_layout((total_wt,), (1,)))

    scale1_val = al.bitcast(scale1_bits, al.f32)

    # Cooperative weight load
    for i in al.range((TOTAL_WT + BLOCK_SIZE - 1) // BLOCK_SIZE):
        idx = tid + i * BLOCK_SIZE
        if idx < TOTAL_WT:
            wt_shared[idx] = wt_global[idx]
    al.syncthreads()

    in_t = al.make_tensor(input_ptr, al.bf16, al.make_layout((total_in,), (1,)))
    bias_t = al.make_tensor(bias_ptr, al.bf16, al.make_layout((total_bias,), (1,)))
    out_t = al.make_tensor(output_ptr, al.bf16, al.make_layout((total_out,), (1,)))

    # Precompute strides
    in_stride_C = D_in * H_in * W_in
    in_stride_D = H_in * W_in
    in_stride_H = W_in

    wt_stride_OC = KD * KH * KW
    wt_stride_KD = KH * KW
    wt_stride_KH = KW

    out_stride_C = D_out * H_out * W_out
    out_stride_D = H_out * W_out
    out_stride_H = W_out

    start = tid + bid * BLOCK_SIZE
    for gid in al.range(start, total_out, GRID_STRIDE):
        w = gid % W_out
        tmp = gid // W_out
        h = tmp % H_out
        tmp = tmp // H_out
        d = tmp % D_out
        tmp = tmp // D_out
        oc = tmp % OC
        b = tmp // OC

        acc = al.convert(0.0, al.f32)

        for ic_i in al.range(IC):
            for kd_i in al.range(KD):
                d_in_val = d + padding_val - kd_i
                d_valid = (d_in_val % stride_val) == 0
                d_in_idx = d_in_val // stride_val
                d_ok = d_valid & (d_in_idx >= 0) & (d_in_idx < D_in)
                for kh_i in al.range(KH):
                    h_in_val = h + padding_val - kh_i
                    h_valid = (h_in_val % stride_val) == 0
                    h_in_idx = h_in_val // stride_val
                    h_ok = h_valid & (h_in_idx >= 0) & (h_in_idx < H_in)
                    for kw_i in al.range(KW):
                        w_in_val = w + padding_val - kw_i
                        w_valid = (w_in_val % stride_val) == 0
                        w_in_idx = w_in_val // stride_val
                        w_ok = w_valid & (w_in_idx >= 0) & (w_in_idx < W_in)
                        ok = d_ok & h_ok & w_ok
                        if ok:
                            in_idx = (b * IC + ic_i) * in_stride_C + d_in_idx * in_stride_D + h_in_idx * in_stride_H + w_in_idx
                            wt_idx = (ic_i * OC + oc) * wt_stride_OC + kd_i * wt_stride_KD + kh_i * wt_stride_KH + kw_i
                            inp_val = al.convert(in_t[in_idx], al.f32)
                            wt_val = al.convert(wt_shared[wt_idx], al.f32)
                            acc = acc + inp_val * wt_val

        # Add ConvT bias and apply scale1
        bias_val = al.convert(bias_t[oc], al.f32)
        acc = acc + bias_val
        acc = acc * scale1_val

        out_idx = (b * OC + oc) * out_stride_C + d * out_stride_D + h * out_stride_H + w
        out_t[out_idx] = al.convert(acc, al.bf16)


# ---------------------------------------------------------------------------
# Kernel 2: Fused avg_pool_3d(2,2,2) + bias + scale2
# ---------------------------------------------------------------------------
@avelang.jit
def scale_pool_bias_scale_kernel(
    input_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    total_in: al.i32,
    total_bias: al.i32,
    total_out: al.i32,
    B: al.i32,
    OC: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    scale2_bits: al.i32,
    POOL_SIZE: al.constexpr,
    BLOCK_SIZE: al.constexpr,
    GRID_STRIDE: al.constexpr,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    in_t = al.make_tensor(input_ptr, al.bf16, al.make_layout((total_in,), (1,)))
    bias_t = al.make_tensor(bias_ptr, al.bf16, al.make_layout((total_bias,), (1,)))
    out_t = al.make_tensor(output_ptr, al.bf16, al.make_layout((total_out,), (1,)))

    scale2_val = al.bitcast(scale2_bits, al.f32)

    in_stride_C = D_in * H_in * W_in
    in_stride_D = H_in * W_in
    in_stride_H = W_in

    out_stride_C = D_out * H_out * W_out
    out_stride_D = H_out * W_out
    out_stride_H = W_out

    start = tid + bid * BLOCK_SIZE
    for gid in al.range(start, total_out, GRID_STRIDE):
        wp = gid % W_out
        tmp = gid // W_out
        hp = tmp % H_out
        tmp = tmp // H_out
        dp = tmp % D_out
        tmp = tmp // D_out
        oc = tmp % OC
        b = tmp // OC

        acc = al.convert(0.0, al.f32)
        count_val = al.convert(0.0, al.f32)

        d_start = dp * POOL_SIZE
        h_start = hp * POOL_SIZE
        w_start = wp * POOL_SIZE

        for di in al.range(POOL_SIZE):
            d_idx = d_start + di
            for hi in al.range(POOL_SIZE):
                h_idx = h_start + hi
                for wi in al.range(POOL_SIZE):
                    w_idx = w_start + wi
                    valid = (d_idx < D_in) & (h_idx < H_in) & (w_idx < W_in)
                    if valid:
                        in_idx = (b * OC + oc) * in_stride_C + d_idx * in_stride_D + h_idx * in_stride_H + w_idx
                        val = al.convert(in_t[in_idx], al.f32)
                        acc = acc + val
                        count_val = count_val + al.convert(1.0, al.f32)

        avg_val = acc / count_val
        bias_val = al.convert(bias_t[oc], al.f32)
        result = avg_val + bias_val
        result = result * scale2_val

        out_idx = (b * OC + oc) * out_stride_C + dp * out_stride_D + hp * out_stride_H + wp
        out_t[out_idx] = al.convert(result, al.bf16)


# ---------------------------------------------------------------------------
# Host-side helpers
# ---------------------------------------------------------------------------
BLOCK_SIZE = 256
POOL_SIZE = 2
NUM_BLOCKS_CONV = 16384
NUM_BLOCKS_POOL = 4096


def _float_to_bits(f: float) -> int:
    return struct.unpack('<i', struct.pack('<f', float(f)))[0]


def _compute_conv_transpose_output_size(size_in, kernel, stride, padding):
    return (size_in - 1) * stride - 2 * padding + kernel


def _init_conv_transpose_weight(in_channels, out_channels, kernel_size):
    w = torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size)
    nn.init.kaiming_uniform_(w, a=math.sqrt(5))
    return nn.Parameter(w)


def _init_conv_transpose_bias(weight, out_channels):
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1.0 / math.sqrt(fan_in) if fan_in != 0 else 0.0
    b = torch.empty(out_channels)
    nn.init.uniform_(b, -bound, bound)
    return nn.Parameter(b)


def _init_post_bias(bias_shape):
    return nn.Parameter(torch.randn(bias_shape))


def run_conv_transpose3d(
    scale1_val: float,
    x_bf16: torch.Tensor,
    weight_bf16: torch.Tensor,
    bias_bf16: torch.Tensor,
    B: int, IC: int, OC: int,
    D_in: int, H_in: int, W_in: int,
    KD: int, KH: int, KW: int,
    stride_val: int, padding_val: int,
    D_out: int, H_out: int, W_out: int,
) -> torch.Tensor:
    total_out = B * OC * D_out * H_out * W_out
    out = torch.empty(total_out, dtype=torch.bfloat16, device=x_bf16.device)

    total_in = B * IC * D_in * H_in * W_in
    total_wt = IC * OC * KD * KH * KW
    total_bias = OC

    max_blocks = NUM_BLOCKS_CONV
    num_blocks = min(max_blocks, (total_out + BLOCK_SIZE - 1) // BLOCK_SIZE)
    actual_stride = num_blocks * BLOCK_SIZE

    conv_transpose3d_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16.data_ptr(), weight_bf16.data_ptr(), bias_bf16.data_ptr(),
        _float_to_bits(scale1_val),
        out.data_ptr(),
        total_in, total_wt, total_bias, total_out,
        B, IC, OC, D_in, H_in, W_in,
        KD, KH, KW, stride_val, padding_val,
        D_out, H_out, W_out,
        BLOCK_SIZE, total_wt, actual_stride,
    )
    return out.reshape(B, OC, D_out, H_out, W_out)


def run_scale_pool_bias_scale(
    scale2_val: float,
    x_bf16: torch.Tensor,
    bias_bf16: torch.Tensor,
    B: int, OC: int,
    D_in: int, H_in: int, W_in: int,
    D_out: int, H_out: int, W_out: int,
) -> torch.Tensor:
    total_out = B * OC * D_out * H_out * W_out
    out = torch.empty(total_out, dtype=torch.bfloat16, device=x_bf16.device)

    total_in = B * OC * D_in * H_in * W_in
    total_bias = OC

    max_blocks = NUM_BLOCKS_POOL
    num_blocks = min(max_blocks, (total_out + BLOCK_SIZE - 1) // BLOCK_SIZE)
    actual_stride = num_blocks * BLOCK_SIZE

    scale_pool_bias_scale_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16.data_ptr(), bias_bf16.data_ptr(), out.data_ptr(),
        total_in, total_bias, total_out,
        B, OC, D_in, H_in, W_in, D_out, H_out, W_out,
        _float_to_bits(scale2_val),
        POOL_SIZE, BLOCK_SIZE, actual_stride,
    )
    return out.reshape(B, OC, D_out, H_out, W_out)


# ---------------------------------------------------------------------------
# ModelNew entrypoint
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale1 = scale1
        self.scale2 = scale2
        self.bias_shape = bias_shape

        self.weight = _init_conv_transpose_weight(in_channels, out_channels, kernel_size)
        self.conv_bias = _init_conv_transpose_bias(self.weight, out_channels)
        self.post_bias = _init_post_bias(bias_shape)

    def forward(self, x):
        B, IC, D_in, H_in, W_in = x.shape
        OC = self.out_channels
        KD = self.kernel_size
        KH = self.kernel_size
        KW = self.kernel_size
        stride_val = self.stride if isinstance(self.stride, int) else self.stride[0]
        padding_val = self.padding if isinstance(self.padding, int) else self.padding[0]

        D_ct = _compute_conv_transpose_output_size(D_in, KD, stride_val, padding_val)
        H_ct = _compute_conv_transpose_output_size(H_in, KH, stride_val, padding_val)
        W_ct = _compute_conv_transpose_output_size(W_in, KW, stride_val, padding_val)

        D_p = D_ct // POOL_SIZE
        H_p = H_ct // POOL_SIZE
        W_p = W_ct // POOL_SIZE

        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = self.weight.to(torch.bfloat16).contiguous()
        conv_bias_bf16 = self.conv_bias.to(torch.bfloat16).contiguous()
        post_bias_bf16 = self.post_bias.to(torch.bfloat16).contiguous().reshape(-1)

        ct_out = run_conv_transpose3d(
            float(self.scale1),
            x_bf16, w_bf16, conv_bias_bf16,
            B, IC, OC, D_in, H_in, W_in,
            KD, KH, KW,
            stride_val, padding_val,
            D_ct, H_ct, W_ct,
        )
        ct_out = ct_out.contiguous()

        result = run_scale_pool_bias_scale(
            float(self.scale2),
            ct_out, post_bias_bf16,
            B, OC,
            D_ct, H_ct, W_ct,
            D_p, H_p, W_p,
        )

        return result
