import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256


@avelang.jit
def conv3d_fused_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    conv_bias_ptr: al.Pointer(al.bf16),
    extra_bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    IC: al.i32,
    OC: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    KD: al.i32,
    KH: al.i32,
    KW: al.i32,
    OD: al.i32,
    OH: al.i32,
    OW: al.i32,
    total_elements: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    idx = bid * BLOCK_SIZE + tid

    if idx < total_elements:
        # Build tensor views
        x_layout = al.make_layout(
            (N, IC, D, H, W),
            (IC * D * H * W, D * H * W, H * W, W, 1),
        )
        x = al.make_tensor(x_ptr, al.bf16, x_layout)

        w_layout = al.make_layout(
            (OC, IC, KD, KH, KW),
            (IC * KD * KH * KW, KD * KH * KW, KH * KW, KW, 1),
        )
        w = al.make_tensor(w_ptr, al.bf16, w_layout)

        cb_layout = al.make_layout((OC,), (1,))
        conv_bias = al.make_tensor(conv_bias_ptr, al.bf16, cb_layout)

        eb_layout = al.make_layout((OC,), (1,))
        extra_bias = al.make_tensor(extra_bias_ptr, al.bf16, eb_layout)

        o_layout = al.make_layout(
            (N, OC, OD, OH, OW),
            (OC * OD * OH * OW, OD * OH * OW, OH * OW, OW, 1),
        )
        out = al.make_tensor(out_ptr, al.bf16, o_layout)

        # Decompose flat index into (n, oc, od, oh, ow)
        t0 = OC * OD * OH * OW
        n = idx // t0
        r0 = idx - n * t0
        t1 = OD * OH * OW
        oc = r0 // t1
        r1 = r0 - oc * t1
        t2 = OH * OW
        od = r1 // t2
        r2 = r1 - od * t2
        t3 = OW
        oh = r2 // t3
        ow = r2 - oh * t3

        # Accumulate conv3d in FP32
        acc = al.convert(0.0, al.f32)
        for ic in al.range(IC):
            for kd in al.range(KD):
                for kh in al.range(KH):
                    for kw in al.range(KW):
                        in_val = al.convert(
                            x[n, ic, od + kd, oh + kh, ow + kw], al.f32
                        )
                        w_val = al.convert(w[oc, ic, kd, kh, kw], al.f32)
                        acc = acc + in_val * w_val

        # Conv bias
        acc = acc + al.convert(conv_bias[oc], al.f32)

        # ReLU
        zero = al.convert(0.0, al.f32)
        if acc < zero:
            acc = zero

        # LeakyReLU(0.01)
        neg_slope = al.convert(0.01, al.f32)
        if acc < zero:
            acc = acc * neg_slope

        # GELU tanh approximation
        sqrt_2_div_pi = al.convert(0.7978845608028654, al.f32)
        gelu_coeff = al.convert(0.044715, al.f32)
        half = al.convert(0.5, al.f32)
        one = al.convert(1.0, al.f32)
        x3 = acc * acc * acc
        inner = sqrt_2_div_pi * (acc + gelu_coeff * x3)
        tanh_val = al.tanh(inner)
        acc = half * acc * (one + tanh_val)

        # Sigmoid
        neg_acc = zero - acc
        exp_neg = al.exp(neg_acc)
        acc = one / (one + exp_neg)

        # Extra bias
        acc = acc + al.convert(extra_bias[oc], al.f32)

        # Store
        out[n, oc, od, oh, ow] = al.convert(acc, al.bf16)


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv3d_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_bias: torch.Tensor,
    extra_bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)
    cb_bf16 = _to_bf16_contiguous(conv_bias)
    eb_bf16 = _to_bf16_contiguous(extra_bias.reshape(-1))

    N_val, IC_val, D_val, H_val, W_val = x_bf16.shape
    OC_val, IC_w, KD_val, KH_val, KW_val = w_bf16.shape

    OD_val = D_val - KD_val + 1
    OH_val = H_val - KH_val + 1
    OW_val = W_val - KW_val + 1

    total_elements = N_val * OC_val * OD_val * OH_val * OW_val
    out = torch.empty(
        (N_val, OC_val, OD_val, OH_val, OW_val),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )

    grid_x = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    conv3d_fused_kernel[lambda: ((grid_x, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, w_bf16, cb_bf16, eb_bf16, out,
        N_val, IC_val, OC_val, D_val, H_val, W_val,
        KD_val, KH_val, KW_val, OD_val, OH_val, OW_val,
        total_elements,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        weight = self.conv.weight.data
        conv_bias = self.conv.bias.data
        extra_bias = self.bias.data
        result = avelang_conv3d_fused(x, weight, conv_bias, extra_bias)
        return result.to(x.dtype)


batch_size = 64
in_channels = 8
out_channels = 32
depth, height, width = 32, 64, 64
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
