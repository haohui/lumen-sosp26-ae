import torch
import torch.nn as nn
import avelang
import avelang.language as al

batch_size = 4
in_channels = 32
out_channels = 64
depth, height, width = 32, 128, 128
kernel_size = 3
stride = 1
padding = 1

BLOCK_SZ: al.constexpr = 256


@avelang.jit
def conv3d_maxpool3d_bias_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    ID: al.i32,
    IH: al.i32,
    IW: al.i32,
    OD: al.i32,
    OH: al.i32,
    OW: al.i32,
    KD: al.i32,
    KH: al.i32,
    KW: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    idx = bid * BLOCK_SZ + tid
    total = B * OC * OD * OH * OW

    if idx < total:
        b = idx // (OC * OD * OH * OW)
        rem = idx - b * (OC * OD * OH * OW)
        oc = rem // (OD * OH * OW)
        rem = rem - oc * (OD * OH * OW)
        dp = rem // (OH * OW)
        rem = rem - dp * (OH * OW)
        hp = rem // OW
        wp = rem - hp * OW

        layout_x = al.make_layout(
            (B, IC, ID, IH, IW),
            (IC * ID * IH * IW, ID * IH * IW, IH * IW, IW, 1),
        )
        x = al.make_tensor(x_ptr, al.bf16, layout_x)
        layout_w = al.make_layout(
            (OC, IC, KD, KH, KW),
            (IC * KD * KH * KW, KD * KH * KW, KH * KW, KW, 1),
        )
        w = al.make_tensor(w_ptr, al.bf16, layout_w)
        layout_bias = al.make_layout((OC,), (1,))
        bias_t = al.make_tensor(bias_ptr, al.bf16, layout_bias)
        layout_out = al.make_layout(
            (B, OC, OD, OH, OW),
            (OC * OD * OH * OW, OD * OH * OW, OH * OW, OW, 1),
        )
        out = al.make_tensor(out_ptr, al.bf16, layout_out)

        zero_i32 = al.convert(0, al.i32)
        one_i32 = al.convert(1, al.i32)
        two_i32 = al.convert(2, al.i32)
        zero_f32 = al.convert(0.0, al.f32)
        worst = al.convert(-1.0e30, al.f32)
        acc_max = worst

        for dd in al.range(2):
            d_pos = dp * two_i32 + dd
            for hh in al.range(2):
                h_pos = hp * two_i32 + hh
                for ww in al.range(2):
                    w_pos = wp * two_i32 + ww
                    acc = zero_f32
                    for ic in al.range(IC):
                        for kd in al.range(KD):
                            d_in = d_pos + kd - one_i32
                            for kh in al.range(KH):
                                h_in = h_pos + kh - one_i32
                                for kw in al.range(KW):
                                    w_in = w_pos + kw - one_i32
                                    if d_in >= zero_i32:
                                        if d_in < ID:
                                            if h_in >= zero_i32:
                                                if h_in < IH:
                                                    if w_in >= zero_i32:
                                                        if w_in < IW:
                                                            inp = al.convert(
                                                                x[b, ic, d_in, h_in, w_in], al.f32
                                                            )
                                                            wgt = al.convert(
                                                                w[oc, ic, kd, kh, kw], al.f32
                                                            )
                                                            acc = acc + inp * wgt
                    if acc > acc_max:
                        acc_max = acc

        bias_val = al.convert(bias_t[oc], al.f32)
        result = acc_max + bias_val
        out[b, oc, dp, hp, wp] = al.convert(result, al.bf16)


@avelang.jit
def logsumexp_relu_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    OC: al.i32,
    OD: al.i32,
    OH: al.i32,
    OW: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    idx = bid * BLOCK_SZ + tid
    total = B * OD * OH * OW

    if idx < total:
        b = idx // (OD * OH * OW)
        rem = idx - b * (OD * OH * OW)
        dp = rem // (OH * OW)
        rem = rem - dp * (OH * OW)
        hp = rem // OW
        wp = rem - hp * OW

        layout_x = al.make_layout(
            (B, OC, OD, OH, OW),
            (OC * OD * OH * OW, OD * OH * OW, OH * OW, OW, 1),
        )
        x = al.make_tensor(x_ptr, al.bf16, layout_x)
        layout_out = al.make_layout(
            (B, al.convert(1, al.i32), OD, OH, OW),
            (OD * OH * OW, OD * OH * OW, OH * OW, OW, 1),
        )
        out = al.make_tensor(out_ptr, al.bf16, layout_out)

        zero_i = al.convert(0, al.i32)
        zero_f = al.convert(0.0, al.f32)

        max_val = al.convert(x[b, zero_i, dp, hp, wp], al.f32)
        for c in al.range(1, OC):
            val = al.convert(x[b, c, dp, hp, wp], al.f32)
            if val > max_val:
                max_val = val

        sum_exp = zero_f
        for c in al.range(OC):
            val = al.convert(x[b, c, dp, hp, wp], al.f32)
            sum_exp = sum_exp + al.exp(val - max_val)

        result = max_val + al.log(sum_exp)
        if result < zero_f:
            result = zero_f

        out[b, zero_i, dp, hp, wp] = al.convert(result, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_pool_logsumexp_relu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = _prepare_bf16_cuda_contiguous(weight)
    bias_bf16 = _prepare_bf16_cuda_contiguous(bias)

    B, IC, ID, IH, IW = x_bf16.shape
    OC, W_IC, KD, KH, KW = w_bf16.shape

    OD = ID // 2
    OH = IH // 2
    OW = IW // 2

    pooled = torch.empty(
        (B, OC, OD, OH, OW), device=x_bf16.device, dtype=torch.bfloat16
    )

    total_conv = B * OC * OD * OH * OW
    num_blocks_conv = (total_conv + BLOCK_SZ - 1) // BLOCK_SZ

    conv3d_maxpool3d_bias_kernel[
        lambda: ((num_blocks_conv, 1, 1), (BLOCK_SZ, 1, 1))
    ](
        x_bf16,
        w_bf16,
        bias_bf16,
        pooled,
        B,
        IC,
        OC,
        ID,
        IH,
        IW,
        OD,
        OH,
        OW,
        KD,
        KH,
        KW,
    )

    final_out = torch.empty(
        (B, 1, OD, OH, OW), device=x_bf16.device, dtype=torch.bfloat16
    )

    total_lse = B * OD * OH * OW
    num_blocks_lse = (total_lse + BLOCK_SZ - 1) // BLOCK_SZ

    logsumexp_relu_kernel[
        lambda: ((num_blocks_lse, 1, 1), (BLOCK_SZ, 1, 1))
    ](
        pooled,
        final_out,
        B,
        OC,
        OD,
        OH,
        OW,
    )

    return final_out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        weight = self.conv.weight.data
        bias = self.conv.bias.data
        return avelang_conv_pool_logsumexp_relu(x, weight, bias)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
