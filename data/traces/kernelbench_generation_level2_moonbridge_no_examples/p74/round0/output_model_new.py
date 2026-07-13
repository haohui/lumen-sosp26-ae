import struct
import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_transpose_3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    stride_val: al.i32,
    padding_val: al.i32,
    SPATIAL_ELEMS: al.i32,
    HW_out: al.i32,
):
    n = al.block_id(0)
    oc = al.block_id(1)
    tid = al.block_id(2) * al.block_dim(0) + al.thread_id(0)

    if tid < SPATIAL_ELEMS:
        d = tid // HW_out
        rem = tid % HW_out
        h = rem // W_out
        w = rem % W_out

        one_i32 = al.convert(1, al.i32)
        zero_i32 = al.convert(0, al.i32)

        in_stride_N = C_in * D_in * H_in * W_in
        in_stride_C = D_in * H_in * W_in
        in_stride_D = H_in * W_in
        in_stride_H = W_in

        input_layout = al.make_layout(
            (N, C_in, D_in, H_in, W_in),
            (in_stride_N, in_stride_C, in_stride_D, in_stride_H, one_i32),
        )
        input_t = al.make_tensor(input_ptr, al.bf16, input_layout)

        kd3 = al.convert(3, al.i32)
        wt_stride_IC = C_out * kd3 * kd3 * kd3
        wt_stride_OC = kd3 * kd3 * kd3
        wt_stride_KD = kd3 * kd3
        wt_stride_KH = kd3

        weight_layout = al.make_layout(
            (C_in, C_out, kd3, kd3, kd3),
            (wt_stride_IC, wt_stride_OC, wt_stride_KD, wt_stride_KH, one_i32),
        )
        weight_t = al.make_tensor(weight_ptr, al.bf16, weight_layout)

        bias_layout = al.make_layout((C_out,), (one_i32,))
        bias_t = al.make_tensor(bias_ptr, al.bf16, bias_layout)

        out_stride_N = C_out * D_out * H_out * W_out
        out_stride_C = D_out * H_out * W_out
        out_stride_D = H_out * W_out
        out_stride_H = W_out

        output_layout = al.make_layout(
            (N, C_out, D_out, H_out, W_out),
            (out_stride_N, out_stride_C, out_stride_D, out_stride_H, one_i32),
        )
        output_t = al.make_tensor(output_ptr, al.bf16, output_layout)

        zero_f32 = al.convert(0.0, al.f32)
        accum = zero_f32

        for ic in al.range(C_in):
            for kd in al.range(kd3):
                d_idx = d + padding_val - kd
                d_rem = d_idx % stride_val
                if d_rem == zero_i32:
                    d_in = d_idx // stride_val
                    if d_in >= zero_i32:
                        if d_in < D_in:
                            for kh in al.range(kd3):
                                h_idx = h + padding_val - kh
                                h_rem = h_idx % stride_val
                                if h_rem == zero_i32:
                                    h_in = h_idx // stride_val
                                    if h_in >= zero_i32:
                                        if h_in < H_in:
                                            for kw in al.range(kd3):
                                                w_idx = w + padding_val - kw
                                                w_rem = w_idx % stride_val
                                                if w_rem == zero_i32:
                                                    w_in = w_idx // stride_val
                                                    if w_in >= zero_i32:
                                                        if w_in < W_in:
                                                            inp_val = al.convert(
                                                                input_t[n, ic, d_in, h_in, w_in],
                                                                al.f32,
                                                            )
                                                            w_val = al.convert(
                                                                weight_t[ic, oc, kd, kh, kw],
                                                                al.f32,
                                                            )
                                                            accum = accum + inp_val * w_val

        accum = accum + al.convert(bias_t[oc], al.f32)
        output_t[n, oc, d, h, w] = al.convert(accum, al.bf16)


@avelang.jit
def fused_leaky_mul_leaky_maxpool_kernel(
    conv_out_ptr: al.Pointer(al.bf16),
    multiplier_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_out: al.i32,
    D_pre: al.i32,
    H_pre: al.i32,
    W_pre: al.i32,
    D_final: al.i32,
    H_final: al.i32,
    W_final: al.i32,
    SPATIAL_FINAL: al.i32,
    HW_final: al.i32,
    neg_slope_bits: al.i32,
):
    n = al.block_id(0)
    oc = al.block_id(1)
    tid = al.block_id(2) * al.block_dim(0) + al.thread_id(0)

    if tid < SPATIAL_FINAL:
        df = tid // HW_final
        rem = tid % HW_final
        hf = rem // W_final
        wf = rem % W_final

        one_i32 = al.convert(1, al.i32)
        zero_i32 = al.convert(0, al.i32)

        conv_stride_N = C_out * D_pre * H_pre * W_pre
        conv_stride_C = D_pre * H_pre * W_pre
        conv_stride_D = H_pre * W_pre
        conv_stride_H = W_pre

        conv_layout = al.make_layout(
            (N, C_out, D_pre, H_pre, W_pre),
            (conv_stride_N, conv_stride_C, conv_stride_D, conv_stride_H, one_i32),
        )
        conv_t = al.make_tensor(conv_out_ptr, al.bf16, conv_layout)

        mult_layout = al.make_layout(
            (C_out, one_i32, one_i32, one_i32),
            (one_i32, one_i32, one_i32, one_i32),
        )
        mult_t = al.make_tensor(multiplier_ptr, al.bf16, mult_layout)

        out_stride_N = C_out * D_final * H_final * W_final
        out_stride_C = D_final * H_final * W_final
        out_stride_D = H_final * W_final
        out_stride_H = W_final

        out_layout = al.make_layout(
            (N, C_out, D_final, H_final, W_final),
            (out_stride_N, out_stride_C, out_stride_D, out_stride_H, one_i32),
        )
        out_t = al.make_tensor(output_ptr, al.bf16, out_layout)

        two_i32 = al.convert(2, al.i32)
        zero_f32 = al.convert(0.0, al.f32)

        neg_slope = al.bitcast(neg_slope_bits, al.f32)

        mult_val = al.convert(mult_t[oc, zero_i32, zero_i32, zero_i32], al.f32)

        dp0 = df * two_i32
        hp0 = hf * two_i32
        wp0 = wf * two_i32

        first = one_i32
        max_val = zero_f32

        for dd in al.range(two_i32):
            dp = dp0 + dd
            for dh in al.range(two_i32):
                hp = hp0 + dh
                for dw in al.range(two_i32):
                    wp = wp0 + dw

                    v = al.convert(conv_t[n, oc, dp, hp, wp], al.f32)

                    if v > zero_f32:
                        v = v
                    else:
                        v = neg_slope * v

                    v = v * mult_val

                    if v > zero_f32:
                        v = v
                    else:
                        v = neg_slope * v

                    if first == one_i32:
                        max_val = v
                        first = zero_i32
                    else:
                        if v > max_val:
                            max_val = v

        out_t[n, oc, df, hf, wf] = al.convert(max_val, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        N = x.shape[0]
        C_in = x.shape[1]
        D_in = x.shape[2]
        H_in = x.shape[3]
        W_in = x.shape[4]

        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        multiplier = self.multiplier

        stride_val = self.conv_transpose.stride[0]
        padding_val = self.conv_transpose.padding[0]

        C_out = weight.shape[1]
        D_out = (D_in - 1) * stride_val - 2 * padding_val + weight.shape[2] + self.conv_transpose.output_padding[0]
        H_out = (H_in - 1) * stride_val - 2 * padding_val + weight.shape[3] + self.conv_transpose.output_padding[1]
        W_out = (W_in - 1) * stride_val - 2 * padding_val + weight.shape[4] + self.conv_transpose.output_padding[2]

        D_final = D_out // 2
        H_final = H_out // 2
        W_final = W_out // 2

        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = weight.to(torch.bfloat16).contiguous()
        bias_bf16 = bias.to(torch.bfloat16).contiguous()
        multiplier_bf16 = multiplier.to(torch.bfloat16).contiguous()

        conv_out = torch.empty(N, C_out, D_out, H_out, W_out, device=x.device, dtype=torch.bfloat16)

        SPATIAL_ELEMS = D_out * H_out * W_out
        HW_out = H_out * W_out
        BLOCK_SIZE = 256
        grid_z = (SPATIAL_ELEMS + BLOCK_SIZE - 1) // BLOCK_SIZE

        conv_transpose_3d_kernel[lambda: ((N, C_out, grid_z), (BLOCK_SIZE, 1, 1))](
            x_bf16.data_ptr(), w_bf16.data_ptr(), bias_bf16.data_ptr(), conv_out.data_ptr(),
            N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
            stride_val, padding_val, SPATIAL_ELEMS, HW_out,
        )

        final_out = torch.empty(N, C_out, D_final, H_final, W_final, device=x.device, dtype=torch.bfloat16)

        SPATIAL_FINAL = D_final * H_final * W_final
        HW_final = H_final * W_final
        grid_z2 = (SPATIAL_FINAL + BLOCK_SIZE - 1) // BLOCK_SIZE

        neg_slope_bits = struct.unpack('<i', struct.pack('<f', 0.2))[0]

        fused_leaky_mul_leaky_maxpool_kernel[lambda: ((N, C_out, grid_z2), (BLOCK_SIZE, 1, 1))](
            conv_out.data_ptr(), multiplier_bf16.data_ptr(), final_out.data_ptr(),
            N, C_out, D_out, H_out, W_out, D_final, H_final, W_final,
            SPATIAL_FINAL, HW_final, neg_slope_bits,
        )

        return final_out
