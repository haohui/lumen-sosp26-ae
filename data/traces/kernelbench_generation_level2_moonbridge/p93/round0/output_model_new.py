import torch
import torch.nn as nn
import avelang
import avelang.language as al

KS = 4
STRIDE = 2
C_IN = 64
C_OUT = 128
BLOCK_SIZE = 256
ELEMS_PER_THREAD = 4

GELU_SQRT_2_PI = 0.7978845608028654
GELU_COEFF = 0.044715


@avelang.jit
def conv_transpose_fused_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    add_val: al.f32,
    mult_val: al.f32,
):
    tid = al.thread_id(0)
    block_idx = al.block_id(0)

    gid = block_idx * BLOCK_SIZE * ELEMS_PER_THREAD + tid
    total = N * C_OUT * H_out * W_out

    if gid < total:
        x_layout = al.make_layout(
            (N, C_IN, H_in, W_in),
            (C_IN * H_in * W_in, H_in * W_in, W_in, 1),
        )
        x_g = al.make_tensor(x_ptr, al.bf16, x_layout)

        w_layout = al.make_layout(
            (C_IN, C_OUT, KS, KS),
            (C_OUT * KS * KS, KS * KS, KS, 1),
        )
        w_g = al.make_tensor(w_ptr, al.bf16, w_layout)

        b_layout = al.make_layout((C_OUT,), (1,))
        b_g = al.make_tensor(b_ptr, al.bf16, b_layout)

        out_layout = al.make_layout(
            (N, C_OUT, H_out, W_out),
            (C_OUT * H_out * W_out, H_out * W_out, W_out, 1),
        )
        out_g = al.make_tensor(out_ptr, al.bf16, out_layout)

        zero_f32 = al.convert(0.0, al.f32)
        zero_i32 = al.convert(0, al.i32)
        one_i32 = al.convert(1, al.i32)
        two_i32 = al.convert(2, al.i32)
        hin_i32 = al.convert(H_in, al.i32)
        win_i32 = al.convert(W_in, al.i32)

        for e in al.range(ELEMS_PER_THREAD):
            elem_gid = gid + e * BLOCK_SIZE
            if elem_gid >= total:
                break

            wo = elem_gid % W_out
            r1 = elem_gid // W_out
            ho = r1 % H_out
            r2 = r1 // H_out
            co = r2 % C_OUT
            n = r2 // C_OUT

            acc = zero_f32

            ho_i32 = al.convert(ho, al.i32)
            wo_i32 = al.convert(wo, al.i32)

            kh_a = ho_i32 % two_i32
            h_in_a = (ho_i32 - kh_a) // two_i32
            kh_b = kh_a + two_i32
            h_in_b = h_in_a - one_i32

            kw_a = wo_i32 % two_i32
            w_in_a = (wo_i32 - kw_a) // two_i32
            kw_b = kw_a + two_i32
            w_in_b = w_in_a - one_i32

            ha_valid = zero_i32
            if h_in_a < zero_i32:
                ha_valid = zero_i32
            else:
                if h_in_a < hin_i32:
                    ha_valid = one_i32

            hb_valid = zero_i32
            if h_in_b < zero_i32:
                hb_valid = zero_i32
            else:
                if h_in_b < hin_i32:
                    hb_valid = one_i32

            wa_valid = zero_i32
            if w_in_a < zero_i32:
                wa_valid = zero_i32
            else:
                if w_in_a < win_i32:
                    wa_valid = one_i32

            wb_valid = zero_i32
            if w_in_b < zero_i32:
                wb_valid = zero_i32
            else:
                if w_in_b < win_i32:
                    wb_valid = one_i32

            for ci in al.range(C_IN):
                ci_i32 = al.convert(ci, al.i32)

                if ha_valid != zero_i32:
                    if wa_valid != zero_i32:
                        w_val = w_g[ci_i32, co, kh_a, kw_a]
                        x_val = x_g[n, ci_i32, h_in_a, w_in_a]
                        acc = acc + al.convert(w_val, al.f32) * al.convert(x_val, al.f32)

                if ha_valid != zero_i32:
                    if wb_valid != zero_i32:
                        w_val = w_g[ci_i32, co, kh_a, kw_b]
                        x_val = x_g[n, ci_i32, h_in_a, w_in_b]
                        acc = acc + al.convert(w_val, al.f32) * al.convert(x_val, al.f32)

                if hb_valid != zero_i32:
                    if wa_valid != zero_i32:
                        w_val = w_g[ci_i32, co, kh_b, kw_a]
                        x_val = x_g[n, ci_i32, h_in_b, w_in_a]
                        acc = acc + al.convert(w_val, al.f32) * al.convert(x_val, al.f32)

                if hb_valid != zero_i32:
                    if wb_valid != zero_i32:
                        w_val = w_g[ci_i32, co, kh_b, kw_b]
                        x_val = x_g[n, ci_i32, h_in_b, w_in_b]
                        acc = acc + al.convert(w_val, al.f32) * al.convert(x_val, al.f32)

            bias_val = al.convert(b_g[co], al.f32)
            acc = acc + bias_val
            acc = acc + al.convert(add_val, al.f32)
            if acc > zero_f32:
                acc = zero_f32
            x3 = acc * acc * acc
            inner = al.convert(GELU_SQRT_2_PI, al.f32) * (acc + al.convert(GELU_COEFF, al.f32) * x3)
            gelu = al.convert(0.5, al.f32) * acc * (al.convert(1.0, al.f32) + al.tanh(inner))
            acc = gelu * al.convert(mult_val, al.f32)

            out_g[n, co, ho, wo] = al.convert(acc, al.bf16)


def avelang_conv_transpose_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    add_value: float,
    multiply_value: float,
) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."

    x_bf16 = x.contiguous().to(dtype=torch.bfloat16)
    w_bf16 = weight.contiguous().to(dtype=torch.bfloat16)
    b_bf16 = bias.contiguous().to(dtype=torch.bfloat16)

    N_batch, C_in_val, H_in_val, W_in_val = x_bf16.shape
    H_out_val = (H_in_val - 1) * STRIDE + KS
    W_out_val = (W_in_val - 1) * STRIDE + KS
    C_out_val = b_bf16.shape[0]

    total_elems = N_batch * C_out_val * H_out_val * W_out_val
    num_blocks = (total_elems + BLOCK_SIZE * ELEMS_PER_THREAD - 1) // (BLOCK_SIZE * ELEMS_PER_THREAD)

    out = torch.empty(
        (N_batch, C_out_val, H_out_val, W_out_val),
        dtype=torch.bfloat16,
        device=x_bf16.device,
    )

    grid = (num_blocks, 1, 1)
    conv_transpose_fused_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](
        x_bf16,
        w_bf16,
        b_bf16,
        out,
        N_batch,
        H_in_val,
        W_in_val,
        H_out_val,
        W_out_val,
        add_value,
        multiply_value,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        result_bf16 = avelang_conv_transpose_fused(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.add_value,
            self.multiply_value,
        )
        return result_bf16.to(dtype=x.dtype)


batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]
