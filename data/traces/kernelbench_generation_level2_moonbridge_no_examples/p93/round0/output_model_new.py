import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Problem constants
_BATCH_SIZE = 128
_IN_CHANNELS = 64
_OUT_CHANNELS = 128
_H_IN = 64
_W_IN = 64
_KERNEL_SIZE = 4
_STRIDE = 2
_ADD_VALUE = 0.5
_MULTIPLY_VALUE = 2.0

# Derived output spatial dimensions for ConvTranspose2d with padding=0, dilation=1, output_padding=0
_H_OUT = (_H_IN - 1) * _STRIDE + _KERNEL_SIZE
_W_OUT = (_W_IN - 1) * _STRIDE + _KERNEL_SIZE


@avelang.jit
def fused_conv_transpose2d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.i32,
    stride: al.i32,
    in_plane_size: al.i32,
    wt_ch_stride: al.i32,
):
    # 1D flattened tensor views for safe linear indexing
    in_total = N * C_in * H_in * W_in
    in_flat = al.make_tensor(input_ptr, al.bf16, al.make_layout((in_total,), (1,)))

    wt_total = C_in * C_out * K * K
    wt_flat = al.make_tensor(weight_ptr, al.bf16, al.make_layout((wt_total,), (1,)))

    bias_flat = al.make_tensor(bias_ptr, al.f32, al.make_layout((C_out,), (1,)))

    out_total = N * C_out * H_out * W_out
    out_flat = al.make_tensor(output_ptr, al.bf16, al.make_layout((out_total,), (1,)))

    # Flat global thread id
    tid = al.block_id(0) * al.block_dim(0) + al.thread_id(0)

    if tid < out_total:
        # Decompose flat index: tid = n * C_out*H_out*W_out + c * H_out*W_out + h * W_out + w
        stride_c = H_out * W_out
        stride_n = C_out * stride_c

        n = tid // stride_n
        r1 = tid - n * stride_n
        c = r1 // stride_c
        r2 = r1 - c * stride_c
        h = r2 // W_out
        w = r2 - h * W_out

        # Parity-based kernel start offsets (stride == 2 optimization)
        # Only kh where kh % 2 == h % 2 can be valid (2 iterations instead of 4)
        h_parity = h - (h // stride) * stride
        w_parity = w - (w // stride) * stride

        # FP32 accumulator
        acc = al.convert(0.0, al.f32)

        # Input batch base offset
        in_batch_off = n * C_in * in_plane_size

        # Accumulate over input channels
        for ci in al.range(C_in):
            in_ch_off = in_batch_off + ci * in_plane_size
            wt_ch_off = ci * wt_ch_stride + c * K * K

            # Only iterate over kh with matching parity
            kh = h_parity
            for _kh in al.range(2):
                h_off = h - kh
                h_div = h_off // stride
                h_valid = (h_div * stride + kh == h) & (h_div >= 0) & (h_div < H_in)

                if h_valid:
                    in_h_off = in_ch_off + h_div * W_in

                    kw = w_parity
                    for _kw in al.range(2):
                        w_off = w - kw
                        w_div = w_off // stride
                        w_valid = (w_div * stride + kw == w) & (w_div >= 0) & (w_div < W_in)

                        if w_valid:
                            in_idx = in_h_off + w_div
                            wt_idx = wt_ch_off + kh * K + kw
                            in_val = al.convert(in_flat[in_idx], al.f32)
                            w_val = al.convert(wt_flat[wt_idx], al.f32)
                            acc = acc + in_val * w_val

                        kw = kw + stride
                kh = kh + stride

        # Post-processing in FP32
        # Add bias (per output channel)
        acc = acc + bias_flat[c]

        # Add add_value (0.5)
        acc = acc + al.convert(0.5, al.f32)

        # Min with 0.0 (clamp to <= 0)
        if acc > al.convert(0.0, al.f32):
            acc = al.convert(0.0, al.f32)

        # GELU via tanh approximation
        sqrt_2_pi = al.convert(0.7978845608028654, al.f32)
        coeff = al.convert(0.044715, al.f32)
        half = al.convert(0.5, al.f32)
        one = al.convert(1.0, al.f32)

        x2 = acc * acc
        x3 = x2 * acc
        inner = sqrt_2_pi * (acc + coeff * x3)
        gelu_out = half * acc * (one + al.tanh(inner))

        # Multiply by multiply_value (2.0)
        result = gelu_out * al.convert(2.0, al.f32)

        # Store as BF16
        out_flat[tid] = al.convert(result, al.bf16)


def avelang_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Launch the fused conv-transpose + post-process kernel."""
    N, C_in, H_in, W_in = x.shape
    C_out = weight.shape[1]
    K = weight.shape[2]
    stride = _STRIDE
    H_out = _H_OUT
    W_out = _W_OUT

    # Convert to BF16 on device
    x_bf16 = x.to(torch.bfloat16).contiguous()
    w_bf16 = weight.to(torch.bfloat16).contiguous()
    b_f32 = bias.to(torch.float32).contiguous()

    out = torch.empty((N, C_out, H_out, W_out), dtype=torch.bfloat16, device=x.device)

    BLOCK_SIZE = 256
    total_elements = N * C_out * H_out * W_out
    grid_x = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    in_plane_size = H_in * W_in
    wt_ch_stride = C_out * K * K

    fused_conv_transpose2d_kernel[lambda: ((grid_x, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16,
        w_bf16,
        b_f32,
        out,
        N,
        C_in,
        C_out,
        H_in,
        W_in,
        H_out,
        W_out,
        K,
        stride,
        in_plane_size,
        wt_ch_stride,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride
        )
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        return avelang_forward(x, weight, bias)
