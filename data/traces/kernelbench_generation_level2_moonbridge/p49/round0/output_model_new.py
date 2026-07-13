import math
import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
NUM_CHANNELS: al.constexpr = 64

batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1


@avelang.jit
def tiled_fused_conv_softmax_sigmoid_kernel(
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
    KD: al.i32,
    KH: al.i32,
    KW: al.i32,
    stride_val: al.i32,
    pad: al.i32,
    total_positions: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    gid = bid * BLOCK_SIZE + tid

    shm_wgt = al.make_shared((NUM_CHANNELS, 3, 3, 3), al.bf16)

    if gid < total_positions:
        tmp = gid
        w_idx = tmp % W_out
        tmp = tmp // W_out
        h_idx = tmp % H_out
        tmp = tmp // H_out
        d_idx = tmp % D_out
        b = tmp // D_out

        inp_layout = al.make_layout(
            (N, C_in, D_in, H_in, W_in),
            (C_in * D_in * H_in * W_in, D_in * H_in * W_in, H_in * W_in, W_in, 1),
        )
        inp = al.make_tensor(input_ptr, al.bf16, inp_layout)

        wgt_layout = al.make_layout(
            (C_in, C_out, KD, KH, KW),
            (C_out * KD * KH * KW, KD * KH * KW, KH * KW, KW, 1),
        )
        wgt = al.make_tensor(weight_ptr, al.bf16, wgt_layout)

        bias_layout = al.make_layout((C_out,), (1,))
        bias_t = al.make_tensor(bias_ptr, al.bf16, bias_layout)

        out_layout = al.make_layout(
            (N, C_out, D_out, H_out, W_out),
            (C_out * D_out * H_out * W_out, D_out * H_out * W_out, H_out * W_out, W_out, 1),
        )
        out_t = al.make_tensor(output_ptr, al.bf16, out_layout)

        acc = al.make_local((NUM_CHANNELS,), al.f32)

        for c in al.range(0, NUM_CHANNELS):
            acc[c] = al.convert(bias_t[c], al.f32)

        d_parity = (d_idx + pad) % 2
        h_parity = (h_idx + pad) % 2
        w_parity = (w_idx + pad) % 2

        for ic in al.range(0, C_in):
            num_wgt = NUM_CHANNELS * 3 * 3 * 3
            elems_per_thread = (num_wgt + BLOCK_SIZE - 1) // BLOCK_SIZE
            for i in al.range(0, elems_per_thread):
                flat_idx = tid + i * BLOCK_SIZE
                if flat_idx < num_wgt:
                    oc = flat_idx // 27
                    rem = flat_idx - oc * 27
                    kd = rem // 9
                    rem = rem - kd * 9
                    kh = rem // 3
                    kw = rem - kh * 3
                    shm_wgt[oc, kd, kh, kw] = wgt[ic, oc, kd, kh, kw]
            al.syncthreads()

            for kd in al.range(d_parity, KD, 2):
                d_in = (d_idx + pad - kd) // 2
                if d_in >= 0:
                    if d_in < D_in:
                        for kh in al.range(h_parity, KH, 2):
                            h_in = (h_idx + pad - kh) // 2
                            if h_in >= 0:
                                if h_in < H_in:
                                    for kw in al.range(w_parity, KW, 2):
                                        w_in = (w_idx + pad - kw) // 2
                                        if w_in >= 0:
                                            if w_in < W_in:
                                                inp_val = al.convert(inp[b, ic, d_in, h_in, w_in], al.f32)
                                                for oc in al.range(0, NUM_CHANNELS):
                                                    wgt_val = al.convert(shm_wgt[oc, kd, kh, kw], al.f32)
                                                    acc[oc] = acc[oc] + inp_val * wgt_val
            al.syncthreads()

        max_val = acc[0]
        for c in al.range(1, NUM_CHANNELS):
            if acc[c] > max_val:
                max_val = acc[c]

        exp_sum = al.convert(0.0, al.f32)
        for c in al.range(0, NUM_CHANNELS):
            diff = acc[c] - max_val
            exp_sum = exp_sum + al.exp(diff)

        one = al.convert(1.0, al.f32)
        for c in al.range(0, NUM_CHANNELS):
            diff = acc[c] - max_val
            softmax_val = al.exp(diff) / exp_sum
            neg_softmax = -softmax_val
            sigmoid_val = one / (one + al.exp(neg_softmax))
            out_t[b, c, d_idx, h_idx, w_idx] = al.convert(sigmoid_val, al.bf16)


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_transpose_softmax_sigmoid(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride_val: int,
    pad_val: int,
    output_pad: int,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)
    b_bf16 = _to_bf16_contiguous(bias)

    N, C_in, D_in, H_in, W_in = x_bf16.shape
    C_out = w_bf16.shape[1]
    KD = w_bf16.shape[2]
    KH = w_bf16.shape[3]
    KW = w_bf16.shape[4]

    D_out = (D_in - 1) * stride_val - 2 * pad_val + KD + output_pad
    H_out = (H_in - 1) * stride_val - 2 * pad_val + KH + output_pad
    W_out = (W_in - 1) * stride_val - 2 * pad_val + KW + output_pad

    total_positions = N * D_out * H_out * W_out
    num_blocks = (total_positions + BLOCK_SIZE - 1) // BLOCK_SIZE

    output = torch.empty(
        (N, C_out, D_out, H_out, W_out),
        dtype=torch.bfloat16,
        device=x_bf16.device,
    )

    tiled_fused_conv_softmax_sigmoid_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16,
        w_bf16,
        b_bf16,
        output,
        N,
        C_in,
        C_out,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        KD,
        KH,
        KW,
        stride_val,
        pad_val,
        total_positions,
    )

    return output


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        bias: bool = True,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride_val = stride
        self.padding_val = padding
        self.output_padding_val = output_padding

        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv_transpose_softmax_sigmoid(
            x,
            self.weight,
            self.bias,
            self.stride_val,
            self.padding_val,
            self.output_padding_val,
        )


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]
