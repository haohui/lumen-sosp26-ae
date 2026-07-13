import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_S = 8


@avelang.jit
def _conv_transpose_epilogue_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    conv_bias_ptr: al.Pointer(al.bf16),
    user_bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    Cin: al.i32,
    Cout: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    kD: al.i32,
    kH: al.i32,
    kW: al.i32,
    stride: al.i32,
    padding: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    tid = al.thread_id(0)
    block_s = al.block_id(0)
    block_n = al.block_id(1)
    cout = tid

    if cout < Cout:
        n = block_n
        total_spatial = D_out * H_out * W_out

        hw_out = H_out * W_out

        in_hw = H_in * W_in
        in_dhw = D_in * in_hw
        cin_dhw = Cin * in_dhw

        w_hw_len = kH * kW
        w_dhw_len = kD * w_hw_len
        cout_dhw = Cout * w_dhw_len

        out_hw = H_out * W_out
        out_dhw = D_out * out_hw
        cout_out_dhw = Cout * out_dhw

        one = al.convert(1, al.i32)
        zero = al.convert(0, al.i32)
        stride_s = al.convert(stride, al.i32)
        padding_s = al.convert(padding, al.i32)

        in_layout = al.make_layout((N * cin_dhw,), (one,))
        input_t = al.make_tensor(input_ptr, al.bf16, in_layout)

        w_layout = al.make_layout((Cin * cout_dhw,), (one,))
        weight_t = al.make_tensor(weight_ptr, al.bf16, w_layout)

        cb_layout = al.make_layout((Cout,), (one,))
        conv_bias_t = al.make_tensor(conv_bias_ptr, al.bf16, cb_layout)

        ub_layout = al.make_layout((Cout,), (one,))
        user_bias_t = al.make_tensor(user_bias_ptr, al.bf16, ub_layout)

        out_layout = al.make_layout((N * cout_out_dhw,), (one,))
        output_t = al.make_tensor(output_ptr, al.bf16, out_layout)

        cb_val = al.convert(conv_bias_t[cout], al.f32)
        ub_val = al.convert(user_bias_t[cout], al.f32)

        n_base_in = n * cin_dhw
        n_base_out = n * cout_out_dhw
        cout_w_base = cout * w_dhw_len
        cout_out_base = cout * out_dhw

        for s in al.range(TILE_S):
            spatial_idx = block_s * TILE_S + s
            if spatial_idx < total_spatial:
                d_out = spatial_idx // hw_out
                rem_h = spatial_idx % hw_out
                h_out = rem_h // W_out
                w_out = rem_h % W_out

                d_val_base = d_out + padding_s
                d_min = (d_val_base - kD + stride_s) // stride_s
                if d_min < zero:
                    d_min = zero
                d_max = d_val_base // stride_s
                if d_max >= D_in:
                    d_max = D_in - one

                h_val_base = h_out + padding_s
                h_min = (h_val_base - kH + stride_s) // stride_s
                if h_min < zero:
                    h_min = zero
                h_max = h_val_base // stride_s
                if h_max >= H_in:
                    h_max = H_in - one

                w_val_base = w_out + padding_s
                w_min = (w_val_base - kW + stride_s) // stride_s
                if w_min < zero:
                    w_min = zero
                w_max = w_val_base // stride_s
                if w_max >= W_in:
                    w_max = W_in - one

                d_range = d_max - d_min + one
                h_range = h_max - h_min + one
                w_range = w_max - w_min + one

                local_acc = al.convert(0.0, al.f32)

                for cin in al.range(Cin):
                    cin_off_val = cin * in_dhw
                    w_cin_off_val = cin * cout_dhw

                    for d_i in al.range(d_range):
                        d_in = d_min + d_i
                        kd = d_val_base - d_in * stride_s
                        d_off_val = d_in * in_hw

                        for h_i in al.range(h_range):
                            h_in = h_min + h_i
                            kh = h_val_base - h_in * stride_s
                            h_off_val = h_in * W_in

                            for w_i in al.range(w_range):
                                w_in = w_min + w_i
                                kw = w_val_base - w_in * stride_s
                                w_idx = w_cin_off_val + cout_w_base + kd * w_hw_len + kh * kW + kw
                                in_idx = n_base_in + cin_off_val + d_off_val + h_off_val + w_in
                                local_acc = local_acc + al.convert(input_t[in_idx], al.f32) * al.convert(weight_t[w_idx], al.f32)

                local_acc = local_acc + cb_val
                original = local_acc
                local_acc = local_acc + ub_val
                local_acc = local_acc + original
                local_acc = local_acc * original
                local_acc = local_acc + original

                out_idx = n_base_out + cout_out_base + d_out * out_hw + h_out * W_out + w_out
                output_t[out_idx] = al.convert(local_acc, al.bf16)


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def _avelang_conv_transpose_epilogue(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_bias: torch.Tensor,
    user_bias: torch.Tensor,
    stride: int,
    padding: int,
    output_padding: int,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _to_bf16_contiguous(x)
    weight_bf16 = _to_bf16_contiguous(weight)
    conv_bias_bf16 = _to_bf16_contiguous(conv_bias)
    user_bias_bf16 = _to_bf16_contiguous(user_bias.reshape(-1))

    N, Cin, D_in, H_in, W_in = x_bf16.shape
    Cout = weight_bf16.shape[1]
    kD = weight_bf16.shape[2]
    kH = weight_bf16.shape[3]
    kW = weight_bf16.shape[4]

    D_out = (D_in - 1) * stride - 2 * padding + kD + output_padding
    H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + kW + output_padding

    out = torch.empty(
        (N, Cout, D_out, H_out, W_out),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )

    total_spatial = D_out * H_out * W_out
    grid_x = (total_spatial + TILE_S - 1) // TILE_S
    grid = (grid_x, N, 1)
    block = (Cout, 1, 1)

    _conv_transpose_epilogue_kernel[lambda: (grid, block)](
        x_bf16,
        weight_bf16,
        conv_bias_bf16,
        user_bias_bf16,
        out,
        N,
        Cin,
        Cout,
        D_in,
        H_in,
        W_in,
        kD,
        kH,
        kW,
        stride,
        padding,
        D_out,
        H_out,
        W_out,
    )

    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        bias_shape: tuple,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._stride = stride
        self._padding = padding
        self._output_padding = output_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _avelang_conv_transpose_epilogue(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.bias,
            self._stride,
            self._padding,
            self._output_padding,
        )
