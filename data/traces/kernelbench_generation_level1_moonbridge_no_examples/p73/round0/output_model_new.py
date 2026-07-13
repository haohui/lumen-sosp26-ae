import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_H = 8
TILE_W = 32


@avelang.jit
def conv_transpose3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
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
    K: al.i32,
    stride: al.i32,
    padding: al.i32,
    groups: al.i32,
    in_ch_per_g: al.i32,
    out_ch_per_g: al.i32,
    TILE_H: al.constexpr,
    TILE_W: al.constexpr,
):
    th = al.thread_id(0)
    tw = al.thread_id(1)

    h_out = al.block_id(0) * TILE_H + th
    w_out = al.block_id(1) * TILE_W + tw

    bz = al.block_id(2)
    b = bz // (groups * D_out)
    rem = bz % (groups * D_out)
    g = rem // D_out
    d_out = rem % D_out

    # Stride-aware start offsets
    kd_start = (d_out + padding) % stride
    kh_start = (h_out + padding) % stride
    kw_start = (w_out + padding) % stride

    # Flat 1D tensor views
    in_size = N * C_in * D_in * H_in * W_in
    in_layout = al.make_layout((in_size,), (1,))
    input_t = al.make_tensor(input_ptr, al.bf16, in_layout)

    w_size = C_in * out_ch_per_g * K * K * K
    w_layout = al.make_layout((w_size,), (1,))
    weight_t = al.make_tensor(weight_ptr, al.bf16, w_layout)

    out_size = N * C_out * D_out * H_out * W_out
    out_layout = al.make_layout((out_size,), (1,))
    output_t = al.make_tensor(output_ptr, al.bf16, out_layout)

    # Precompute strides for flat indexing
    in_stride_n = C_in * D_in * H_in * W_in
    in_stride_c = D_in * H_in * W_in
    in_stride_d = H_in * W_in
    in_stride_h = W_in

    w_stride_cin = out_ch_per_g * K * K * K
    w_stride_co = K * K * K
    w_stride_kd = K * K
    w_stride_kh = K

    out_stride_n = C_out * D_out * H_out * W_out
    out_stride_c = D_out * H_out * W_out
    out_stride_d = H_out * W_out
    out_stride_h = W_out

    zero_f32 = al.convert(0.0, al.f32)

    if h_out < H_out:
        if w_out < W_out:
            for c_out_loc in al.range(out_ch_per_g):
                c_out = g * out_ch_per_g + c_out_loc
                acc = zero_f32

                for c_in_loc in al.range(in_ch_per_g):
                    c_in = g * in_ch_per_g + c_in_loc
                    in_c_base = b * in_stride_n + c_in * in_stride_c
                    w_cin_base = c_in * w_stride_cin + c_out_loc * w_stride_co

                    for kd in al.range(kd_start, K, stride):
                        d_in_idx = (d_out + padding - kd) // stride
                        if (d_in_idx >= 0) and (d_in_idx < D_in):
                            in_d_base = in_c_base + d_in_idx * in_stride_d
                            w_kd_base = w_cin_base + kd * w_stride_kd

                            for kh in al.range(kh_start, K, stride):
                                h_in_idx = (h_out + padding - kh) // stride
                                if (h_in_idx >= 0) and (h_in_idx < H_in):
                                    in_h_base = in_d_base + h_in_idx * in_stride_h
                                    w_kh_base = w_kd_base + kh * w_stride_kh

                                    for kw in al.range(kw_start, K, stride):
                                        w_in_idx = (w_out + padding - kw) // stride
                                        if (w_in_idx >= 0) and (w_in_idx < W_in):
                                            in_val = al.convert(
                                                input_t[in_h_base + w_in_idx],
                                                al.f32,
                                            )
                                            w_val = al.convert(
                                                weight_t[w_kh_base + kw],
                                                al.f32,
                                            )
                                            acc = acc + in_val * w_val

                out_idx = b * out_stride_n + c_out * out_stride_c + d_out * out_stride_d + h_out * out_stride_h + w_out
                output_t[out_idx] = al.convert(acc, al.bf16)


def _run_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    N: int,
    C_in: int,
    C_out: int,
    D_in: int,
    H_in: int,
    W_in: int,
    D_out: int,
    H_out: int,
    W_out: int,
    K: int,
    stride: int,
    padding: int,
    groups: int,
    in_ch_per_g: int,
    out_ch_per_g: int,
) -> None:
    x_bf16 = x.to(torch.bfloat16).contiguous()
    w_bf16 = weight.to(torch.bfloat16).contiguous()

    grid_h = (H_out + TILE_H - 1) // TILE_H
    grid_w = (W_out + TILE_W - 1) // TILE_W
    grid_z = N * groups * D_out

    conv_transpose3d_kernel[lambda: ((grid_h, grid_w, grid_z), (TILE_H, TILE_W, 1))](
        x_bf16.data_ptr(),
        w_bf16.data_ptr(),
        output.data_ptr(),
        N,
        C_in,
        C_out,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        K,
        stride,
        padding,
        groups,
        in_ch_per_g,
        out_ch_per_g,
        TILE_H,
        TILE_W,
    )


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.in_ch_per_g = in_channels // groups
        self.out_ch_per_g = out_channels // groups

        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5.0 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size ** 3
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C_in, D_in, H_in, W_in = x.shape
        stride = self.stride
        padding = self.padding
        K = self.kernel_size
        C_out = self.out_channels

        D_out = (D_in - 1) * stride - 2 * padding + K
        H_out = (H_in - 1) * stride - 2 * padding + K
        W_out = (W_in - 1) * stride - 2 * padding + K

        out = torch.empty(N, C_out, D_out, H_out, W_out, dtype=torch.bfloat16, device=x.device)

        _run_conv_transpose3d(
            x,
            self.weight,
            out,
            N,
            C_in,
            C_out,
            D_in,
            H_in,
            W_in,
            D_out,
            H_out,
            W_out,
            K,
            stride,
            padding,
            self.groups,
            self.in_ch_per_g,
            self.out_ch_per_g,
        )

        if self.bias is not None:
            out = out + self.bias.to(torch.bfloat16).view(1, C_out, 1, 1, 1)

        return out


# Test code
batch_size = 4
in_channels = 32
out_channels = 32
kernel_size = 3
depth = 32
height = 64
width = 128
stride = 2
padding = 1
groups = 4


def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups]
