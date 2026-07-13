import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_transpose3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
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
    BLOCK_SPATIAL: al.constexpr,
):
    tid = al.thread_id(0)

    spatial_block_idx = al.block_id(0)
    batch_idx = al.block_id(1)

    spatial_idx = spatial_block_idx * BLOCK_SPATIAL + tid
    total_spatial = D_out * H_out * W_out

    if spatial_idx >= total_spatial:
        return

    H_out_W_out = H_out * W_out
    d_out = spatial_idx // H_out_W_out
    tmp = spatial_idx - d_out * H_out_W_out
    h_out = tmp // W_out
    w_out = tmp - h_out * W_out

    # Input: (B, C_in, D_in, H_in, W_in)
    in_layout = al.make_layout(
        (B, C_in, D_in, H_in, W_in),
        (C_in * D_in * H_in * W_in, D_in * H_in * W_in, H_in * W_in, W_in, 1),
    )
    input_t = al.make_tensor(input_ptr, al.bf16, in_layout)

    # Weight: (C_in, C_out, KD, KH, KW), row-major
    wgt_layout = al.make_layout(
        (C_in, C_out, KD, KH, KW),
        (C_out * KD * KH * KW, KD * KH * KW, KH * KW, KW, 1),
    )
    weight_t = al.make_tensor(weight_ptr, al.bf16, wgt_layout)

    # Output: (B, C_out, D_out, H_out, W_out)
    out_layout = al.make_layout(
        (B, C_out, D_out, H_out, W_out),
        (C_out * D_out * H_out * W_out, D_out * H_out * W_out, H_out * W_out, W_out, 1),
    )
    output_t = al.make_tensor(output_ptr, al.bf16, out_layout)

    # Precompute valid kernel-offset ranges for this output position.
    zero_i32 = al.convert(0, al.i32)
    one_i32 = al.convert(1, al.i32)
    kd_start = al.max(zero_i32, d_out - D_in + one_i32)
    kd_end = al.min(KD, d_out + one_i32)
    kh_start = al.max(zero_i32, h_out - H_in + one_i32)
    kh_end = al.min(KH, h_out + one_i32)
    kw_start = al.max(zero_i32, w_out - W_in + one_i32)
    kw_end = al.min(KW, w_out + one_i32)

    # Each thread handles one spatial position and iterates over all C_out.
    # Unroll C_out in groups of 8 to reuse input loads.
    for c_out_grp in al.range(0, C_out, 8):
        accum0 = al.convert(0.0, al.f32)
        accum1 = al.convert(0.0, al.f32)
        accum2 = al.convert(0.0, al.f32)
        accum3 = al.convert(0.0, al.f32)
        accum4 = al.convert(0.0, al.f32)
        accum5 = al.convert(0.0, al.f32)
        accum6 = al.convert(0.0, al.f32)
        accum7 = al.convert(0.0, al.f32)

        for ic in al.range(C_in):
            for kd in al.range(kd_start, kd_end):
                d_in = d_out - kd
                for kh in al.range(kh_start, kh_end):
                    h_in = h_out - kh
                    for kw in al.range(kw_start, kw_end):
                        w_in = w_out - kw
                        inp_val = al.convert(
                            input_t[batch_idx, ic, d_in, h_in, w_in], al.f32
                        )
                        # 8-way unrolled weight load and accumulate
                        wgt0 = al.convert(weight_t[ic, c_out_grp, kd, kh, kw], al.f32)
                        wgt1 = al.convert(weight_t[ic, c_out_grp + 1, kd, kh, kw], al.f32)
                        wgt2 = al.convert(weight_t[ic, c_out_grp + 2, kd, kh, kw], al.f32)
                        wgt3 = al.convert(weight_t[ic, c_out_grp + 3, kd, kh, kw], al.f32)
                        wgt4 = al.convert(weight_t[ic, c_out_grp + 4, kd, kh, kw], al.f32)
                        wgt5 = al.convert(weight_t[ic, c_out_grp + 5, kd, kh, kw], al.f32)
                        wgt6 = al.convert(weight_t[ic, c_out_grp + 6, kd, kh, kw], al.f32)
                        wgt7 = al.convert(weight_t[ic, c_out_grp + 7, kd, kh, kw], al.f32)
                        accum0 = accum0 + inp_val * wgt0
                        accum1 = accum1 + inp_val * wgt1
                        accum2 = accum2 + inp_val * wgt2
                        accum3 = accum3 + inp_val * wgt3
                        accum4 = accum4 + inp_val * wgt4
                        accum5 = accum5 + inp_val * wgt5
                        accum6 = accum6 + inp_val * wgt6
                        accum7 = accum7 + inp_val * wgt7

        output_t[batch_idx, c_out_grp, d_out, h_out, w_out] = al.convert(accum0, al.bf16)
        output_t[batch_idx, c_out_grp + 1, d_out, h_out, w_out] = al.convert(accum1, al.bf16)
        output_t[batch_idx, c_out_grp + 2, d_out, h_out, w_out] = al.convert(accum2, al.bf16)
        output_t[batch_idx, c_out_grp + 3, d_out, h_out, w_out] = al.convert(accum3, al.bf16)
        output_t[batch_idx, c_out_grp + 4, d_out, h_out, w_out] = al.convert(accum4, al.bf16)
        output_t[batch_idx, c_out_grp + 5, d_out, h_out, w_out] = al.convert(accum5, al.bf16)
        output_t[batch_idx, c_out_grp + 6, d_out, h_out, w_out] = al.convert(accum6, al.bf16)
        output_t[batch_idx, c_out_grp + 7, d_out, h_out, w_out] = al.convert(accum7, al.bf16)


def _compute_output_dims(
    D_in: int, H_in: int, W_in: int,
    KD: int, KH: int, KW: int,
    stride, padding, output_padding, dilation,
):
    """Compute ConvTranspose3d output spatial dimensions."""
    def _out_dim(L_in, k, s, p, d, op):
        return (L_in - 1) * s - 2 * p + d * (k - 1) + op + 1

    D_out = _out_dim(D_in, KD, stride[0], padding[0], dilation[0], output_padding[0])
    H_out = _out_dim(H_in, KH, stride[1], padding[1], dilation[1], output_padding[1])
    W_out = _out_dim(W_in, KW, stride[2], padding[2], dilation[2], output_padding[2])
    return D_out, H_out, W_out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._output_padding = output_padding
        self._groups = groups
        self._bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, D_in, H_in, W_in = x.shape
        KD, KH, KW = self._kernel_size
        C_out = self._out_channels

        dilation = self.conv_transpose3d.dilation

        D_out, H_out, W_out = _compute_output_dims(
            D_in, H_in, W_in, KD, KH, KW,
            self._stride, self._padding, self._output_padding, dilation,
        )

        x_bf16 = x.contiguous().to(torch.bfloat16)
        w_bf16 = self.conv_transpose3d.weight.contiguous().to(torch.bfloat16)

        out_bf16 = torch.empty(
            B, C_out, D_out, H_out, W_out,
            dtype=torch.bfloat16, device=x.device,
        )

        total_spatial = D_out * H_out * W_out
        BLOCK_SPATIAL = 128
        num_spatial_blocks = (total_spatial + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL

        conv_transpose3d_kernel[
            lambda: ((num_spatial_blocks, B, 1), (BLOCK_SPATIAL, 1, 1))
        ](
            x_bf16.data_ptr(),
            w_bf16.data_ptr(),
            out_bf16.data_ptr(),
            B,
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
            BLOCK_SPATIAL,
        )

        return out_bf16.to(x.dtype)
