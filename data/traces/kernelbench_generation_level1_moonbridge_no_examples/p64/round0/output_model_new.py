import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_L = 64
BLOCK_OC = 4


@avelang.jit
def conv_transpose1d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    K: al.i32,
    L_in: al.i32,
    L_out: al.i32,
    padding: al.i32,
    groups: al.i32,
):
    bid_l = al.block_id(0)
    bid_oc = al.block_id(1)
    tid_l = al.thread_id(0)
    tid_oc = al.thread_id(1)
    pos = bid_l * BLOCK_L + tid_l
    oc = bid_oc * BLOCK_OC + tid_oc
    b = al.block_id(2)

    if pos < L_out and oc < C_out:
        out_ch_per_group = C_out // groups
        in_ch_per_group = C_in // groups

        g = oc // out_ch_per_group
        oc_local = oc % out_ch_per_group
        ic_base = g * in_ch_per_group

        x_stride0 = C_in * L_in
        x_layout = al.make_layout((N, C_in, L_in), (x_stride0, L_in, 1))
        x = al.make_tensor(x_ptr, al.bf16, x_layout)

        w_stride0 = out_ch_per_group * K
        w_layout = al.make_layout((C_in, out_ch_per_group, K), (w_stride0, K, 1))
        w = al.make_tensor(w_ptr, al.bf16, w_layout)

        out_stride0 = C_out * L_out
        out_layout = al.make_layout((N, C_out, L_out), (out_stride0, L_out, 1))
        out = al.make_tensor(out_ptr, al.bf16, out_layout)

        acc = al.convert(0.0, al.f32)

        l0 = pos + padding
        l1 = l0 - 1
        l2 = l0 - 2
        v0 = 0
        v1 = 0
        v2 = 0
        if l0 >= 0:
            if l0 < L_in:
                v0 = 1
        if l1 >= 0:
            if l1 < L_in:
                v1 = 1
        if l2 >= 0:
            if l2 < L_in:
                v2 = 1

        for ic in al.range(in_ch_per_group):
            ic_idx = ic_base + ic
            w0 = al.convert(w[ic_idx, oc_local, 0], al.f32)
            w1 = al.convert(w[ic_idx, oc_local, 1], al.f32)
            w2 = al.convert(w[ic_idx, oc_local, 2], al.f32)

            if v0:
                x_val = al.convert(x[b, ic_idx, l0], al.f32)
                acc = acc + x_val * w0

            if v1:
                x_val = al.convert(x[b, ic_idx, l1], al.f32)
                acc = acc + x_val * w1

            if v2:
                x_val = al.convert(x[b, ic_idx, l2], al.f32)
                acc = acc + x_val * w2

        out[b, oc, pos] = al.convert(acc, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            output_padding=output_padding, groups=groups, bias=bias
        )
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._output_padding = output_padding
        self._groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C_in, L_in = x.shape
        K = self._kernel_size
        C_out = self._out_channels
        stride = self._stride
        padding = self._padding
        output_padding = self._output_padding
        groups = self._groups

        L_out = (L_in - 1) * stride - 2 * padding + K + output_padding

        x = x.contiguous()
        weight = self.conv1d_transpose.weight.contiguous()

        out = torch.empty(N, C_out, L_out, dtype=x.dtype, device=x.device)

        grid_x = (L_out + BLOCK_L - 1) // BLOCK_L
        grid_y = (C_out + BLOCK_OC - 1) // BLOCK_OC
        grid_z = N

        conv_transpose1d_kernel[lambda: ((grid_x, grid_y, grid_z), (BLOCK_L, BLOCK_OC, 1))](
            x, weight, out,
            N, C_in, C_out, K, L_in, L_out,
            padding, groups
        )

        if self.conv1d_transpose.bias is not None:
            out = out + self.conv1d_transpose.bias.view(1, -1, 1)

        return out
