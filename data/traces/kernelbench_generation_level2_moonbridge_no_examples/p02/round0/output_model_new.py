import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_transpose_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
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
    padding: al.i32,
    H_TILE: al.constexpr,
    W_TILE: al.constexpr,
    OC_TILE: al.constexpr,
    num_h_tiles: al.i32,
    num_w_tiles: al.i32,
):
    th = al.thread_id(0)
    tw = al.thread_id(1)

    n = al.block_id(0)
    oc_group = al.block_id(1)
    oc_base = oc_group * OC_TILE

    spatial_group = al.block_id(2)
    h_tile = spatial_group // num_w_tiles
    w_tile = spatial_group % num_w_tiles
    h_start = h_tile * H_TILE
    w_start = w_tile * W_TILE

    h_out = h_start + th
    w_out = w_start + tw

    wt_layout = al.make_layout(
        (C_in, C_out, K, K),
        (C_out * K * K, K * K, K, 1),
    )
    wt = al.make_tensor(weight_ptr, al.bf16, wt_layout)

    weight_smem = al.make_shared((2304,), al.bf16)

    tid_flat = th * W_TILE + tw
    for i in al.range(tid_flat, 2304, H_TILE * W_TILE):
        loc = i
        oc_off = loc // (C_in * 9)
        loc = loc % (C_in * 9)
        ic = loc // 9
        loc = loc % 9
        kh = loc // 3
        kw = loc % 3
        oc_val = oc_base + oc_off
        if oc_val < C_out:
            weight_smem[i] = wt[ic, oc_val, kh, kw]

    al.syncthreads()

    wt_smem_layout = al.make_layout(
        (OC_TILE, C_in, K, K),
        (C_in * 9, 9, 3, 1),
    )
    wt_smem_view = al.view(weight_smem, al.bf16, wt_smem_layout)

    in_layout = al.make_layout(
        (N, C_in, H_in, W_in),
        (C_in * H_in * W_in, H_in * W_in, W_in, 1),
    )
    inp = al.make_tensor(input_ptr, al.bf16, in_layout)

    bias_view = al.make_tensor(bias_ptr, al.bf16, al.make_layout((C_out,), (1,)))

    out_layout = al.make_layout(
        (N, C_out, H_out, W_out),
        (C_out * H_out * W_out, H_out * W_out, W_out, 1),
    )
    out = al.make_tensor(output_ptr, al.bf16, out_layout)

    if h_out < H_out and w_out < W_out:
        acc0 = al.convert(0.0, al.f32)
        acc1 = al.convert(0.0, al.f32)
        acc2 = al.convert(0.0, al.f32)
        acc3 = al.convert(0.0, al.f32)

        for ic in al.range(C_in):
            for kh in al.range(K):
                h_in_src = h_out + padding - kh
                h_rem = h_in_src % stride
                if h_rem == 0:
                    h_in = h_in_src // stride
                    if h_in >= 0 and h_in < H_in:
                        for kw in al.range(K):
                            w_in_src = w_out + padding - kw
                            w_rem = w_in_src % stride
                            if w_rem == 0:
                                w_in = w_in_src // stride
                                if w_in >= 0 and w_in < W_in:
                                    in_val = al.convert(inp[n, ic, h_in, w_in], al.f32)

                                    w0 = al.convert(wt_smem_view[0, ic, kh, kw], al.f32)
                                    acc0 = acc0 + in_val * w0
                                    w1 = al.convert(wt_smem_view[1, ic, kh, kw], al.f32)
                                    acc1 = acc1 + in_val * w1
                                    w2 = al.convert(wt_smem_view[2, ic, kh, kw], al.f32)
                                    acc2 = acc2 + in_val * w2
                                    w3 = al.convert(wt_smem_view[3, ic, kh, kw], al.f32)
                                    acc3 = acc3 + in_val * w3

        zero = al.convert(0.0, al.f32)
        one = al.convert(1.0, al.f32)
        half = al.convert(0.5, al.f32)

        oc_val0 = oc_base
        if oc_val0 < C_out:
            acc0 = acc0 + al.convert(bias_view[oc_val0], al.f32)
            if acc0 < zero: acc0 = zero
            if acc0 > one: acc0 = one
            if acc0 > half: acc0 = half
            out[n, oc_val0, h_out, w_out] = al.convert(acc0, al.bf16)

        oc_val1 = oc_base + 1
        if oc_val1 < C_out:
            acc1 = acc1 + al.convert(bias_view[oc_val1], al.f32)
            if acc1 < zero: acc1 = zero
            if acc1 > one: acc1 = one
            if acc1 > half: acc1 = half
            out[n, oc_val1, h_out, w_out] = al.convert(acc1, al.bf16)

        oc_val2 = oc_base + 2
        if oc_val2 < C_out:
            acc2 = acc2 + al.convert(bias_view[oc_val2], al.f32)
            if acc2 < zero: acc2 = zero
            if acc2 > one: acc2 = one
            if acc2 > half: acc2 = half
            out[n, oc_val2, h_out, w_out] = al.convert(acc2, al.bf16)

        oc_val3 = oc_base + 3
        if oc_val3 < C_out:
            acc3 = acc3 + al.convert(bias_view[oc_val3], al.f32)
            if acc3 < zero: acc3 = zero
            if acc3 > one: acc3 = one
            if acc3 > half: acc3 = half
            out[n, oc_val3, h_out, w_out] = al.convert(acc3, al.bf16)


def _run_conv_transpose(
    x: torch.Tensor,
    weight: torch.Tensor,
    combined_bias: torch.Tensor,
    H_TILE: int = 16,
    W_TILE: int = 16,
    OC_TILE: int = 4,
) -> torch.Tensor:
    N, C_in, H_in, W_in = x.shape
    C_in_wt, C_out, K, _ = weight.shape
    assert C_in == C_in_wt

    stride = 2
    padding = 1
    output_padding = 1
    H_out = (H_in - 1) * stride - 2 * padding + K + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + K + output_padding

    x_bf16 = x.to(torch.bfloat16).contiguous()
    w_bf16 = weight.to(torch.bfloat16).contiguous()
    b_bf16 = combined_bias.to(torch.bfloat16).contiguous()

    out_bf16 = torch.empty((N, C_out, H_out, W_out), dtype=torch.bfloat16, device=x.device)

    oc_groups = (C_out + OC_TILE - 1) // OC_TILE
    num_h_tiles = (H_out + H_TILE - 1) // H_TILE
    num_w_tiles = (W_out + W_TILE - 1) // W_TILE
    grid_z = num_h_tiles * num_w_tiles
    conv_transpose_kernel[lambda: ((N, oc_groups, grid_z), (H_TILE, W_TILE, 1))](
        x_bf16,
        w_bf16,
        b_bf16,
        out_bf16,
        N, C_in, C_out, H_in, W_in, H_out, W_out, K, stride, padding,
        H_TILE, W_TILE, OC_TILE,
        num_h_tiles, num_w_tiles,
    )

    return out_bf16


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
        scaling_factor,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        wt = self.conv_transpose.weight.data
        conv_bias = self.conv_transpose.bias.data
        extra_bias = self.bias.data

        combined_bias = (conv_bias.view(-1, 1, 1) + extra_bias).view(-1)

        out_bf16 = _run_conv_transpose(x, wt, combined_bias)
        return out_bf16
