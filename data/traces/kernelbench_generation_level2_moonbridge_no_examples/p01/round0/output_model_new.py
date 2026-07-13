import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_H = 8
TILE_W = 8
TILE_OC = 8
K = 3


@avelang.jit
def conv2d_relu_bias_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    conv_bias_ptr: al.Pointer(al.bf16),
    add_bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N_val: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    oh_tiles: al.i32,
    ow_tiles: al.i32,
):
    input_t = al.make_tensor(
        input_ptr, al.bf16,
        al.make_layout(
            (N_val, C_in, H, W),
            (C_in * H * W, H * W, W, 1),
        ),
    )

    weight_t = al.make_tensor(
        weight_ptr, al.bf16,
        al.make_layout(
            (C_out, C_in, K, K),
            (C_in * K * K, K * K, K, 1),
        ),
    )

    conv_bias_t = al.make_tensor(
        conv_bias_ptr, al.bf16,
        al.make_layout((C_out,), (1,)),
    )

    add_bias_t = al.make_tensor(
        add_bias_ptr, al.bf16,
        al.make_layout((C_out, 1, 1), (1, 1, 1)),
    )

    output_t = al.make_tensor(
        output_ptr, al.bf16,
        al.make_layout(
            (N_val, C_out, H_out, W_out),
            (C_out * H_out * W_out, H_out * W_out, W_out, 1),
        ),
    )

    n = al.block_id(0)
    oc_block = al.block_id(1)
    spatial_block = al.block_id(2)

    tid = al.thread_id(0)
    th = tid / TILE_W
    tw = tid % TILE_W

    oh_tile = spatial_block / ow_tiles
    ow_tile = spatial_block % ow_tiles

    oh = oh_tile * TILE_H + th
    ow = ow_tile * TILE_W + tw

    oc_base = oc_block * TILE_OC

    if (oh < H_out) and (ow < W_out):
        acc0 = al.convert(0.0, al.f32)
        acc1 = al.convert(0.0, al.f32)
        acc2 = al.convert(0.0, al.f32)
        acc3 = al.convert(0.0, al.f32)
        acc4 = al.convert(0.0, al.f32)
        acc5 = al.convert(0.0, al.f32)
        acc6 = al.convert(0.0, al.f32)
        acc7 = al.convert(0.0, al.f32)

        if oc_base + 0 < C_out:
            acc0 = al.convert(conv_bias_t[oc_base + 0], al.f32)
        if oc_base + 1 < C_out:
            acc1 = al.convert(conv_bias_t[oc_base + 1], al.f32)
        if oc_base + 2 < C_out:
            acc2 = al.convert(conv_bias_t[oc_base + 2], al.f32)
        if oc_base + 3 < C_out:
            acc3 = al.convert(conv_bias_t[oc_base + 3], al.f32)
        if oc_base + 4 < C_out:
            acc4 = al.convert(conv_bias_t[oc_base + 4], al.f32)
        if oc_base + 5 < C_out:
            acc5 = al.convert(conv_bias_t[oc_base + 5], al.f32)
        if oc_base + 6 < C_out:
            acc6 = al.convert(conv_bias_t[oc_base + 6], al.f32)
        if oc_base + 7 < C_out:
            acc7 = al.convert(conv_bias_t[oc_base + 7], al.f32)

        for ic in al.range(0, C_in):
            in_00 = al.convert(input_t[n, ic, oh + 0, ow + 0], al.f32)
            in_01 = al.convert(input_t[n, ic, oh + 0, ow + 1], al.f32)
            in_02 = al.convert(input_t[n, ic, oh + 0, ow + 2], al.f32)
            in_10 = al.convert(input_t[n, ic, oh + 1, ow + 0], al.f32)
            in_11 = al.convert(input_t[n, ic, oh + 1, ow + 1], al.f32)
            in_12 = al.convert(input_t[n, ic, oh + 1, ow + 2], al.f32)
            in_20 = al.convert(input_t[n, ic, oh + 2, ow + 0], al.f32)
            in_21 = al.convert(input_t[n, ic, oh + 2, ow + 1], al.f32)
            in_22 = al.convert(input_t[n, ic, oh + 2, ow + 2], al.f32)

            if oc_base + 0 < C_out:
                w_00 = al.convert(weight_t[oc_base + 0, ic, 0, 0], al.f32)
                w_01 = al.convert(weight_t[oc_base + 0, ic, 0, 1], al.f32)
                w_02 = al.convert(weight_t[oc_base + 0, ic, 0, 2], al.f32)
                w_10 = al.convert(weight_t[oc_base + 0, ic, 1, 0], al.f32)
                w_11 = al.convert(weight_t[oc_base + 0, ic, 1, 1], al.f32)
                w_12 = al.convert(weight_t[oc_base + 0, ic, 1, 2], al.f32)
                w_20 = al.convert(weight_t[oc_base + 0, ic, 2, 0], al.f32)
                w_21 = al.convert(weight_t[oc_base + 0, ic, 2, 1], al.f32)
                w_22 = al.convert(weight_t[oc_base + 0, ic, 2, 2], al.f32)
                acc0 = acc0 + in_00 * w_00 + in_01 * w_01 + in_02 * w_02
                acc0 = acc0 + in_10 * w_10 + in_11 * w_11 + in_12 * w_12
                acc0 = acc0 + in_20 * w_20 + in_21 * w_21 + in_22 * w_22

            if oc_base + 1 < C_out:
                w_00 = al.convert(weight_t[oc_base + 1, ic, 0, 0], al.f32)
                w_01 = al.convert(weight_t[oc_base + 1, ic, 0, 1], al.f32)
                w_02 = al.convert(weight_t[oc_base + 1, ic, 0, 2], al.f32)
                w_10 = al.convert(weight_t[oc_base + 1, ic, 1, 0], al.f32)
                w_11 = al.convert(weight_t[oc_base + 1, ic, 1, 1], al.f32)
                w_12 = al.convert(weight_t[oc_base + 1, ic, 1, 2], al.f32)
                w_20 = al.convert(weight_t[oc_base + 1, ic, 2, 0], al.f32)
                w_21 = al.convert(weight_t[oc_base + 1, ic, 2, 1], al.f32)
                w_22 = al.convert(weight_t[oc_base + 1, ic, 2, 2], al.f32)
                acc1 = acc1 + in_00 * w_00 + in_01 * w_01 + in_02 * w_02
                acc1 = acc1 + in_10 * w_10 + in_11 * w_11 + in_12 * w_12
                acc1 = acc1 + in_20 * w_20 + in_21 * w_21 + in_22 * w_22

            if oc_base + 2 < C_out:
                w_00 = al.convert(weight_t[oc_base + 2, ic, 0, 0], al.f32)
                w_01 = al.convert(weight_t[oc_base + 2, ic, 0, 1], al.f32)
                w_02 = al.convert(weight_t[oc_base + 2, ic, 0, 2], al.f32)
                w_10 = al.convert(weight_t[oc_base + 2, ic, 1, 0], al.f32)
                w_11 = al.convert(weight_t[oc_base + 2, ic, 1, 1], al.f32)
                w_12 = al.convert(weight_t[oc_base + 2, ic, 1, 2], al.f32)
                w_20 = al.convert(weight_t[oc_base + 2, ic, 2, 0], al.f32)
                w_21 = al.convert(weight_t[oc_base + 2, ic, 2, 1], al.f32)
                w_22 = al.convert(weight_t[oc_base + 2, ic, 2, 2], al.f32)
                acc2 = acc2 + in_00 * w_00 + in_01 * w_01 + in_02 * w_02
                acc2 = acc2 + in_10 * w_10 + in_11 * w_11 + in_12 * w_12
                acc2 = acc2 + in_20 * w_20 + in_21 * w_21 + in_22 * w_22

            if oc_base + 3 < C_out:
                w_00 = al.convert(weight_t[oc_base + 3, ic, 0, 0], al.f32)
                w_01 = al.convert(weight_t[oc_base + 3, ic, 0, 1], al.f32)
                w_02 = al.convert(weight_t[oc_base + 3, ic, 0, 2], al.f32)
                w_10 = al.convert(weight_t[oc_base + 3, ic, 1, 0], al.f32)
                w_11 = al.convert(weight_t[oc_base + 3, ic, 1, 1], al.f32)
                w_12 = al.convert(weight_t[oc_base + 3, ic, 1, 2], al.f32)
                w_20 = al.convert(weight_t[oc_base + 3, ic, 2, 0], al.f32)
                w_21 = al.convert(weight_t[oc_base + 3, ic, 2, 1], al.f32)
                w_22 = al.convert(weight_t[oc_base + 3, ic, 2, 2], al.f32)
                acc3 = acc3 + in_00 * w_00 + in_01 * w_01 + in_02 * w_02
                acc3 = acc3 + in_10 * w_10 + in_11 * w_11 + in_12 * w_12
                acc3 = acc3 + in_20 * w_20 + in_21 * w_21 + in_22 * w_22

            if oc_base + 4 < C_out:
                w_00 = al.convert(weight_t[oc_base + 4, ic, 0, 0], al.f32)
                w_01 = al.convert(weight_t[oc_base + 4, ic, 0, 1], al.f32)
                w_02 = al.convert(weight_t[oc_base + 4, ic, 0, 2], al.f32)
                w_10 = al.convert(weight_t[oc_base + 4, ic, 1, 0], al.f32)
                w_11 = al.convert(weight_t[oc_base + 4, ic, 1, 1], al.f32)
                w_12 = al.convert(weight_t[oc_base + 4, ic, 1, 2], al.f32)
                w_20 = al.convert(weight_t[oc_base + 4, ic, 2, 0], al.f32)
                w_21 = al.convert(weight_t[oc_base + 4, ic, 2, 1], al.f32)
                w_22 = al.convert(weight_t[oc_base + 4, ic, 2, 2], al.f32)
                acc4 = acc4 + in_00 * w_00 + in_01 * w_01 + in_02 * w_02
                acc4 = acc4 + in_10 * w_10 + in_11 * w_11 + in_12 * w_12
                acc4 = acc4 + in_20 * w_20 + in_21 * w_21 + in_22 * w_22

            if oc_base + 5 < C_out:
                w_00 = al.convert(weight_t[oc_base + 5, ic, 0, 0], al.f32)
                w_01 = al.convert(weight_t[oc_base + 5, ic, 0, 1], al.f32)
                w_02 = al.convert(weight_t[oc_base + 5, ic, 0, 2], al.f32)
                w_10 = al.convert(weight_t[oc_base + 5, ic, 1, 0], al.f32)
                w_11 = al.convert(weight_t[oc_base + 5, ic, 1, 1], al.f32)
                w_12 = al.convert(weight_t[oc_base + 5, ic, 1, 2], al.f32)
                w_20 = al.convert(weight_t[oc_base + 5, ic, 2, 0], al.f32)
                w_21 = al.convert(weight_t[oc_base + 5, ic, 2, 1], al.f32)
                w_22 = al.convert(weight_t[oc_base + 5, ic, 2, 2], al.f32)
                acc5 = acc5 + in_00 * w_00 + in_01 * w_01 + in_02 * w_02
                acc5 = acc5 + in_10 * w_10 + in_11 * w_11 + in_12 * w_12
                acc5 = acc5 + in_20 * w_20 + in_21 * w_21 + in_22 * w_22

            if oc_base + 6 < C_out:
                w_00 = al.convert(weight_t[oc_base + 6, ic, 0, 0], al.f32)
                w_01 = al.convert(weight_t[oc_base + 6, ic, 0, 1], al.f32)
                w_02 = al.convert(weight_t[oc_base + 6, ic, 0, 2], al.f32)
                w_10 = al.convert(weight_t[oc_base + 6, ic, 1, 0], al.f32)
                w_11 = al.convert(weight_t[oc_base + 6, ic, 1, 1], al.f32)
                w_12 = al.convert(weight_t[oc_base + 6, ic, 1, 2], al.f32)
                w_20 = al.convert(weight_t[oc_base + 6, ic, 2, 0], al.f32)
                w_21 = al.convert(weight_t[oc_base + 6, ic, 2, 1], al.f32)
                w_22 = al.convert(weight_t[oc_base + 6, ic, 2, 2], al.f32)
                acc6 = acc6 + in_00 * w_00 + in_01 * w_01 + in_02 * w_02
                acc6 = acc6 + in_10 * w_10 + in_11 * w_11 + in_12 * w_12
                acc6 = acc6 + in_20 * w_20 + in_21 * w_21 + in_22 * w_22

            if oc_base + 7 < C_out:
                w_00 = al.convert(weight_t[oc_base + 7, ic, 0, 0], al.f32)
                w_01 = al.convert(weight_t[oc_base + 7, ic, 0, 1], al.f32)
                w_02 = al.convert(weight_t[oc_base + 7, ic, 0, 2], al.f32)
                w_10 = al.convert(weight_t[oc_base + 7, ic, 1, 0], al.f32)
                w_11 = al.convert(weight_t[oc_base + 7, ic, 1, 1], al.f32)
                w_12 = al.convert(weight_t[oc_base + 7, ic, 1, 2], al.f32)
                w_20 = al.convert(weight_t[oc_base + 7, ic, 2, 0], al.f32)
                w_21 = al.convert(weight_t[oc_base + 7, ic, 2, 1], al.f32)
                w_22 = al.convert(weight_t[oc_base + 7, ic, 2, 2], al.f32)
                acc7 = acc7 + in_00 * w_00 + in_01 * w_01 + in_02 * w_02
                acc7 = acc7 + in_10 * w_10 + in_11 * w_11 + in_12 * w_12
                acc7 = acc7 + in_20 * w_20 + in_21 * w_21 + in_22 * w_22

        zero_f32 = al.convert(0.0, al.f32)

        if oc_base + 0 < C_out:
            if acc0 < zero_f32:
                acc0 = zero_f32
            acc0 = acc0 + al.convert(add_bias_t[oc_base + 0, 0, 0], al.f32)
            output_t[n, oc_base + 0, oh, ow] = al.convert(acc0, al.bf16)

        if oc_base + 1 < C_out:
            if acc1 < zero_f32:
                acc1 = zero_f32
            acc1 = acc1 + al.convert(add_bias_t[oc_base + 1, 0, 0], al.f32)
            output_t[n, oc_base + 1, oh, ow] = al.convert(acc1, al.bf16)

        if oc_base + 2 < C_out:
            if acc2 < zero_f32:
                acc2 = zero_f32
            acc2 = acc2 + al.convert(add_bias_t[oc_base + 2, 0, 0], al.f32)
            output_t[n, oc_base + 2, oh, ow] = al.convert(acc2, al.bf16)

        if oc_base + 3 < C_out:
            if acc3 < zero_f32:
                acc3 = zero_f32
            acc3 = acc3 + al.convert(add_bias_t[oc_base + 3, 0, 0], al.f32)
            output_t[n, oc_base + 3, oh, ow] = al.convert(acc3, al.bf16)

        if oc_base + 4 < C_out:
            if acc4 < zero_f32:
                acc4 = zero_f32
            acc4 = acc4 + al.convert(add_bias_t[oc_base + 4, 0, 0], al.f32)
            output_t[n, oc_base + 4, oh, ow] = al.convert(acc4, al.bf16)

        if oc_base + 5 < C_out:
            if acc5 < zero_f32:
                acc5 = zero_f32
            acc5 = acc5 + al.convert(add_bias_t[oc_base + 5, 0, 0], al.f32)
            output_t[n, oc_base + 5, oh, ow] = al.convert(acc5, al.bf16)

        if oc_base + 6 < C_out:
            if acc6 < zero_f32:
                acc6 = zero_f32
            acc6 = acc6 + al.convert(add_bias_t[oc_base + 6, 0, 0], al.f32)
            output_t[n, oc_base + 6, oh, ow] = al.convert(acc6, al.bf16)

        if oc_base + 7 < C_out:
            if acc7 < zero_f32:
                acc7 = zero_f32
            acc7 = acc7 + al.convert(add_bias_t[oc_base + 7, 0, 0], al.f32)
            output_t[n, oc_base + 7, oh, ow] = al.convert(acc7, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        conv_ref = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)
        self.conv_weight = nn.Parameter(conv_ref.weight.data.clone())
        self.conv_bias = nn.Parameter(conv_ref.bias.data.clone())
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        N_val, C_in, H, W = x.shape
        C_out = self.out_channels
        K_size = self.kernel_size
        H_out = H - K_size + 1
        W_out = W - K_size + 1

        ow_tiles = (W_out + TILE_W - 1) // TILE_W
        oh_tiles = (H_out + TILE_H - 1) // TILE_H
        spatial_tiles = oh_tiles * ow_tiles
        oc_tiles = (C_out + TILE_OC - 1) // TILE_OC

        x_bf16 = x.contiguous().to(torch.bfloat16)
        w_bf16 = self.conv_weight.contiguous().to(torch.bfloat16)
        cb_bf16 = self.conv_bias.contiguous().to(torch.bfloat16)
        ab_bf16 = self.bias.contiguous().to(torch.bfloat16)

        out = torch.empty(
            (N_val, C_out, H_out, W_out),
            dtype=torch.bfloat16,
            device=x.device,
        )

        conv2d_relu_bias_kernel[
            lambda: (
                (N_val, oc_tiles, spatial_tiles),
                (TILE_H * TILE_W, 1, 1),
            )
        ](
            x_bf16, w_bf16, cb_bf16, ab_bf16, out,
            N_val, C_in, C_out, H, W, H_out, W_out, oh_tiles, ow_tiles,
        )
        return out
