import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_OC = 8
TILE_D = 1
TILE_H = 8
TILE_W = 8
K = 3


@avelang.jit
def conv3d_fused_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    conv_bias_ptr: al.Pointer(al.bf16),
    scale_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N_val: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    od_tiles: al.i32,
    oh_tiles: al.i32,
    ow_tiles: al.i32,
):
    input_t = al.make_tensor(
        input_ptr, al.bf16,
        al.make_layout(
            (N_val, C_in, D, H, W),
            (C_in * D * H * W, D * H * W, H * W, W, 1),
        ),
    )

    weight_t = al.make_tensor(
        weight_ptr, al.bf16,
        al.make_layout(
            (C_out, C_in, K, K, K),
            (C_in * K * K * K, K * K * K, K * K, K, 1),
        ),
    )

    conv_bias_t = al.make_tensor(
        conv_bias_ptr, al.bf16,
        al.make_layout((C_out,), (1,)),
    )

    scale_t = al.make_tensor(
        scale_ptr, al.bf16,
        al.make_layout((C_out,), (1,)),
    )

    bias_t = al.make_tensor(
        bias_ptr, al.bf16,
        al.make_layout((C_out,), (1,)),
    )

    output_t = al.make_tensor(
        output_ptr, al.bf16,
        al.make_layout(
            (N_val, C_out, D_out, H_out, W_out),
            (C_out * D_out * H_out * W_out, D_out * H_out * W_out, H_out * W_out, W_out, 1),
        ),
    )

    n = al.block_id(0)
    oc_block = al.block_id(1)
    spatial_block = al.block_id(2)

    tid = al.thread_id(0)

    td = tid // (TILE_H * TILE_W)
    rem_hw = tid % (TILE_H * TILE_W)
    th = rem_hw // TILE_W
    tw = rem_hw % TILE_W

    od_tile = spatial_block // (oh_tiles * ow_tiles)
    rem_ohw = spatial_block % (oh_tiles * ow_tiles)
    oh_tile = rem_ohw // ow_tiles
    ow_tile = rem_ohw % ow_tiles

    od_pos = od_tile * TILE_D + td
    oh_pos = oh_tile * TILE_H + th
    ow_pos = ow_tile * TILE_W + tw

    oc_base = oc_block * TILE_OC

    if (od_pos < D_out) and (oh_pos < H_out) and (ow_pos < W_out):
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
            for kd in al.range(0, K):
                for kh in al.range(0, K):
                    for kw in al.range(0, K):
                        in_val = al.convert(
                            input_t[n, ic, od_pos + kd, oh_pos + kh, ow_pos + kw],
                            al.f32,
                        )

                        if oc_base + 0 < C_out:
                            w_val = al.convert(
                                weight_t[oc_base + 0, ic, kd, kh, kw], al.f32
                            )
                            acc0 = acc0 + in_val * w_val
                        if oc_base + 1 < C_out:
                            w_val = al.convert(
                                weight_t[oc_base + 1, ic, kd, kh, kw], al.f32
                            )
                            acc1 = acc1 + in_val * w_val
                        if oc_base + 2 < C_out:
                            w_val = al.convert(
                                weight_t[oc_base + 2, ic, kd, kh, kw], al.f32
                            )
                            acc2 = acc2 + in_val * w_val
                        if oc_base + 3 < C_out:
                            w_val = al.convert(
                                weight_t[oc_base + 3, ic, kd, kh, kw], al.f32
                            )
                            acc3 = acc3 + in_val * w_val
                        if oc_base + 4 < C_out:
                            w_val = al.convert(
                                weight_t[oc_base + 4, ic, kd, kh, kw], al.f32
                            )
                            acc4 = acc4 + in_val * w_val
                        if oc_base + 5 < C_out:
                            w_val = al.convert(
                                weight_t[oc_base + 5, ic, kd, kh, kw], al.f32
                            )
                            acc5 = acc5 + in_val * w_val
                        if oc_base + 6 < C_out:
                            w_val = al.convert(
                                weight_t[oc_base + 6, ic, kd, kh, kw], al.f32
                            )
                            acc6 = acc6 + in_val * w_val
                        if oc_base + 7 < C_out:
                            w_val = al.convert(
                                weight_t[oc_base + 7, ic, kd, kh, kw], al.f32
                            )
                            acc7 = acc7 + in_val * w_val

        zero_f32 = al.convert(0.0, al.f32)
        one_f32 = al.convert(1.0, al.f32)

        if oc_base + 0 < C_out:
            acc0 = acc0 * al.convert(scale_t[oc_base + 0], al.f32)
            acc0 = al.tanh(acc0)
            acc0 = acc0 * al.convert(bias_t[oc_base + 0], al.f32)
            neg0 = zero_f32 - acc0
            sig0 = one_f32 / (one_f32 + al.exp(neg0))
            output_t[n, oc_base + 0, od_pos, oh_pos, ow_pos] = al.convert(sig0, al.bf16)

        if oc_base + 1 < C_out:
            acc1 = acc1 * al.convert(scale_t[oc_base + 1], al.f32)
            acc1 = al.tanh(acc1)
            acc1 = acc1 * al.convert(bias_t[oc_base + 1], al.f32)
            neg1 = zero_f32 - acc1
            sig1 = one_f32 / (one_f32 + al.exp(neg1))
            output_t[n, oc_base + 1, od_pos, oh_pos, ow_pos] = al.convert(sig1, al.bf16)

        if oc_base + 2 < C_out:
            acc2 = acc2 * al.convert(scale_t[oc_base + 2], al.f32)
            acc2 = al.tanh(acc2)
            acc2 = acc2 * al.convert(bias_t[oc_base + 2], al.f32)
            neg2 = zero_f32 - acc2
            sig2 = one_f32 / (one_f32 + al.exp(neg2))
            output_t[n, oc_base + 2, od_pos, oh_pos, ow_pos] = al.convert(sig2, al.bf16)

        if oc_base + 3 < C_out:
            acc3 = acc3 * al.convert(scale_t[oc_base + 3], al.f32)
            acc3 = al.tanh(acc3)
            acc3 = acc3 * al.convert(bias_t[oc_base + 3], al.f32)
            neg3 = zero_f32 - acc3
            sig3 = one_f32 / (one_f32 + al.exp(neg3))
            output_t[n, oc_base + 3, od_pos, oh_pos, ow_pos] = al.convert(sig3, al.bf16)

        if oc_base + 4 < C_out:
            acc4 = acc4 * al.convert(scale_t[oc_base + 4], al.f32)
            acc4 = al.tanh(acc4)
            acc4 = acc4 * al.convert(bias_t[oc_base + 4], al.f32)
            neg4 = zero_f32 - acc4
            sig4 = one_f32 / (one_f32 + al.exp(neg4))
            output_t[n, oc_base + 4, od_pos, oh_pos, ow_pos] = al.convert(sig4, al.bf16)

        if oc_base + 5 < C_out:
            acc5 = acc5 * al.convert(scale_t[oc_base + 5], al.f32)
            acc5 = al.tanh(acc5)
            acc5 = acc5 * al.convert(bias_t[oc_base + 5], al.f32)
            neg5 = zero_f32 - acc5
            sig5 = one_f32 / (one_f32 + al.exp(neg5))
            output_t[n, oc_base + 5, od_pos, oh_pos, ow_pos] = al.convert(sig5, al.bf16)

        if oc_base + 6 < C_out:
            acc6 = acc6 * al.convert(scale_t[oc_base + 6], al.f32)
            acc6 = al.tanh(acc6)
            acc6 = acc6 * al.convert(bias_t[oc_base + 6], al.f32)
            neg6 = zero_f32 - acc6
            sig6 = one_f32 / (one_f32 + al.exp(neg6))
            output_t[n, oc_base + 6, od_pos, oh_pos, ow_pos] = al.convert(sig6, al.bf16)

        if oc_base + 7 < C_out:
            acc7 = acc7 * al.convert(scale_t[oc_base + 7], al.f32)
            acc7 = al.tanh(acc7)
            acc7 = acc7 * al.convert(bias_t[oc_base + 7], al.f32)
            neg7 = zero_f32 - acc7
            sig7 = one_f32 / (one_f32 + al.exp(neg7))
            output_t[n, oc_base + 7, od_pos, oh_pos, ow_pos] = al.convert(sig7, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        conv_ref = nn.Conv3d(in_channels, out_channels, kernel_size, bias=True)
        self.conv_weight = nn.Parameter(conv_ref.weight.data.clone())
        self.conv_bias = nn.Parameter(conv_ref.bias.data.clone())
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        N_val, C_in, D, H, W = x.shape
        C_out = self.out_channels
        K_size = self.kernel_size
        D_out = D - K_size + 1
        H_out = H - K_size + 1
        W_out = W - K_size + 1

        od_tiles = (D_out + TILE_D - 1) // TILE_D
        oh_tiles = (H_out + TILE_H - 1) // TILE_H
        ow_tiles = (W_out + TILE_W - 1) // TILE_W
        spatial_tiles = od_tiles * oh_tiles * ow_tiles
        oc_tiles = (C_out + TILE_OC - 1) // TILE_OC

        x_bf16 = x.contiguous().to(torch.bfloat16)
        w_bf16 = self.conv_weight.contiguous().to(torch.bfloat16)
        cb_bf16 = self.conv_bias.contiguous().to(torch.bfloat16)
        sc_bf16 = self.scaling_factor.contiguous().to(torch.bfloat16)
        bi_bf16 = self.bias.contiguous().to(torch.bfloat16)

        out = torch.empty(
            (N_val, C_out, D_out, H_out, W_out),
            dtype=torch.bfloat16,
            device=x.device,
        )

        conv3d_fused_kernel[
            lambda: (
                (N_val, oc_tiles, spatial_tiles),
                (TILE_D * TILE_H * TILE_W, 1, 1),
            )
        ](
            x_bf16, w_bf16, cb_bf16, sc_bf16, bi_bf16, out,
            N_val, C_in, C_out, D, H, W, D_out, H_out, W_out,
            od_tiles, oh_tiles, ow_tiles,
        )
        return out
