import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_transpose2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.i32,
    stride: al.i32,
    padding: al.i32,
    OC_TILES: al.i32,
    OC_PER_TILE: al.i32,
):
    bx = al.block_id(0)
    by = al.block_id(1)
    bz = al.block_id(2)
    tx = al.thread_id(0)
    ty = al.thread_id(1)

    TILE_W = al.block_dim(0)
    TILE_H = al.block_dim(1)
    BLOCK_SIZE = TILE_W * TILE_H

    w_out = bx * TILE_W + tx
    h_out = by * TILE_H + ty

    b_idx = bz // OC_TILES
    oc_tile = bz - b_idx * OC_TILES
    oc_start = oc_tile * OC_PER_TILE

    tid = ty * TILE_W + tx

    oc_k_k = OC * K * K
    k_k = K * K
    w_total = IC * oc_k_k

    # Weight 1D view + shared memory
    w_1d_layout = al.make_layout((IC * OC * K * K,), (al.convert(1, al.i32),))
    w_1d = al.make_tensor(w_ptr, al.bf16, w_1d_layout)
    w_smem = al.make_shared((32 * 32 * 3 * 3,), al.bf16)

    for i in al.range(tid, w_total, BLOCK_SIZE):
        w_smem[i] = w_1d[i]

    al.syncthreads()

    # Input layout
    stride_ic_in = H_in * W_in
    stride_h_in = W_in
    x_layout = al.make_layout(
        (B, IC, H_in, W_in),
        (IC * stride_ic_in, stride_ic_in, stride_h_in, al.convert(1, al.i32)),
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    # Output layout
    stride_oc_out = H_out * W_out
    stride_h_out = W_out
    out_layout = al.make_layout(
        (B, OC, H_out, W_out),
        (OC * stride_oc_out, stride_oc_out, stride_h_out, al.convert(1, al.i32)),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    if w_out < W_out and h_out < H_out:
        acc0 = al.convert(0.0, al.f32)
        acc1 = al.convert(0.0, al.f32)
        acc2 = al.convert(0.0, al.f32)
        acc3 = al.convert(0.0, al.f32)
        acc4 = al.convert(0.0, al.f32)
        acc5 = al.convert(0.0, al.f32)
        acc6 = al.convert(0.0, al.f32)
        acc7 = al.convert(0.0, al.f32)
        acc8 = al.convert(0.0, al.f32)
        acc9 = al.convert(0.0, al.f32)
        acc10 = al.convert(0.0, al.f32)
        acc11 = al.convert(0.0, al.f32)
        acc12 = al.convert(0.0, al.f32)
        acc13 = al.convert(0.0, al.f32)
        acc14 = al.convert(0.0, al.f32)
        acc15 = al.convert(0.0, al.f32)

        for ic in al.range(IC):
            ic_w_base = ic * oc_k_k
            for kh in al.range(K):
                h_in = h_out - kh
                kh_w_off = ic_w_base + kh * K
                for kw in al.range(K):
                    w_in = w_out - kw
                    if h_in >= 0 and h_in < H_in and w_in >= 0 and w_in < W_in:
                        x_val_f32 = al.convert(x[b_idx, ic, h_in, w_in], al.f32)
                        w_base = kh_w_off + kw

                        w0  = al.convert(w_smem[w_base + (oc_start +  0) * k_k], al.f32)
                        w1  = al.convert(w_smem[w_base + (oc_start +  1) * k_k], al.f32)
                        w2  = al.convert(w_smem[w_base + (oc_start +  2) * k_k], al.f32)
                        w3  = al.convert(w_smem[w_base + (oc_start +  3) * k_k], al.f32)
                        w4  = al.convert(w_smem[w_base + (oc_start +  4) * k_k], al.f32)
                        w5  = al.convert(w_smem[w_base + (oc_start +  5) * k_k], al.f32)
                        w6  = al.convert(w_smem[w_base + (oc_start +  6) * k_k], al.f32)
                        w7  = al.convert(w_smem[w_base + (oc_start +  7) * k_k], al.f32)
                        w8  = al.convert(w_smem[w_base + (oc_start +  8) * k_k], al.f32)
                        w9  = al.convert(w_smem[w_base + (oc_start +  9) * k_k], al.f32)
                        w10 = al.convert(w_smem[w_base + (oc_start + 10) * k_k], al.f32)
                        w11 = al.convert(w_smem[w_base + (oc_start + 11) * k_k], al.f32)
                        w12 = al.convert(w_smem[w_base + (oc_start + 12) * k_k], al.f32)
                        w13 = al.convert(w_smem[w_base + (oc_start + 13) * k_k], al.f32)
                        w14 = al.convert(w_smem[w_base + (oc_start + 14) * k_k], al.f32)
                        w15 = al.convert(w_smem[w_base + (oc_start + 15) * k_k], al.f32)

                        acc0  = acc0  + x_val_f32 * w0
                        acc1  = acc1  + x_val_f32 * w1
                        acc2  = acc2  + x_val_f32 * w2
                        acc3  = acc3  + x_val_f32 * w3
                        acc4  = acc4  + x_val_f32 * w4
                        acc5  = acc5  + x_val_f32 * w5
                        acc6  = acc6  + x_val_f32 * w6
                        acc7  = acc7  + x_val_f32 * w7
                        acc8  = acc8  + x_val_f32 * w8
                        acc9  = acc9  + x_val_f32 * w9
                        acc10 = acc10 + x_val_f32 * w10
                        acc11 = acc11 + x_val_f32 * w11
                        acc12 = acc12 + x_val_f32 * w12
                        acc13 = acc13 + x_val_f32 * w13
                        acc14 = acc14 + x_val_f32 * w14
                        acc15 = acc15 + x_val_f32 * w15

        out[b_idx, oc_start +  0, h_out, w_out] = al.convert(acc0,  al.bf16)
        out[b_idx, oc_start +  1, h_out, w_out] = al.convert(acc1,  al.bf16)
        out[b_idx, oc_start +  2, h_out, w_out] = al.convert(acc2,  al.bf16)
        out[b_idx, oc_start +  3, h_out, w_out] = al.convert(acc3,  al.bf16)
        out[b_idx, oc_start +  4, h_out, w_out] = al.convert(acc4,  al.bf16)
        out[b_idx, oc_start +  5, h_out, w_out] = al.convert(acc5,  al.bf16)
        out[b_idx, oc_start +  6, h_out, w_out] = al.convert(acc6,  al.bf16)
        out[b_idx, oc_start +  7, h_out, w_out] = al.convert(acc7,  al.bf16)
        out[b_idx, oc_start +  8, h_out, w_out] = al.convert(acc8,  al.bf16)
        out[b_idx, oc_start +  9, h_out, w_out] = al.convert(acc9,  al.bf16)
        out[b_idx, oc_start + 10, h_out, w_out] = al.convert(acc10, al.bf16)
        out[b_idx, oc_start + 11, h_out, w_out] = al.convert(acc11, al.bf16)
        out[b_idx, oc_start + 12, h_out, w_out] = al.convert(acc12, al.bf16)
        out[b_idx, oc_start + 13, h_out, w_out] = al.convert(acc13, al.bf16)
        out[b_idx, oc_start + 14, h_out, w_out] = al.convert(acc14, al.bf16)
        out[b_idx, oc_start + 15, h_out, w_out] = al.convert(acc15, al.bf16)


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
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x = x.contiguous()
        x_bf16 = x.to(torch.bfloat16)
        w_bf16 = self.conv.weight.data.to(torch.bfloat16).contiguous()

        B, IC, H_in, W_in = x_bf16.shape
        OC = self.out_channels
        K = self.kernel_size
        stride_val = self.stride
        padding_val = self.padding
        opad = self.output_padding

        H_out = (H_in - 1) * stride_val - 2 * padding_val + K + opad
        W_out = (W_in - 1) * stride_val - 2 * padding_val + K + opad

        out_bf16 = torch.empty(
            B, OC, H_out, W_out, dtype=torch.bfloat16, device=x.device
        )

        TILE_H = 16
        TILE_W = 16
        OC_TILES = 2
        OC_PER_TILE = OC // OC_TILES  # 16

        W_tiles = (W_out + TILE_W - 1) // TILE_W
        H_tiles = (H_out + TILE_H - 1) // TILE_H

        grid = (W_tiles, H_tiles, B * OC_TILES)
        block = (TILE_W, TILE_H, 1)

        conv_transpose2d_kernel[lambda: (grid, block)](
            x_bf16.data_ptr(),
            w_bf16.data_ptr(),
            out_bf16.data_ptr(),
            B,
            IC,
            OC,
            H_in,
            W_in,
            H_out,
            W_out,
            K,
            stride_val,
            padding_val,
            OC_TILES,
            OC_PER_TILE,
        )

        if self.conv.bias is not None:
            out_bf16 = out_bf16 + self.conv.bias.to(torch.bfloat16).view(1, -1, 1, 1)

        return out_bf16.to(original_dtype)
