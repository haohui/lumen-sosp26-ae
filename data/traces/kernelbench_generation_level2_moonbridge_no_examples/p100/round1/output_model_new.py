import torch
import torch.nn as nn
import struct
import avelang
import avelang.language as al


@avelang.jit
def conv_transpose3d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    # Total element counts for flat buffers
    n_in: al.i32,
    n_w: al.i32,
    n_out: al.i32,
    # Dimensions for index computation
    IC: al.i32, OC: al.i32,
    D: al.i32, H: al.i32, W: al.i32,
    OD: al.i32, OH: al.i32, OW: al.i32,
    KD: al.i32, KH: al.i32, KW: al.i32,
    stride: al.i32,
    padding: al.i32,
    min_value_i32: al.i32,
    divisor_i32: al.i32,
    # Precomputed strides for flat indexing
    s_out_B: al.i32, s_out_OC: al.i32, s_out_OD: al.i32, s_out_OH: al.i32, s_out_OW: al.i32,
    s_in_B: al.i32, s_in_IC: al.i32, s_in_D: al.i32, s_in_H: al.i32, s_in_W: al.i32,
    s_w_IC: al.i32, s_w_OC: al.i32, s_w_KD: al.i32, s_w_KH: al.i32, s_w_KW: al.i32,
    # Tile dims and derived values
    TILE_W: al.i32, TILE_H: al.i32, TILE_D: al.i32,
    OD_TILES: al.i32,
):
    # Flat 1D layouts
    one = al.convert(1, al.i32)
    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((n_in,), (one,)))
    w = al.make_tensor(w_ptr, al.bf16, al.make_layout((n_w,), (one,)))
    bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((OC,), (one,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((n_out,), (one,)))

    # Thread/block ids
    tid_w = al.thread_id(0)
    tid_h = al.thread_id(1)
    tid_d = al.thread_id(2)
    bid_x = al.block_id(0)
    bid_y = al.block_id(1)
    bid_z = al.block_id(2)

    # Compute this thread's output spatial position
    ow = bid_x * TILE_W + tid_w
    oh = bid_y * TILE_H + tid_h

    # Decode bid_z into (b, oc, od_tile_idx)
    b_idx = bid_z // (OC * OD_TILES)
    rem = bid_z % (OC * OD_TILES)
    oc = rem // OD_TILES
    od_tile = rem % OD_TILES
    od = od_tile * TILE_D + tid_d

    # Bounds check
    if ow < OW and oh < OH and od < OD:
        acc = al.convert(0.0, al.f32)
        min_f32 = al.bitcast(min_value_i32, al.f32)
        div_f32 = al.bitcast(divisor_i32, al.f32)

        for ic in al.range(IC):
            for kd in al.range(KD):
                id_val = od + padding - kd
                if id_val >= 0:
                    id_q = id_val // stride
                    id_chk = id_q * stride
                    if id_chk == id_val:
                        if id_q < D:
                            for kh in al.range(KH):
                                ih_val = oh + padding - kh
                                if ih_val >= 0:
                                    ih_q = ih_val // stride
                                    ih_chk = ih_q * stride
                                    if ih_chk == ih_val:
                                        if ih_q < H:
                                            for kw in al.range(KW):
                                                iw_val = ow + padding - kw
                                                if iw_val >= 0:
                                                    iw_q = iw_val // stride
                                                    iw_chk = iw_q * stride
                                                    if iw_chk == iw_val:
                                                        if iw_q < W:
                                                            xi = b_idx * s_in_B + ic * s_in_IC + id_q * s_in_D + ih_q * s_in_H + iw_q * s_in_W
                                                            wi = ic * s_w_IC + oc * s_w_OC + kd * s_w_KD + kh * s_w_KH + kw * s_w_KW
                                                            x_val = al.convert(x[xi], al.f32)
                                                            w_val = al.convert(w[wi], al.f32)
                                                            acc = acc + x_val * w_val

        acc = acc + al.convert(bias[oc], al.f32)
        if acc < min_f32:
            acc = min_f32
        acc = acc / div_f32
        oi = b_idx * s_out_B + oc * s_out_OC + od * s_out_OD + oh * s_out_OH + ow * s_out_OW
        out[oi] = al.convert(acc, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        ref_conv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
        )
        self.weight = nn.Parameter(ref_conv.weight.detach().clone())
        self.bias = nn.Parameter(ref_conv.bias.detach().clone())
        self.stride = stride
        self.padding = padding
        self.min_value = min_value
        self.divisor = divisor

    def forward(self, x):
        x_bf16 = x.contiguous().to(torch.bfloat16)
        w_bf16 = self.weight.contiguous().to(torch.bfloat16)
        bias_bf16 = self.bias.contiguous().to(torch.bfloat16)

        B, IC, D_in, H_in, W_in = x_bf16.shape
        _w_IC, OC, KD, KH, KW = w_bf16.shape

        OD = (D_in - 1) * self.stride - 2 * self.padding + KD
        OH = (H_in - 1) * self.stride - 2 * self.padding + KH
        OW = (W_in - 1) * self.stride - 2 * self.padding + KW

        out_bf16 = torch.empty(B, OC, OD, OH, OW, dtype=torch.bfloat16, device=x.device)

        # Total element counts
        n_in = B * IC * D_in * H_in * W_in
        n_w = IC * OC * KD * KH * KW
        n_out = B * OC * OD * OH * OW

        # Precompute strides
        s_out_OW = 1
        s_out_OH = OW
        s_out_OD = OH * OW
        s_out_OC = OD * OH * OW
        s_out_B = OC * OD * OH * OW

        s_in_W = 1
        s_in_H = W_in
        s_in_D = H_in * W_in
        s_in_IC = D_in * H_in * W_in
        s_in_B = IC * D_in * H_in * W_in

        s_w_KW = 1
        s_w_KH = KW
        s_w_KD = KH * KW
        s_w_OC = KD * KH * KW
        s_w_IC = OC * KD * KH * KW

        # Tile and grid dimensions
        TILE_W = 8
        TILE_H = 8
        TILE_D = 4
        BLOCK_W = TILE_W
        BLOCK_H = TILE_H
        BLOCK_D = TILE_D

        OW_TILES = (OW + TILE_W - 1) // TILE_W
        OH_TILES = (OH + TILE_H - 1) // TILE_H
        OD_TILES = (OD + TILE_D - 1) // TILE_D

        grid_x = OW_TILES
        grid_y = OH_TILES
        grid_z = B * OC * OD_TILES

        min_bits = struct.unpack('<i', struct.pack('<f', float(self.min_value)))[0]
        div_bits = struct.unpack('<i', struct.pack('<f', float(self.divisor)))[0]

        conv_transpose3d_kernel[lambda: ((grid_x, grid_y, grid_z), (BLOCK_W, BLOCK_H, BLOCK_D))](
            x_bf16, w_bf16, bias_bf16, out_bf16,
            n_in, n_w, n_out,
            IC, OC, D_in, H_in, W_in, OD, OH, OW, KD, KH, KW,
            self.stride, self.padding,
            min_bits, div_bits,
            s_out_B, s_out_OC, s_out_OD, s_out_OH, s_out_OW,
            s_in_B, s_in_IC, s_in_D, s_in_H, s_in_W,
            s_w_IC, s_w_OC, s_w_KD, s_w_KH, s_w_KW,
            TILE_W, TILE_H, TILE_D,
            OD_TILES,
        )

        return out_bf16


# Helpers for harness compatibility
batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]
