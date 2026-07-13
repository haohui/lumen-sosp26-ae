import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_transpose2d_bias_tanh_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    conv_bias_ptr: al.Pointer(al.bf16),
    ext_bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    KH: al.i32,
    KW: al.i32,
    stride: al.i32,
    padding: al.i32,
    num_h_tiles: al.i32,
    num_w_tiles: al.i32,
    num_oc_tiles: al.i32,
    x_sn: al.i32,
    x_sc: al.i32,
    x_sh: al.i32,
    w_sic: al.i32,
    w_soc: al.i32,
    w_skh: al.i32,
    out_sn: al.i32,
    out_soc: al.i32,
    out_soh: al.i32,
):
    # Create tensor views
    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((N, C_in, H, W), (x_sn, x_sc, x_sh, 1)))
    w = al.make_tensor(w_ptr, al.bf16, al.make_layout((C_in, C_out, KH, KW), (w_sic, w_soc, w_skh, 1)))
    conv_bias = al.make_tensor(conv_bias_ptr, al.bf16, al.make_layout((C_out,), (1,)))
    ext_bias = al.make_tensor(ext_bias_ptr, al.bf16, al.make_layout((C_out,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((N, C_out, H_out, W_out), (out_sn, out_soc, out_soh, 1)))

    # Shared memory: weight (4 oc x 64 ic x 4 x 4) = 8KB, input (64 ic x 12 x 12) = 18.4KB
    w_shared = al.make_shared((4, 64, 4, 4), al.bf16)
    x_shared = al.make_shared((64, 12, 12), al.bf16)

    TILE_H = 16
    TILE_W = 16
    TILE_OC = 4
    IN_H = 12
    IN_W = 12
    BLOCK_SIZE = 256

    tid = al.thread_id(0)
    local_h = tid // TILE_W
    local_w = tid % TILE_W

    bid = al.block_id(0)
    oc_tile = bid % num_oc_tiles
    rest = bid // num_oc_tiles
    w_tile = rest % num_w_tiles
    rest = rest // num_w_tiles
    h_tile = rest % num_h_tiles
    n = rest // num_h_tiles

    oh = h_tile * TILE_H + local_h
    ow = w_tile * TILE_W + local_w

    oh_start = h_tile * TILE_H
    ow_start = w_tile * TILE_W

    # Input region bounds for this output tile
    ih_min = (oh_start + padding - 3) // stride
    if ih_min < 0:
        ih_min = 0
    ih_max = (oh_start + TILE_H - 1 + padding) // stride + 1
    if ih_max > H:
        ih_max = H
    ih_load_start = ih_min

    iw_min = (ow_start + padding - 3) // stride
    if iw_min < 0:
        iw_min = 0
    iw_max = (ow_start + TILE_W - 1 + padding) // stride + 1
    if iw_max > W:
        iw_max = W
    iw_load_start = iw_min

    # Cooperative load of input tile into shared memory
    x_total = 64 * IN_H * IN_W
    for idx in al.range(tid, x_total, BLOCK_SIZE):
        lic = idx // (IN_H * IN_W)
        r = idx % (IN_H * IN_W)
        lih = r // IN_W
        liw = r % IN_W
        g_ih = ih_load_start + lih
        g_iw = iw_load_start + liw
        if lic < C_in and g_ih < H and g_iw < W:
            x_shared[lic, lih, liw] = x[n, lic, g_ih, g_iw]

    # Cooperative load of weight tile
    w_total = TILE_OC * 64 * 4 * 4
    for idx in al.range(tid, w_total, BLOCK_SIZE):
        loc = idx // (64 * 4 * 4)
        r = idx % (64 * 4 * 4)
        lic = r // (4 * 4)
        r = r % (4 * 4)
        lkh = r // 4
        lkw = r % 4
        g_oc = oc_tile * TILE_OC + loc
        if g_oc < C_out and lic < C_in:
            w_shared[loc, lic, lkh, lkw] = w[lic, g_oc, lkh, lkw]

    al.syncthreads()

    if oh < H_out and ow < W_out:
        oh_div2 = oh // 2
        ow_div2 = ow // 2
        oh_plus1_div2 = (oh + 1) // 2
        ow_plus1_div2 = (ow + 1) // 2

        a0 = al.convert(0.0, al.f32)
        a1 = al.convert(0.0, al.f32)
        a2 = al.convert(0.0, al.f32)
        a3 = al.convert(0.0, al.f32)

        oh_par = oh % 2
        ow_par = ow % 2

        # Parity-unrolled compute: exactly 4 valid kernel positions per output pixel
        if oh_par == 0:
            if ow_par == 0:
                # oh even, ow even: (kh,kw) in {(1,1),(1,3),(3,1),(3,3)}
                for ic in al.range(C_in):
                    ih = oh_div2; iw = ow_div2
                    if ih < H and iw < W:
                        lih = ih - ih_load_start; liw = iw - iw_load_start
                        xv = al.convert(x_shared[ic, lih, liw], al.f32)
                        a0 += xv * al.convert(w_shared[0, ic, 1, 1], al.f32)
                        a1 += xv * al.convert(w_shared[1, ic, 1, 1], al.f32)
                        a2 += xv * al.convert(w_shared[2, ic, 1, 1], al.f32)
                        a3 += xv * al.convert(w_shared[3, ic, 1, 1], al.f32)
                    ih = oh_div2; iw = ow_div2 - 1
                    if ih < H and iw >= 0:
                        lih = ih - ih_load_start; liw = iw - iw_load_start
                        xv = al.convert(x_shared[ic, lih, liw], al.f32)
                        a0 += xv * al.convert(w_shared[0, ic, 1, 3], al.f32)
                        a1 += xv * al.convert(w_shared[1, ic, 1, 3], al.f32)
                        a2 += xv * al.convert(w_shared[2, ic, 1, 3], al.f32)
                        a3 += xv * al.convert(w_shared[3, ic, 1, 3], al.f32)
                    ih = oh_div2 - 1; iw = ow_div2
                    if ih >= 0 and iw < W:
                        lih = ih - ih_load_start; liw = iw - iw_load_start
                        xv = al.convert(x_shared[ic, lih, liw], al.f32)
                        a0 += xv * al.convert(w_shared[0, ic, 3, 1], al.f32)
                        a1 += xv * al.convert(w_shared[1, ic, 3, 1], al.f32)
                        a2 += xv * al.convert(w_shared[2, ic, 3, 1], al.f32)
                        a3 += xv * al.convert(w_shared[3, ic, 3, 1], al.f32)
                    ih = oh_div2 - 1; iw = ow_div2 - 1
                    if ih >= 0 and iw >= 0:
                        lih = ih - ih_load_start; liw = iw - iw_load_start
                        xv = al.convert(x_shared[ic, lih, liw], al.f32)
                        a0 += xv * al.convert(w_shared[0, ic, 3, 3], al.f32)
                        a1 += xv * al.convert(w_shared[1, ic, 3, 3], al.f32)
                        a2 += xv * al.convert(w_shared[2, ic, 3, 3], al.f32)
                        a3 += xv * al.convert(w_shared[3, ic, 3, 3], al.f32)
            else:
                # oh even, ow odd: (kh,kw) in {(1,0),(1,2),(3,0),(3,2)}
                for ic in al.range(C_in):
                    ih = oh_div2; iw = ow_plus1_div2
                    if ih < H and iw < W:
                        lih = ih - ih_load_start; liw = iw - iw_load_start
                        xv = al.convert(x_shared[ic, lih, liw], al.f32)
                        a0 += xv * al.convert(w_shared[0, ic, 1, 0], al.f32)
                        a1 += xv * al.convert(w_shared[1, ic, 1, 0], al.f32)
                        a2 += xv * al.convert(w_shared[2, ic, 1, 0], al.f32)
                        a3 += xv * al.convert(w_shared[3, ic, 1, 0], al.f32)
                    ih = oh_div2; iw = ow_div2
                    if ih < H and iw < W:
                        lih = ih - ih_load_start; liw = iw - iw_load_start
                        xv = al.convert(x_shared[ic, lih, liw], al.f32)
                        a0 += xv * al.convert(w_shared[0, ic, 1, 2], al.f32)
                        a1 += xv * al.convert(w_shared[1, ic, 1, 2], al.f32)
                        a2 += xv * al.convert(w_shared[2, ic, 1, 2], al.f32)
                        a3 += xv * al.convert(w_shared[3, ic, 1, 2], al.f32)
                    ih = oh_div2 - 1; iw = ow_plus1_div2
                    if ih >= 0 and iw < W:
                        lih = ih - ih_load_start; liw = iw - iw_load_start
                        xv = al.convert(x_shared[ic, lih, liw], al.f32)
                        a0 += xv * al.convert(w_shared[0, ic, 3, 0], al.f32)
                        a1 += xv * al.convert(w_shared[1, ic, 3, 0], al.f32)
                        a2 += xv * al.convert(w_shared[2, ic, 3, 0], al.f32)
                        a3 += xv * al.convert(w_shared[3, ic, 3, 0], al.f32)
                    ih = oh_div2 - 1; iw = ow_div2
                    if ih >= 0 and iw < W:
                        lih = ih - ih_load_start; liw = iw - iw_load_start
                        xv = al.convert(x_shared[ic, lih, liw], al.f32)
                        a0 += xv * al.convert(w_shared[0, ic, 3, 2], al.f32)
                        a1 += xv * al.convert(w_shared[1, ic, 3, 2], al.f32)
                        a2 += xv * al.convert(w_shared[2, ic, 3, 2], al.f32)
                        a3 += xv * al.convert(w_shared[3, ic, 3, 2], al.f32)
        else:
            if ow_par == 0:
                # oh odd, ow even: (kh,kw) in {(0,1),(0,3),(2,1),(2,3)}
                for ic in al.range(C_in):
                    ih = oh_plus1_div2; iw = ow_div2
                    if ih < H and iw < W:
                        lih = ih - ih_load_start; liw = iw - iw_load_start
                        xv = al.convert(x_shared[ic, lih, liw], al.f32)
                        a0 += xv * al.convert(w_shared[0, ic, 0, 1], al.f32)
                        a1 += xv * al.convert(w_shared[1, ic, 0, 1], al.f32)
                        a2 += xv * al.convert(w_shared[2, ic, 0, 1], al.f32)
                        a3 += xv * al.convert(w_shared[3, ic, 0, 1], al.f32)
                    ih = oh_plus1_div2; iw = ow_div2 - 1
                    if ih < H and iw >= 0:
                        lih = ih - ih_load_start; liw = iw - iw_load_start
                        xv = al.convert(x_shared[ic, lih, liw], al.f32)
                        a0 += xv * al.convert(w_shared[0, ic, 0, 3], al.f32)
                        a1 += xv * al.convert(w_shared[1, ic, 0, 3], al.f32)
                        a2 += xv * al.convert(w_shared[2, ic, 0, 3], al.f32)
                        a3 += xv * al.convert(w_shared[3, ic, 0, 3], al.f32)
                    ih = oh_div2; iw = ow_div2
                    if ih < H and iw < W:
                        lih = ih - ih_load_start; liw = iw - iw_load_start
                        xv = al.convert(x_shared[ic, lih, liw], al.f32)
                        a0 += xv * al.convert(w_shared[0, ic, 2, 1], al.f32)
                        a1 += xv * al.convert(w_shared[1, ic, 2, 1], al.f32)
                        a2 += xv * al.convert(w_shared[2, ic, 2, 1], al.f32)
                        a3 += xv * al.convert(w_shared[3, ic, 2, 1], al.f32)
                    ih = oh_div2; iw = ow_div2 - 1
                    if ih < H and iw >= 0:
                        lih = ih - ih_load_start; liw = iw - iw_load_start
                        xv = al.convert(x_shared[ic, lih, liw], al.f32)
                        a0 += xv * al.convert(w_shared[0, ic, 2, 3], al.f32)
                        a1 += xv * al.convert(w_shared[1, ic, 2, 3], al.f32)
                        a2 += xv * al.convert(w_shared[2, ic, 2, 3], al.f32)
                        a3 += xv * al.convert(w_shared[3, ic, 2, 3], al.f32)
            else:
                # oh odd, ow odd: (kh,kw) in {(0,0),(0,2),(2,0),(2,2)}
                for ic in al.range(C_in):
                    ih = oh_plus1_div2; iw = ow_plus1_div2
                    if ih < H and iw < W:
                        lih = ih - ih_load_start; liw = iw - iw_load_start
                        xv = al.convert(x_shared[ic, lih, liw], al.f32)
                        a0 += xv * al.convert(w_shared[0, ic, 0, 0], al.f32)
                        a1 += xv * al.convert(w_shared[1, ic, 0, 0], al.f32)
                        a2 += xv * al.convert(w_shared[2, ic, 0, 0], al.f32)
                        a3 += xv * al.convert(w_shared[3, ic, 0, 0], al.f32)
                    ih = oh_plus1_div2; iw = ow_div2
                    if ih < H and iw < W:
                        lih = ih - ih_load_start; liw = iw - iw_load_start
                        xv = al.convert(x_shared[ic, lih, liw], al.f32)
                        a0 += xv * al.convert(w_shared[0, ic, 0, 2], al.f32)
                        a1 += xv * al.convert(w_shared[1, ic, 0, 2], al.f32)
                        a2 += xv * al.convert(w_shared[2, ic, 0, 2], al.f32)
                        a3 += xv * al.convert(w_shared[3, ic, 0, 2], al.f32)
                    ih = oh_div2; iw = ow_plus1_div2
                    if ih < H and iw < W:
                        lih = ih - ih_load_start; liw = iw - iw_load_start
                        xv = al.convert(x_shared[ic, lih, liw], al.f32)
                        a0 += xv * al.convert(w_shared[0, ic, 2, 0], al.f32)
                        a1 += xv * al.convert(w_shared[1, ic, 2, 0], al.f32)
                        a2 += xv * al.convert(w_shared[2, ic, 2, 0], al.f32)
                        a3 += xv * al.convert(w_shared[3, ic, 2, 0], al.f32)
                    ih = oh_div2; iw = ow_div2
                    if ih < H and iw < W:
                        lih = ih - ih_load_start; liw = iw - iw_load_start
                        xv = al.convert(x_shared[ic, lih, liw], al.f32)
                        a0 += xv * al.convert(w_shared[0, ic, 2, 2], al.f32)
                        a1 += xv * al.convert(w_shared[1, ic, 2, 2], al.f32)
                        a2 += xv * al.convert(w_shared[2, ic, 2, 2], al.f32)
                        a3 += xv * al.convert(w_shared[3, ic, 2, 2], al.f32)

        # Bias + tanh + store
        go0 = oc_tile * 4 + 0
        if go0 < C_out:
            cb = al.convert(conv_bias[go0], al.f32)
            eb = al.convert(ext_bias[go0], al.f32)
            out[n, go0, oh, ow] = al.convert(al.tanh(a0 + cb - eb), al.bf16)
        go1 = oc_tile * 4 + 1
        if go1 < C_out:
            cb = al.convert(conv_bias[go1], al.f32)
            eb = al.convert(ext_bias[go1], al.f32)
            out[n, go1, oh, ow] = al.convert(al.tanh(a1 + cb - eb), al.bf16)
        go2 = oc_tile * 4 + 2
        if go2 < C_out:
            cb = al.convert(conv_bias[go2], al.f32)
            eb = al.convert(ext_bias[go2], al.f32)
            out[n, go2, oh, ow] = al.convert(al.tanh(a2 + cb - eb), al.bf16)
        go3 = oc_tile * 4 + 3
        if go3 < C_out:
            cb = al.convert(conv_bias[go3], al.f32)
            eb = al.convert(ext_bias[go3], al.f32)
            out[n, go3, oh, ow] = al.convert(al.tanh(a3 + cb - eb), al.bf16)


def _compute_output_size(H, W, stride, padding, kernel_size, output_padding, dilation=1):
    H_out = (H - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    W_out = (W - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    return H_out, W_out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        N, C_in, H, W = x.shape
        C_out = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        stride = self.stride
        padding = self.padding
        output_padding = self.output_padding

        H_out, W_out = _compute_output_size(H, W, stride, padding, KH, output_padding)

        weight = self.conv_transpose.weight.data
        conv_bias = self.conv_transpose.bias.data
        ext_bias = self.bias.data.view(-1)

        out = torch.empty(N, C_out, H_out, W_out, dtype=torch.bfloat16, device=x.device)

        x_sn = C_in * H * W
        x_sc = H * W
        x_sh = W

        w_sic = C_out * KH * KW
        w_soc = KH * KW
        w_skh = KW

        out_sn = C_out * H_out * W_out
        out_soc = H_out * W_out
        out_soh = W_out

        TILE_H = 16
        TILE_W = 16
        TILE_OC = 4

        num_h_tiles = (H_out + TILE_H - 1) // TILE_H
        num_w_tiles = (W_out + TILE_W - 1) // TILE_W
        num_oc_tiles = (C_out + TILE_OC - 1) // TILE_OC

        grid_x = N * num_h_tiles * num_w_tiles * num_oc_tiles

        conv_transpose2d_bias_tanh_kernel[lambda: ((grid_x, 1, 1), (TILE_H * TILE_W, 1, 1))](
            x, weight, conv_bias, ext_bias, out,
            N, C_in, C_out, H, W, H_out, W_out, KH, KW, stride, padding,
            num_h_tiles, num_w_tiles, num_oc_tiles,
            x_sn, x_sc, x_sh,
            w_sic, w_soc, w_skh,
            out_sn, out_soc, out_soh,
        )

        return out
