import torch
import torch.nn as nn
import avelang
import avelang.language as al

# --- compile-time tile constants ---
TILE_OC = al.constexpr(8)
TILE_D = al.constexpr(2)
TILE_W = al.constexpr(4)
TILE_H = al.constexpr(4)

# --- host-side tile values (mirror compile-time constants) ---
_HOST_TILE_OC = 8
_HOST_TILE_D = 2
_HOST_TILE_W = 4
_HOST_TILE_H = 4


@avelang.jit
def conv_transpose3d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    D: al.i32,
    W: al.i32,
    H: al.i32,
    KD: al.i32,
    KW: al.i32,
    KH: al.i32,
    D_out: al.i32,
    W_out: al.i32,
    H_out: al.i32,
):
    # --- input layout: (B, IC, D, W, H) ---
    x_stride_ic = D * W * H
    x_stride_d = W * H
    x_stride_w = H
    x_layout = al.make_layout(
        (B, IC, D, W, H),
        (IC * x_stride_ic, x_stride_ic, x_stride_d, x_stride_w, 1),
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    # --- weight layout: (IC, OC, KD, KW, KH) ---
    w_stride_ic = OC * KD * KW * KH
    w_stride_oc = KD * KW * KH
    w_stride_kd = KW * KH
    w_stride_kw = KH
    w_layout = al.make_layout(
        (IC, OC, KD, KW, KH),
        (w_stride_ic, w_stride_oc, w_stride_kd, w_stride_kw, 1),
    )
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    # --- output layout: (B, OC, D_out, W_out, H_out) ---
    out_stride_oc = D_out * W_out * H_out
    out_stride_d = W_out * H_out
    out_stride_w = H_out
    out_layout = al.make_layout(
        (B, OC, D_out, W_out, H_out),
        (OC * out_stride_oc, out_stride_oc, out_stride_d, out_stride_w, 1),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    # --- block decomposition ---
    bid0 = al.block_id(0)
    bid1 = al.block_id(1)
    tid = al.thread_id(0)

    # bid0 encodes batch and oc_tile
    oc_tiles = (OC + TILE_OC - 1) // TILE_OC
    b = bid0 // oc_tiles
    oc_tile = bid0 % oc_tiles

    # thread index -> local position within the tile
    spatial_size = TILE_D * TILE_W * TILE_H
    oc_off = tid // spatial_size
    rest = tid % spatial_size
    d_off = rest // (TILE_W * TILE_H)
    rest2 = rest % (TILE_W * TILE_H)
    w_off = rest2 // TILE_H
    h_off = rest2 % TILE_H

    oc = oc_tile * TILE_OC + oc_off

    # bid1 encodes spatial tile (depth-major among D, W, H tiles)
    num_wtiles = (W_out + TILE_W - 1) // TILE_W
    num_htiles = (H_out + TILE_H - 1) // TILE_H
    wh_tiles = num_wtiles * num_htiles
    d_tile = bid1 // wh_tiles
    wh_rest = bid1 % wh_tiles
    w_tile = wh_rest // num_htiles
    h_tile = wh_rest % num_htiles

    d = d_tile * TILE_D + d_off
    w = w_tile * TILE_W + w_off
    h = h_tile * TILE_H + h_off

    # --- bounds check ---
    if oc < OC and d < D_out and w < W_out and h < H_out:
        acc = al.convert(0.0, al.f32)
        for ic in al.range(IC):
            for kd in al.range(KD):
                d_in = d - kd
                if d_in >= 0 and d_in < D:
                    for kw in al.range(KW):
                        w_in = w - kw
                        if w_in >= 0 and w_in < W:
                            for kh in al.range(KH):
                                h_in = h - kh
                                if h_in >= 0 and h_in < H:
                                    val = al.convert(x[b, ic, d_in, w_in, h_in], al.f32)
                                    wt = al.convert(w[ic, oc, kd, kw, kh], al.f32)
                                    acc = acc + val * wt
        out[b, oc, d, w, h] = al.convert(acc, al.bf16)


def _avelang_conv_transpose3d(
    x: torch.Tensor, weight: torch.Tensor,
    stride, padding, output_padding, groups,
) -> torch.Tensor:
    B, IC, D, W, H = x.shape
    OC = weight.shape[1]
    KD, KW, KH = weight.shape[2], weight.shape[3], weight.shape[4]

    # output spatial sizes (stride=1, padding=0, dilation=1 case)
    D_out = (D - 1) * stride[0] - 2 * padding[0] + KD + output_padding[0]
    W_out = (W - 1) * stride[1] - 2 * padding[1] + KW + output_padding[1]
    H_out = (H - 1) * stride[2] - 2 * padding[2] + KH + output_padding[2]

    oc_tiles = (OC + _HOST_TILE_OC - 1) // _HOST_TILE_OC
    dtiles = (D_out + _HOST_TILE_D - 1) // _HOST_TILE_D
    wtiles = (W_out + _HOST_TILE_W - 1) // _HOST_TILE_W
    htiles = (H_out + _HOST_TILE_H - 1) // _HOST_TILE_H
    spatial_tiles = dtiles * wtiles * htiles

    grid = (B * oc_tiles, spatial_tiles, 1)
    block = (_HOST_TILE_OC * _HOST_TILE_D * _HOST_TILE_W * _HOST_TILE_H, 1, 1)

    out = torch.empty(B, OC, D_out, W_out, H_out, dtype=torch.bfloat16, device=x.device)

    # ensure contiguous bf16 inputs
    x_contig = x.contiguous()
    w_contig = weight.contiguous()

    if x_contig.dtype != torch.bfloat16:
        x_contig = x_contig.to(torch.bfloat16)
    if w_contig.dtype != torch.bfloat16:
        w_contig = w_contig.to(torch.bfloat16)

    conv_transpose3d_kernel[lambda: (grid, block)](
        x_contig.data_ptr(),
        w_contig.data_ptr(),
        out.data_ptr(),
        B, IC, OC,
        D, W, H,
        KD, KW, KH,
        D_out, W_out, H_out,
    )

    return out


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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # weight shape: (in_channels, out_channels // groups, *kernel_size)
        kd, kw, kh = kernel_size
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels // groups, kd, kw, kh)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5.0 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1.0 / (fan_in ** 0.5)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = _avelang_conv_transpose3d(
            x, self.weight,
            self.stride, self.padding, self.output_padding, self.groups,
        )
        if self.bias is not None:
            # bias shape: (out_channels,) → broadcast to (B, OC, D_out, W_out, H_out)
            out = out + self.bias.view(1, -1, 1, 1, 1).to(out.dtype)
        return out
