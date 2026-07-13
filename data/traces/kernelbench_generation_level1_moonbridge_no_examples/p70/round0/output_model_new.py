import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_transpose3d_tiled_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    IC: al.constexpr,
    OC: al.constexpr,
    N: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    OD: al.i32,
    OH: al.i32,
    OW: al.i32,
    TILE_D: al.constexpr,
    TILE_H: al.constexpr,
    TILE_W: al.constexpr,
    TILE_IC: al.constexpr,
    KD: al.constexpr,
    IN_D: al.constexpr,
    IN_H: al.constexpr,
    IN_W: al.constexpr,
    PAD: al.constexpr,
):
    KH = KD
    KW = KD
    IN_SPATIAL = IN_D * IN_H * IN_W
    IN_HW = IN_H * IN_W

    in_stride_n = IC * D * H * W
    in_stride_c = D * H * W
    in_stride_d = H * W
    in_stride_h = W
    in_layout = al.make_layout(
        (N, IC, D, H, W),
        (in_stride_n, in_stride_c, in_stride_d, in_stride_h, 1),
    )
    input_t = al.make_tensor(input_ptr, al.bf16, in_layout)

    w_stride_ic = OC * KD * KH * KW
    w_stride_oc = KD * KH * KW
    w_stride_kd = KH * KW
    w_stride_kh = KW
    w_layout = al.make_layout(
        (IC, OC, KD, KH, KW),
        (w_stride_ic, w_stride_oc, w_stride_kd, w_stride_kh, 1),
    )
    weight_t = al.make_tensor(weight_ptr, al.bf16, w_layout)

    out_stride_n = OC * OD * OH * OW
    out_stride_oc = OD * OH * OW
    out_stride_od = OH * OW
    out_stride_oh = OW
    out_layout = al.make_layout(
        (N, OC, OD, OH, OW),
        (out_stride_n, out_stride_oc, out_stride_od, out_stride_oh, 1),
    )
    output_t = al.make_tensor(output_ptr, al.bf16, out_layout)

    input_smem = al.make_shared((TILE_IC, IN_SPATIAL), al.bf16)

    b = al.block_id(0)
    spatial_tile = al.block_id(1)

    tiles_per_d = (OD + TILE_D - 1) // TILE_D
    tiles_per_h = (OH + TILE_H - 1) // TILE_H
    tiles_per_w = (OW + TILE_W - 1) // TILE_W

    tile_d = spatial_tile // (tiles_per_h * tiles_per_w)
    tile_rem = spatial_tile % (tiles_per_h * tiles_per_w)
    tile_h = tile_rem // tiles_per_w
    tile_w = tile_rem % tiles_per_w

    d_origin = tile_d * TILE_D
    h_origin = tile_h * TILE_H
    w_origin = tile_w * TILE_W

    tid = al.thread_id(0)
    td = tid // (TILE_H * TILE_W)
    th = (tid % (TILE_H * TILE_W)) // TILE_W
    tw = tid % TILE_W

    od = d_origin + td
    oh = h_origin + th
    ow = w_origin + tw

    valid_thread = (td < TILE_D) and (th < TILE_H) and (tw < TILE_W)
    valid_output = valid_thread and (od < OD) and (oh < OH) and (ow < OW)

    acc = al.make_local((OC,), al.f32)
    for ocl in al.range(OC):
        acc[ocl] = al.convert(0.0, al.f32)

    num_ic_tiles = (IC + TILE_IC - 1) // TILE_IC

    for t in al.range(num_ic_tiles):
        ic_start = t * TILE_IC
        ic_actual = TILE_IC
        ic_rem = IC - ic_start
        if ic_rem < TILE_IC:
            ic_actual = ic_rem

        smem_total = TILE_IC * IN_SPATIAL
        for i in al.range(tid, smem_total, al.block_dim(0)):
            ic_local = i // IN_SPATIAL
            spat = i % IN_SPATIAL
            in_d_idx = spat // IN_HW
            rem_hw = spat % IN_HW
            in_h_idx = rem_hw // IN_W
            in_w_idx = rem_hw % IN_W

            ic_global = ic_start + ic_local
            d_global = d_origin - PAD + in_d_idx
            h_global = h_origin - PAD + in_h_idx
            w_global = w_origin - PAD + in_w_idx

            is_in_bounds = (
                (ic_global < IC)
                and (d_global >= 0)
                and (d_global < D)
                and (h_global >= 0)
                and (h_global < H)
                and (w_global >= 0)
                and (w_global < W)
            )
            if is_in_bounds:
                input_smem[ic_local, spat] = input_t[
                    b, ic_global, d_global, h_global, w_global
                ]
            else:
                input_smem[ic_local, spat] = al.convert(0.0, al.bf16)

        al.syncthreads()

        if valid_output:
            # Precompute base spatial index (+ PAD offset) once per thread
            base_idx = PAD * (IN_HW + IN_W + 1) + td * IN_HW + th * IN_W + tw
            for ic_local in al.range(ic_actual):
                ic_global = ic_start + ic_local
                for kd in al.range(KD):
                    kd_off = kd * IN_HW
                    for kh in al.range(KH):
                        kh_off = kd_off + kh * IN_W
                        for kw in al.range(KW):
                            smem_idx = base_idx - (kh_off + kw)
                            in_val = al.convert(
                                input_smem[ic_local, smem_idx], al.f32
                            )
                            for ocl in al.range(OC):
                                w_val = al.convert(
                                    weight_t[ic_global, ocl, kd, kh, kw],
                                    al.f32,
                                )
                                acc[ocl] = acc[ocl] + in_val * w_val

        al.syncthreads()

    if valid_output:
        for ocl in al.range(OC):
            output_t[b, ocl, od, oh, ow] = al.convert(acc[ocl], al.bf16)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias_flag = bias

        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            self.stride == 1
            and self.padding == 0
            and self.dilation == 1
            and self.output_padding == 0
            and self.groups == 1
        ), (
            "Optimized kernel requires stride=1, padding=0, dilation=1, "
            "output_padding=0, groups=1"
        )

        w = self.conv_transpose3d.weight
        b = self.conv_transpose3d.bias if self.bias_flag else None

        N, IC, D, H, W = x.shape
        K = self.kernel_size

        OD = D + K - 1
        OH = H + K - 1
        OW = W + K - 1

        out_dtype = x.dtype

        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = w.to(torch.bfloat16).contiguous()
        out_bf16 = torch.empty(
            N, self.out_channels, OD, OH, OW,
            dtype=torch.bfloat16, device=x.device,
        )

        TILE_D = 8
        TILE_H = 8
        TILE_W = 4
        TILE_IC = 8
        PAD = K - 1
        IN_D = TILE_D + 2 * PAD
        IN_H = TILE_H + 2 * PAD
        IN_W = TILE_W + 2 * PAD

        tiles_d = (OD + TILE_D - 1) // TILE_D
        tiles_h = (OH + TILE_H - 1) // TILE_H
        tiles_w = (OW + TILE_W - 1) // TILE_W
        num_spatial_tiles = tiles_d * tiles_h * tiles_w

        grid_x = N
        grid_y = num_spatial_tiles
        block_threads = TILE_D * TILE_H * TILE_W

        conv_transpose3d_tiled_kernel[
            lambda: ((grid_x, grid_y, 1), (block_threads, 1, 1))
        ](
            x_bf16,
            w_bf16,
            out_bf16,
            IC,
            self.out_channels,
            N,
            D,
            H,
            W,
            OD,
            OH,
            OW,
            TILE_D,
            TILE_H,
            TILE_W,
            TILE_IC,
            K,
            IN_D,
            IN_H,
            IN_W,
            PAD,
        )

        out = out_bf16.to(out_dtype)

        if b is not None:
            out = out + b.view(1, -1, 1, 1, 1)

        return out
