import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_H = 4
TILE_W = 64
TILE_OC = 8
THREADS = 256
SHM_W_BF16 = 10752
LOADS_PER_THREAD = 42


@avelang.jit
def _conv_transpose2d_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    IC: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
    KH: al.i32,
    KW: al.i32,
    OH: al.i32,
    OW: al.i32,
    stride_h: al.i32,
    stride_w: al.i32,
    pad_h: al.i32,
    pad_w: al.i32,
    dil_h: al.i32,
    dil_w: al.i32,
    has_bias: al.i32,
    groups: al.i32,
    oc_tiles: al.i32,
):
    tid = al.thread_id(0)
    block_ow = al.block_id(0)
    block_oh = al.block_id(1)
    block_z = al.block_id(2)

    groups_oc_tiles = groups * oc_tiles
    n = block_z // groups_oc_tiles
    rem = block_z - n * groups_oc_tiles
    g = rem // oc_tiles
    oc_tile = rem - g * oc_tiles
    oc_base = oc_tile * TILE_OC

    oc_per_group = OC // groups
    ic_per_group = IC // groups

    oh = block_oh * TILE_H + tid // TILE_W
    ow = block_ow * TILE_W + tid % TILE_W

    if n >= N or g >= groups:
        return

    ic_group_off = g * ic_per_group
    oc_group_off = g * oc_per_group

    x_n_stride = IC * H * W
    x_ic_stride = H * W
    x_h_stride = W

    w_ic_stride = oc_per_group * KH * KW
    w_oc_stride = KH * KW
    w_kh_stride = KW

    o_n_stride = OC * OH * OW
    o_oc_stride = OH * OW
    o_oh_stride = OW

    # Phase 1: all threads load weight
    shm_w = al.make_shared((SHM_W_BF16,), al.bf16)

    w_size = IC * oc_per_group * KH * KW
    w = al.make_tensor(w_ptr, al.bf16, al.make_layout((w_size,), (1,)))

    for r in al.range(LOADS_PER_THREAD):
        bf16_idx = tid * LOADS_PER_THREAD + r
        if bf16_idx >= SHM_W_BF16:
            break
        tmp = bf16_idx // KW
        kw_i = bf16_idx - tmp * KW
        tmp2 = tmp // KH
        kh_i = tmp - tmp2 * KH
        oc_i = tmp2 % TILE_OC
        ic_i = tmp2 // TILE_OC
        w_gidx = (
            (ic_group_off + ic_i) * w_ic_stride
            + (oc_base + oc_i) * w_oc_stride
            + kh_i * w_kh_stride
            + kw_i
        )
        shm_w[bf16_idx] = w[w_gidx]

    al.syncthreads()

    if oh >= OH or ow >= OW:
        return

    # Phase 2: compute
    x_size = N * IC * H * W
    o_size = N * OC * OH * OW

    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((x_size,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((o_size,), (1,)))
    bias_t = al.make_tensor(b_ptr, al.bf16, al.make_layout((OC,), (1,)))

    x_n_base = n * x_n_stride
    o_n_base = n * o_n_stride

    shm_oc_stride = KH * KW
    shm_ic_stride = TILE_OC * shm_oc_stride

    zero = al.convert(0.0, al.f32)

    # Precompute valid kernel bounds for this (oh, ow)
    # kh valid range: max(0, oh+H-KH), min(KH, oh+1) with stride=1, pad=0, dil=1
    # General: kh where 0 <= oh+pad-kh*dil < H*stride and divisible by stride

    for t_oc in al.range(TILE_OC):
        oc_local = oc_base + t_oc
        if oc_local >= oc_per_group:
            break

        oc = oc_group_off + oc_local
        acc = zero

        shm_oc_base = t_oc * shm_oc_stride

        for ic_local in al.range(ic_per_group):
            ic = ic_group_off + ic_local
            shm_ic_base = ic_local * shm_ic_stride + shm_oc_base
            x_ic_base = x_n_base + ic * x_ic_stride

            for kh in al.range(KH):
                raw_h = oh + pad_h - kh * dil_h
                # Equivalent to: raw_h >= 0, raw_h % stride == 0, raw_h/stride < H
                if raw_h >= 0:
                    ih = raw_h // stride_h
                    if ih < H:
                        if ih * stride_h == raw_h:
                            x_ih_base = x_ic_base + ih * x_h_stride
                            shm_kh_base = shm_ic_base + kh * KW

                            for kw in al.range(KW):
                                raw_w = ow + pad_w - kw * dil_w
                                if raw_w >= 0:
                                    iw = raw_w // stride_w
                                    if iw < W:
                                        if iw * stride_w == raw_w:
                                            x_idx = x_ih_base + iw
                                            shm_w_idx = shm_kh_base + kw
                                            x_val = al.convert(x[x_idx], al.f32)
                                            w_val = al.convert(shm_w[shm_w_idx], al.f32)
                                            acc = acc + x_val * w_val

        if has_bias != 0:
            acc = acc + al.convert(bias_t[oc], al.f32)

        o_idx = o_n_base + oc * o_oc_stride + oh * o_oh_stride + ow
        out[o_idx] = al.convert(acc, al.bf16)


def _maybe_tuple(v):
    if isinstance(v, int):
        return (v, v)
    return tuple(v)


def _conv_transpose2d_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = x.contiguous().to(device=x.device, dtype=torch.bfloat16)
    w_bf16 = weight.contiguous().to(device=x.device, dtype=torch.bfloat16)

    N, IC, H, W_in = x_bf16.shape
    w_IC, w_OC_per_group, KH, KW = w_bf16.shape
    OC = w_OC_per_group * groups

    stride_h, stride_w = _maybe_tuple(stride)
    pad_h, pad_w = _maybe_tuple(padding)
    dil_h, dil_w = _maybe_tuple(dilation)
    opad_h, opad_w = _maybe_tuple(output_padding)

    OH = (H - 1) * stride_h - 2 * pad_h + dil_h * (KH - 1) + opad_h + 1
    OW = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (KW - 1) + opad_w + 1

    ic_per_group = IC // groups
    shm_bf16_needed = ic_per_group * TILE_OC * KH * KW
    if shm_bf16_needed > SHM_W_BF16:
        raise ValueError(
            f"Weight tile needs {shm_bf16_needed} bf16 but SHM_W_BF16={SHM_W_BF16}"
        )

    if bias is not None:
        b_bf16 = bias.contiguous().to(device=x.device, dtype=torch.bfloat16)
        has_bias = 1
    else:
        b_bf16 = torch.zeros((OC,), device=x.device, dtype=torch.bfloat16)
        has_bias = 0

    out = torch.empty((N, OC, OH, OW), device=x_bf16.device, dtype=torch.bfloat16)

    oc_tiles = (OC + TILE_OC - 1) // TILE_OC
    grid_w = (OW + TILE_W - 1) // TILE_W
    grid_h = (OH + TILE_H - 1) // TILE_H
    grid_z = N * groups * oc_tiles

    _conv_transpose2d_bf16_kernel[lambda: ((grid_w, grid_h, grid_z), (THREADS, 1, 1))](
        x_bf16,
        w_bf16,
        b_bf16,
        out,
        N,
        IC,
        OC,
        H,
        W_in,
        KH,
        KW,
        OH,
        OW,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        has_bias,
        groups,
        oc_tiles,
    )

    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_transpose2d.weight
        bias = self.conv_transpose2d.bias
        return _conv_transpose2d_bf16(
            x,
            weight,
            bias,
            self.conv_transpose2d.stride,
            self.conv_transpose2d.padding,
            self.conv_transpose2d.output_padding,
            self.conv_transpose2d.groups,
            self.conv_transpose2d.dilation,
        )


# Test code
batch_size = 8
in_channels = 64
out_channels = 64
kernel_size = (3, 7)
width = 512
height = 512


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
