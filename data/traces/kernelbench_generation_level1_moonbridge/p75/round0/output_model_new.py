import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_H = 16
TILE_W = 16
THREADS = TILE_H * TILE_W  # 256
MAX_WEIGHT_PER_GROUP = 4096


@avelang.jit
def conv_transpose2d_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    H: al.i32,
    W: al.i32,
    kH: al.i32,
    kW: al.i32,
    stride_h: al.i32,
    stride_w: al.i32,
    pad_h: al.i32,
    pad_w: al.i32,
    dil_h: al.i32,
    dil_w: al.i32,
    groups: al.i32,
    C_in_g: al.i32,
    C_out_g: al.i32,
    C_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    W_tiles: al.i32,
    w_group_stride: al.i32,
    weight_elems: al.i32,
    x_sN: al.i32,
    x_sC: al.i32,
    x_sH: al.i32,
    out_sN: al.i32,
    out_sC: al.i32,
    out_sH: al.i32,
):
    tid = al.thread_id(0)
    bid_ng = al.block_id(0)
    bid_spatial = al.block_id(1)

    n = bid_ng // groups
    g = bid_ng - n * groups

    h_tile = bid_spatial // W_tiles
    w_tile = bid_spatial - h_tile * W_tiles

    local_h = tid // TILE_W
    local_w = tid - local_h * TILE_W

    oh = h_tile * TILE_H + local_h
    ow = w_tile * TILE_W + local_w

    if oh >= H_out:
        return
    if ow >= W_out:
        return

    shm_w = al.make_shared((MAX_WEIGHT_PER_GROUP,), al.bf16)

    w_flat = al.make_tensor(
        w_ptr, al.bf16,
        al.make_layout((C_in * C_out_g * kH * kW,), (1,))
    )
    w_base = g * w_group_stride

    load_idx = tid
    while load_idx < weight_elems:
        shm_w[load_idx] = w_flat[w_base + load_idx]
        load_idx = load_idx + THREADS
    al.syncthreads()

    x_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((N * C_in * H * W,), (1,)))
    out_flat = al.make_tensor(out_ptr, al.bf16, al.make_layout((N * C_out * H_out * W_out,), (1,)))

    ic_start = g * C_in_g
    oc_start = g * C_out_g
    zero_f32 = al.convert(0.0, al.f32)

    w_sm_oc = kH * kW
    w_sm_kh = kW
    w_sm_ic_stride = C_out_g * kH * kW

    x_n_base = n * x_sN
    out_n_base = n * out_sN
    out_spatial_base = out_n_base + oh * out_sH + ow

    for oc_local in al.range(C_out_g):
        oc = oc_start + oc_local
        result = zero_f32
        w_oc_base = oc_local * w_sm_oc
        out_oc_base = out_spatial_base + oc * out_sC

        for kh in al.range(kH):
            oh_src = oh + pad_h - kh * dil_h
            oh_div = oh_src // stride_h
            oh_rem = oh_src - oh_div * stride_h
            if oh_rem == 0:
                ih = oh_div
                if ih >= 0:
                    if ih < H:
                        w_kh_base = w_oc_base + kh * w_sm_kh
                        x_ih_base = x_n_base + ih * x_sH

                        for kw in al.range(kW):
                            ow_src = ow + pad_w - kw * dil_w
                            ow_div = ow_src // stride_w
                            ow_rem = ow_src - ow_div * stride_w
                            if ow_rem == 0:
                                iw = ow_div
                                if iw >= 0:
                                    if iw < W:
                                        w_kw_base = w_kh_base + kw
                                        x_ih_iw_base = x_ih_base + iw

                                        for ic_local in al.range(C_in_g):
                                            ic = ic_start + ic_local
                                            w_shm_idx = ic_local * w_sm_ic_stride + w_kw_base
                                            x_idx = x_ih_iw_base + ic * x_sC
                                            x_val = al.convert(x_flat[x_idx], al.f32)
                                            w_val = al.convert(shm_w[w_shm_idx], al.f32)
                                            result = result + x_val * w_val

        out_flat[out_oc_base] = al.convert(result, al.bf16)


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    return t.contiguous().to(dtype=torch.bfloat16)


def avelang_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    groups: int,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device required for AveLang kernels.")

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)

    N_val, C_in_val, H_val, W_val = x_bf16.shape
    C_in_w, C_out_g_w, kH_val, kW_val = w_bf16.shape

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    C_out_val = C_out_g_w * groups
    C_in_g_val = C_in_val // groups
    C_out_g_val = C_out_g_w

    H_out_val = (H_val - 1) * stride_h - 2 * pad_h + dil_h * (kH_val - 1) + 1
    W_out_val = (W_val - 1) * stride_w - 2 * pad_w + dil_w * (kW_val - 1) + 1

    assert C_in_val == C_in_w, f"Weight C_in mismatch: {C_in_val} vs {C_in_w}"

    H_tiles = (H_out_val + TILE_H - 1) // TILE_H
    W_tiles = (W_out_val + TILE_W - 1) // TILE_W

    out = torch.empty((N_val, C_out_val, H_out_val, W_out_val), device=x_bf16.device, dtype=torch.bfloat16)

    x_sN = C_in_val * H_val * W_val
    x_sC = H_val * W_val
    x_sH = W_val

    out_sN = C_out_val * H_out_val * W_out_val
    out_sC = H_out_val * W_out_val
    out_sH = W_out_val

    w_group_stride = C_in_g_val * C_out_g_val * kH_val * kW_val
    weight_elems = w_group_stride

    grid = (N_val * groups, H_tiles * W_tiles, 1)
    block = (THREADS, 1, 1)

    conv_transpose2d_bf16_kernel[lambda: (grid, block)](
        x_bf16,
        w_bf16,
        out,
        N_val, C_in_val, H_val, W_val,
        kH_val, kW_val,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        groups, C_in_g_val, C_out_g_val, C_out_val,
        H_out_val, W_out_val,
        W_tiles,
        w_group_stride, weight_elems,
        x_sN, x_sC, x_sH,
        out_sN, out_sC, out_sH,
    )

    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
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
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_transpose2d.weight.data
        return avelang_conv_transpose2d(
            x, weight, self.stride, self.padding, self.dilation, self.groups
        )


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height = 128
width = 256
stride = (2, 3)
padding = (1, 2)
dilation = (2, 1)
groups = 4


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, groups]
