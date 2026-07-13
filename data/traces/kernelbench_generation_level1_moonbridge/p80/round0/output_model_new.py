import torch
import torch.nn as nn
import avelang
import avelang.language as al

BATCH_SIZE = 8
IN_CHANNELS = 32
OUT_CHANNELS = 64
K_H = 5
K_W = 9
H_IN = 512
W_IN = 512
STRIDE_H = 1
STRIDE_W = 1
PAD_H = 2
PAD_W = 4
DIL_H = 2
DIL_W = 3

H_OUT = (H_IN + 2 * PAD_H - DIL_H * (K_H - 1) - 1) // STRIDE_H + 1
W_OUT = (W_IN + 2 * PAD_W - DIL_W * (K_W - 1) - 1) // STRIDE_W + 1

OH_TILE = 16
OW_TILE = 16
IC_TILE = 4
OC_TILE = 16
THREADS = 256

OH_TILES = (H_OUT + OH_TILE - 1) // OH_TILE
OW_TILES = (W_OUT + OW_TILE - 1) // OW_TILE
IC_TILES = IN_CHANNELS // IC_TILE
OC_TILES = (OUT_CHANNELS + OC_TILE - 1) // OC_TILE

SHM_WT_SIZE = OC_TILE * IC_TILE * K_H * K_W
WT_LOADS = (SHM_WT_SIZE + THREADS - 1) // THREADS


@avelang.jit
def conv2d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    C_out: al.i32,
    K_h: al.i32,
    K_w: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    stride_h: al.i32,
    stride_w: al.i32,
    pad_h: al.i32,
    pad_w: al.i32,
    dilation_h: al.i32,
    dilation_w: al.i32,
):
    tid = al.thread_id(0)
    ow_tile = al.block_id(0)
    oh_tile = al.block_id(1)
    batch_oc = al.block_id(2)

    n_idx = batch_oc // OC_TILES
    oc_group = batch_oc % OC_TILES
    oc_base = oc_group * OC_TILE
    oh_base = oh_tile * OH_TILE
    ow_base = ow_tile * OW_TILE

    oh_local = tid // OW_TILE
    ow_local = tid % OW_TILE
    oh_global = oh_base + oh_local
    ow_global = ow_base + ow_local

    in_stride_n = C * H * W
    in_stride_c = H * W
    in_stride_h = W
    wt_stride_oc = C * K_h * K_w
    wt_stride_ic = K_h * K_w
    wt_stride_kh = K_w
    out_stride_n = C_out * H_out * W_out
    out_stride_c = H_out * W_out
    out_stride_h = W_out

    in_flat = al.make_tensor(input_ptr, al.bf16, al.make_layout((N * C * H * W,), (1,)))
    wt_flat = al.make_tensor(weight_ptr, al.bf16, al.make_layout((C_out * C * K_h * K_w,), (1,)))
    out_flat = al.make_tensor(output_ptr, al.bf16, al.make_layout((N * C_out * H_out * W_out,), (1,)))

    shm_wt = al.make_shared((SHM_WT_SIZE,), al.bf16)

    zero_i32 = al.convert(0, al.i32)
    one_i32 = al.convert(1, al.i32)

    wt_oc_stride = IC_TILE * K_h * K_w
    wt_ic_stride = K_h * K_w
    wt_kh_stride = K_w

    valid = al.convert(0, al.i32)
    if oh_global < H_out:
        if ow_global < W_out:
            valid = one_i32

    acc = al.make_local((OC_TILE,), al.f32)
    for i in al.range(OC_TILE):
        acc[i] = al.convert(0.0, al.f32)

    for ic_tile in al.range(IC_TILES):
        ic_base = ic_tile * IC_TILE

        for i in al.range(WT_LOADS):
            idx = tid + i * THREADS
            if idx < SHM_WT_SIZE:
                oc_w = idx // wt_oc_stride
                rem_w = idx % wt_oc_stride
                ic_w = rem_w // wt_ic_stride
                rem_w = rem_w % wt_ic_stride
                kh_w = rem_w // wt_kh_stride
                kw_w = rem_w % wt_kh_stride
                wt_off = (oc_base + oc_w) * wt_stride_oc + (ic_base + ic_w) * wt_stride_ic + kh_w * wt_stride_kh + kw_w
                shm_wt[idx] = wt_flat[wt_off]

        al.syncthreads()

        if valid:
            for oc_local in al.range(OC_TILE):
                wt_oc_base = oc_local * wt_oc_stride
                for ic_local in al.range(IC_TILE):
                    in_off_base = n_idx * in_stride_n + (ic_base + ic_local) * in_stride_c
                    wt_ic_base = wt_oc_base + ic_local * wt_ic_stride
                    for kh in al.range(K_h):
                        ih_val = oh_global * stride_h + kh * dilation_h - pad_h
                        ok_ih = al.convert(0, al.i32)
                        if ih_val >= zero_i32:
                            if ih_val < H:
                                ok_ih = one_i32
                        if ok_ih:
                            in_off = in_off_base + ih_val * in_stride_h
                            wt_kh_base = wt_ic_base + kh * wt_kh_stride
                            for kw in al.range(K_w):
                                iw_val = ow_global * stride_w + kw * dilation_w - pad_w
                                ok_iw = al.convert(0, al.i32)
                                if iw_val >= zero_i32:
                                    if iw_val < W:
                                        ok_iw = one_i32
                                if ok_iw:
                                    in_val = al.convert(in_flat[in_off + iw_val], al.f32)
                                    wt_val = al.convert(shm_wt[wt_kh_base + kw], al.f32)
                                    acc[oc_local] = acc[oc_local] + in_val * wt_val

        al.syncthreads()

    if valid:
        for oc_local in al.range(OC_TILE):
            oc_global_val = oc_base + oc_local
            out_off = n_idx * out_stride_n + oc_global_val * out_stride_c + oh_global * out_stride_h + ow_global
            out_flat[out_off] = al.convert(acc[oc_local], al.bf16)


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride,
    padding,
    dilation,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)

    N, C, H, W = x_bf16.shape
    C_out, _, K_h, K_w = w_bf16.shape

    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = stride
    if isinstance(padding, int):
        pad_h, pad_w = padding, padding
    else:
        pad_h, pad_w = padding
    if isinstance(dilation, int):
        dil_h, dil_w = dilation, dilation
    else:
        dil_h, dil_w = dilation

    H_out = (H + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1

    oh_tiles = (H_out + OH_TILE - 1) // OH_TILE
    ow_tiles = (W_out + OW_TILE - 1) // OW_TILE
    oc_tiles = (C_out + OC_TILE - 1) // OC_TILE

    out = torch.empty((N, C_out, H_out, W_out), device=x_bf16.device, dtype=torch.bfloat16)

    grid = (ow_tiles, oh_tiles, N * oc_tiles)
    conv2d_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, w_bf16, out,
        N, C, H, W,
        C_out, K_h, K_w,
        H_out, W_out,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
    )
    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv2d(
            x, self.weight, self.stride, self.padding, self.dilation
        )


def get_inputs():
    x = torch.rand(BATCH_SIZE, IN_CHANNELS, H_IN, W_IN)
    return [x]


def get_init_inputs():
    return [
        IN_CHANNELS,
        OUT_CHANNELS,
        (K_H, K_W),
        STRIDE_H,
        (PAD_H, PAD_W),
        (DIL_H, DIL_W),
    ]
