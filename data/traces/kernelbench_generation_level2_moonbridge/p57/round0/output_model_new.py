import torch
import torch.nn as nn
import avelang
import avelang.language as al

batch_size = 128
in_channels = 8
out_channels = 64
height = 128
width = 128
kernel_size = 3

TILE_OH: al.constexpr = 14
TILE_OW: al.constexpr = 14
C_IN: al.constexpr = 8
HK: al.constexpr = 3
WK: al.constexpr = 3
THREADS: al.constexpr = 256
MAX_WEIGHT: al.constexpr = 4608
MAX_IN_TILE: al.constexpr = 2048


@avelang.jit
def conv_relu_hardswish_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    H: al.i32,
    W: al.i32,
    C_out: al.i32,
    OH: al.i32,
    OW: al.i32,
    OH_tiles: al.i32,
    K: al.i32,
):
    tid = al.convert(al.thread_id(0), al.i32)
    bid = al.convert(al.block_id(0), al.i32)

    ow_tile = bid % OH_tiles
    tmp = bid // OH_tiles
    oh_tile = tmp % OH_tiles
    n_batch = tmp // OH_tiles

    oh_start = oh_tile * TILE_OH
    ow_start = ow_tile * TILE_OW
    oh_end = oh_start + TILE_OH
    ow_end = ow_start + TILE_OW
    if oh_end > OH:
        oh_end = OH
    if ow_end > OW:
        ow_end = OW

    layout_x = al.make_layout((N * C_IN * H * W,), (1,))
    g_x = al.make_tensor(x_ptr, al.bf16, layout_x)

    layout_w = al.make_layout((C_out * K,), (1,))
    g_w = al.make_tensor(w_ptr, al.bf16, layout_w)

    layout_bias = al.make_layout((C_out,), (1,))
    g_bias = al.make_tensor(bias_ptr, al.bf16, layout_bias)

    layout_out = al.make_layout((N * C_out * OH * OW,), (1,))
    g_out = al.make_tensor(out_ptr, al.bf16, layout_out)

    shm_w = al.make_shared((MAX_WEIGHT,), al.bf16)
    shm_in = al.make_shared((MAX_IN_TILE,), al.bf16)

    for i in al.range(tid, MAX_WEIGHT, THREADS):
        shm_w[i] = g_w[i]

    in_h_start = oh_start
    in_w_start = ow_start
    in_h_end = oh_end + HK - 1
    in_w_end = ow_end + WK - 1
    in_tile_h = in_h_end - in_h_start
    in_tile_w = in_w_end - in_w_start
    in_tile_size = C_IN * in_tile_h * in_tile_w

    for i in al.range(tid, MAX_IN_TILE, THREADS):
        if i < in_tile_size:
            ic = i % C_IN
            tmp_i = i // C_IN
            iy = tmp_i % in_tile_h
            ix = tmp_i // in_tile_h
            in_h = in_h_start + iy
            in_w = in_w_start + ix
            in_idx = ((n_batch * C_IN + ic) * H + in_h) * W + in_w
            shm_in[i] = g_x[in_idx]
    al.syncthreads()

    zero_f32 = al.convert(0.0, al.f32)
    three_f32 = al.convert(3.0, al.f32)
    six_f32 = al.convert(6.0, al.f32)
    one_f32 = al.convert(1.0, al.f32)

    tile_oh_act = oh_end - oh_start
    tile_ow_act = ow_end - ow_start
    total_work = tile_oh_act * tile_ow_act * C_out

    for work_idx in al.range(tid, total_work, THREADS):
        co = work_idx % C_out
        tmp_w = work_idx // C_out
        oy = tmp_w % tile_oh_act
        ox = tmp_w // tile_oh_act

        oh_pos = oh_start + oy
        ow_pos = ow_start + ox

        acc = al.convert(0.0, al.f32)
        for ci in al.range(C_IN):
            for ky in al.range(HK):
                for kx in al.range(WK):
                    in_h_rel = oy + ky
                    in_w_rel = ox + kx
                    in_tile_idx = ci + in_h_rel * C_IN + in_w_rel * C_IN * in_tile_h
                    w_idx = co * K + ci * HK * WK + ky * WK + kx
                    x_val = al.convert(shm_in[in_tile_idx], al.f32)
                    w_val = al.convert(shm_w[w_idx], al.f32)
                    acc = acc + x_val * w_val

        bias_val = al.convert(g_bias[co], al.f32)
        acc = acc + bias_val

        if acc < zero_f32:
            acc = zero_f32

        hs = (acc + three_f32) / six_f32
        if hs < zero_f32:
            hs = zero_f32
        if hs > one_f32:
            hs = one_f32
        result = acc * hs

        out_idx = ((n_batch * C_out + co) * OH + oh_pos) * OW + ow_pos
        g_out[out_idx] = al.convert(result, al.bf16)


def avelang_conv_relu_hardswish(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = x.contiguous().cuda().to(dtype=torch.bfloat16)
    w_bf16 = weight.contiguous().cuda().to(dtype=torch.bfloat16)
    b_bf16 = bias.contiguous().cuda().to(dtype=torch.bfloat16)

    N = x_bf16.shape[0]
    H = x_bf16.shape[2]
    W = x_bf16.shape[3]
    C_out = w_bf16.shape[0]
    HK_val = w_bf16.shape[2]
    WK_val = w_bf16.shape[3]
    OH = H - HK_val + 1
    OW = W - WK_val + 1
    K = 8 * HK_val * WK_val

    OH_tiles = (OH + TILE_OH - 1) // TILE_OH
    OW_tiles = (OW + TILE_OW - 1) // TILE_OW

    out = torch.empty((N, C_out, OH, OW), device=x_bf16.device, dtype=torch.bfloat16)

    grid = (OH_tiles * OW_tiles * N, 1, 1)

    conv_relu_hardswish_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, w_bf16, b_bf16, out,
        N, H, W, C_out, OH, OW, OH_tiles, K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        weight = self.conv.weight.data
        bias = self.conv.bias.data
        result = avelang_conv_relu_hardswish(x, weight, bias)
        return result.to(x.dtype)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
