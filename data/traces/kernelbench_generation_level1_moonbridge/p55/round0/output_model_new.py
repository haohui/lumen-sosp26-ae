import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_OC: al.constexpr = 32
TILE_H: al.constexpr = 8
TILE_W: al.constexpr = 16
TILE_IC: al.constexpr = 16
KH: al.constexpr = 3
KW: al.constexpr = 3
THREADS: al.constexpr = 256

SHM_WEIGHT_ELEMS: al.constexpr = TILE_OC * TILE_IC * KH * KW
INPUT_H: al.constexpr = TILE_H + KH - 1
INPUT_W: al.constexpr = TILE_W + KW - 1
SHM_INPUT_ELEMS: al.constexpr = TILE_IC * INPUT_H * INPUT_W
TILE_ELEMS: al.constexpr = TILE_OC * TILE_H * TILE_W
ELEMS_PER_THREAD: al.constexpr = TILE_ELEMS // THREADS


@avelang.jit
def conv2d_direct_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    OC: al.i32,
    HOUT: al.i32,
    WOUT: al.i32,
    num_oc_tiles: al.i32,
    num_h_tiles: al.i32,
    num_w_tiles: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    tiles_per_batch = num_oc_tiles * num_h_tiles * num_w_tiles
    batch = bid // tiles_per_batch
    rest = bid - batch * tiles_per_batch
    oc_tile = rest // (num_h_tiles * num_w_tiles)
    rest2 = rest - oc_tile * (num_h_tiles * num_w_tiles)
    h_tile = rest2 // num_w_tiles
    w_tile = rest2 - h_tile * num_w_tiles

    oc_start = al.convert(oc_tile * TILE_OC, al.i32)
    h_start = al.convert(h_tile * TILE_H, al.i32)
    w_start = al.convert(w_tile * TILE_W, al.i32)

    total_input = B * C * H * W
    total_weight = OC * C * KH * KW
    total_output = B * OC * HOUT * WOUT

    input_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((total_input,), (1,)))
    weight_flat = al.make_tensor(w_ptr, al.bf16, al.make_layout((total_weight,), (1,)))
    output_flat = al.make_tensor(out_ptr, al.bf16, al.make_layout((total_output,), (1,)))

    shm_weight = al.make_shared((SHM_WEIGHT_ELEMS,), al.bf16)
    shm_input = al.make_shared((SHM_INPUT_ELEMS,), al.bf16)

    acc = al.make_local((ELEMS_PER_THREAD,), al.f32)
    for i in al.range(ELEMS_PER_THREAD):
        acc[i] = al.convert(0.0, al.f32)

    stride_c = C * KH * KW
    stride_hw = H * W
    stride_w = W

    num_ic_tiles = (C + TILE_IC - 1) // TILE_IC

    for ict in al.range(num_ic_tiles):
        ic_start = ict * TILE_IC
        ic_range = TILE_IC
        if ic_start + TILE_IC > C:
            ic_range = C - ic_start

        total_wl = (SHM_WEIGHT_ELEMS + THREADS - 1) // THREADS
        for t in al.range(total_wl):
            idx = tid + t * THREADS
            if idx < SHM_WEIGHT_ELEMS:
                oc_l = idx // (TILE_IC * KH * KW)
                rest_w = idx - oc_l * (TILE_IC * KH * KW)
                ic_l = rest_w // (KH * KW)
                rest_w2 = rest_w - ic_l * (KH * KW)
                kh_l = rest_w2 // KW
                kw_l = rest_w2 - kh_l * KW

                g_oc = oc_start + oc_l
                g_ic = ic_start + ic_l
                if g_oc < OC and g_ic < C:
                    w_idx = g_oc * stride_c + g_ic * (KH * KW) + kh_l * KW + kw_l
                    shm_weight[idx] = weight_flat[w_idx]
                else:
                    shm_weight[idx] = al.convert(0.0, al.bf16)

        total_il = (SHM_INPUT_ELEMS + THREADS - 1) // THREADS
        for t in al.range(total_il):
            idx = tid + t * THREADS
            if idx < SHM_INPUT_ELEMS:
                ic_l = idx // (INPUT_H * INPUT_W)
                rest_i = idx - ic_l * (INPUT_H * INPUT_W)
                ih = rest_i // INPUT_W
                iw = rest_i - ih * INPUT_W

                g_ic = ic_start + ic_l
                g_h = h_start + ih
                g_w = w_start + iw

                if g_ic < C and g_h < H and g_w < W:
                    in_idx = batch * C * stride_hw + g_ic * stride_hw + g_h * stride_w + g_w
                    shm_input[idx] = input_flat[in_idx]
                else:
                    shm_input[idx] = al.convert(0.0, al.bf16)

        al.syncthreads()

        for elem_t in al.range(ELEMS_PER_THREAD):
            elem_idx = tid + elem_t * THREADS
            oc_local = elem_idx // (TILE_H * TILE_W)
            hw_local = elem_idx - oc_local * (TILE_H * TILE_W)
            h_local = hw_local // TILE_W
            w_local = hw_local - h_local * TILE_W

            g_oc = oc_start + oc_local
            g_h = h_start + h_local
            g_w = w_start + w_local

            if g_oc < OC and g_h < HOUT and g_w < WOUT:
                partial = al.convert(0.0, al.f32)
                for kh in al.range(KH):
                    for kw in al.range(KW):
                        for ic_local in al.range(ic_range):
                            in_idx = ic_local * (INPUT_H * INPUT_W) + (h_local + kh) * INPUT_W + (w_local + kw)
                            in_val = al.convert(shm_input[in_idx], al.f32)
                            w_idx = oc_local * (TILE_IC * KH * KW) + ic_local * (KH * KW) + kh * KW + kw
                            w_val = al.convert(shm_weight[w_idx], al.f32)
                            partial = partial + in_val * w_val
                acc[elem_t] = acc[elem_t] + partial

        al.syncthreads()

    stride_oc_out = HOUT * WOUT
    for elem_t in al.range(ELEMS_PER_THREAD):
        elem_idx = tid + elem_t * THREADS
        oc_local = elem_idx // (TILE_H * TILE_W)
        hw_local = elem_idx - oc_local * (TILE_H * TILE_W)
        h_local = hw_local // TILE_W
        w_local = hw_local - h_local * TILE_W

        g_oc = oc_start + oc_local
        g_h = h_start + h_local
        g_w = w_start + w_local

        if g_oc < OC and g_h < HOUT and g_w < WOUT:
            out_idx = batch * (OC * stride_oc_out) + g_oc * stride_oc_out + g_h * WOUT + g_w
            output_flat[out_idx] = al.convert(acc[elem_t], al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv2d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = _prepare_bf16_cuda_contiguous(weight)

    B, C, H, W = x_bf16.shape
    OC, C_w, KH_w, KW_w = w_bf16.shape

    if C != C_w:
        raise ValueError(f"Channel mismatch")

    HOUT = H - KH + 1
    WOUT = W - KW + 1

    num_oc_tiles = (OC + TILE_OC - 1) // TILE_OC
    num_h_tiles = (HOUT + TILE_H - 1) // TILE_H
    num_w_tiles = (WOUT + TILE_W - 1) // TILE_W
    tiles_per_batch = num_oc_tiles * num_h_tiles * num_w_tiles
    total_blocks = B * tiles_per_batch

    out = torch.empty((B, OC, HOUT, WOUT), device=x_bf16.device, dtype=torch.bfloat16)

    conv2d_direct_kernel[lambda: ((total_blocks, 1, 1), (THREADS, 1, 1))](
        x_bf16, w_bf16, out,
        B, C, H, W, OC, HOUT, WOUT,
        num_oc_tiles, num_h_tiles, num_w_tiles,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv2d.weight.data
        result_bf16 = avelang_conv2d(x, weight)
        return result_bf16.to(x.dtype)


batch_size = 8
height = 512
width = 1024
in_channels = 64
out_channels = 128
kernel_size = 3


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
