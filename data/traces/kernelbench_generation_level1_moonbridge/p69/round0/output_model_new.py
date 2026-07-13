import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Problem dimensions (fixed for this benchmark)
BATCH_SIZE = 64
IN_CHANNELS = 64
OUT_CHANNELS = 128
H_IN = 128
W_IN = 256
KH = 3
KW = 5
H_OUT = 130
W_OUT = 260

# Tiling parameters
TILE_H = 16
TILE_W = 16
TILE_C_OUT = 8
C_IN_GROUP = 8
THREADS = TILE_H * TILE_W  # 256

# Derived constants
IN_WIN_H = TILE_H + KH - 1  # 18
IN_WIN_W = TILE_W + KW - 1  # 20
SHM_INPUT_ELEMS = IN_WIN_H * IN_WIN_W * C_IN_GROUP  # 2880
SHM_WEIGHT_ELEMS = C_IN_GROUP * TILE_C_OUT * KH * KW  # 960
C_IN_GROUPS = IN_CHANNELS // C_IN_GROUP  # 8
C_OUT_GROUPS = OUT_CHANNELS // TILE_C_OUT  # 16
H_TILES = (H_OUT + TILE_H - 1) // TILE_H  # 9
W_TILES = (W_OUT + TILE_W - 1) // TILE_W  # 17


@avelang.jit
def _load_input_to_shm(
    shm_input: al.Tensor((SHM_INPUT_ELEMS,), al.bf16),
    x_flat: al.Tensor((BATCH_SIZE * IN_CHANNELS * H_IN * W_IN,), al.bf16),
    n_idx: al.i32,
    h_start: al.i32,
    w_start: al.i32,
    c_in_start: al.i32,
    tid: al.i32,
):
    elems_per_thread = (SHM_INPUT_ELEMS + THREADS - 1) // THREADS
    zero_bf16 = al.convert(0.0, al.bf16)
    for i in al.range(elems_per_thread):
        idx = tid + i * THREADS
        if idx < SHM_INPUT_ELEMS:
            c_in_local = idx % C_IN_GROUP
            w_off = (idx // C_IN_GROUP) % IN_WIN_W
            h_off = idx // (C_IN_GROUP * IN_WIN_W)
            h_in_val = h_start + h_off
            w_in_val = w_start + w_off
            c_in_val = c_in_start + c_in_local
            if h_in_val >= 0:
                if h_in_val < H_IN:
                    if w_in_val >= 0:
                        if w_in_val < W_IN:
                            g_idx = ((n_idx * IN_CHANNELS + c_in_val) * H_IN + h_in_val) * W_IN + w_in_val
                            shm_input[idx] = x_flat[g_idx]


@avelang.jit
def _load_weight_to_shm(
    shm_weight: al.Tensor((SHM_WEIGHT_ELEMS,), al.bf16),
    w_flat: al.Tensor((IN_CHANNELS * OUT_CHANNELS * KH * KW,), al.bf16),
    c_in_start: al.i32,
    c_out_start: al.i32,
    tid: al.i32,
):
    elems_per_thread = (SHM_WEIGHT_ELEMS + THREADS - 1) // THREADS
    for i in al.range(elems_per_thread):
        idx = tid + i * THREADS
        if idx < SHM_WEIGHT_ELEMS:
            kw_local = idx % KW
            kh_local = (idx // KW) % KH
            c_out_local = (idx // (KH * KW)) % TILE_C_OUT
            c_in_local = idx // (TILE_C_OUT * KH * KW)
            c_in_val = c_in_start + c_in_local
            c_out_val = c_out_start + c_out_local
            g_idx = ((c_in_val * OUT_CHANNELS + c_out_val) * KH + kh_local) * KW + kw_local
            shm_weight[idx] = w_flat[g_idx]


@avelang.jit
def conv_transpose2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
):
    flat_nc = al.block_id(0)
    h_tile = al.block_id(1)
    w_tile = al.block_id(2)
    tid = al.thread_id(0)

    n_idx = flat_nc // C_OUT_GROUPS
    c_out_group = flat_nc % C_OUT_GROUPS
    c_out_start = c_out_group * TILE_C_OUT

    ty = tid // TILE_W
    tx = tid % TILE_W
    h_out = h_tile * TILE_H + ty
    w_out = w_tile * TILE_W + tx

    x_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((BATCH_SIZE * IN_CHANNELS * H_IN * W_IN,), (1,)))
    w_flat = al.make_tensor(w_ptr, al.bf16, al.make_layout((IN_CHANNELS * OUT_CHANNELS * KH * KW,), (1,)))
    out_flat = al.make_tensor(out_ptr, al.bf16, al.make_layout((BATCH_SIZE * OUT_CHANNELS * H_OUT * W_OUT,), (1,)))

    h_start = h_tile * TILE_H - (KH - 1)
    w_start = w_tile * TILE_W - (KW - 1)

    shm_input = al.make_shared((SHM_INPUT_ELEMS,), al.bf16)
    shm_weight = al.make_shared((SHM_WEIGHT_ELEMS,), al.bf16)

    acc = al.make_local((TILE_C_OUT,), al.f32)
    for ci in al.range(TILE_C_OUT):
        acc[ci] = al.convert(0.0, al.f32)

    for cig in al.range(C_IN_GROUPS):
        c_in_start = cig * C_IN_GROUP

        _load_input_to_shm(shm_input, x_flat, n_idx, h_start, w_start, c_in_start, tid)
        _load_weight_to_shm(shm_weight, w_flat, c_in_start, c_out_start, tid)
        al.syncthreads()

        if h_out < H_OUT:
            if w_out < W_OUT:
                for kh in al.range(KH):
                    h_in_val = h_out - kh
                    h_off = h_in_val - h_start
                    for kw in al.range(KW):
                        w_in_val = w_out - kw
                        w_off = w_in_val - w_start
                        if h_in_val >= 0:
                            if h_in_val < H_IN:
                                if w_in_val >= 0:
                                    if w_in_val < W_IN:
                                        for cil in al.range(C_IN_GROUP):
                                            shm_in_idx = (h_off * IN_WIN_W + w_off) * C_IN_GROUP + cil
                                            input_val = al.convert(shm_input[shm_in_idx], al.f32)
                                            for col in al.range(TILE_C_OUT):
                                                shm_w_idx = ((cil * TILE_C_OUT + col) * KH + kh) * KW + kw
                                                weight_val = al.convert(shm_weight[shm_w_idx], al.f32)
                                                acc[col] = acc[col] + input_val * weight_val

        al.syncthreads()

    if h_out < H_OUT:
        if w_out < W_OUT:
            for col in al.range(TILE_C_OUT):
                c_out_val = c_out_start + col
                out_idx = ((n_idx * OUT_CHANNELS + c_out_val) * H_OUT + h_out) * W_OUT + w_out
                out_flat[out_idx] = al.convert(acc[col], al.bf16)


def _prepare_bf16(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16(x)
    w_bf16 = _prepare_bf16(weight)

    N_val, C_in_val, H_in_val, W_in_val = x_bf16.shape
    C_in_w, C_out_val, Kh_val, Kw_val = w_bf16.shape

    if C_in_val != C_in_w:
        raise ValueError(f"Input/weight channel mismatch: {C_in_val} vs {C_in_w}")
    if H_in_val != H_IN or W_in_val != W_IN:
        raise ValueError(
            f"Expected input spatial dims ({H_IN}, {W_IN}), got ({H_in_val}, {W_in_val})"
        )
    if Kh_val != KH or Kw_val != KW:
        raise ValueError(
            f"Expected kernel size ({KH}, {KW}), got ({Kh_val}, {Kw_val})"
        )

    out = torch.empty((N_val, OUT_CHANNELS, H_OUT, W_OUT), device=x_bf16.device, dtype=torch.bfloat16)
    grid_dim0 = N_val * C_OUT_GROUPS
    grid = (grid_dim0, H_TILES, W_TILES)
    conv_transpose2d_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, w_bf16, out
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv_transpose2d(x, self.conv_transpose2d.weight)


def get_inputs():
    x = torch.rand(BATCH_SIZE, IN_CHANNELS, H_IN, W_IN)
    return [x]


def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, (KH, KW)]
