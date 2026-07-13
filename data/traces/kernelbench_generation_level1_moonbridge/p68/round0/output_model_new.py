import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── Problem-specific compile-time constants ────────────────────────
KD = 3
KW = 5
KH = 5

# ── Tile dimensions ────────────────────────────────────────────────
TILE_D = 8
TILE_W = 8
TILE_H = 4
OC_TILE = 16

# ── Derived compile-time constants ─────────────────────────────────
_TILE_SPATIAL = TILE_D * TILE_W * TILE_H  # 256
THREADS = _TILE_SPATIAL  # 256
REG_D = TILE_D + KD - 1  # 10
REG_W = TILE_W + KW - 1  # 12
REG_H = TILE_H + KH - 1  # 8
REG_SIZE = REG_D * REG_W * REG_H  # 960
WT_KD_KW_KH = KD * KW * KH  # 75
WT_TILE_SIZE = OC_TILE * WT_KD_KW_KH  # 1200
# Blocked loading: each thread loads contiguous chunk
IN_LOADS = (REG_SIZE + THREADS - 1) // THREADS  # 4
WT_LOADS = (WT_TILE_SIZE + THREADS - 1) // THREADS  # 5


@avelang.jit
def conv_transpose3d_bf16_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    D_in: al.i32,
    W_in: al.i32,
    H_in: al.i32,
    D_out: al.i32,
    W_out: al.i32,
    H_out: al.i32,
    H_grid: al.i32,
    W_grid: al.i32,
    OC_grid: al.i32,
):
    tid = al.thread_id(0)
    spatial_block_id = al.block_id(0)
    batch_oc_block_id = al.block_id(1)

    h_tile = spatial_block_id % H_grid
    wh_rem = spatial_block_id // H_grid
    w_tile = wh_rem % W_grid
    d_tile = wh_rem // W_grid

    oc_tile = batch_oc_block_id % OC_grid
    batch_idx = batch_oc_block_id // OC_grid

    h_local = tid % TILE_H
    wd_rem = tid // TILE_H
    w_local = wd_rem % TILE_W
    d_local = wd_rem // TILE_W

    d_out = d_tile * TILE_D + d_local
    w_out = w_tile * TILE_W + w_local
    h_out = h_tile * TILE_H + h_local

    valid = al.convert(1, al.i32)
    if batch_idx >= B:
        valid = al.convert(0, al.i32)
    if d_out >= D_out:
        valid = al.convert(0, al.i32)
    if w_out >= W_out:
        valid = al.convert(0, al.i32)
    if h_out >= H_out:
        valid = al.convert(0, al.i32)

    oc_base = oc_tile * OC_TILE
    in_d_start = d_tile * TILE_D - (KD - 1)
    in_w_start = w_tile * TILE_W - (KW - 1)
    in_h_start = h_tile * TILE_H - (KH - 1)

    in_spatial_stride = W_in * H_in
    in_channel_stride = D_in * in_spatial_stride
    in_batch_stride = IC * in_channel_stride

    in_flat = al.make_tensor(
        input_ptr, al.bf16,
        al.make_layout((B * IC * D_in * W_in * H_in,), (1,)),
    )

    wt_flat = al.make_tensor(
        weight_ptr, al.bf16,
        al.make_layout((IC * OC * WT_KD_KW_KH,), (1,)),
    )

    out_spatial_stride = W_out * H_out
    out_channel_stride = D_out * out_spatial_stride
    out_batch_stride = OC * out_channel_stride

    out_flat = al.make_tensor(
        output_ptr, al.bf16,
        al.make_layout((B * OC * D_out * W_out * H_out,), (1,)),
    )

    input_shm = al.make_shared((REG_SIZE,), al.bf16)
    weight_shm = al.make_shared((WT_TILE_SIZE,), al.bf16)

    acc = al.make_local((OC_TILE,), al.f32)
    for o in al.range(OC_TILE):
        acc[o] = al.convert(0.0, al.f32)

    zero_i32 = al.convert(0, al.i32)

    for ic in al.range(IC):
        # ── Blocked load of input region (coalesced) ──────────────
        in_base_idx = tid * IN_LOADS
        for e in al.range(IN_LOADS):
            idx = in_base_idx + e
            if idx < REG_SIZE:
                hh = idx % REG_H
                wh_rem_load = idx // REG_H
                ww = wh_rem_load % REG_W
                dd = wh_rem_load // REG_W
                in_d = in_d_start + dd
                in_w = in_w_start + ww
                in_h = in_h_start + hh
                load_valid = al.convert(1, al.i32)
                if in_d < zero_i32:
                    load_valid = al.convert(0, al.i32)
                if in_d >= D_in:
                    load_valid = al.convert(0, al.i32)
                if in_w < zero_i32:
                    load_valid = al.convert(0, al.i32)
                if in_w >= W_in:
                    load_valid = al.convert(0, al.i32)
                if in_h < zero_i32:
                    load_valid = al.convert(0, al.i32)
                if in_h >= H_in:
                    load_valid = al.convert(0, al.i32)
                if load_valid != zero_i32:
                    in_flat_idx = (
                        batch_idx * in_batch_stride
                        + ic * in_channel_stride
                        + in_d * in_spatial_stride
                        + in_w * H_in
                        + in_h
                    )
                    input_shm[idx] = in_flat[in_flat_idx]
                else:
                    input_shm[idx] = al.convert(0.0, al.bf16)

        # ── Blocked load of weight tile ───────────────────────────
        wt_gbase = ic * OC * WT_KD_KW_KH + oc_base * WT_KD_KW_KH
        wt_base_idx = tid * WT_LOADS
        for e in al.range(WT_LOADS):
            wi = wt_base_idx + e
            if wi < WT_TILE_SIZE:
                weight_shm[wi] = wt_flat[wt_gbase + wi]

        al.syncthreads()

        # ── Compute ───────────────────────────────────────────────
        for kd in al.range(KD):
            for kw in al.range(KW):
                for kh in al.range(KH):
                    in_d_elt = d_out - kd
                    in_w_elt = w_out - kw
                    in_h_elt = h_out - kh
                    shm_d = in_d_elt - in_d_start
                    shm_w = in_w_elt - in_w_start
                    shm_h = in_h_elt - in_h_start

                    if valid != zero_i32:
                        shm_in_idx = (
                            shm_d * (REG_W * REG_H)
                            + shm_w * REG_H
                            + shm_h
                        )
                        in_val = al.convert(input_shm[shm_in_idx], al.f32)
                        wt_kp_base = kd * (KW * KH) + kw * KH + kh

                        for o in al.range(OC_TILE):
                            oc_out = oc_base + o
                            if oc_out < OC:
                                wt_idx = o * WT_KD_KW_KH + wt_kp_base
                                wt_val = al.convert(weight_shm[wt_idx], al.f32)
                                acc[o] = acc[o] + in_val * wt_val

        al.syncthreads()

    # ── Write outputs ─────────────────────────────────────────────
    if valid != zero_i32:
        out_base = (
            batch_idx * out_batch_stride
            + d_out * out_spatial_stride
            + w_out * H_out
            + h_out
        )
        for o in al.range(OC_TILE):
            oc_out = oc_base + o
            if oc_out < OC:
                out_idx = out_base + oc_out * out_channel_stride
                out_flat[out_idx] = al.convert(acc[o], al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple = (1, 1, 1),
    padding: tuple = (0, 0, 0),
    output_padding: tuple = (0, 0, 0),
    groups: int = 1,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight)

    B, IC, D_in, W_in, H_in = x_bf16.shape
    wt_IC, OC, wt_KD, wt_KW, wt_KH = weight_bf16.shape

    if wt_IC != IC or wt_KD != KD or wt_KW != KW or wt_KH != KH:
        raise ValueError("Shape mismatch")

    stride_d, stride_w, stride_h = stride
    padding_d, padding_w, padding_h = padding
    op_d, op_w, op_h = output_padding

    D_out = (D_in - 1) * stride_d - 2 * padding_d + (KD - 1) + op_d + 1
    W_out = (W_in - 1) * stride_w - 2 * padding_w + (KW - 1) + op_w + 1
    H_out = (H_in - 1) * stride_h - 2 * padding_h + (KH - 1) + op_h + 1

    D_grid = (D_out + TILE_D - 1) // TILE_D
    W_grid = (W_out + TILE_W - 1) // TILE_W
    H_grid = (H_out + TILE_H - 1) // TILE_H
    OC_grid = (OC + OC_TILE - 1) // OC_TILE

    num_spatial_tiles = D_grid * W_grid * H_grid

    out = torch.empty(
        (B, OC, D_out, W_out, H_out),
        device=x_bf16.device, dtype=torch.bfloat16,
    )

    grid = (num_spatial_tiles, B * OC_grid, 1)
    block = (THREADS, 1, 1)

    conv_transpose3d_bf16_kernel[lambda: (grid, block)](
        x_bf16, weight_bf16, out,
        B, IC, OC,
        D_in, W_in, H_in,
        D_out, W_out, H_out,
        H_grid, W_grid, OC_grid,
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
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias

        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            output_padding=output_padding, groups=groups, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_transpose3d.weight
        bias = self.conv_transpose3d.bias

        x_bf16 = x.to(dtype=torch.bfloat16)
        w_bf16 = weight.to(dtype=torch.bfloat16)

        result_bf16 = avelang_conv_transpose3d(
            x_bf16, w_bf16,
            stride=self.stride, padding=self.padding,
            output_padding=self.output_padding, groups=self.groups,
        )

        if bias is not None:
            bias_bf16 = bias.to(dtype=torch.bfloat16)
            result_bf16 = result_bf16 + bias_bf16.view(1, -1, 1, 1, 1)

        return result_bf16.to(x.dtype)


batch_size = 16
in_channels = 32
out_channels = 64
kernel_depth = 3
kernel_width = 5
kernel_height = 5
depth = 64
width = 64
height = 64


def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, width, height)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, (kernel_depth, kernel_width, kernel_height)]
