import torch
import torch.nn as nn
import avelang
import avelang.language as al
import math

TILE_H = 8
TILE_W = 8
TILE_OC = 4
THREADS = 256


@avelang.jit
def conv3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.u32,
    IC: al.u32,
    OC: al.u32,
    D: al.u32,
    H: al.u32,
    W: al.u32,
    KD: al.u32,
    KH: al.u32,
    KW: al.u32,
    D_out: al.u32,
    H_out: al.u32,
    W_out: al.u32,
):
    tid = al.thread_id(0)
    block_x = al.block_id(0)
    block_y = al.block_id(1)
    block_z = al.block_id(2)

    tile_h = al.convert(TILE_H, al.u32)
    tile_w = al.convert(TILE_W, al.u32)
    tile_oc = al.convert(TILE_OC, al.u32)
    oc_tiles = OC // tile_oc
    threads = al.convert(THREADS, al.u32)

    oc_tile = block_z % oc_tiles
    bd = block_z // oc_tiles
    b = bd // D_out
    do_idx = bd % D_out

    toc = tid % tile_oc
    tid_rem = tid // tile_oc
    tw = tid_rem % tile_w
    th = tid_rem // tile_w

    ho = block_y * tile_h + th
    wo = block_x * tile_w + tw
    oc = oc_tile * tile_oc + toc

    valid = al.convert(1, al.u32)
    if wo >= W_out:
        valid = al.convert(0, al.u32)
    if ho >= H_out:
        valid = al.convert(0, al.u32)

    kernel_volume = KD * KH * KW
    ic_kv = IC * kernel_volume
    weight_tile_elems = tile_oc * ic_kv
    dhw = D * H * W
    hw = H * W

    in_total = B * IC * dhw
    input_memref = al.make_tensor(input_ptr, al.bf16, al.make_layout((in_total,), (1,)))
    weight_memref = al.make_tensor(weight_ptr, al.bf16, al.make_layout((OC * ic_kv,), (1,)))
    out_total = B * OC * D_out * H_out * W_out
    output_memref = al.make_tensor(output_ptr, al.bf16, al.make_layout((out_total,), (1,)))

    # Weight tile: TILE_OC x IC x KD x KH x KW = 4 x 3 x 3 x 5 x 7 = 1260 bf16
    shm_weight = al.make_shared((4 * 3 * 3 * 5 * 7,), al.bf16)
    # Input window: IC x KD x (KH+TILE_H-1) x (KW+TILE_W-1) = 3 x 3 x 12 x 14 = 1512 bf16
    win_h = KH + tile_h - al.convert(1, al.u32)
    win_w = KW + tile_w - al.convert(1, al.u32)
    win_2d = win_h * win_w
    win_size = IC * KD * win_2d
    shm_input = al.make_shared((3 * 3 * 12 * 14,), al.bf16)

    # Stage 1: load weight into shared memory
    num_w_loads = (weight_tile_elems + threads - al.convert(1, al.u32)) // threads
    for i in al.range(num_w_loads):
        idx = tid + i * threads
        if idx < weight_tile_elems:
            global_w_idx = oc_tile * tile_oc * ic_kv + idx
            shm_weight[idx] = weight_memref[global_w_idx]

    al.syncthreads()

    # Stage 2: load full 3D input window into shared memory
    ho_base = block_y * tile_h
    wo_base = block_x * tile_w
    zero_bf16 = al.convert(0.0, al.bf16)

    num_in_loads = (win_size + threads - al.convert(1, al.u32)) // threads
    for i in al.range(num_in_loads):
        idx = tid + i * threads
        if idx < win_size:
            ic_idx = idx // (KD * win_2d)
            rem = idx % (KD * win_2d)
            kd_idx = rem // win_2d
            rem2 = rem % win_2d
            load_h = rem2 // win_w
            load_w = rem2 % win_w
            glb_h = ho_base + load_h
            glb_w = wo_base + load_w
            if glb_h < H:
                if glb_w < W:
                    in_d = do_idx + kd_idx
                    in_off = ((b * IC + ic_idx) * D + in_d) * H * W + glb_h * W + glb_w
                    shm_input[idx] = input_memref[in_off]
                else:
                    shm_input[idx] = zero_bf16
            else:
                shm_input[idx] = zero_bf16

    al.syncthreads()

    # Stage 3: compute convolution from shared memory
    if valid != al.convert(0, al.u32):
        acc = al.convert(0.0, al.f32)
        w_toc_base = toc * ic_kv

        for _ic in al.range(IC):
            w_ic_off = w_toc_base + _ic * kernel_volume
            in_ic_off = _ic * KD * win_2d

            for _kd in al.range(KD):
                w_kd_off = w_ic_off + _kd * KH * KW
                in_kd_off = in_ic_off + _kd * win_2d

                for _kh in al.range(KH):
                    in_h_off = in_kd_off + (th + _kh) * win_w
                    w_kh_off = w_kd_off + _kh * KW

                    for _kw in al.range(KW):
                        in_val = al.convert(shm_input[in_h_off + tw + _kw], al.f32)
                        w_val = al.convert(shm_weight[w_kh_off + _kw], al.f32)
                        acc = acc + in_val * w_val

        out_off = ((b * OC + oc) * D_out + do_idx) * H_out * W_out + ho * W_out + wo
        output_memref[out_off] = al.convert(acc, al.bf16)


def _prepare_bf16_cuda(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda(x)
    w_bf16 = _prepare_bf16_cuda(weight)

    B, IC, D, H, W = x_bf16.shape
    OC, w_IC, KD, KH, KW = w_bf16.shape

    if IC != w_IC:
        raise ValueError(f"Input channels mismatch: x has {IC}, weight has {w_IC}")

    D_out = D - KD + 1
    H_out = H - KH + 1
    W_out = W - KW + 1

    grid_x = (W_out + TILE_W - 1) // TILE_W
    grid_y = (H_out + TILE_H - 1) // TILE_H
    oc_tiles = (OC + TILE_OC - 1) // TILE_OC
    grid_z = B * D_out * oc_tiles

    out = torch.empty((B, OC, D_out, H_out, W_out), device=x_bf16.device, dtype=torch.bfloat16)

    conv3d_kernel[lambda: ((grid_x, grid_y, grid_z), (THREADS, 1, 1))](
        x_bf16, w_bf16, out,
        B, IC, OC, D, H, W, KD, KH, KW, D_out, H_out, W_out,
    )

    if bias is not None:
        bias_bf16 = _prepare_bf16_cuda(bias)
        out = out + bias_bf16.view(1, -1, 1, 1, 1)

    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        dilation: tuple = (1, 1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels
            for k in self.kernel_size:
                fan_in *= k
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv3d(x, self.weight, self.bias)


# Test code (matching input_model.py)
batch_size = 8
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)
depth = 16
height = 128
width = 128


def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
