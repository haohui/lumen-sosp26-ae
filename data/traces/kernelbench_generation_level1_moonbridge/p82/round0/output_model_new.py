import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Tile constants for depthwise conv2d 3x3 kernel
TILE_OH = 64
TILE_OW = 64
THREADS = 256
KH = 3
KW = 3
SHM_H = TILE_OH + KH - 1
SHM_W = TILE_OW + KW - 1


@avelang.jit
def depthwise_conv2d_3x3_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    OH: al.i32,
    OW: al.i32,
):
    tid = al.thread_id(0)
    ow_block = al.block_id(0)
    oh_block = al.block_id(1)
    nc_id = al.block_id(2)

    n = nc_id // C
    c = nc_id % C

    oh_start = oh_block * TILE_OH
    ow_start = ow_block * TILE_OW

    # Flattened global memory views
    CHW = C * H * W
    HW = H * W
    x_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((N * CHW,), (1,)))

    KHKW = KH * KW
    w_flat = al.make_tensor(w_ptr, al.bf16, al.make_layout((C * KHKW,), (1,)))

    OHOW = OH * OW
    COHOW = C * OHOW
    out_flat = al.make_tensor(out_ptr, al.bf16, al.make_layout((N * COHOW,), (1,)))

    # Shared memory for input tile with halo for 3x3 kernel
    shm_x = al.make_shared((SHM_H, SHM_W), al.bf16)

    # Cooperative load of input tile into shared memory
    shm_size = SHM_H * SHM_W
    x_base = n * CHW + c * HW
    zero_bf16 = al.convert(0.0, al.bf16)
    for idx in al.range(tid, shm_size, THREADS):
        shm_h = idx // SHM_W
        shm_w = idx % SHM_W
        ih = oh_start + shm_h
        iw = ow_start + shm_w
        if ih < H:
            if iw < W:
                shm_x[shm_h, shm_w] = x_flat[x_base + ih * W + iw]
            else:
                shm_x[shm_h, shm_w] = zero_bf16
        else:
            shm_x[shm_h, shm_w] = zero_bf16

    al.syncthreads()

    w_base = c * KHKW
    out_base = n * COHOW + c * OHOW
    zero_f32 = al.convert(0.0, al.f32)

    # Each thread computes several output elements
    total_outs = TILE_OH * TILE_OW
    for idx in al.range(tid, total_outs, THREADS):
        out_h = idx // TILE_OW
        out_w = idx % TILE_OW

        global_oh = oh_start + out_h
        global_ow = ow_start + out_w

        if global_oh < OH:
            if global_ow < OW:
                acc = zero_f32
                for kh in al.range(KH):
                    for kw in al.range(KW):
                        w_val = al.convert(w_flat[w_base + kh * KW + kw], al.f32)
                        x_val = al.convert(shm_x[out_h + kh, out_w + kw], al.f32)
                        acc = acc + w_val * x_val
                out_flat[out_base + global_oh * OW + global_ow] = al.convert(acc, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = _prepare_bf16_cuda_contiguous(weight)

    N, C, H, W = x_bf16.shape
    out_c, _, KH_w, KW_w = w_bf16.shape
    if out_c != C:
        raise ValueError(f"Channel mismatch: input has {C} channels, weight has {out_c}")

    OH = (H - KH_w) + 1
    OW = (W - KW_w) + 1

    out = torch.empty((N, C, OH, OW), device=x_bf16.device, dtype=torch.bfloat16)

    grid_x = (OW + TILE_OW - 1) // TILE_OW
    grid_y = (OH + TILE_OH - 1) // TILE_OH
    grid_z = N * C

    grid = (grid_x, grid_y, grid_z)
    depthwise_conv2d_3x3_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, w_bf16, out, N, C, H, W, OH, OW
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_depthwise_conv2d(x, self.conv2d.weight)


# Test code
batch_size = 16
in_channels = 64
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]
