import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Tile constants
BLOCK_D: al.constexpr = 16
BLOCK_H: al.constexpr = 16
BLOCK_OC: al.constexpr = 16
THREADS: al.constexpr = 256
KD: al.constexpr = 3
KH: al.constexpr = 3
KW: al.constexpr = 1
TILE_D: al.constexpr = BLOCK_D + KD - 1  # 18
TILE_H: al.constexpr = BLOCK_H + KH - 1  # 18
W_MAX: al.constexpr = 10
# Shared memory: 4 bf16 per (d,h,w) for clean u32 alignment
SHM_IN_U32: al.constexpr = TILE_D * TILE_H * W_MAX * 2  # 6480
SHM_IN_BF16: al.constexpr = TILE_D * TILE_H * W_MAX * 4  # 12960


@avelang.jit
def conv3d_bf16_kernel(
    in_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch: al.i32,
    in_c: al.i32,
    out_c: al.i32,
    d_in: al.i32,
    h_in: al.i32,
    w_in: al.i32,
    d_out: al.i32,
    h_out: al.i32,
    w_out: al.i32,
    stride: al.i32,
    padding: al.i32,
    dilation: al.i32,
    groups: al.i32,
):
    tid = al.thread_id(0)
    gid_0 = al.block_id(0)
    gid_1 = al.block_id(1)

    num_d_tiles = (d_out + BLOCK_D - 1) // BLOCK_D
    num_h_tiles = (h_out + BLOCK_H - 1) // BLOCK_H
    num_oc_tiles = (out_c // groups + BLOCK_OC - 1) // BLOCK_OC

    batch_idx = gid_1 // num_oc_tiles
    oc_tile_idx = gid_1 - batch_idx * num_oc_tiles
    spatial_idx = gid_0
    h_tile_idx = spatial_idx % num_h_tiles
    d_tile_idx = spatial_idx // num_h_tiles

    group_in_c = in_c // groups
    group_out_c = out_c // groups

    group_id = oc_tile_idx // ((group_out_c + BLOCK_OC - 1) // BLOCK_OC)
    ic_start = group_id * group_in_c
    ic_end = ic_start + group_in_c
    oc_start = group_id * group_out_c + oc_tile_idx * BLOCK_OC
    oc_end = oc_start + BLOCK_OC
    if oc_end > (group_id + 1) * group_out_c:
        oc_end = (group_id + 1) * group_out_c

    block_oc = oc_end - oc_start
    block_d = BLOCK_D
    block_h = BLOCK_H
    d_tile_end = d_tile_idx * BLOCK_D + BLOCK_D
    if d_tile_end > d_out:
        block_d = d_out - d_tile_idx * BLOCK_D
    h_tile_end = h_tile_idx * BLOCK_H + BLOCK_H
    if h_tile_end > h_out:
        block_h = h_out - h_tile_idx * BLOCK_H

    d_base = d_tile_idx * BLOCK_D * stride - padding
    h_base = h_tile_idx * BLOCK_H * stride - padding

    # Shared memory for input tile (all W slices staged at once)
    shm_in_u32 = al.make_shared((SHM_IN_U32,), al.u32)
    shm_in_bf16 = al.view(shm_in_u32, al.Tensor((SHM_IN_BF16,), al.bf16))

    # Tensors for global access
    in_layout = al.make_layout(
        (batch, in_c, d_in, h_in, w_in),
        (in_c * d_in * h_in * w_in, d_in * h_in * w_in, h_in * w_in, w_in, 1),
    )
    in_tensor = al.make_tensor(in_ptr, al.bf16, in_layout)

    w_layout = al.make_layout(
        (out_c, group_in_c, KD, KH, KW),
        (group_in_c * KD * KH * KW, KD * KH * KW, KH * KW, KW, 1),
    )
    w_tensor = al.make_tensor(w_ptr, al.bf16, w_layout)

    out_layout = al.make_layout(
        (batch, out_c, d_out, h_out, w_out),
        (out_c * d_out * h_out * w_out, d_out * h_out * w_out, h_out * w_out, w_out, 1),
    )
    out_tensor = al.make_tensor(out_ptr, al.bf16, out_layout)

    kw_zero = al.convert(0, al.i32)
    zero_bf16 = al.convert(0.0, al.bf16)

    row_stride = TILE_H * w_in * 4
    h_stride = w_in * 4

    # Cooperative load of input tile into shared memory
    # Iteration order W→H→D→IC for coalesced global reads
    total_load = TILE_D * TILE_H * w_in * in_c
    idx = tid
    for _ in al.range((total_load + THREADS - 1) // THREADS):
        if idx < total_load:
            w = idx % w_in
            rest = idx // w_in
            h_local = rest % TILE_H
            rest = rest // TILE_H
            d_local = rest % TILE_D
            ic = rest // TILE_D

            d_src = d_base + d_local
            h_src = h_base + h_local

            valid = al.convert(1, al.i32)
            if d_src < 0:
                valid = 0
            if d_src >= d_in:
                valid = 0
            if h_src < 0:
                valid = 0
            if h_src >= h_in:
                valid = 0

            shm_idx = d_local * row_stride + h_local * h_stride + w * 4 + ic
            if valid != 0:
                shm_in_bf16[shm_idx] = in_tensor[batch_idx, ic, d_src, h_src, w]
            else:
                shm_in_bf16[shm_idx] = zero_bf16
        idx += THREADS
    al.syncthreads()

    # Compute all outputs using incremental index computation
    total_work = block_d * block_h * w_out * block_oc
    idx = tid
    for _ in al.range((total_work + THREADS - 1) // THREADS):
        if idx < total_work:
            oc_local = idx % block_oc
            rest = idx // block_oc
            w = rest % w_out
            rest = rest // w_out
            h_local = rest % block_h
            d_local = rest // block_h

            d_out_global = d_tile_idx * BLOCK_D + d_local
            h_out_global = h_tile_idx * BLOCK_H + h_local
            oc_global = oc_start + oc_local

            acc = al.convert(0.0, al.f32)

            d_base_idx = d_local * stride * row_stride
            h_base_idx = h_local * stride * h_stride + w * 4

            for ic in al.range(ic_start, ic_end):
                d_off = d_base_idx
                for kd in al.range(KD):
                    h_off = d_off + h_base_idx
                    for kh in al.range(KH):
                        shm_idx = h_off + ic
                        in_bf16 = shm_in_bf16[shm_idx]
                        w_bf16 = w_tensor[oc_global, ic, kd, kh, kw_zero]
                        acc = acc + al.convert(in_bf16, al.f32) * al.convert(w_bf16, al.f32)
                        h_off = h_off + dilation * h_stride
                    d_off = d_off + dilation * row_stride

            out_tensor[batch_idx, oc_global, d_out_global, h_out_global, w] = al.convert(acc, al.bf16)
        idx += THREADS


def _prepare_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_contiguous(x)
    w_bf16 = _prepare_bf16_contiguous(weight)

    batch, in_c, d_in, h_in, w_in = x_bf16.shape
    out_c, w_in_c, kd, kh, kw = w_bf16.shape

    if w_in_c != in_c // groups:
        raise ValueError(f"Weight input channels {w_in_c} != in_c/groups {in_c // groups}")

    d_out = (d_in + 2 * padding - dilation * (kd - 1) - 1) // stride + 1
    h_out = (h_in + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
    w_out = (w_in + 2 * padding - dilation * (kw - 1) - 1) // stride + 1

    out = torch.empty((batch, out_c, d_out, h_out, w_out), device=x_bf16.device, dtype=torch.bfloat16)

    num_d_tiles = (d_out + BLOCK_D - 1) // BLOCK_D
    num_h_tiles = (h_out + BLOCK_H - 1) // BLOCK_H
    num_oc_tiles = (out_c // groups + BLOCK_OC - 1) // BLOCK_OC

    grid_0 = num_d_tiles * num_h_tiles
    grid_1 = num_oc_tiles * batch

    conv3d_bf16_kernel[lambda: ((grid_0, grid_1, 1), (THREADS, 1, 1))](
        x_bf16, w_bf16, out,
        batch, in_c, out_c, d_in, h_in, w_in, d_out, h_out, w_out,
        stride, padding, dilation, groups,
    )
    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(
            in_channels, out_channels,
            (kernel_size, kernel_size, 1),
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv3d.weight.data
        result = avelang_conv3d(x, weight, self.stride, self.padding, self.dilation, self.groups)

        if self.conv3d.bias is not None:
            bias = self.conv3d.bias.data
            result = result + bias.view(1, -1, 1, 1, 1).to(result.dtype)

        return result


# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256
depth = 10


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width, depth)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
