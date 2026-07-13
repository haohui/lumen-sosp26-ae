import torch
import torch.nn as nn
import math
import avelang
import avelang.language as al

TILE_H: al.constexpr = 8
TILE_W: al.constexpr = 8
TILE_OC: al.constexpr = 4
THREADS: al.constexpr = 256
SHM_SIZE: al.constexpr = 8192


@avelang.jit
def conv2d_shm_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_in: al.i32,
    H: al.i32,
    W: al.i32,
    OC: al.i32,
    KH: al.i32,
    KW: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    stride_h: al.i32,
    stride_w: al.i32,
    pad_h: al.i32,
    pad_w: al.i32,
    dil_h: al.i32,
    dil_w: al.i32,
    groups: al.i32,
    oc_groups: al.i32,
    C_in_per_group: al.i32,
    OC_per_group: al.i32,
):
    tid = al.thread_id(0)
    block_w = al.block_id(0)
    block_h = al.block_id(1)
    block_bgoc = al.block_id(2)

    g_oc_total = groups * oc_groups
    b = block_bgoc // g_oc_total
    residual = block_bgoc % g_oc_total
    g = residual // oc_groups
    oc_block = residual % oc_groups
    oc_start = oc_block * TILE_OC

    h_start = block_h * TILE_H
    w_start = block_w * TILE_W

    local_hw = tid // TILE_OC
    local_h = local_hw // TILE_W
    local_w = local_hw % TILE_W
    local_oc = tid % TILE_OC

    h_out = h_start + local_h
    w_out = w_start + local_w
    oc_local = oc_start + local_oc

    ic_start = g * C_in_per_group
    oc_global = g * OC_per_group + oc_local

    win_h = (TILE_H - 1) * stride_h + dil_h * (KH - 1) + 1
    win_w = (TILE_W - 1) * stride_w + dil_w * (KW - 1) + 1
    win_total = C_in_per_group * win_h * win_w

    shm = al.make_shared((SHM_SIZE,), al.bf16)
    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((B, C_in, H, W), (C_in * H * W, H * W, W, 1)))
    w = al.make_tensor(w_ptr, al.bf16, al.make_layout((OC, C_in_per_group, KH, KW), (C_in_per_group * KH * KW, KH * KW, KW, 1)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((B, OC, H_out, W_out), (OC * H_out * W_out, H_out * W_out, W_out, 1)))

    load_iters = (win_total + THREADS - 1) // THREADS
    idx = tid
    for _ in al.range(load_iters):
        if idx < win_total:
            ic = idx // (win_h * win_w)
            hw = idx % (win_h * win_w)
            h_off = hw // win_w
            w_off = hw % win_w
            h_in = h_start * stride_h + h_off - pad_h
            w_in = w_start * stride_w + w_off - pad_w
            ic_global = ic_start + ic
            if h_in >= 0:
                if h_in < H:
                    if w_in >= 0:
                        if w_in < W:
                            shm[idx] = x[b, ic_global, h_in, w_in]
                        else:
                            shm[idx] = al.convert(0, al.bf16)
                    else:
                        shm[idx] = al.convert(0, al.bf16)
                else:
                    shm[idx] = al.convert(0, al.bf16)
            else:
                shm[idx] = al.convert(0, al.bf16)
        idx = idx + THREADS

    al.syncthreads()

    if oc_local < OC_per_group:
        if h_out < H_out:
            if w_out < W_out:
                acc = al.convert(0.0, al.f32)
                for ic in al.range(C_in_per_group):
                    for kh in al.range(KH):
                        h_off = local_h * stride_h + kh * dil_h
                        for kw in al.range(KW):
                            w_off = local_w * stride_w + kw * dil_w
                            shm_idx = ic * win_h * win_w + h_off * win_w + w_off
                            inp = al.convert(shm[shm_idx], al.f32)
                            wt = al.convert(w[oc_global, ic, kh, kw], al.f32)
                            acc = acc + inp * wt
                out[b, oc_global, h_out, w_out] = al.convert(acc, al.bf16)


def _prepare_bf16_cuda_contiguous(t):
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def _normalize_pair(value, name):
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, tuple) and len(value) == 2:
        return value
    raise ValueError(f"{name} must be int or tuple of length 2, got {value}")


def avelang_conv2d(x, weight, stride, padding, dilation, groups):
    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = _prepare_bf16_cuda_contiguous(weight)
    B, C_in, H, W = x_bf16.shape
    OC, _, KH, KW = w_bf16.shape
    sh, sw = _normalize_pair(stride, "stride")
    ph, pw = _normalize_pair(padding, "padding")
    dh, dw = _normalize_pair(dilation, "dilation")
    C_in_per_group = C_in // groups
    OC_per_group = OC // groups
    H_out = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    W_out = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1
    oc_groups = (OC_per_group + TILE_OC - 1) // TILE_OC
    out = torch.empty((B, OC, H_out, W_out), device=x_bf16.device, dtype=torch.bfloat16)
    grid_x = (W_out + TILE_W - 1) // TILE_W
    grid_y = (H_out + TILE_H - 1) // TILE_H
    grid_z = B * groups * oc_groups
    conv2d_shm_kernel[lambda: ((grid_x, grid_y, grid_z), (THREADS, 1, 1))](
        x_bf16, w_bf16, out,
        B, C_in, H, W, OC, KH, KW, H_out, W_out,
        sh, sw, ph, pw, dh, dw,
        groups, oc_groups, C_in_per_group, OC_per_group,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        out = avelang_conv2d(x, self.weight, self.stride, self.padding, self.dilation, self.groups)
        if self.bias is not None:
            out = out + self.bias.to(dtype=out.dtype).view(1, -1, 1, 1)
        return out
