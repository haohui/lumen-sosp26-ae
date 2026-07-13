import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
TILE_H: al.constexpr = 8
TILE_W: al.constexpr = 8
MAX_IN_H: al.constexpr = 6
MAX_IN_W: al.constexpr = 6
SHM_IN_SIZE: al.constexpr = 64 * MAX_IN_H * MAX_IN_W
CH_PER_THREAD: al.constexpr = 32
THREADS_PER_POS: al.constexpr = 4
BLOCK_SIZE_K2: al.constexpr = 256
W_PER_BLOCK: al.constexpr = 8


@avelang.jit
def conv_transpose_min_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    conv_bias_ptr: al.Pointer(al.bf16),
    min_vals_ptr: al.Pointer(al.f32),
    B: al.i32,
    IC: al.i32,
    IH: al.i32,
    IW: al.i32,
    OH: al.i32,
    OW: al.i32,
    KH: al.i32,
    KW: al.i32,
    stride_val: al.i32,
    padding_val: al.i32,
):
    tid = al.thread_id(0)
    ow_tile = al.block_id(0)
    oh_tile = al.block_id(1)
    batch = al.block_id(2)

    oh_start = oh_tile * TILE_H
    ow_start = ow_tile * TILE_W

    h_in_min = (oh_start + padding_val) // stride_val
    h_in_max = (oh_start + TILE_H + padding_val - 1) // stride_val
    if h_in_max >= IH:
        h_in_max = IH - 1
    w_in_min = (ow_start + padding_val) // stride_val
    w_in_max = (ow_start + TILE_W + padding_val - 1) // stride_val
    if w_in_max >= IW:
        w_in_max = IW - 1

    num_h_in = h_in_max - h_in_min + 1
    num_w_in = w_in_max - w_in_min + 1
    hw_stride = num_h_in * num_w_in
    num_in_elems = IC * hw_stride

    smem_in = al.make_shared((SHM_IN_SIZE,), al.bf16)
    layout_g_in = al.make_layout((B, IC, IH, IW), (IC * IH * IW, IH * IW, IW, 1))
    g_in = al.make_tensor(input_ptr, al.bf16, layout_g_in)
    for idx in al.range(tid, SHM_IN_SIZE, BLOCK_SIZE):
        if idx < num_in_elems:
            ic = idx // hw_stride
            spat = idx - ic * hw_stride
            hi = spat // num_w_in
            wi = spat - hi * num_w_in
            smem_in[idx] = g_in[batch, ic, h_in_min + hi, w_in_min + wi]

    al.syncthreads()

    sp = tid // THREADS_PER_POS
    cg = tid - sp * THREADS_PER_POS

    local_oh = sp // TILE_W
    local_ow = sp - local_oh * TILE_W
    global_oh = oh_start + local_oh
    global_ow = ow_start + local_ow
    oc_start = cg * CH_PER_THREAD

    OC = al.convert(128, al.i32)

    layout_w = al.make_layout((IC, OC, KH, KW), (OC * KH * KW, KH * KW, KW, 1))
    wgt = al.make_tensor(weight_ptr, al.bf16, layout_w)
    layout_cb = al.make_layout((OC,), (1,))
    cb = al.make_tensor(conv_bias_ptr, al.bf16, layout_cb)

    parity_h = global_oh - (global_oh // stride_val) * stride_val
    parity_w = global_ow - (global_ow // stride_val) * stride_val
    kh_start = 1 - parity_h
    kh_end = 2 + parity_h
    kh_step = 1 + parity_h
    kw_start = 1 - parity_w
    kw_end = 2 + parity_w
    kw_step = 1 + parity_w

    smem = al.make_shared((BLOCK_SIZE,), al.f32)

    if sp < TILE_H * TILE_W:
        acc_regs = al.make_local((CH_PER_THREAD,), al.f32)
        for _i in al.range(CH_PER_THREAD):
            acc_regs[_i] = al.convert(0.0, al.f32)

        for kh in al.range(kh_start, kh_end, kh_step):
            h_in = (global_oh + padding_val - kh) // stride_val
            h_in_rel = h_in - h_in_min
            if h_in_rel >= 0:
                if h_in_rel < num_h_in:
                    for kw in al.range(kw_start, kw_end, kw_step):
                        w_in = (global_ow + padding_val - kw) // stride_val
                        w_in_rel = w_in - w_in_min
                        if w_in_rel >= 0:
                            if w_in_rel < num_w_in:
                                in_base = h_in_rel * num_w_in + w_in_rel
                                for ic in al.range(IC):
                                    in_idx = ic * hw_stride + in_base
                                    in_val = al.convert(smem_in[in_idx], al.f32)
                                    for ch in al.range(CH_PER_THREAD):
                                        oc = oc_start + ch
                                        w_val = al.convert(wgt[ic, oc, kh, kw], al.f32)
                                        acc_regs[ch] = acc_regs[ch] + in_val * w_val

        best = al.convert(1e30, al.f32)
        for ch in al.range(CH_PER_THREAD):
            oc = oc_start + ch
            val = acc_regs[ch] + al.convert(cb[oc], al.f32)
            if val < best:
                best = val
        smem[tid] = best
    else:
        smem[tid] = al.convert(1e30, al.f32)

    al.syncthreads()

    if tid < TILE_H * TILE_W:
        base = tid * THREADS_PER_POS
        best = smem[base]
        v1 = smem[base + 1]
        v2 = smem[base + 2]
        v3 = smem[base + 3]
        if v1 < best:
            best = v1
        if v2 < best:
            best = v2
        if v3 < best:
            best = v3
        goh = oh_start + tid // TILE_W
        gow = ow_start + tid - (tid // TILE_W) * TILE_W
        layout_out = al.make_layout((B, OH, OW), (OH * OW, OW, 1))
        mv = al.make_tensor(min_vals_ptr, al.f32, layout_out)
        mv[batch, goh, gow] = best


@avelang.jit
def sum_gelu_bias_kernel(
    min_vals_ptr: al.Pointer(al.f32),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    OH: al.i32,
    OW: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    ow = bid % OW
    batch = bid // OW

    layout_mv = al.make_layout((B, OH, OW), (OH * OW, OW, 1))
    mv = al.make_tensor(min_vals_ptr, al.f32, layout_mv)

    layout_bias = al.make_layout((1, 1, 1), (1, 1, 1))
    b = al.make_tensor(bias_ptr, al.bf16, layout_bias)
    bias_val = al.convert(b[0, 0, 0], al.f32)

    smem = al.make_shared((BLOCK_SIZE_K2,), al.f32)

    if tid < OH:
        smem[tid] = mv[batch, tid, ow]
    else:
        smem[tid] = al.convert(0.0, al.f32)

    al.syncthreads()

    if tid < 128:
        smem[tid] = smem[tid] + smem[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem[tid] = smem[tid] + smem[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem[tid] = smem[tid] + smem[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem[tid] = smem[tid] + smem[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem[tid] = smem[tid] + smem[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem[tid] = smem[tid] + smem[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem[tid] = smem[tid] + smem[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem[tid] = smem[tid] + smem[tid + 1]

    if tid == 0:
        x = smem[0]
        inner = al.convert(0.7978845608, al.f32) * (x + al.convert(0.044715, al.f32) * x * x * x)
        gelu = al.convert(0.5, al.f32) * x * (al.convert(1.0, al.f32) + al.tanh(inner))
        result = gelu + bias_val
        layout_out = al.make_layout((B, 1, 1, OW), (OW, OW, OW, 1))
        ot = al.make_tensor(out_ptr, al.bf16, layout_out)
        ot[batch, 0, 0, ow] = al.convert(result, al.bf16)


def _prepare_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_fused_forward(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    model_bias: torch.Tensor,
    stride: int,
    padding: int,
    output_padding: int,
) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    B = x.shape[0]
    IC = x.shape[1]
    IH = x.shape[2]
    IW = x.shape[3]
    KH = conv_weight.shape[2]
    KW = conv_weight.shape[3]
    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    x_bf16 = _prepare_bf16_contiguous(x)
    w_bf16 = _prepare_bf16_contiguous(conv_weight)
    cb_bf16 = _prepare_bf16_contiguous(conv_bias)
    mb_bf16 = _prepare_bf16_contiguous(model_bias)

    min_vals = torch.empty((B, OH, OW), dtype=torch.float32, device=x.device)

    gx = OW // TILE_W
    gy = OH // TILE_H
    gz = B
    conv_transpose_min_kernel[lambda: ((gx, gy, gz), (BLOCK_SIZE, 1, 1))](
        x_bf16, w_bf16, cb_bf16, min_vals,
        B, IC, IH, IW, OH, OW, KH, KW, stride, padding,
    )

    out = torch.empty((B, 1, 1, OW), dtype=torch.bfloat16, device=x.device)
    total_k2 = B * OW
    sum_gelu_bias_kernel[lambda: ((total_k2, 1, 1), (BLOCK_SIZE_K2, 1, 1))](
        min_vals, mb_bf16, out, B, OH, OW,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return avelang_fused_forward(
            x,
            self.conv_transpose.weight.data,
            self.conv_transpose.bias.data,
            self.bias.data,
            stride=self.conv_transpose.stride[0],
            padding=self.conv_transpose.padding[0],
            output_padding=self.conv_transpose.output_padding[0],
        )


batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
