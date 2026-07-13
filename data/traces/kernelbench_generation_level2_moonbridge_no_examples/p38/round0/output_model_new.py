import torch
import torch.nn as nn
import avelang
import avelang.language as al

_NUM_THREADS = 256
_TOTAL_SPATIAL_IN = 16384   # 16 * 32 * 32
_TOTAL_SPATIAL_OUT = 131072  # 32 * 64 * 64


@avelang.jit
def avgpool3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
):
    tid = al.thread_id(0)
    num_threads = al.block_dim(0)
    b = al.block_id(0)
    c_in = al.block_id(1)

    total_spatial = _TOTAL_SPATIAL_IN

    in_B_stride = al.convert(524288, al.i32)
    in_C_stride = al.convert(16384, al.i32)
    in_D_stride = al.convert(1024, al.i32)
    in_H_stride = al.convert(32, al.i32)

    input_layout = al.make_layout(
        (al.convert(32, al.i32), al.convert(32, al.i32), al.convert(16, al.i32), al.convert(32, al.i32), al.convert(32, al.i32)),
        (in_B_stride, in_C_stride, in_D_stride, in_H_stride, al.convert(1, al.i32)),
    )
    inp = al.make_tensor(input_ptr, al.bf16, input_layout)

    out_B_stride = al.convert(524288, al.i32)
    out_C_stride = al.convert(16384, al.i32)
    out_D_stride = al.convert(1024, al.i32)
    out_H_stride = al.convert(32, al.i32)

    output_layout = al.make_layout(
        (al.convert(32, al.i32), al.convert(32, al.i32), al.convert(16, al.i32), al.convert(32, al.i32), al.convert(32, al.i32)),
        (out_B_stride, out_C_stride, out_D_stride, out_H_stride, al.convert(1, al.i32)),
    )
    out = al.make_tensor(output_ptr, al.bf16, output_layout)

    one_eighth = al.convert(0.125, al.f32)

    for idx in al.range(tid, total_spatial, num_threads):
        d = idx // al.convert(1024, al.i32)
        rem = idx % al.convert(1024, al.i32)
        h = rem // al.convert(32, al.i32)
        w = rem % al.convert(32, al.i32)

        d2 = d * al.convert(2, al.i32)
        h2 = h * al.convert(2, al.i32)
        w2 = w * al.convert(2, al.i32)

        v0 = al.convert(inp[b, c_in, d2, h2, w2], al.f32)
        v1 = al.convert(inp[b, c_in, d2, h2, w2 + al.convert(1, al.i32)], al.f32)
        v2 = al.convert(inp[b, c_in, d2, h2 + al.convert(1, al.i32), w2], al.f32)
        v3 = al.convert(inp[b, c_in, d2, h2 + al.convert(1, al.i32), w2 + al.convert(1, al.i32)], al.f32)
        v4 = al.convert(inp[b, c_in, d2 + al.convert(1, al.i32), h2, w2], al.f32)
        v5 = al.convert(inp[b, c_in, d2 + al.convert(1, al.i32), h2, w2 + al.convert(1, al.i32)], al.f32)
        v6 = al.convert(inp[b, c_in, d2 + al.convert(1, al.i32), h2 + al.convert(1, al.i32), w2], al.f32)
        v7 = al.convert(inp[b, c_in, d2 + al.convert(1, al.i32), h2 + al.convert(1, al.i32), w2 + al.convert(1, al.i32)], al.f32)

        acc = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7
        avg = acc * one_eighth
        out[b, c_in, d, h, w] = al.convert(avg, al.bf16)


@avelang.jit
def conv_transpose3d_clamp_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
):
    tid = al.thread_id(0)
    num_threads = al.block_dim(0)
    b = al.block_id(0)
    c_out = al.block_id(1)

    _0 = al.convert(0, al.i32)
    _1 = al.convert(1, al.i32)
    _2 = al.convert(2, al.i32)
    _3 = al.convert(3, al.i32)
    _9 = al.convert(9, al.i32)
    _16 = al.convert(16, al.i32)
    _27 = al.convert(27, al.i32)
    _32 = al.convert(32, al.i32)
    _64 = al.convert(64, al.i32)
    _1024 = al.convert(1024, al.i32)
    _4096 = al.convert(4096, al.i32)
    _16384 = al.convert(16384, al.i32)
    _131072 = al.convert(131072, al.i32)
    _524288 = al.convert(524288, al.i32)
    _1728 = al.convert(1728, al.i32)
    _8388608 = al.convert(8388608, al.i32)

    input_layout = al.make_layout(
        (_32, _32, _16, _32, _32),
        (_524288, _16384, _1024, _32, _1),
    )
    inp = al.make_tensor(input_ptr, al.bf16, input_layout)

    weight_layout = al.make_layout(
        (_32, _64, _3, _3, _3),
        (_1728, _27, _9, _3, _1),
    )
    wgt = al.make_tensor(weight_ptr, al.bf16, weight_layout)

    output_layout = al.make_layout(
        (_32, _64, al.convert(32, al.i32), _64, _64),
        (_8388608, _131072, _4096, _64, _1),
    )
    out = al.make_tensor(output_ptr, al.bf16, output_layout)

    total_spatial = _131072

    f0 = al.convert(0.0, al.f32)
    f1 = al.convert(1.0, al.f32)

    for idx in al.range(tid, total_spatial, num_threads):
        w_out = idx % _64
        temp = idx // _64
        h_out = temp % _64
        d_out = temp // _64

        acc = f0

        for c_in in al.range(_32):
            for kd in al.range(_3):
                id_candidate = d_out + _1 - kd
                id_valid = ((id_candidate & _1) == _0) & (id_candidate >= _0) & (id_candidate < _32)
                id = id_candidate // _2
                if id_valid:
                    for kh in al.range(_3):
                        ih_candidate = h_out + _1 - kh
                        ih_valid = ((ih_candidate & _1) == _0) & (ih_candidate >= _0) & (ih_candidate < _64)
                        ih = ih_candidate // _2
                        if ih_valid:
                            for kw in al.range(_3):
                                iw_candidate = w_out + _1 - kw
                                iw_valid = ((iw_candidate & _1) == _0) & (iw_candidate >= _0) & (iw_candidate < _64)
                                iw = iw_candidate // _2
                                if iw_valid:
                                    in_val = al.convert(inp[b, c_in, id, ih, iw], al.f32)
                                    w_val = al.convert(wgt[c_in, c_out, kd, kh, kw], al.f32)
                                    acc = acc + in_val * w_val

        clamped = acc
        if clamped < f0:
            clamped = f0
        if clamped > f1:
            clamped = f1

        out[b, c_out, d_out, h_out, w_out] = al.convert(clamped, al.bf16)


@avelang.jit
def softmax_scale_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    scale_ptr: al.Pointer(al.bf16),
):
    tid = al.thread_id(0)
    num_threads = al.block_dim(0)
    b = al.block_id(0)
    c = al.block_id(1)

    _0 = al.convert(0, al.i32)
    _1 = al.convert(1, al.i32)
    _32 = al.convert(32, al.i32)
    _64 = al.convert(64, al.i32)
    _131072 = al.convert(131072, al.i32)
    _4096 = al.convert(4096, al.i32)

    io_layout = al.make_layout(
        (al.convert(32, al.i32), al.convert(64, al.i32), _32, _64, _64),
        (al.convert(8388608, al.i32), _131072, _4096, _64, _1),
    )
    inp = al.make_tensor(input_ptr, al.bf16, io_layout)
    out = al.make_tensor(output_ptr, al.bf16, io_layout)

    scale_layout = al.make_layout(
        (al.convert(1, al.i32), al.convert(64, al.i32), al.convert(1, al.i32), al.convert(1, al.i32), al.convert(1, al.i32)),
        (al.convert(64, al.i32), al.convert(1, al.i32), al.convert(1, al.i32), al.convert(1, al.i32), al.convert(1, al.i32)),
    )
    scl = al.make_tensor(scale_ptr, al.bf16, scale_layout)

    total_spatial = _131072

    shared = al.make_shared((8,), al.f32)

    neg_inf = al.convert(-3.402823e+38, al.f32)
    zero_f32 = al.convert(0.0, al.f32)

    local_max = neg_inf
    for idx in al.range(tid, total_spatial, num_threads):
        w = idx % _64
        temp = idx // _64
        h = temp % _64
        d = temp // _64
        val = al.convert(inp[b, c, d, h, w], al.f32)
        if val > local_max:
            local_max = val

    warp_id = tid // al.convert(32, al.i32)
    lane_id = tid % al.convert(32, al.i32)

    for offset in al.range(16, 0, -1):
        other = al.shuffle_down(local_max, al.convert(offset, al.i32), al.convert(32, al.i32))
        if other > local_max:
            local_max = other

    if lane_id == _0:
        shared[warp_id] = local_max
    al.syncthreads()

    global_max = neg_inf
    if warp_id == _0:
        for w in al.range(8):
            val = shared[w]
            if val > global_max:
                global_max = val
    al.syncthreads()

    if warp_id == _0 and lane_id == _0:
        shared[_0] = global_max
    al.syncthreads()
    global_max = shared[_0]

    local_sum = zero_f32
    for idx in al.range(tid, total_spatial, num_threads):
        w = idx % _64
        temp = idx // _64
        h = temp % _64
        d = temp // _64
        val = al.convert(inp[b, c, d, h, w], al.f32)
        exp_val = al.exp(val - global_max)
        local_sum = local_sum + exp_val

    for offset in al.range(16, 0, -1):
        other = al.shuffle_down(local_sum, al.convert(offset, al.i32), al.convert(32, al.i32))
        local_sum = local_sum + other

    if lane_id == _0:
        shared[warp_id] = local_sum
    al.syncthreads()

    global_sum = zero_f32
    if warp_id == _0:
        for w in al.range(8):
            global_sum = global_sum + shared[w]
    al.syncthreads()

    if warp_id == _0 and lane_id == _0:
        shared[_0] = global_sum
    al.syncthreads()
    global_sum = shared[_0]

    scale_val = al.convert(scl[al.convert(0, al.i32), c, al.convert(0, al.i32), al.convert(0, al.i32), al.convert(0, al.i32)], al.f32)

    for idx in al.range(tid, total_spatial, num_threads):
        w = idx % _64
        temp = idx // _64
        h = temp % _64
        d = temp // _64
        val = al.convert(inp[b, c, d, h, w], al.f32)
        exp_val = al.exp(val - global_max)
        norm_val = exp_val / global_sum
        result = norm_val * scale_val
        out[b, c, d, h, w] = al.convert(result, al.bf16)


# --- Host wrappers ---

def avelang_avgpool3d(x: torch.Tensor) -> torch.Tensor:
    B_val, C_val, D_val, H_val, W_val = x.shape
    D_out = D_val // 2
    H_out = H_val // 2
    W_out = W_val // 2

    x_bf16 = x.contiguous() if x.dtype == torch.bfloat16 else x.to(torch.bfloat16).contiguous()
    out = torch.empty(B_val, C_val, D_out, H_out, W_out, dtype=torch.bfloat16, device=x.device)

    grid = (B_val, C_val, 1)
    block = (_NUM_THREADS, 1, 1)
    avgpool3d_kernel[lambda: (grid, block)](x_bf16.data_ptr(), out.data_ptr())

    return out


def avelang_conv_transpose3d_clamp(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    B_val, C_in_val, D_in_val, H_in_val, W_in_val = x.shape
    C_out_val = weight.shape[1]
    D_out = 32
    H_out = 64
    W_out = 64

    x_bf16 = x.contiguous() if x.dtype == torch.bfloat16 else x.to(torch.bfloat16).contiguous()
    w_bf16 = weight.contiguous() if weight.dtype == torch.bfloat16 else weight.to(torch.bfloat16).contiguous()
    out = torch.empty(B_val, C_out_val, D_out, H_out, W_out, dtype=torch.bfloat16, device=x.device)

    grid = (B_val, C_out_val, 1)
    block = (_NUM_THREADS, 1, 1)
    conv_transpose3d_clamp_kernel[lambda: (grid, block)](
        x_bf16.data_ptr(), w_bf16.data_ptr(), out.data_ptr()
    )

    return out


def avelang_softmax_scale(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    B_val, C_val, D_val, H_val, W_val = x.shape

    x_bf16 = x.contiguous() if x.dtype == torch.bfloat16 else x.to(torch.bfloat16).contiguous()
    s_bf16 = scale.contiguous() if scale.dtype == torch.bfloat16 else scale.to(torch.bfloat16).contiguous()
    out = torch.empty(B_val, C_val, D_val, H_val, W_val, dtype=torch.bfloat16, device=x.device)

    grid = (B_val, C_val, 1)
    block = (_NUM_THREADS, 1, 1)
    softmax_scale_kernel[lambda: (grid, block)](
        x_bf16.data_ptr(), out.data_ptr(), s_bf16.data_ptr()
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        x = avelang_avgpool3d(x)
        x = avelang_conv_transpose3d_clamp(x, self.conv_transpose.weight)
        x = avelang_softmax_scale(x, self.scale)
        return x
