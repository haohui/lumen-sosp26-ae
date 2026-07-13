import torch
import torch.nn as nn
import avelang
import avelang.language as al

_C_IN = 64
_KH = 3
_KW = 3
_TILE_H = 8
_TILE_W = 8
_TILE_OC = 4
_THREADS = _TILE_H * _TILE_W * _TILE_OC  # 256

_PATCH_H = _TILE_H + _KH - 1   # 10
_PATCH_W = _TILE_W + _KW - 1   # 10
_SHM_INPUT_SIZE = _C_IN * _PATCH_H * _PATCH_W   # 6400
_SHM_WEIGHT_SIZE = _TILE_OC * _C_IN * _KH * _KW  # 2304


@avelang.jit
def direct_conv2d_relu_bias_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    conv_bias_ptr: al.Pointer(al.bf16),
    extra_bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H: al.i32,
    W: al.i32,
    OH: al.i32,
    OW: al.i32,
    KH: al.i32,
    KW: al.i32,
    oc_blocks: al.i32,
):
    tid = al.thread_id(0)
    block_ow = al.block_id(0)
    block_oh = al.block_id(1)
    block_z = al.block_id(2)

    batch_idx = block_z // oc_blocks
    oc_block = block_z - batch_idx * oc_blocks

    start_h = block_oh * _TILE_H
    start_w = block_ow * _TILE_W
    oc_start = oc_block * _TILE_OC

    shm_input = al.make_shared((_SHM_INPUT_SIZE,), al.bf16)
    shm_weight = al.make_shared((_SHM_WEIGHT_SIZE,), al.bf16)

    x_memref = al.make_tensor(
        x_ptr, al.bf16, al.make_layout((N * C_in * H * W,), (1,)),
    )
    w_memref = al.make_tensor(
        w_ptr, al.bf16, al.make_layout((C_out * C_in * KH * KW,), (1,)),
    )

    for idx in al.range(tid, _SHM_INPUT_SIZE, _THREADS):
        ic = idx // (_PATCH_H * _PATCH_W)
        rem = idx - ic * _PATCH_H * _PATCH_W
        ph = rem // _PATCH_W
        pw = rem - ph * _PATCH_W
        h_in = start_h + ph
        w_in = start_w + pw
        if h_in < H:
            if w_in < W:
                x_idx = batch_idx * C_in * H * W + ic * H * W + h_in * W + w_in
                shm_input[idx] = x_memref[x_idx]

    for idx in al.range(tid, _SHM_WEIGHT_SIZE, _THREADS):
        w_idx = oc_start * C_in * KH * KW + idx
        shm_weight[idx] = w_memref[w_idx]

    al.syncthreads()

    tid_oc = tid // (_TILE_H * _TILE_W)
    tid_spatial = tid - tid_oc * _TILE_H * _TILE_W
    tid_oh = tid_spatial // _TILE_W
    tid_ow = tid_spatial - tid_oh * _TILE_W

    oc = oc_start + tid_oc
    oh = start_h + tid_oh
    ow = start_w + tid_ow

    if batch_idx < N:
        if oc < C_out:
            if oh < OH:
                if ow < OW:
                    cb_memref = al.make_tensor(
                        conv_bias_ptr, al.bf16, al.make_layout((C_out,), (1,)),
                    )
                    eb_memref = al.make_tensor(
                        extra_bias_ptr, al.bf16, al.make_layout((C_out,), (1,)),
                    )
                    out_memref = al.make_tensor(
                        out_ptr, al.bf16,
                        al.make_layout((N * C_out * OH * OW,), (1,)),
                    )

                    acc = al.convert(0.0, al.f32)

                    for ic in al.range(_C_IN):
                        for kh in al.range(_KH):
                            for kw in al.range(_KW):
                                in_idx = (
                                    ic * _PATCH_H * _PATCH_W
                                    + (tid_oh + kh) * _PATCH_W
                                    + (tid_ow + kw)
                                )
                                w_idx = (
                                    tid_oc * _C_IN * _KH * _KW
                                    + ic * _KH * _KW
                                    + kh * _KW
                                    + kw
                                )
                                x_val = al.convert(shm_input[in_idx], al.f32)
                                w_val = al.convert(shm_weight[w_idx], al.f32)
                                acc = acc + x_val * w_val

                    cb_val = al.convert(cb_memref[oc], al.f32)
                    acc = acc + cb_val

                    zero = al.convert(0.0, al.f32)
                    if acc < zero:
                        acc = zero

                    eb_val = al.convert(eb_memref[oc], al.f32)
                    acc = acc + eb_val

                    out_idx = (
                        batch_idx * C_out * OH * OW
                        + oc * OH * OW
                        + oh * OW
                        + ow
                    )
                    out_memref[out_idx] = al.convert(acc, al.bf16)


def _to_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv2d_relu_bias(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_bias: torch.Tensor,
    extra_bias: torch.Tensor,
) -> torch.Tensor:
    N, C_in, H, W_in = x.shape
    C_out, _, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W_in - KW + 1

    x_bf16 = _to_bf16_cuda_contiguous(x)
    w_bf16 = _to_bf16_cuda_contiguous(weight)
    cb_bf16 = _to_bf16_cuda_contiguous(conv_bias)
    eb_bf16 = _to_bf16_cuda_contiguous(extra_bias.reshape(C_out))

    out = torch.empty((N, C_out, OH, OW), device=x_bf16.device, dtype=torch.bfloat16)

    oc_blocks = (C_out + _TILE_OC - 1) // _TILE_OC
    grid_x = (OW + _TILE_W - 1) // _TILE_W
    grid_y = (OH + _TILE_H - 1) // _TILE_H
    grid_z = N * oc_blocks

    direct_conv2d_relu_bias_kernel[lambda: ((grid_x, grid_y, grid_z), (_THREADS, 1, 1))](
        x_bf16, w_bf16, cb_bf16, eb_bf16, out,
        N, C_in, C_out, H, W_in, OH, OW, KH, KW, oc_blocks,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        orig_dtype = x.dtype
        out_bf16 = avelang_conv2d_relu_bias(
            x, self.conv.weight, self.conv.bias, self.bias,
        )
        return out_bf16.to(orig_dtype)


# Preserve the original module-level contract from input_model.py
batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
