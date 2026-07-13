import math
import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_H = 12
TILE_W = 12
OC_PER_BLOCK = 64
THREADS = 256
TILE_SIZE = OC_PER_BLOCK * TILE_H * TILE_W
SPATIAL_PER_TILE = TILE_H * TILE_W
NUM_ELEMS_PER_THREAD = (TILE_SIZE + THREADS - 1) // THREADS


@avelang.jit
def tconv_bias_tanh_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    IC: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
    K: al.i32,
    OH: al.i32,
    OW: al.i32,
    stride: al.i32,
    pad: al.i32,
    num_w_tiles: al.i32,
):
    tid = al.thread_id(0)
    batch = al.block_id(0)
    spatial_tile = al.block_id(1)
    oc_tile = al.block_id(2)

    h_tile = spatial_tile // num_w_tiles
    w_tile = spatial_tile % num_w_tiles

    oh_start = h_tile * TILE_H
    ow_start = w_tile * TILE_W
    oc_start = oc_tile * OC_PER_BLOCK

    x_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((N * IC * H * W,), (1,)))
    w_flat = al.make_tensor(w_ptr, al.bf16, al.make_layout((IC * OC * K * K,), (1,)))
    bias_flat = al.make_tensor(bias_ptr, al.bf16, al.make_layout((OC,), (1,)))
    out_flat = al.make_tensor(out_ptr, al.bf16, al.make_layout((N * OC * OH * OW,), (1,)))

    ic_stride_w = OC * K * K
    kh_stride_w = K * K
    ic_stride_x = H * W
    batch_stride_x = IC * H * W
    oc_stride_out = OH * OW
    batch_stride_out = OC * OH * OW

    batch_offset_x = batch * batch_stride_x
    batch_offset_out = batch * batch_stride_out

    stride_i32 = al.convert(stride, al.i32)
    pad_i32 = al.convert(pad, al.i32)
    zero_i32 = al.convert(0, al.i32)
    zero_f32 = al.convert(0.0, al.f32)

    for elem_i in al.range(NUM_ELEMS_PER_THREAD):
        idx = tid + elem_i * THREADS
        if idx < TILE_SIZE:
            oc_local = idx // SPATIAL_PER_TILE
            spatial_local = idx % SPATIAL_PER_TILE
            oh_local = spatial_local // TILE_W
            ow_local = spatial_local % TILE_W

            oc = oc_start + oc_local
            oh = oh_start + oh_local
            ow = ow_start + ow_local

            if (batch < N) and (oc < OC) and (oh < OH) and (ow < OW):
                acc = zero_f32
                oc_w_base = oc * kh_stride_w

                for ic in al.range(IC):
                    ic_w_base = ic * ic_stride_w + oc_w_base
                    ic_x_base = batch_offset_x + ic * ic_stride_x

                    for kh in al.range(K):
                        oh_idx = oh + pad_i32 - kh
                        oh_rem = oh_idx % stride_i32
                        if oh_rem == zero_i32:
                            ih = oh_idx // stride_i32
                            if (ih >= zero_i32) and (ih < H):
                                for kw in al.range(K):
                                    ow_idx = ow + pad_i32 - kw
                                    ow_rem = ow_idx % stride_i32
                                    if ow_rem == zero_i32:
                                        iw = ow_idx // stride_i32
                                        if (iw >= zero_i32) and (iw < W):
                                            w_idx = ic_w_base + kh * K + kw
                                            x_idx = ic_x_base + ih * W + iw
                                            w_val = al.convert(w_flat[w_idx], al.f32)
                                            x_val = al.convert(x_flat[x_idx], al.f32)
                                            acc = acc + w_val * x_val

                bias_val = al.convert(bias_flat[oc], al.f32)
                acc = acc + bias_val
                result = al.tanh(acc)

                out_idx = batch_offset_out + oc * oc_stride_out + oh * OW + ow
                out_flat[out_idx] = al.convert(result, al.bf16)


def _ensure_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    return t.contiguous().to(dtype=torch.bfloat16)


def _launch_tconv(
    x: torch.Tensor,
    weight: torch.Tensor,
    combined_bias: torch.Tensor,
) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda and combined_bias.is_cuda

    N, IC, H, W_in = x.shape
    w_IC, OC, K, w_K = weight.shape
    assert IC == w_IC and K == w_K

    stride = 2
    pad = 1
    output_padding = 1
    OH = (H - 1) * stride - 2 * pad + K + output_padding
    OW = (W_in - 1) * stride - 2 * pad + K + output_padding

    x_bf16 = _ensure_bf16_contiguous(x)
    w_bf16 = _ensure_bf16_contiguous(weight)
    bias_bf16 = _ensure_bf16_contiguous(combined_bias)

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.bfloat16)

    num_h_tiles = (OH + TILE_H - 1) // TILE_H
    num_w_tiles = (OW + TILE_W - 1) // TILE_W
    num_oc_tiles = (OC + OC_PER_BLOCK - 1) // OC_PER_BLOCK
    num_spatial_tiles = num_h_tiles * num_w_tiles

    grid = (N, num_spatial_tiles, num_oc_tiles)
    block = (THREADS, 1, 1)

    tconv_bias_tanh_kernel[lambda: (grid, block)](
        x_bf16,
        w_bf16,
        bias_bf16,
        out,
        N, IC, OC, H, W_in, K, OH, OW,
        stride, pad,
        num_w_tiles,
    )
    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias_shape: tuple,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
    ):
        super(ModelNew, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        self.conv_weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        )
        self.conv_bias = nn.Parameter(torch.empty(out_channels))

        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
        fan_in = in_channels * kernel_size * kernel_size
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.conv_bias, -bound, bound)

        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        combined_bias = self.conv_bias - self.bias.view(self.out_channels)
        return _launch_tconv(x, self.conv_weight, combined_bias)
