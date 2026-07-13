import torch
import torch.nn as nn
import avelang
import avelang.language as al

batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

TILE_H = 8
TILE_W = 8
TILE_OC = 4
THREADS = TILE_H * TILE_W * TILE_OC
KH = 3
KW = 3


@avelang.jit
def direct_conv_fused_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    conv_bias_ptr: al.Pointer(al.bf16),
    epilogue_bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    IC: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    const_val: al.f32,
    scale_val: al.f32,
):
    tid = al.thread_id(0)
    bid_x = al.block_id(0)
    bid_y = al.block_id(1)
    bid_z = al.block_id(2)

    n_idx = bid_x
    oc_group = bid_y
    spatial_idx = bid_z

    w_tiles_per_row = (W_out + TILE_W - 1) // TILE_W
    h_tile = spatial_idx // w_tiles_per_row
    w_tile = spatial_idx - h_tile * w_tiles_per_row

    h_base = h_tile * TILE_H
    w_base = w_tile * TILE_W
    oc_base = oc_group * TILE_OC

    local_h = tid // (TILE_W * TILE_OC)
    rem = tid - local_h * TILE_W * TILE_OC
    local_w = rem // TILE_OC
    local_oc = rem - local_w * TILE_OC

    h_out_idx = h_base + local_h
    w_out_idx = w_base + local_w
    oc = oc_base + local_oc

    if h_out_idx < H_out and w_out_idx < W_out and oc < OC and n_idx < N:
        inp = al.make_tensor(
            input_ptr, al.bf16,
            al.make_layout((N, IC, H, W), (IC * H * W, H * W, W, 1)),
        )
        wt_flat = al.make_tensor(
            weight_ptr, al.bf16,
            al.make_layout((OC * IC * KH * KW,), (1,)),
        )
        cb_t = al.make_tensor(
            conv_bias_ptr, al.bf16,
            al.make_layout((OC,), (1,)),
        )
        eb_t = al.make_tensor(
            epilogue_bias_ptr, al.bf16,
            al.make_layout((OC,), (1,)),
        )
        out_t = al.make_tensor(
            out_ptr, al.bf16,
            al.make_layout(
                (N, OC, H_out, W_out),
                (OC * H_out * W_out, H_out * W_out, W_out, 1),
            ),
        )

        acc = al.convert(0.0, al.f32)
        oc_base_k = oc * IC * KH * KW

        for ic in al.range(IC):
            ic_base = oc_base_k + ic * KH * KW
            for kh in al.range(KH):
                in_h = h_out_idx + kh
                kh_base = ic_base + kh * KW
                for kw in al.range(KW):
                    in_w = w_out_idx + kw
                    w_idx = kh_base + kw
                    acc = acc + al.convert(wt_flat[w_idx], al.f32) * al.convert(inp[n_idx, ic, in_h, in_w], al.f32)

        acc = acc + al.convert(cb_t[oc], al.f32)
        if acc > const_val:
            acc = const_val
        acc = acc + al.convert(eb_t[oc], al.f32)
        acc = acc * scale_val

        out_t[n_idx, oc, h_out_idx, w_out_idx] = al.convert(acc, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_bias: torch.Tensor,
    epilogue_bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight).reshape(-1).contiguous()
    cb_bf16 = _prepare_bf16_cuda_contiguous(conv_bias).reshape(-1).contiguous()
    eb_bf16 = _prepare_bf16_cuda_contiguous(epilogue_bias).reshape(-1).contiguous()

    N_val, C_val, H_val, W_val = x_bf16.shape
    OC_val = weight_bf16.shape[0] // (C_val * KH * KW)
    H_out_val = H_val - KH + 1
    W_out_val = W_val - KW + 1

    out = torch.empty(
        N_val, OC_val, H_out_val, W_out_val,
        device=x_bf16.device, dtype=torch.bfloat16,
    )

    h_tiles = (H_out_val + TILE_H - 1) // TILE_H
    w_tiles = (W_out_val + TILE_W - 1) // TILE_W
    oc_groups = (OC_val + TILE_OC - 1) // TILE_OC
    grid = (N_val, oc_groups, h_tiles * w_tiles)

    direct_conv_fused_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, weight_bf16, cb_bf16, eb_bf16, out,
        N_val, C_val, OC_val, H_val, W_val,
        H_out_val, W_out_val,
        float(constant_value), float(scaling_factor),
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        return avelang_conv_fused(x, self.conv.weight, self.conv.bias, self.bias)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]
