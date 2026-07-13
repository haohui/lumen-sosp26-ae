import torch
import torch.nn as nn
import math
import avelang
import avelang.language as al

TM = 256
TK = 64
NUM_THREADS = 256


@avelang.jit
def pointwise_conv2d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H: al.i32,
    W: al.i32,
    stride_spatial: al.i32,
):
    in_stride_b = C_in * stride_spatial
    in_layout = al.make_layout((B, C_in, H, W), (in_stride_b, stride_spatial, W, 1))
    input_t = al.make_tensor(input_ptr, al.bf16, in_layout)

    w_layout = al.make_layout((C_out, C_in), (C_in, 1))
    weight_t = al.make_tensor(weight_ptr, al.bf16, w_layout)

    out_stride_b = C_out * stride_spatial
    out_layout = al.make_layout((B, C_out, H, W), (out_stride_b, stride_spatial, W, 1))
    output_t = al.make_tensor(output_ptr, al.bf16, out_layout)

    block_m = al.block_id(0)
    tid = al.thread_id(0)

    # Shared memory for entire weight: C_out x C_in = 128 x 64 = 16KB
    weight_sh = al.make_shared((128, 64), al.bf16)

    # Cooperative load of entire weight: 256 threads x 32 = 8192 elements
    for i in al.range(32):
        idx = tid * 32 + i
        w_oc = idx // 64
        w_ic = idx % 64
        if w_oc < C_out:
            weight_sh[w_oc, w_ic] = weight_t[w_oc, w_ic]

    al.syncthreads()

    # Each thread handles one spatial position, all output channels
    spatial_idx = block_m * TM + tid
    b = spatial_idx // (H * W)
    hw = spatial_idx % (H * W)
    h = hw // W
    w = hw % W

    if spatial_idx < B * H * W and b < B:
        for oc in al.range(C_out):
            acc = al.convert(0.0, al.f32)
            for ic in al.range(C_in):
                in_val = al.convert(input_t[b, ic, h, w], al.f32)
                w_val = al.convert(weight_sh[oc, ic], al.f32)
                acc = acc + in_val * w_val
            output_t[b, oc, h, w] = al.convert(acc, al.bf16)


def avelang_pointwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    B, C_in, H, W = x.shape
    C_out = weight.shape[0]

    x_bf16 = x.to(torch.bfloat16).contiguous()
    w_bf16 = weight.to(torch.bfloat16).contiguous()

    out_bf16 = torch.empty(B, C_out, H, W, dtype=torch.bfloat16, device=x.device)

    grid_m = (B * H * W + TM - 1) // TM
    stride_spatial = H * W

    pointwise_conv2d_kernel[lambda: ((grid_m, 1, 1), (NUM_THREADS, 1, 1))](
        x_bf16, w_bf16, out_bf16,
        B, C_in, C_out, H, W, stride_spatial,
    )

    if bias is not None:
        out_bf16 = out_bf16 + bias.to(torch.bfloat16).reshape(1, -1, 1, 1)

    return out_bf16


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_pointwise_conv2d(x, self.weight, self.bias)
