import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def epilogue_kernel(
    x_ptr: al.Pointer(al.bf16),
    sum_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    total: al.i32,
    out_c: al.i32,
    out_spatial: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    gid = bid * BLOCK_SIZE + tid

    if gid >= total:
        return

    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((total,), (1,)))
    s = al.make_tensor(sum_ptr, al.bf16, al.make_layout((out_c,), (1,)))
    o = al.make_tensor(out_ptr, al.bf16, al.make_layout((total,), (1,)))

    # Decode flat index: layout is (B, C, D, H, W)
    # gid = n * (C*D*H*W) + c * (D*H*W) + d * (H*W) + h * W + w
    # For epilogue, we just need the channel index
    channel_batch_idx = gid // out_spatial
    oc = channel_batch_idx % out_c

    acc = al.convert(x[gid], al.f32)

    # LeakyReLU(negative_slope=0.2)
    zero_f = al.convert(0.0, al.f32)
    slope_f = al.convert(0.2, al.f32)
    if acc < zero_f:
        acc = acc * slope_f

    # Add sum_tensor
    s_val = al.convert(s[oc], al.f32)
    acc = acc + s_val

    # Clamp to [-1.0, 1.0]
    cmin = al.convert(-1.0, al.f32)
    cmax = al.convert(1.0, al.f32)
    if acc < cmin:
        acc = cmin
    if acc > cmax:
        acc = cmax

    # GELU activation (tanh approximation)
    sqrt_2_pi = al.convert(0.7978845608028654, al.f32)
    coeff = al.convert(0.044715, al.f32)
    half = al.convert(0.5, al.f32)
    one_f = al.convert(1.0, al.f32)

    x3 = acc * acc * acc
    inner = sqrt_2_pi * (acc + coeff * x3)
    tanh_val = al.tanh(inner)
    gelu = half * acc * (one_f + tanh_val)

    o[gid] = al.convert(gelu, al.bf16)


def _prepare_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_epilogue(
    x: torch.Tensor,
    sum_tensor: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    batch_size, out_c, out_d, out_h, out_w = x.shape
    out_spatial = out_d * out_h * out_w
    total = batch_size * out_c * out_spatial

    x_bf16 = _prepare_bf16_contiguous(x)
    s_bf16 = _prepare_bf16_contiguous(sum_tensor.contiguous().view(-1))

    grid_x = (total + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x_bf16)

    epilogue_kernel[lambda: ((grid_x, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16,
        s_bf16,
        out,
        total,
        out_c,
        out_spatial,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = self.conv(x)
        x_bf16 = x.contiguous().to(torch.bfloat16)
        result_bf16 = avelang_epilogue(x_bf16, self.sum_tensor.data)
        return result_bf16.to(x.dtype)


batch_size = 128
in_channels = 8
out_channels = 64
depth = 16
height = 64
width = 64
kernel_size = 3
sum_tensor_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]
