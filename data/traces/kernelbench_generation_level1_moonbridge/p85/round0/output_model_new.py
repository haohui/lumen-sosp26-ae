import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def depthwise_conv2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    in_channels: al.i32,
    in_h: al.i32,
    in_w: al.i32,
    out_h: al.i32,
    out_w: al.i32,
    kernel_h: al.i32,
    kernel_w: al.i32,
):
    x_flat = al.make_tensor(
        x_ptr, al.bf16,
        al.make_layout((batch_size * in_channels * in_h * in_w,), (1,)),
    )
    w_flat = al.make_tensor(
        w_ptr, al.bf16,
        al.make_layout((in_channels * kernel_h * kernel_w,), (1,)),
    )
    out_flat = al.make_tensor(
        out_ptr, al.bf16,
        al.make_layout((batch_size * in_channels * out_h * out_w,), (1,)),
    )

    tid = al.thread_id(0)
    gid = al.block_id(0) * BLOCK_SIZE + tid

    total_out = batch_size * in_channels * out_h * out_w
    if gid < total_out:
        ow = gid % out_w
        tmp = gid // out_w
        oh = tmp % out_h
        tmp = tmp // out_h
        c = tmp % in_channels
        b = tmp // in_channels

        acc = al.convert(0.0, al.f32)
        for kh in al.range(kernel_h):
            for kw in al.range(kernel_w):
                ih = oh + kh
                iw = ow + kw
                x_idx = ((b * in_channels + c) * in_h + ih) * in_w + iw
                w_idx = (c * kernel_h + kh) * kernel_w + kw
                x_val = al.convert(x_flat[x_idx], al.f32)
                w_val = al.convert(w_flat[w_idx], al.f32)
                acc = acc + x_val * w_val

        out_idx = ((b * in_channels + c) * out_h + oh) * out_w + ow
        out_flat[out_idx] = al.convert(acc, al.bf16)


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)

    batch_size, in_channels, in_h, in_w = x_bf16.shape
    kernel_h = w_bf16.shape[2]
    kernel_w = w_bf16.shape[3]

    out_h = in_h - kernel_h + 1
    out_w = in_w - kernel_w + 1

    out = torch.empty(
        (batch_size, in_channels, out_h, out_w),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )

    total_out = batch_size * in_channels * out_h * out_w
    num_blocks = (total_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks, 1, 1)

    depthwise_conv2d_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](
        x_bf16, w_bf16, out,
        batch_size, in_channels, in_h, in_w, out_h, out_w,
        kernel_h, kernel_w,
    )

    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_h: int,
        kernel_size_w: int,
        stride_h: int = 1,
        stride_w: int = 1,
        padding_h: int = 0,
        padding_w: int = 0,
        dilation_h: int = 1,
        dilation_w: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels, in_channels,
            (kernel_size_h, kernel_size_w),
            stride=(stride_h, stride_w),
            padding=(padding_h, padding_w),
            dilation=(dilation_h, dilation_w),
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_depthwise_conv2d(x, self.conv2d.weight)


# Test code
batch_size = 32
in_channels = 128
out_channels = 128
kernel_size_h = 3
kernel_size_w = 7
width = 256
height = 128
stride_h = 1
stride_w = 1
padding_h = 0
padding_w = 0
dilation_h = 1
dilation_w = 1
groups = in_channels


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size_h,
        kernel_size_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
    ]
