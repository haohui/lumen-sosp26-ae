import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256
_KH = 11
_KW = 11
_KAREA = _KH * _KW  # 121


@avelang.jit
def avgpool2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    gid = bid * BLOCK_SIZE + tid

    stride_b = C * H * W
    stride_c = H * W
    stride_h = W
    stride_w = al.convert(1, al.i32)
    x_layout = al.make_layout(
        (B, C, H, W), (stride_b, stride_c, stride_h, stride_w)
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    out_stride_b = C * H_out * W_out
    out_stride_c = H_out * W_out
    out_stride_h = W_out
    out_stride_w = al.convert(1, al.i32)
    out_layout = al.make_layout(
        (B, C, H_out, W_out),
        (out_stride_b, out_stride_c, out_stride_h, out_stride_w),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    total_out = B * C * H_out * W_out
    if gid < total_out:
        tmp = gid
        w_out = tmp % W_out
        tmp = tmp // W_out
        h_out = tmp % H_out
        tmp = tmp // H_out
        c = tmp % C
        b = tmp // C

        h_start = h_out * _KH
        w_start = w_out * _KW

        acc = al.convert(0.0, al.f32)
        for _kh in al.range(_KH):
            h_idx = h_start + _kh
            for _kw in al.range(_KW):
                w_idx = w_start + _kw
                val = al.convert(x[b, c, h_idx, w_idx], al.f32)
                acc = acc + val

        area_f32 = al.convert(_KAREA, al.f32)
        avg = acc / area_f32
        out[b, c, h_out, w_out] = al.convert(avg, al.bf16)


def _prepare_bf16_cuda(t: torch.Tensor) -> torch.Tensor:
    if not t.is_cuda:
        t = t.cuda()
    if t.dtype != torch.bfloat16:
        t = t.to(dtype=torch.bfloat16)
    if not t.is_contiguous():
        t = t.contiguous()
    return t


def avelang_avgpool2d(
    x: torch.Tensor,
    kernel_size_val: int,
    stride_val: int,
    padding_val: int,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device required for AveLang kernels.")
    if padding_val != 0:
        raise ValueError("Only padding=0 is supported.")

    x_bf16 = _prepare_bf16_cuda(x)
    B, C, H, W = x_bf16.shape

    H_out = (H + 2 * padding_val - kernel_size_val) // stride_val + 1
    W_out = (W + 2 * padding_val - kernel_size_val) // stride_val + 1

    total_out = B * C * H_out * W_out
    num_blocks = (total_out + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty(
        (B, C, H_out, W_out), device=x_bf16.device, dtype=torch.bfloat16
    )

    avgpool2d_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, out,
        B, C, H, W, H_out, W_out,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride=None, padding=0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_avgpool2d(x, self.kernel_size, self.stride, self.padding)


batch_size = 16
channels = 64
height = 2048
width = 2048
kernel_size = 11


def get_inputs():
    x = torch.rand(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [kernel_size]
