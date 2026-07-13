import torch
import torch.nn as nn
import torch.nn.functional as F
import avelang
import avelang.language as al

BLOCK_SIZE = 256

BATCH_SIZE = 128
IN_CHANNELS = 3
OUT_CHANNELS = 16
IN_D = 16
IN_H = 32
IN_W = 32
KERNEL_D = 3
STRIDE = 2
PADDING = 1
GROUPS = 4


@avelang.jit
def hardswish_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    global_idx = bid * BLOCK_SIZE + tid

    if global_idx < N:
        x_layout = al.make_layout((N,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, x_layout)
        out = al.make_tensor(out_ptr, al.bf16, x_layout)

        val = al.convert(x[global_idx], al.f32)

        three_f = al.convert(3.0, al.f32)
        six_f = al.convert(6.0, al.f32)
        zero_f = al.convert(0.0, al.f32)

        x_plus_3 = val + three_f
        hs_val = al.convert(0.0, al.f32)
        if x_plus_3 < zero_f:
            hs_val = zero_f
        elif x_plus_3 > six_f:
            hs_val = val
        else:
            hs_val = val * x_plus_3 / six_f

        out[global_idx] = al.convert(hs_val, al.bf16)


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


class ModelNew(nn.Module):
    """
    Hybrid: PyTorch ConvTranspose3d+Swish+GroupNorm, AveLang HardSwish.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )
        self.group_norm = nn.GroupNorm(
            num_groups=groups, num_channels=out_channels, eps=eps,
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.sigmoid(x) * x
        x = self.group_norm(x)

        x_bf16 = _to_bf16_contiguous(x)
        N = x_bf16.numel()
        out = torch.empty(N, device=x.device, dtype=torch.bfloat16)
        grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)

        hardswish_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](
            x_bf16, out, N,
        )

        return out.reshape(x.shape)


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W)]


def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_D, STRIDE, PADDING, GROUPS, 1e-5]
