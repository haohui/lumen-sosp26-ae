import torch
import torch.nn as nn
import avelang
import avelang.language as al

_BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def bias_scale_sigmoid_kernel(
    x_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    scale_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    HW: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    gid = bid * _BLOCK_SIZE + tid
    total = N * C * HW

    if gid < total:
        n_idx = gid // (C * HW)
        rem = gid - n_idx * C * HW
        c_idx = rem // HW
        hw_idx = rem - c_idx * HW

        layout_x = al.make_layout((N, C, HW), (C * HW, HW, 1))
        x = al.make_tensor(x_ptr, al.bf16, layout_x)

        layout_p = al.make_layout((C,), (1,))
        bias = al.make_tensor(bias_ptr, al.bf16, layout_p)
        scale = al.make_tensor(scale_ptr, al.bf16, layout_p)

        xv = al.convert(x[n_idx, c_idx, hw_idx], al.f32)
        bv = al.convert(bias[c_idx], al.f32)
        sv_f32 = al.convert(scale[c_idx], al.f32)

        tmp = xv + bv
        tmp_bf16 = al.convert(tmp, al.bf16)
        tmp_f32 = al.convert(tmp_bf16, al.f32)
        res = tmp_f32 * sv_f32
        res_bf16 = al.convert(res, al.bf16)
        res_f32 = al.convert(res_bf16, al.f32)

        neg = al.convert(0.0, al.f32) - res_f32
        exp_neg = al.exp(neg)
        denom = al.convert(1.0, al.f32) + exp_neg
        sig = al.convert(1.0, al.f32) / denom

        layout_out = al.make_layout((N, C, HW), (C * HW, HW, 1))
        ot = al.make_tensor(out_ptr, al.bf16, layout_out)
        ot[n_idx, c_idx, hw_idx] = al.convert(sig, al.bf16)


def _ensure_bf16_contig(t: torch.Tensor) -> torch.Tensor:
    if t.dtype != torch.bfloat16:
        t = t.to(dtype=torch.bfloat16)
    if not t.is_contiguous():
        t = t.contiguous()
    return t


def avelang_bias_scale_sigmoid(
    x: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    x_bf16 = _ensure_bf16_contig(x)
    x_shape = x_bf16.shape
    x_flat = x_bf16.reshape(x_shape[0], x_shape[1], -1).contiguous()

    bias_bf16 = _ensure_bf16_contig(bias)
    scale_bf16 = _ensure_bf16_contig(scale)

    N, C, HW = x_flat.shape
    total = N * C * HW
    num_blocks = (total + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    out_flat = torch.empty_like(x_flat)

    bias_scale_sigmoid_kernel[lambda: ((num_blocks, 1, 1), (_BLOCK_SIZE, 1, 1))](
        x_flat, bias_bf16, scale_bf16, out_flat,
        N, C, HW,
    )

    return out_flat.reshape(x_shape)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x_bf16 = x.to(dtype=torch.bfloat16, device=x.device)
        conv_out = self.conv(x_bf16)
        mid = avelang_bias_scale_sigmoid(conv_out, self.bias.data, self.scale.data)
        result = self.group_norm(mid)
        if x.dtype != torch.bfloat16:
            result = result.to(x.dtype)
        return result
