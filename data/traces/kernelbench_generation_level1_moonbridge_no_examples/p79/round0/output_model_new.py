import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_transpose1d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    L_in: al.i32,
    L_out: al.i32,
    K: al.i32,
    stride: al.i32,
    padding: al.i32,
    dilation: al.i32,
    BLOCK_L: al.constexpr,
):
    n_id = al.block_id(2)
    l_block = al.block_id(0)

    tid = al.thread_id(0)

    i = l_block * BLOCK_L + tid

    if i >= L_out:
        return

    x_layout = al.make_layout((N, C_in, L_in), (C_in * L_in, L_in, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    w_layout = al.make_layout((C_in, C_out, K), (C_out * K, K, 1))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    out_layout = al.make_layout((N, C_out, L_out), (C_out * L_out, L_out, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    zero_i32 = al.convert(0, al.i32)

    for oc in al.range(C_out):
        acc = al.convert(0.0, al.f32)

        for kk in al.range(K):
            kd = kk * dilation
            num = i + padding - kd
            if num >= zero_i32:
                q = num // stride
                r = num - q * stride
                if r == zero_i32:
                    if q < L_in:
                        for ic in al.range(C_in):
                            x_val = al.convert(x[n_id, ic, q], al.f32)
                            w_val = al.convert(w[ic, oc, kk], al.f32)
                            acc = acc + x_val * w_val

        out[n_id, oc, i] = al.convert(acc, al.bf16)


def avelang_conv_transpose1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    x_bf16 = x.to(torch.bfloat16).contiguous()
    w_bf16 = weight.to(torch.bfloat16).contiguous()

    N, C_in, L_in = x_bf16.shape
    C_out = w_bf16.shape[1]
    K = w_bf16.shape[2]

    L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1

    out = torch.zeros(N, C_out, L_out, dtype=torch.bfloat16, device=x_bf16.device)

    BLOCK_L = 256
    grid_l = (L_out + BLOCK_L - 1) // BLOCK_L

    conv_transpose1d_kernel[lambda: ((grid_l, 1, N), (BLOCK_L, 1, 1))](
        x_bf16, w_bf16, out,
        N, C_in, C_out, L_in, L_out, K,
        stride, padding, dilation,
        BLOCK_L,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv1d_transpose.weight
        s = int(self.conv1d_transpose.stride[0])
        p = int(self.conv1d_transpose.padding[0])
        d = int(self.conv1d_transpose.dilation[0])

        out = avelang_conv_transpose1d(x, weight, s, p, d)

        if self.conv1d_transpose.bias is not None:
            out = out + self.conv1d_transpose.bias.to(out.dtype).view(1, -1, 1)

        return out
