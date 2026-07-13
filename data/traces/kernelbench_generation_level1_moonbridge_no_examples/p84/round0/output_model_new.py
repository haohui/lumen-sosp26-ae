import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def depthwise_conv3x3_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    bx = al.block_id(0)
    by = al.block_id(1)
    bz = al.block_id(2)

    n = bz // C
    c = bz % C

    tx = al.thread_id(0)
    ty = al.thread_id(1)

    h_out = by * 16 + ty
    w_out = bx * 16 + tx

    if h_out < H_out and w_out < W_out:
        x_layout = al.make_layout((N, C, H, W), (C * H * W, H * W, W, 1))
        x = al.make_tensor(x_ptr, al.bf16, x_layout)

        w_layout = al.make_layout((C, 1, 3, 3), (9, 9, 3, 1))
        w = al.make_tensor(w_ptr, al.bf16, w_layout)

        out_layout = al.make_layout((N, C, H_out, W_out), (C * H_out * W_out, H_out * W_out, W_out, 1))
        out = al.make_tensor(out_ptr, al.bf16, out_layout)

        acc = al.convert(0.0, al.f32)

        for kh in al.range(3):
            for kw in al.range(3):
                h_in = h_out + kh
                w_in = w_out + kw
                x_val = al.convert(x[n, c, h_in, w_in], al.f32)
                w_val = al.convert(w[c, 0, kh, kw], al.f32)
                acc = acc + x_val * w_val

        out[n, c, h_out, w_out] = al.convert(acc, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv2d = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=stride, padding=padding,
            groups=in_channels, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv2d.weight
        N, C, H, W = x.shape

        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        out = torch.empty(N, self.out_channels, H_out, W_out, dtype=x.dtype, device=x.device)

        BLOCK_H = 16
        BLOCK_W = 16

        grid_x = (W_out + BLOCK_W - 1) // BLOCK_W
        grid_y = (H_out + BLOCK_H - 1) // BLOCK_H
        grid_z = N * C

        depthwise_conv3x3_kernel[lambda: ((grid_x, grid_y, grid_z), (BLOCK_W, BLOCK_H, 1))](
            x, weight, out,
            N, C, H, W, H_out, W_out,
        )

        return out
