import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def maxpool2d_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.i32,
    stride: al.i32,
    padding: al.i32,
    dilation: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    total = B * C * H_out * W_out
    gid = bid * BLOCK_SIZE + tid

    if gid < total:
        layout_in = al.make_layout((B, C, H, W), (C * H * W, H * W, W, 1))
        x = al.make_tensor(input_ptr, al.bf16, layout_in)

        layout_out = al.make_layout((B, C, H_out, W_out), (C * H_out * W_out, H_out * W_out, W_out, 1))
        y = al.make_tensor(output_ptr, al.bf16, layout_out)

        w_out = gid % W_out
        tmp = gid // W_out
        h_out = tmp % H_out
        tmp = tmp // H_out
        c = tmp % C
        b = tmp // C

        h_start = h_out * stride - padding
        w_start = w_out * stride - padding

        first = al.convert(1, al.i32)
        best = al.convert(0.0, al.f32)

        for kh in al.range(K):
            h_in = h_start + kh * dilation
            if h_in >= 0 and h_in < H:
                for kw in al.range(K):
                    w_in = w_start + kw * dilation
                    if w_in >= 0 and w_in < W:
                        val = al.convert(x[b, c, h_in, w_in], al.f32)
                        if first != 0:
                            best = val
                            first = al.convert(0, al.i32)
                        else:
                            if val > best:
                                best = val

        y[b, c, h_out, w_out] = al.convert(best, al.bf16)


def avelang_maxpool2d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA/HIP device."

    B, C, H, W = x.shape
    K = kernel_size

    H_out = (H + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (K - 1) - 1) // stride + 1

    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    x = x.contiguous()

    out = torch.empty((B, C, H_out, W_out), dtype=torch.bfloat16, device=x.device)

    total = B * C * H_out * W_out
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)

    maxpool2d_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](
        x, out, B, C, H, W, H_out, W_out, K, stride, padding, dilation
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized Max Pooling 2D model using AveLang DSL.
    """

    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_maxpool2d(
            x, self.kernel_size, self.stride, self.padding, self.dilation
        )


batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1


def get_inputs():
    x = torch.rand(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation]
