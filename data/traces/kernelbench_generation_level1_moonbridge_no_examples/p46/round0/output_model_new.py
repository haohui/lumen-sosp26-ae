import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def avgpool3d_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    KERNEL: al.constexpr,
    STRIDE: al.constexpr,
    PADDING: al.constexpr,
):
    b = al.block_id(0)
    c = al.block_id(1)
    d_out = al.block_id(2) // H_out
    h_out = al.block_id(2) % H_out
    w_out = al.thread_id(0)

    if b < B and c < C and d_out < D_out and h_out < H_out and w_out < W_out:
        x_layout = al.make_layout(
            (B, C, D, H, W),
            (C * D * H * W, D * H * W, H * W, W, 1),
        )
        x = al.make_tensor(x_ptr, al.bf16, x_layout)

        out_layout = al.make_layout(
            (B, C, D_out, H_out, W_out),
            (C * D_out * H_out * W_out, D_out * H_out * W_out, H_out * W_out, W_out, 1),
        )
        out = al.make_tensor(out_ptr, al.bf16, out_layout)

        d_in_base = d_out * STRIDE - PADDING
        h_in_base = h_out * STRIDE - PADDING
        w_in_base = w_out * STRIDE - PADDING

        acc = al.convert(0.0, al.f32)

        for kd in al.range(KERNEL):
            d_idx = d_in_base + kd
            for kh in al.range(KERNEL):
                h_idx = h_in_base + kh
                for kw in al.range(KERNEL):
                    w_idx = w_in_base + kw
                    if d_idx >= 0 and d_idx < D and h_idx >= 0 and h_idx < H and w_idx >= 0 and w_idx < W:
                        val = x[b, c, d_idx, h_idx, w_idx]
                        acc = acc + al.convert(val, al.f32)

        result = acc / al.convert(KERNEL * KERNEL * KERNEL, al.f32)
        out[b, c, d_out, h_out, w_out] = al.convert(result, al.bf16)


def _compute_output_dim(dim: int, kernel: int, stride: int, padding: int) -> int:
    return (dim + 2 * padding - kernel) // stride + 1


def avelang_avgpool3d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA/HIP device."
    orig_dtype = x.dtype
    x = x.to(torch.bfloat16).contiguous()

    B, C, D, H, W = x.shape

    D_out = _compute_output_dim(D, kernel_size, stride, padding)
    H_out = _compute_output_dim(H, kernel_size, stride, padding)
    W_out = _compute_output_dim(W, kernel_size, stride, padding)

    out = torch.empty(
        (B, C, D_out, H_out, W_out), dtype=torch.bfloat16, device=x.device
    )

    grid = (B, C, D_out * H_out)
    block = (W_out, 1, 1)

    avgpool3d_kernel[lambda: (grid, block)](
        x, out,
        B, C, D, H, W,
        D_out, H_out, W_out,
        kernel_size, stride, padding,
    )

    return out.to(orig_dtype)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_avgpool3d(x, self.kernel_size, self.stride, self.padding)
