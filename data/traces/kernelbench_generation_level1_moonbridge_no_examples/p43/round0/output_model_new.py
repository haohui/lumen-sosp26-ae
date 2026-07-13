import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def maxpool3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    in_total: al.i32,
    out_total: al.i32,
    B: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    kernel_size: al.i32,
    stride: al.i32,
    padding: al.i32,
    dilation: al.i32,
):
    input_t = al.make_tensor(input_ptr, al.bf16, al.make_layout((in_total,), (1,)))
    output_t = al.make_tensor(output_ptr, al.bf16, al.make_layout((out_total,), (1,)))

    tid = al.thread_id(0)
    bid = al.block_id(0)
    bdim = al.block_dim(0)

    gid = bid * bdim + tid

    if gid < out_total:
        spatial_out = H_out * W_out
        slice_out = D_out * spatial_out
        channel_out = C * slice_out

        b = gid // channel_out
        rem = gid % channel_out
        c = rem // slice_out
        rem = rem % slice_out
        d = rem // spatial_out
        rem = rem % spatial_out
        h = rem // W_out
        w = rem % W_out

        in_C_stride = D * H * W
        in_D_stride = H * W
        in_H_stride = W

        in_base = b * C * in_C_stride + c * in_C_stride

        max_val = al.convert(-3.4e38, al.bf16)

        for kd in al.range(kernel_size):
            in_d = stride * d + dilation * kd - padding
            if in_d >= 0 and in_d < D:
                for kh in al.range(kernel_size):
                    in_h = stride * h + dilation * kh - padding
                    if in_h >= 0 and in_h < H:
                        for kw in al.range(kernel_size):
                            in_w = stride * w + dilation * kw - padding
                            if in_w >= 0 and in_w < W:
                                in_idx = in_base + in_d * in_D_stride + in_h * in_H_stride + in_w
                                val = input_t[in_idx]
                                if val > max_val:
                                    max_val = val

        out_idx = b * channel_out + c * slice_out + d * spatial_out + h * W_out + w
        output_t[out_idx] = max_val


def _compute_output_size(dim, kernel_size, stride, padding, dilation):
    return (dim + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def avelang_maxpool3d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    B, C, D, H, W = x.shape

    D_out = _compute_output_size(D, kernel_size, stride, padding, dilation)
    H_out = _compute_output_size(H, kernel_size, stride, padding, dilation)
    W_out = _compute_output_size(W, kernel_size, stride, padding, dilation)

    in_total = B * C * D * H * W
    out_total = B * C * D_out * H_out * W_out

    x_bf16 = x.to(torch.bfloat16).contiguous()
    out_bf16 = torch.empty(B, C, D_out, H_out, W_out, device=x.device, dtype=torch.bfloat16)

    BLOCK_SIZE = 256
    grid = ((out_total + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
    block = (BLOCK_SIZE, 1, 1)

    maxpool3d_kernel[lambda: (grid, block)](
        x_bf16,
        out_bf16,
        in_total,
        out_total,
        B, C, D, H, W,
        D_out, H_out, W_out,
        kernel_size, stride, padding, dilation,
    )

    return out_bf16.to(x.dtype)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_maxpool3d(
            x, self.kernel_size, self.stride, self.padding, self.dilation
        )
