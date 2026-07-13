import torch
import torch.nn as nn
import avelang
import avelang.language as al

@avelang.jit
def maxpool1d_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    L_in: al.i32,
    L_out: al.i32,
    kernel_size: al.i32,
    stride: al.i32,
    padding: al.i32,
    dilation: al.i32,
):
    b = al.block_id(0) // C
    c = al.block_id(0) % C

    stride_b = C * L_in
    stride_out_b = C * L_out

    total_in = B * stride_b
    total_out = B * stride_out_b

    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((total_in,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((total_out,), (1,)))

    x_base = b * stride_b + c * L_in
    out_base = b * stride_out_b + c * L_out

    tid = al.thread_id(0)
    neg_inf = al.convert(-1.0e30, al.f32)

    for out_idx in al.range(tid, L_out, 256):
        in_start = out_idx * stride - padding
        max_val = neg_inf
        max_val_out = al.convert(0.0, al.bf16)

        for k in al.range(kernel_size):
            pos = in_start + k * dilation
            if pos >= 0 and pos < L_in:
                val_bf16 = x[x_base + pos]
                val_f32 = al.convert(val_bf16, al.f32)
                if val_f32 > max_val:
                    max_val = val_f32
                    max_val_out = val_bf16

        out[out_base + out_idx] = max_val_out


def avelang_maxpool1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    B, C, L_in = x.shape

    L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    x_bf16 = x.contiguous().to(torch.bfloat16)
    out_bf16 = torch.empty(B, C, L_out, dtype=torch.bfloat16, device=x.device)

    grid = (B * C, 1, 1)
    block = (256, 1, 1)

    maxpool1d_kernel[lambda: (grid, block)](
        x_bf16.data_ptr(),
        out_bf16.data_ptr(),
        B, C, L_in, L_out,
        kernel_size, stride, padding, dilation,
    )

    return out_bf16.to(x.dtype)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_maxpool1d(x, self.kernel_size, self.stride, self.padding, self.dilation)
