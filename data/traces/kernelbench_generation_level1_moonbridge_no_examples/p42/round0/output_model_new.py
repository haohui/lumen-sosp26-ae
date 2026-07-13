import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_H = 16
TILE_W = 16
SHMEM_H = 19
SHMEM_W = 19


@avelang.jit
def maxpool2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    kernel_size: al.i32,
    stride: al.i32,
    padding: al.i32,
    dilation: al.i32,
):
    tid_x = al.thread_id(0)
    tid_y = al.thread_id(1)

    block_x = al.block_id(0)
    block_y = al.block_id(1)
    slice_idx = al.block_id(2)

    h0 = block_y * TILE_H
    w0 = block_x * TILE_W

    h_in_start = h0 * stride - padding
    w_in_start = w0 * stride - padding

    n = slice_idx // C
    c = slice_idx % C

    x_stride_n = C * H * W
    x_stride_c = H * W
    x_stride_h = W
    x_stride_w = 1
    x_layout = al.make_layout(
        (N, C, H, W),
        (x_stride_n, x_stride_c, x_stride_h, x_stride_w),
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    shared_in = al.make_shared((SHMEM_H, SHMEM_W), al.bf16)
    neg_inf_bf16 = al.convert(-3.389531e38, al.bf16)

    zero = al.convert(0, al.i32)
    tid = tid_y * TILE_W + tid_x
    total_threads = TILE_H * TILE_W
    total_shmem = SHMEM_H * SHMEM_W

    for idx in al.range(tid, total_shmem, total_threads):
        si = idx // SHMEM_W
        sj = idx % SHMEM_W
        h_in = h_in_start + si
        w_in = w_in_start + sj
        if h_in >= zero and h_in < H and w_in >= zero and w_in < W:
            shared_in[si, sj] = x[n, c, h_in, w_in]
        else:
            shared_in[si, sj] = neg_inf_bf16

    al.syncthreads()

    w_out = w0 + tid_x
    h_out = h0 + tid_y

    if w_out < W_out and h_out < H_out:
        neg_inf = al.convert(-3.389531e38, al.f32)
        max_val = neg_inf

        for kh in al.range(kernel_size):
            for kw in al.range(kernel_size):
                si = tid_y * stride + kh * dilation
                sj = tid_x * stride + kw * dilation
                val = al.convert(shared_in[si, sj], al.f32)
                if val > max_val:
                    max_val = val

        out_stride_n = C * H_out * W_out
        out_stride_c = H_out * W_out
        out_stride_h = W_out
        out_stride_w = 1
        out_layout = al.make_layout(
            (N, C, H_out, W_out),
            (out_stride_n, out_stride_c, out_stride_h, out_stride_w),
        )
        out = al.make_tensor(out_ptr, al.bf16, out_layout)
        out[n, c, h_out, w_out] = al.convert(max_val, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
        N, C, H, W = x.shape

        H_out = (H + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        W_out = (W + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        x_bf16 = x.contiguous().to(torch.bfloat16)
        out = torch.empty(N, C, H_out, W_out, dtype=torch.bfloat16, device=x.device)

        grid_x = (W_out + TILE_W - 1) // TILE_W
        grid_y = (H_out + TILE_H - 1) // TILE_H
        grid_z = N * C

        maxpool2d_kernel[lambda: ((grid_x, grid_y, grid_z), (TILE_W, TILE_H, 1))](
            x_bf16.data_ptr(),
            out.data_ptr(),
            N,
            C,
            H,
            W,
            H_out,
            W_out,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )

        return out
