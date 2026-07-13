import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def avg_pool1d_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    L: al.i32,
    OL: al.i32,
    KS: al.constexpr,
    BLOCK: al.constexpr,
    stride: al.i32,
    padding: al.i32,
):
    b = al.block_id(0)
    c = al.block_id(1)
    tid = al.thread_id(0)

    total_in = B * C * L
    total_out = B * C * OL
    in_layout = al.make_layout((total_in,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, in_layout)
    out_layout = al.make_layout((total_out,), (1,))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    base_in = (b * C + c) * L
    base_out = (b * C + c) * OL

    tile_in = BLOCK + KS - 1
    shared = al.make_shared((tile_in,), al.f32)

    num_tiles = (OL + BLOCK - 1) // BLOCK
    for tile_idx in al.range(num_tiles):
        out_start = tile_idx * BLOCK
        in_start = out_start * stride - padding

        # Cooperative load of input tile into shared memory
        for i in al.range(tid, tile_in, BLOCK):
            global_idx = in_start + i
            if global_idx >= 0 and global_idx < L:
                shared[i] = al.convert(x[base_in + global_idx], al.f32)
            else:
                shared[i] = al.convert(0.0, al.f32)

        al.syncthreads()

        # Each thread computes its output element
        out_pos = out_start + tid
        if out_pos < OL:
            sum_val = al.convert(0.0, al.f32)
            for k in al.range(KS):
                sum_val = sum_val + shared[tid + k]
            avg = sum_val / al.convert(KS, al.f32)
            out[base_out + out_pos] = al.convert(avg, al.bf16)

        al.syncthreads()


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
        batch_size, in_channels, input_length = x.shape
        output_length = (
            (input_length + 2 * self.padding - self.kernel_size) // self.stride + 1
        )

        out = torch.empty(
            batch_size, in_channels, output_length, dtype=x.dtype, device=x.device
        )

        grid = (batch_size, in_channels, 1)
        block = (256, 1, 1)

        avg_pool1d_kernel[lambda: (grid, block)](
            x.data_ptr(),
            out.data_ptr(),
            batch_size,
            in_channels,
            input_length,
            output_length,
            self.kernel_size,
            256,
            self.stride,
            self.padding,
        )

        return out
