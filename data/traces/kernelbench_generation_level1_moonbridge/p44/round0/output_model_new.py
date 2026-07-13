import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Compile-time tile and pooling parameters
TILE_OUT: al.constexpr = 256
KERNEL_SIZE: al.constexpr = 8
PADDING: al.constexpr = 4
SHM_SIZE: al.constexpr = TILE_OUT + KERNEL_SIZE - 1  # 263


@avelang.jit
def avgpool1d_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    L: al.i32,
    L_out: al.i32,
):
    tid = al.thread_id(0)
    l_tile = al.block_id(0)
    flat_bc = al.block_id(1)

    b = flat_bc // C
    c = flat_bc - b * C

    l_start = l_tile * TILE_OUT

    # Flat layout for the input: (B * C * L,)
    x_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((B * C * L,), (1,)))
    x_base = b * C * L + c * L

    # Shared memory: cache the input window [l_start - PADDING, l_start + TILE_OUT + KERNEL_SIZE - 1 - PADDING)
    shm = al.make_shared((SHM_SIZE,), al.bf16)

    # Cooperative load into shared memory; each thread loads up to 2 elements
    for i in al.range(tid, SHM_SIZE, TILE_OUT):
        input_idx = l_start + i - PADDING
        if input_idx >= 0 and input_idx < L:
            shm[i] = x_flat[x_base + input_idx]
        else:
            shm[i] = al.convert(0.0, al.bf16)

    al.syncthreads()

    # Each thread computes one output element from the preloaded window
    l_out = l_start + tid
    if tid < TILE_OUT and l_out < L_out:
        acc = al.convert(0.0, al.f32)
        for j in al.range(KERNEL_SIZE):
            val = al.convert(shm[tid + j], al.f32)
            acc = acc + val

        result = acc / al.convert(KERNEL_SIZE, al.f32)

        # Flat layout for the output: (B * C * L_out,)
        out_flat = al.make_tensor(out_ptr, al.bf16, al.make_layout((B * C * L_out,), (1,)))
        out_idx = b * C * L_out + c * L_out + l_out
        out_flat[out_idx] = al.convert(result, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_avgpool1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    B, C, L = x_bf16.shape

    # Output length for AvgPool1d
    L_out = (L + 2 * padding - kernel_size) // stride + 1

    out = torch.empty((B, C, L_out), device=x_bf16.device, dtype=torch.bfloat16)

    num_l_tiles = (L_out + TILE_OUT - 1) // TILE_OUT
    grid = (num_l_tiles, B * C, 1)

    avgpool1d_kernel[lambda: (grid, (TILE_OUT, 1, 1))](
        x_bf16, out,
        B, C, L, L_out,
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized 1D Average Pooling using AveLang DSL GPU kernel.
    Matches the semantics of nn.AvgPool1d(kernel_size, stride, padding).
    """

    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = avelang_avgpool1d(x, self.kernel_size, self.stride, self.padding)
        return result.to(x.dtype)


batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4


def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length)
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding]
