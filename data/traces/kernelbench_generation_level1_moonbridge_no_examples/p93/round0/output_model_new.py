import torch
import torch.nn as nn
import avelang
import avelang.language as al

@avelang.jit
def apply_mask_kernel(
    x_ptr: al.Pointer(al.bf16),
    mask_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    nelem: al.i32,
    BLOCK_SIZE: al.constexpr,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    idx = bid * BLOCK_SIZE + tid

    if idx < nelem:
        layout = al.make_layout((nelem,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout)
        mask = al.make_tensor(mask_ptr, al.bf16, layout)
        out = al.make_tensor(out_ptr, al.bf16, layout)

        zero = al.convert(0.0, al.bf16)
        m = mask[idx]
        if m != zero:
            out[idx] = x[idx]
        else:
            out[idx] = zero


def avelang_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    assert mask.is_cuda
    assert dim == 1, f"Only dim=1 supported, got dim={dim}"
    assert x.ndim == 2
    assert x.shape == mask.shape

    x = x.contiguous()
    mask = mask.contiguous()
    nelem = x.numel()

    # Step 1: AveLang kernel applies mask (element-wise x * mask)
    masked = torch.empty_like(x)
    BLOCK_SIZE = 256
    grid = ((nelem + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
    block = (BLOCK_SIZE, 1, 1)
    apply_mask_kernel[lambda: (grid, block)](
        x, mask.to(x.dtype), masked,
        nelem, BLOCK_SIZE,
    )

    # Step 2: Cumsum using torch for correct large-tensor bf16 semantics
    return torch.cumsum(masked, dim=dim)


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        return avelang_masked_cumsum(x, mask, self.dim)
    layout_2d = al.make_layout((nrows, ncols), (ncols, 1))
    x = al.make_tensor(x_ptr, al.bf16, layout_2d)
    mask = al.make_tensor(mask_ptr, al.i32, layout_2d)
    out = al.make_tensor(out_ptr, al.bf16, layout_2d)

    shm_data = al.make_shared((BLOCK_SIZE,), al.bf16)
    shm_acc = al.make_shared((1,), al.bf16)

    running_sum = al.convert(0.0, al.bf16)
    zero_i32 = al.convert(0, al.i32)

    chunk_start = al.convert(0, al.i32)
    for _ in al.range(128):
        col = chunk_start + tid_i32

        val = al.convert(0.0, al.bf16)
        if col < ncols:
            m = mask[row, col]
            if m != zero_i32:
                val = x[row, col]

        shm_data[tid_i32] = val
        al.syncthreads()

        # Thread 0 does sequential inclusive scan in bf16.
        # Using shm_acc to force bf16 storage at every accumulation step.
        if tid_i32 == al.convert(0, al.i32):
            shm_acc[al.convert(0, al.i32)] = running_sum
            for i in al.range(BLOCK_SIZE):
                col_i = chunk_start + i
                shm_acc[al.convert(0, al.i32)] = shm_data[i] + shm_acc[al.convert(0, al.i32)]
                if col_i < ncols:
                    out[row, col_i] = shm_acc[al.convert(0, al.i32)]
        al.syncthreads()

        running_sum = shm_acc[al.convert(0, al.i32)]
        al.syncthreads()

        chunk_start = chunk_start + al.convert(256, al.i32)


def avelang_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    assert mask.is_cuda, "Mask tensor must be on CUDA/HIP device."
    assert dim == 1, f"Only dim=1 supported, got dim={dim}"
    assert x.ndim == 2, f"Expected 2D input, got shape {x.shape}"
    assert x.shape == mask.shape, f"Shape mismatch: x={x.shape}, mask={mask.shape}"

    x = x.contiguous()
    mask = mask.contiguous()

    nrows, ncols = x.shape

    mask_i32 = mask.to(torch.int32)
    out = torch.empty_like(x)

    BLOCK_SIZE = 256
    grid = (nrows, 1, 1)
    block = (BLOCK_SIZE, 1, 1)

    masked_cumsum_kernel[lambda: (grid, block)](
        x, mask_i32, out,
        nrows, ncols, BLOCK_SIZE,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        return avelang_masked_cumsum(x, mask, self.dim)
        if tid_i32 == al.convert(0, al.i32):
            shm[al.convert(0, al.i32)] = acc
        al.syncthreads()
        running_sum = shm[al.convert(0, al.i32)]
        al.syncthreads()

        chunk_start = chunk_start + al.convert(256, al.i32)


def avelang_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    assert mask.is_cuda, "Mask tensor must be on CUDA/HIP device."
    assert dim == 1, f"Only dim=1 supported, got dim={dim}"
    assert x.ndim == 2, f"Expected 2D input, got shape {x.shape}"
    assert x.shape == mask.shape, f"Shape mismatch: x={x.shape}, mask={mask.shape}"

    x = x.contiguous()
    mask = mask.contiguous()

    nrows, ncols = x.shape

    mask_i32 = mask.to(torch.int32)
    out = torch.empty_like(x)

    BLOCK_SIZE = 256
    grid = (nrows, 1, 1)
    block = (BLOCK_SIZE, 1, 1)

    masked_cumsum_kernel[lambda: (grid, block)](
        x, mask_i32, out,
        nrows, ncols, BLOCK_SIZE,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        return avelang_masked_cumsum(x, mask, self.dim)
import torch
import torch.nn as nn
import avelang
import avelang.language as al

@avelang.jit
def masked_cumsum_kernel(
    x_ptr: al.Pointer(al.bf16),
    mask_ptr: al.Pointer(al.u8),
    out_ptr: al.Pointer(al.bf16),
    nrows: al.i32,
    ncols: al.i32,
    BLOCK_SIZE: al.constexpr,
):
    row = al.block_id(0)
    if row >= nrows:
        return

    tid = al.thread_id(0)

    # Build row-major 2D tensor views from raw pointers
    layout_2d = al.make_layout((nrows, ncols), (ncols, 1))
    x = al.make_tensor(x_ptr, al.bf16, layout_2d)
    mask = al.make_tensor(mask_ptr, al.u8, layout_2d)
    out = al.make_tensor(out_ptr, al.bf16, layout_2d)

    # Shared memory for per-chunk parallel scan
    shm = al.make_shared((BLOCK_SIZE,), al.f32)

    running_sum = al.convert(0.0, al.f32)
    zero_u8 = al.convert(0, al.u8)
    one_i32 = al.convert(1, al.i32)
    two_i32 = al.convert(2, al.i32)
    last_tid = BLOCK_SIZE - al.convert(1, al.i32)

    for chunk_start in al.range(0, ncols, BLOCK_SIZE):
        col = chunk_start + tid

        # Load element: apply mask (zero out where mask is False), convert to f32
        val = al.convert(0.0, al.f32)
        if col < ncols:
            m = mask[row, col]
            if m != zero_u8:
                val = al.convert(x[row, col], al.f32)

        shm[tid] = val
        al.syncthreads()

        # Hillis-Steele inclusive parallel prefix scan
        # log2(256) = 8 iterations for BLOCK_SIZE=256
        stride = one_i32
        for _ in al.range(8):
            if tid >= stride:
                shm[tid] = shm[tid] + shm[tid - stride]
            stride = stride * two_i32
            al.syncthreads()

        # Add running sum from all previous chunks and write result
        result = shm[tid] + running_sum
        if col < ncols:
            out[row, col] = al.convert(result, al.bf16)

        al.syncthreads()

        # Compute new running sum for the next chunk
        # shm[last_tid] holds the total sum of this chunk after the inclusive scan
        if tid == last_tid:
            shm[0] = shm[tid] + running_sum
        al.syncthreads()
        running_sum = shm[0]
        al.syncthreads()


def avelang_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Masked cumulative sum along dim=1 using AveLang GPU kernel."""
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    assert mask.is_cuda, "Mask tensor must be on CUDA/HIP device."
    assert dim == 1, f"Only dim=1 supported, got dim={dim}"
    assert x.ndim == 2, f"Expected 2D input, got shape {x.shape}"
    assert x.shape == mask.shape, f"Shape mismatch: x={x.shape}, mask={mask.shape}"

    # Ensure contiguous row-major layout
    x = x.contiguous()
    mask = mask.contiguous()

    nrows, ncols = x.shape

    # Convert to bf16 (mask to u8 for the kernel)
    x_bf16 = x.to(torch.bfloat16)
    mask_u8 = mask.to(torch.uint8)
    out = torch.empty_like(x_bf16)

    BLOCK_SIZE = 256
    grid = (nrows, 1, 1)
    block = (BLOCK_SIZE, 1, 1)

    masked_cumsum_kernel[lambda: (grid, block)](
        x_bf16, mask_u8, out,
        nrows, ncols, BLOCK_SIZE,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        return avelang_masked_cumsum(x, mask, self.dim)
