import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def cumprod_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
):
    row = al.block_id(0)
    tid = al.thread_id(0)

    layout = al.make_layout((M, N), (N, 1))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    if row < M:
        # thread_totals[0..255] = per-thread inclusive scan
        # thread_totals[256] = cross-tile carry (FP32)
        thread_totals = al.make_shared((257,), al.f32)
        prefixes = al.make_shared((256,), al.f32)
        local_vals = al.make_local((8,), al.f32)

        # Initialize carry to 1.0
        if tid == 0:
            thread_totals[256] = al.convert(1.0, al.f32)
        al.syncthreads()

        for tile_start in al.range(0, N, 2048):
            # --- 1. Load tile and compute local inclusive scan ---
            base_col = tile_start + tid * 8

            for i in al.range(8):
                col = base_col + i
                if col < N:
                    local_vals[i] = al.convert(x[row, col], al.f32)
                else:
                    local_vals[i] = al.convert(1.0, al.f32)

            for i in al.range(1, 8):
                local_vals[i] = local_vals[i] * local_vals[i - 1]

            # --- 2. Block-wide Hillis-Steele inclusive scan ---
            thread_totals[tid] = local_vals[7]
            al.syncthreads()

            if tid >= 1:
                thread_totals[tid] = thread_totals[tid] * thread_totals[tid - 1]
            al.syncthreads()
            if tid >= 2:
                thread_totals[tid] = thread_totals[tid] * thread_totals[tid - 2]
            al.syncthreads()
            if tid >= 4:
                thread_totals[tid] = thread_totals[tid] * thread_totals[tid - 4]
            al.syncthreads()
            if tid >= 8:
                thread_totals[tid] = thread_totals[tid] * thread_totals[tid - 8]
            al.syncthreads()
            if tid >= 16:
                thread_totals[tid] = thread_totals[tid] * thread_totals[tid - 16]
            al.syncthreads()
            if tid >= 32:
                thread_totals[tid] = thread_totals[tid] * thread_totals[tid - 32]
            al.syncthreads()
            if tid >= 64:
                thread_totals[tid] = thread_totals[tid] * thread_totals[tid - 64]
            al.syncthreads()
            if tid >= 128:
                thread_totals[tid] = thread_totals[tid] * thread_totals[tid - 128]
            al.syncthreads()

            # --- 3. Compute per-thread prefix (carry * exclusive scan) ---
            prefixes[tid] = thread_totals[256]
            if tid > 0:
                prefixes[tid] = prefixes[tid] * thread_totals[tid - 1]
            al.syncthreads()

            # --- 4. Apply prefix to local values and store ---
            for i in al.range(8):
                local_vals[i] = local_vals[i] * prefixes[tid]

            for i in al.range(8):
                col = base_col + i
                if col < N:
                    out[row, col] = al.convert(local_vals[i], al.bf16)

            # --- 5. Propagate carry across tiles ---
            if tid == 0:
                thread_totals[256] = thread_totals[256] * thread_totals[255]
            al.syncthreads()


def avelang_cumprod(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA/HIP device."
    assert x.ndim == 2, "Expected 2D input tensor for cumprod kernel."
    assert dim == 1, "Only dim=1 is supported in this kernel."

    M, N = x.shape
    x_contiguous = x.contiguous()
    out = torch.empty_like(x_contiguous)

    grid = (M, 1, 1)
    block = (256, 1, 1)
    cumprod_kernel[lambda: (grid, block)](x_contiguous, out, M, N)

    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return avelang_cumprod(x, self.dim)
