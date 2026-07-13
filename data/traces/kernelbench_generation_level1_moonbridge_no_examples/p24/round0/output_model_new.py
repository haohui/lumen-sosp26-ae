import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def log_softmax_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
):
    row = al.block_id(0)
    tid = al.thread_id(0)
    BLOCK_SIZE = al.block_dim(0)

    # Create row-major tensor views from raw pointers
    x_layout = al.make_layout((M, N), (N, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    out_layout = al.make_layout((M, N), (N, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    # Shared memory for cross-warp reduction (256 threads = 8 warps of 32)
    smax = al.make_shared((256,), al.f32)
    ssum = al.make_shared((256,), al.f32)

    warp_id = tid // 32
    lane_id = tid % 32

    # ============================================================
    # Phase 1: row-wise max reduction
    # ============================================================
    local_max = al.convert(-1.0e30, al.f32)
    for i in al.range(tid, N, BLOCK_SIZE):
        val = al.convert(x[row, i], al.f32)
        if val > local_max:
            local_max = val

    # Warp-level max reduction (32 lanes -> 1)
    other = al.shuffle_down(local_max, 16, 32)
    if other > local_max:
        local_max = other
    other = al.shuffle_down(local_max, 8, 32)
    if other > local_max:
        local_max = other
    other = al.shuffle_down(local_max, 4, 32)
    if other > local_max:
        local_max = other
    other = al.shuffle_down(local_max, 2, 32)
    if other > local_max:
        local_max = other
    other = al.shuffle_down(local_max, 1, 32)
    if other > local_max:
        local_max = other

    # Block-level reduction: lane 0 of each warp writes to shared memory
    if lane_id == 0:
        smax[warp_id] = local_max
    al.syncthreads()

    # First warp reduces the 8 warp-level results
    if warp_id == 0:
        if lane_id < 8:
            warp_max = smax[lane_id]
            other = al.shuffle_down(warp_max, 4, 8)
            if other > warp_max:
                warp_max = other
            other = al.shuffle_down(warp_max, 2, 8)
            if other > warp_max:
                warp_max = other
            other = al.shuffle_down(warp_max, 1, 8)
            if other > warp_max:
                warp_max = other
            if lane_id == 0:
                smax[0] = warp_max
    al.syncthreads()

    global_max = smax[0]

    # ============================================================
    # Phase 2: row-wise sum(exp(x - max)) reduction
    # ============================================================
    local_sum = al.convert(0.0, al.f32)
    for i in al.range(tid, N, BLOCK_SIZE):
        val = al.convert(x[row, i], al.f32)
        diff = val - global_max
        local_sum = local_sum + al.exp(diff)

    # Warp-level sum reduction
    other = al.shuffle_down(local_sum, 16, 32)
    local_sum = local_sum + other
    other = al.shuffle_down(local_sum, 8, 32)
    local_sum = local_sum + other
    other = al.shuffle_down(local_sum, 4, 32)
    local_sum = local_sum + other
    other = al.shuffle_down(local_sum, 2, 32)
    local_sum = local_sum + other
    other = al.shuffle_down(local_sum, 1, 32)
    local_sum = local_sum + other

    # Block-level sum reduction
    if lane_id == 0:
        ssum[warp_id] = local_sum
    al.syncthreads()

    if warp_id == 0:
        if lane_id < 8:
            warp_sum = ssum[lane_id]
            other = al.shuffle_down(warp_sum, 4, 8)
            warp_sum = warp_sum + other
            other = al.shuffle_down(warp_sum, 2, 8)
            warp_sum = warp_sum + other
            other = al.shuffle_down(warp_sum, 1, 8)
            warp_sum = warp_sum + other
            if lane_id == 0:
                ssum[0] = warp_sum
    al.syncthreads()

    global_sum = ssum[0]
    log_sum = al.log(global_sum)

    # ============================================================
    # Phase 3: compute output = x - max - log(sum(exp(x - max)))
    # ============================================================
    for i in al.range(tid, N, BLOCK_SIZE):
        val = al.convert(x[row, i], al.f32)
        result = val - global_max - log_sum
        out[row, i] = al.convert(result, al.bf16)


def avelang_log_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    assert dim == 1, "Only dim=1 is supported by the AveLang kernel."

    M = x.shape[0]
    N = x.shape[1]
    orig_dtype = x.dtype

    # Ensure contiguous BF16 input for the kernel
    x_bf16 = x.contiguous().to(torch.bfloat16)
    out_bf16 = torch.empty_like(x_bf16)

    BLOCK_SIZE = 256
    grid = (M, 1, 1)
    block = (BLOCK_SIZE, 1, 1)

    log_softmax_kernel[lambda: (grid, block)](x_bf16, out_bf16, M, N)

    return out_bf16.to(orig_dtype)


class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_log_softmax(x, dim=self.dim)
