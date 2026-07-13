import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
WARP_SIZE: al.constexpr = 64
NUM_WARPS: al.constexpr = 4


@avelang.jit
def cumprod_local_scan_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    block_totals_ptr: al.Pointer(al.f32),
    num_rows: al.i32,
    num_cols: al.i32,
    num_segments: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    row_idx = bid // num_segments
    seg_idx = bid - row_idx * num_segments

    col_start = seg_idx * BLOCK_SIZE
    col = col_start + tid

    layout_2d = al.make_layout((num_rows, num_cols), (num_cols, 1))
    input_tensor = al.make_tensor(input_ptr, al.bf16, layout_2d)
    output_tensor = al.make_tensor(output_ptr, al.bf16, layout_2d)

    # Load element (identity for OOB)
    val = al.convert(1.0, al.f32)
    if col < num_cols:
        val = al.convert(input_tensor[row_idx, col], al.f32)

    warp_id = tid // WARP_SIZE
    lane_id = tid - warp_id * WARP_SIZE

    # Intra-warp inclusive prefix scan via butterfly shuffle
    other = al.shuffle_up(val, 1, WARP_SIZE)
    if lane_id >= 1:
        val = val * other
    other = al.shuffle_up(val, 2, WARP_SIZE)
    if lane_id >= 2:
        val = val * other
    other = al.shuffle_up(val, 4, WARP_SIZE)
    if lane_id >= 4:
        val = val * other
    other = al.shuffle_up(val, 8, WARP_SIZE)
    if lane_id >= 8:
        val = val * other
    other = al.shuffle_up(val, 16, WARP_SIZE)
    if lane_id >= 16:
        val = val * other
    other = al.shuffle_up(val, 32, WARP_SIZE)
    if lane_id >= 32:
        val = val * other

    # Cross-warp scan via shared memory
    shared = al.make_shared((NUM_WARPS,), al.f32)

    if lane_id == WARP_SIZE - 1:
        shared[warp_id] = val
    al.syncthreads()

    # Serial scan of the warp totals (NUM_WARPS = 4)
    if warp_id == 0:
        if tid == 0:
            shared[1] = shared[1] * shared[0]
        if tid == 0:
            shared[2] = shared[2] * shared[1]
        if tid == 0:
            shared[3] = shared[3] * shared[2]
    al.syncthreads()

    # Apply warp-level prefix
    if warp_id > 0:
        val = val * shared[warp_id - 1]

    # Write result
    if col < num_cols:
        output_tensor[row_idx, col] = al.convert(val, al.bf16)

    # Write block total
    if tid == BLOCK_SIZE - 1:
        layout_totals = al.make_layout((num_rows, num_segments), (num_segments, 1))
        totals_tensor = al.make_tensor(block_totals_ptr, al.f32, layout_totals)
        totals_tensor[row_idx, seg_idx] = val


@avelang.jit
def cumprod_carry_scan_kernel(
    block_totals_ptr: al.Pointer(al.f32),
    num_rows: al.i32,
    num_segments: al.i32,
):
    tid = al.thread_id(0)
    row_idx = al.block_id(0)

    shared = al.make_shared((BLOCK_SIZE,), al.f32)

    layout_totals = al.make_layout((num_rows, num_segments), (num_segments, 1))
    totals_tensor = al.make_tensor(block_totals_ptr, al.f32, layout_totals)

    # Load segment totals, pad with identity
    shared[tid] = al.convert(1.0, al.f32)
    if tid < num_segments:
        shared[tid] = totals_tensor[row_idx, tid]

    # Hillis-Steele inclusive scan
    al.syncthreads()
    if tid >= 1:
        shared[tid] = shared[tid] * shared[tid - 1]
    al.syncthreads()
    if tid >= 2:
        shared[tid] = shared[tid] * shared[tid - 2]
    al.syncthreads()
    if tid >= 4:
        shared[tid] = shared[tid] * shared[tid - 4]
    al.syncthreads()
    if tid >= 8:
        shared[tid] = shared[tid] * shared[tid - 8]
    al.syncthreads()
    if tid >= 16:
        shared[tid] = shared[tid] * shared[tid - 16]
    al.syncthreads()
    if tid >= 32:
        shared[tid] = shared[tid] * shared[tid - 32]
    al.syncthreads()
    if tid >= 64:
        shared[tid] = shared[tid] * shared[tid - 64]
    al.syncthreads()
    if tid >= 128:
        shared[tid] = shared[tid] * shared[tid - 128]
    al.syncthreads()

    # Write back
    if tid < num_segments:
        totals_tensor[row_idx, tid] = shared[tid]


@avelang.jit
def cumprod_apply_carry_kernel(
    output_ptr: al.Pointer(al.bf16),
    block_totals_ptr: al.Pointer(al.f32),
    num_rows: al.i32,
    num_cols: al.i32,
    num_segments: al.i32,
):
    tid = al.thread_id(0)
    row_idx = al.block_id(0)

    layout_2d = al.make_layout((num_rows, num_cols), (num_cols, 1))
    output_tensor = al.make_tensor(output_ptr, al.bf16, layout_2d)

    layout_totals = al.make_layout((num_rows, num_segments), (num_segments, 1))
    totals_tensor = al.make_tensor(block_totals_ptr, al.f32, layout_totals)

    # Strided loop: each thread processes one element per segment
    for col in al.range(tid, num_cols, BLOCK_SIZE):
        seg_idx = col // BLOCK_SIZE

        carry = al.convert(1.0, al.f32)
        if seg_idx > 0:
            carry = totals_tensor[row_idx, seg_idx - 1]

        val = al.convert(output_tensor[row_idx, col], al.f32)
        result = val * carry
        output_tensor[row_idx, col] = al.convert(result, al.bf16)


def avelang_cumprod(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Input tensor must be bfloat16"

    if dim != 1:
        raise ValueError(f"Only dim=1 is supported, got dim={dim}")

    num_rows = x.shape[0]
    num_cols = x.shape[1]

    x_contig = x.contiguous()
    output = torch.empty_like(x_contig)

    num_segments = (num_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = num_rows * num_segments

    block_totals = torch.empty(
        (num_rows, num_segments), dtype=torch.float32, device=x.device
    )

    # Phase 1: local prefix scan within each segment
    cumprod_local_scan_kernel[lambda: ((total_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contig, output, block_totals, num_rows, num_cols, num_segments
    )

    # Phase 2: scan block totals across segments within each row
    cumprod_carry_scan_kernel[lambda: ((num_rows, 1, 1), (BLOCK_SIZE, 1, 1))](
        block_totals, num_rows, num_segments
    )

    # Phase 3: apply cross-segment carries
    cumprod_apply_carry_kernel[lambda: ((num_rows, 1, 1), (BLOCK_SIZE, 1, 1))](
        output, block_totals, num_rows, num_cols, num_segments
    )

    return output


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return avelang_cumprod(x, self.dim)
