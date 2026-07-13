import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def cumsum_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    segment_sums_ptr: al.Pointer(al.f32),
    cum_offsets_ptr: al.Pointer(al.f32),
    num_rows: al.i32,
    num_cols: al.i32,
    num_segments: al.i32,
):
    """
    Pass 1+3 combined: each block handles one segment of one row.
    All threads load data into shared memory, then thread 0 does a
    sequential BF16-rounded scan starting from the cumulative offset,
    replicating torch.cumsum's element-by-element BF16 behavior.
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    row = bid // num_segments
    seg = bid - row * num_segments

    if row >= num_rows:
        return

    col_base = seg * BLOCK_SIZE
    col = col_base + tid

    layout_2d = al.make_layout((num_rows, num_cols), (num_cols, 1))
    x = al.make_tensor(x_ptr, al.bf16, layout_2d)
    out = al.make_tensor(out_ptr, al.bf16, layout_2d)

    smem = al.make_shared((BLOCK_SIZE,), al.f32)

    # Coalesced load into shared memory
    val = al.convert(0.0, al.f32)
    if col < num_cols:
        val = al.convert(x[row, col], al.f32)
    smem[tid] = val
    al.syncthreads()

    # Thread 0: sequential BF16-rounded scan from cumulative offset
    if tid == 0:
        # Get cumulative offset for this segment
        offset = al.convert(0.0, al.f32)
        if seg > 0:
            layout_co = al.make_layout((num_rows, num_segments), (num_segments, 1))
            co = al.make_tensor(cum_offsets_ptr, al.f32, layout_co)
            offset = co[row, seg]

        acc = offset
        i = al.convert(0, al.i32)
        for _ in al.range(0, 256):
            if i < BLOCK_SIZE:
                cur = smem[i]
                acc = al.convert(al.convert(acc + cur, al.bf16), al.f32)
                smem[i] = acc
            i = i + 1

        # Store segment total (BF16-rounded inclusive scan of the segment)
        if seg < num_segments:
            layout_ps = al.make_layout((num_rows, num_segments), (num_segments, 1))
            ps = al.make_tensor(segment_sums_ptr, al.f32, layout_ps)
            ps[row, seg] = acc
    al.syncthreads()

    # All threads write results
    if col < num_cols:
        out[row, col] = al.convert(smem[tid], al.bf16)


@avelang.jit
def cumsum_aggregate_kernel(
    segment_sums_ptr: al.Pointer(al.f32),
    cum_offsets_ptr: al.Pointer(al.f32),
    num_rows: al.i32,
    num_segments: al.i32,
):
    """
    Pass 2: thread 0 does sequential BF16-rounded exclusive scan of
    segment totals → cumulative offsets.
    """
    tid = al.thread_id(0)
    row = al.block_id(0)

    if row >= num_rows:
        return

    smem = al.make_shared((BLOCK_SIZE,), al.f32)

    layout_ps = al.make_layout((num_rows, num_segments), (num_segments, 1))
    ps = al.make_tensor(segment_sums_ptr, al.f32, layout_ps)

    val = al.convert(0.0, al.f32)
    if tid < num_segments:
        val = ps[row, tid]
    smem[tid] = val
    al.syncthreads()

    if tid == 0:
        acc = al.convert(0.0, al.f32)
        i = al.convert(0, al.i32)
        for _ in al.range(0, 256):
            if i < num_segments:
                cur = smem[i]
                # exclusive = acc (sum before adding cur)
                smem[i] = acc
                # inclusive for next iteration
                acc = al.convert(al.convert(acc + cur, al.bf16), al.f32)
            i = i + 1
    al.syncthreads()

    if tid < num_segments:
        layout_co = al.make_layout((num_rows, num_segments), (num_segments, 1))
        co = al.make_tensor(cum_offsets_ptr, al.f32, layout_co)
        co[row, tid] = smem[tid]


def _cumsum_dim1(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA/HIP device."
    num_rows, num_cols = x.shape

    x_bf16 = x.to(torch.bfloat16).contiguous()
    num_segments = (num_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x_bf16)
    segment_sums = torch.empty(
        (num_rows, num_segments), dtype=torch.float32, device=x.device
    )
    cum_offsets = torch.zeros(
        (num_rows, num_segments), dtype=torch.float32, device=x.device
    )

    block = (BLOCK_SIZE, 1, 1)

    # Pass 1: sequential scan per segment (writes segment totals)
    grid_pass1 = (num_rows * num_segments, 1, 1)
    cumsum_kernel[lambda: (grid_pass1, block)](
        x_bf16, out, segment_sums, cum_offsets,
        num_rows, num_cols, num_segments
    )
    # NOTE: pass 1 uses cum_offsets which is all zeros at this point
    # (since seg > 0 check means offset=0 for the first pass)

    # Pass 2: aggregate scan → cumulative offsets
    grid_pass2 = (num_rows, 1, 1)
    cumsum_aggregate_kernel[lambda: (grid_pass2, block)](
        segment_sums, cum_offsets, num_rows, num_segments
    )

    # Pass 3: re-run segment scans with correct offsets
    cumsum_kernel[lambda: (grid_pass1, block)](
        x_bf16, out, segment_sums, cum_offsets,
        num_rows, num_cols, num_segments
    )

    return out.to(x.dtype)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1:
            return _cumsum_dim1(x)
        if self.dim == 0:
            xt = x.transpose(0, 1).contiguous()
            result_t = _cumsum_dim1(xt)
            return result_t.transpose(0, 1).contiguous()
        raise ValueError(f"Unsupported cumsum dimension: {self.dim}")
