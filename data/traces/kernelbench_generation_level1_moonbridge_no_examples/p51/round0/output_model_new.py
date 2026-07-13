import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def argmax_dim1_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.i64),
    B: al.i32,
    D1: al.i32,
    D2: al.i32,
):
    batch_idx = al.block_id(0)
    dim2_idx = al.block_id(1)
    tid = al.thread_id(0)
    lane = tid & 31

    # Row-major layout for (B, D1, D2): strides = (D1*D2, D2, 1)
    layout = al.make_layout((B, D1, D2), (D1 * D2, D2, 1))
    x = al.make_tensor(x_ptr, al.bf16, layout)

    # Output layout for (B, D2): strides = (D2, 1)
    out_layout = al.make_layout((B, D2), (D2, 1))
    out = al.make_tensor(out_ptr, al.i64, out_layout)

    # Each thread processes a strided chunk of dim1 (64 threads, 64 elements each)
    local_max = al.convert(-1.0e30, al.f32)
    local_idx = al.convert(-1, al.i32)

    for chunk_start in al.range(0, D1, 64):
        idx = chunk_start + tid
        if idx < D1:
            val = al.convert(x[batch_idx, idx, dim2_idx], al.f32)
            if val > local_max:
                local_max = val
                local_idx = idx
            else:
                if val == local_max:
                    if idx < local_idx:
                        local_max = val
                        local_idx = idx

    val = local_max
    idx = local_idx

    # Half-wavefront shuffle reduction (within each 32-thread group)
    other_val = al.shuffle_down(val, 16, 32)
    other_idx = al.shuffle_down(idx, 16, 32)
    if lane < 16:
        if other_val > val:
            val = other_val
            idx = other_idx
        else:
            if other_val == val:
                if other_idx < idx:
                    val = other_val
                    idx = other_idx

    other_val = al.shuffle_down(val, 8, 32)
    other_idx = al.shuffle_down(idx, 8, 32)
    if lane < 8:
        if other_val > val:
            val = other_val
            idx = other_idx
        else:
            if other_val == val:
                if other_idx < idx:
                    val = other_val
                    idx = other_idx

    other_val = al.shuffle_down(val, 4, 32)
    other_idx = al.shuffle_down(idx, 4, 32)
    if lane < 4:
        if other_val > val:
            val = other_val
            idx = other_idx
        else:
            if other_val == val:
                if other_idx < idx:
                    val = other_val
                    idx = other_idx

    other_val = al.shuffle_down(val, 2, 32)
    other_idx = al.shuffle_down(idx, 2, 32)
    if lane < 2:
        if other_val > val:
            val = other_val
            idx = other_idx
        else:
            if other_val == val:
                if other_idx < idx:
                    val = other_val
                    idx = other_idx

    other_val = al.shuffle_down(val, 1, 32)
    other_idx = al.shuffle_down(idx, 1, 32)
    if lane < 1:
        if other_val > val:
            val = other_val
            idx = other_idx
        else:
            if other_val == val:
                if other_idx < idx:
                    val = other_val
                    idx = other_idx

    # Cross-half-wavefront reduction via shared memory
    smem_vals = al.make_shared((2,), al.f32)
    smem_idxs = al.make_shared((2,), al.i32)

    if lane == 0:
        hw_id = tid >> 5
        smem_vals[hw_id] = val
        smem_idxs[hw_id] = idx
    al.syncthreads()

    if tid == 0:
        v0 = smem_vals[0]
        i0 = smem_idxs[0]
        v1 = smem_vals[1]
        i1 = smem_idxs[1]
        if v1 > v0:
            idx = i1
        else:
            if v1 == v0:
                if i1 < i0:
                    idx = i1
                else:
                    idx = i0
            else:
                idx = i0

        out[batch_idx, dim2_idx] = al.convert(idx, al.i64)


class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D1, D2 = x.shape
        x_bf16 = x.contiguous().to(torch.bfloat16)
        out = torch.empty(B, D2, dtype=torch.int64, device=x.device)

        argmax_dim1_kernel[lambda: ((B, D2, 1), (64, 1, 1))](
            x_bf16, out, B, D1, D2
        )
        return out
