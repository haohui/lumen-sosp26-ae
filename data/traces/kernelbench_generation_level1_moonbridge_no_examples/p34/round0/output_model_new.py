import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def instancenorm_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    HW: al.i32,
    stride_n: al.i32,
    stride_c: al.i32,
):
    """One block per (batch, channel) pair.

    Two passes over HW elements:
      1) compute sum and sum-of-squares -> mean and variance
      2) normalize and write output
    """
    bid = al.block_id(0)
    tid = al.thread_id(0)

    c = bid % C
    n = bid // C

    off = n * stride_n + c * stride_c
    total = N * stride_n

    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((total,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((total,), (1,)))

    smem_sum = al.make_shared((256,), al.f32)
    smem_sq = al.make_shared((256,), al.f32)

    # -- Pass 1: compute sum and sum-of-squares in one pass -------------------
    local_sum = al.convert(0.0, al.f32)
    local_sq = al.convert(0.0, al.f32)
    for idx in al.range(tid, HW, 256):
        val = x[off + idx]
        fval = al.convert(val, al.f32)
        local_sum = local_sum + fval
        local_sq = local_sq + fval * fval

    smem_sum[tid] = local_sum
    smem_sq[tid] = local_sq
    al.syncthreads()

    # Unrolled tree reduction for 256 -> 1 (both sum and sq)
    if tid < 128:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 128]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 64]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 32]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 16]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 8]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 4]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 2]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 1]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 1]
    al.syncthreads()

    count_f = al.convert(HW, al.f32)
    mean = smem_sum[0] / count_f
    # var = E[x^2] - E[x]^2
    var = smem_sq[0] / count_f - mean * mean
    # Clamp variance to non-negative to avoid sqrt of negative due to fp rounding
    zero = al.convert(0.0, al.f32)
    if var < zero:
        var = zero

    # -- Compute normalization factor -----------------------------------------
    eps = al.convert(1e-5, al.f32)
    inv_std = al.convert(1.0, al.f32) / al.sqrt(var + eps)

    # -- Pass 2: normalize and write output -----------------------------------
    for idx in al.range(tid, HW, 256):
        val = x[off + idx]
        norm_val = (al.convert(val, al.f32) - mean) * inv_std
        out[off + idx] = al.convert(norm_val, al.bf16)


# -- Host wrapper -------------------------------------------------------------


def avelang_instancenorm(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W
    stride_c = HW
    stride_n = C * HW

    out = torch.empty_like(x)

    grid = (N * C, 1, 1)
    block = (256, 1, 1)
    instancenorm_kernel[lambda: (grid, block)](
        x, out, N, C, HW, stride_n, stride_c,
    )
    return out


# -- ModelNew -----------------------------------------------------------------


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        # affine=False, so no learnable parameters needed.
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_instancenorm(x)
