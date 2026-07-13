import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def instancenorm_row_add_mul_kernel(
    linear_out_ptr: al.Pointer(al.bf16),
    y_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32, N: al.i32,
    BLOCK_SIZE: al.constexpr,
):
    row = al.block_id(0)
    tid = al.thread_id(0)

    linear = al.make_tensor(linear_out_ptr, al.bf16, al.make_layout((B, N), (N, 1)))
    y = al.make_tensor(y_ptr, al.bf16, al.make_layout((B, N), (N, 1)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((B, N), (N, 1)))

    smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
    smem_sumsq = al.make_shared((BLOCK_SIZE,), al.f32)
    elems_per_thread = N // BLOCK_SIZE

    partial_sum = al.convert(0.0, al.f32)
    partial_sumsq = al.convert(0.0, al.f32)
    for i in al.range(elems_per_thread):
        col = tid + i * BLOCK_SIZE
        val = al.convert(linear[row, col], al.f32)
        partial_sum = partial_sum + val
        partial_sumsq = partial_sumsq + val * val

    smem_sum[tid] = partial_sum
    smem_sumsq[tid] = partial_sumsq
    al.syncthreads()

    if tid < 128:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 128]
        smem_sumsq[tid] = smem_sumsq[tid] + smem_sumsq[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 64]
        smem_sumsq[tid] = smem_sumsq[tid] + smem_sumsq[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 32]
        smem_sumsq[tid] = smem_sumsq[tid] + smem_sumsq[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 16]
        smem_sumsq[tid] = smem_sumsq[tid] + smem_sumsq[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 8]
        smem_sumsq[tid] = smem_sumsq[tid] + smem_sumsq[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 4]
        smem_sumsq[tid] = smem_sumsq[tid] + smem_sumsq[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 2]
        smem_sumsq[tid] = smem_sumsq[tid] + smem_sumsq[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 1]
        smem_sumsq[tid] = smem_sumsq[tid] + smem_sumsq[tid + 1]
    al.syncthreads()

    mean = smem_sum[0] / al.convert(N, al.f32)
    var = smem_sumsq[0] / al.convert(N, al.f32) - mean * mean
    var = al.max(var, al.convert(0.0, al.f32))

    eps_val = al.convert(1e-5, al.f32)
    inv_std = al.convert(1.0, al.f32) / al.sqrt(var + eps_val)

    for i in al.range(elems_per_thread):
        col = tid + i * BLOCK_SIZE
        x_val = al.convert(linear[row, col], al.f32)
        norm_val = (x_val - mean) * inv_std
        y_val = al.convert(y[row, col], al.f32)
        result = (norm_val + y_val) * y_val
        out[row, col] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.eps = eps

    def forward(self, x, y):
        B, N = y.shape

        # Linear layer (PyTorch eager — both models get same weights from same seed)
        linear_out = self.linear(x)

        y_contig = y.contiguous()
        linear_contig = linear_out.contiguous()

        BLOCK_SIZE = 256
        final_out = torch.empty(B, N, dtype=x.dtype, device=x.device)

        instancenorm_row_add_mul_kernel[lambda: ((B, 1, 1), (BLOCK_SIZE, 1, 1))](
            linear_contig, y_contig, final_out, B, N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return final_out
