import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def postprocess_kernel(
    c_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    scale: al.constexpr,
):
    """Compute out[row, 0] = scale * sum_j(C[row, j] / 2)."""
    row = al.block_id(0)
    tid = al.thread_id(0)

    c_layout = al.make_layout((M, N), (N, 1))
    c = al.make_tensor(c_ptr, al.bf16, c_layout)

    acc = al.convert(0.0, al.f32)
    for i in al.range(tid, N, 256):
        val = al.convert(c[row, i], al.f32) / al.convert(2.0, al.f32)
        acc = acc + val

    smem = al.make_shared((256,), al.f32)
    smem[tid] = acc
    al.syncthreads()

    if tid < 128:
        smem[tid] = smem[tid] + smem[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem[tid] = smem[tid] + smem[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem[tid] = smem[tid] + smem[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem[tid] = smem[tid] + smem[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem[tid] = smem[tid] + smem[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem[tid] = smem[tid] + smem[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem[tid] = smem[tid] + smem[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem[tid] = smem[tid] + smem[tid + 1]
    al.syncthreads()

    if tid == 0:
        result = smem[0] * al.convert(scale, al.bf16)
        out_layout = al.make_layout((M, 1), (1, 1))
        out = al.make_tensor(out_ptr, al.bf16, out_layout)
        out[row, 0] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        M = x.shape[0]
        N = self.weight.shape[0]

        x = x.contiguous()
        w = self.weight.contiguous()

        # Step 1: GEMM via torch.matmul (matches reference exactly)
        c = torch.matmul(x, w.T)

        # Steps 2-4: post-processing via AveLang kernel
        out = torch.empty(M, 1, dtype=torch.bfloat16, device=x.device)
        postprocess_kernel[lambda: ((M, 1, 1), (256, 1, 1))](
            c, out, M, N, self.scaling_factor
        )

        return out
