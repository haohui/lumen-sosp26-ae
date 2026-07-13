import torch
import torch.nn as nn
import avelang
import avelang.language as al

_REDUCE_BLOCK = 256
REDUCE_BLOCK = al.constexpr(_REDUCE_BLOCK)


@avelang.jit
def fused_reduce_kernel(
    c_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
):
    row = al.block_id(0)
    tid = al.thread_id(0)

    c = al.make_tensor(c_ptr, al.bf16, al.make_layout((M, N), (N, 1)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((M,), (1,)))

    acc = al.convert(0.0, al.f32)
    for col in al.range(tid, N, REDUCE_BLOCK):
        acc = acc + al.convert(c[row, col], al.f32)

    smem = al.make_shared((REDUCE_BLOCK,), al.f32)
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
        out[row] = al.convert(smem[0], al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        M, K = x.shape
        N = self.linear.out_features
        device = x.device

        # Use the linear layer's own F.linear path for bf16 matmul (matches reference exactly)
        lin_out = self.linear(x)

        # AveLang reduction kernel for sum over dim=1
        out = torch.empty(M, dtype=torch.bfloat16, device=device)
        grid = (M, 1, 1)
        block = (_REDUCE_BLOCK, 1, 1)

        fused_reduce_kernel[lambda: (grid, block)](lin_out, out, M, N)

        return out.unsqueeze(1)
