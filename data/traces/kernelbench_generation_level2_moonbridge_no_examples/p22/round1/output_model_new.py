import torch
import torch.nn as nn
import avelang
import avelang.language as al

NUM_THREADS = 256


@avelang.jit
def row_reduce_max_kernel(
    x_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    M: al.i32,
    N: al.i32,
):
    row = al.block_id(0)
    tid = al.thread_id(0)

    x = al.make_tensor(x_ptr, al.f32, al.make_layout((M, N), (N, 1)))
    sh = al.make_shared((NUM_THREADS,), al.f32)

    local_max = al.convert(-1.0, al.f32) * al.convert(1e30, al.f32)
    for i in al.range(0, N, NUM_THREADS):
        idx = tid + i
        v = x[row, idx]
        if v > local_max:
            local_max = v

    sh[tid] = local_max
    al.syncthreads()

    if tid < 128:
        v = sh[tid + 128]
        if v > sh[tid]:
            sh[tid] = v
    al.syncthreads()
    if tid < 64:
        v = sh[tid + 64]
        if v > sh[tid]:
            sh[tid] = v
    al.syncthreads()
    if tid < 32:
        v = sh[tid + 32]
        if v > sh[tid]:
            sh[tid] = v
    al.syncthreads()
    if tid < 16:
        v = sh[tid + 16]
        if v > sh[tid]:
            sh[tid] = v
    al.syncthreads()
    if tid < 8:
        v = sh[tid + 8]
        if v > sh[tid]:
            sh[tid] = v
    al.syncthreads()
    if tid < 4:
        v = sh[tid + 4]
        if v > sh[tid]:
            sh[tid] = v
    al.syncthreads()
    if tid < 2:
        v = sh[tid + 2]
        if v > sh[tid]:
            sh[tid] = v
    al.syncthreads()
    if tid < 1:
        v = sh[tid + 1]
        if v > sh[tid]:
            sh[tid] = v
    al.syncthreads()

    out = al.make_tensor(out_ptr, al.f32, al.make_layout((M,), (1,)))
    out[row] = sh[0]


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        # Matmul + elementwise ops (PyTorch, matches reference)
        inter = self.linear(x)
        inter = inter * self.scale_factor
        inter = inter + inter
        inter = torch.clamp(inter, self.clamp_min, self.clamp_max)

        M, N_val = inter.shape

        # AveLang kernel: per-row max reduction (numerically exact)
        inter_f32 = inter.to(torch.float32).contiguous()
        out_max = torch.empty((M,), dtype=torch.float32, device=x.device)
        row_reduce_max_kernel[lambda: ((M, 1, 1), (NUM_THREADS, 1, 1))](
            inter_f32, out_max, M, N_val,
        )

        # Stable logsumexp using kernel-computed max, in BF16 to match reference
        row_max = out_max.to(torch.bfloat16).unsqueeze(1)
        shifted = inter - row_max
        lse = torch.logsumexp(shifted, dim=1, keepdim=True) + row_max

        # Mish activation (BF16, matches reference)
        result = lse * torch.nn.functional.mish(lse)
        return result
