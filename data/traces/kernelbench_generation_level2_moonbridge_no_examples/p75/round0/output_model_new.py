import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import avelang
import avelang.language as al

# ── Compile-time block constant ──────────────────────────────────────────────
REDUCE_BLOCK = 256


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 1: Row-wise minimum reduction  (M×C) → (M,)
# ═══════════════════════════════════════════════════════════════════════════════
@avelang.jit
def min_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    C: al.i32,
):
    row = al.block_id(0)
    tid = al.thread_id(0)

    elems_per_thread = C // REDUCE_BLOCK
    start = tid * elems_per_thread

    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((M, C), (C, 1)))

    local_min = al.convert(3.402823466e+38, al.f32)
    for j in al.range(elems_per_thread):
        c = start + j
        val = al.convert(x[row, c], al.f32)
        if val < local_min:
            local_min = val

    shared = al.make_shared((REDUCE_BLOCK,), al.f32)
    shared[tid] = local_min
    al.syncthreads()

    if tid < 128:
        if shared[tid + 128] < shared[tid]:
            shared[tid] = shared[tid + 128]
    al.syncthreads()
    if tid < 64:
        if shared[tid + 64] < shared[tid]:
            shared[tid] = shared[tid + 64]
    al.syncthreads()
    if tid < 32:
        if shared[tid + 32] < shared[tid]:
            shared[tid] = shared[tid + 32]
    al.syncthreads()
    if tid < 16:
        if shared[tid + 16] < shared[tid]:
            shared[tid] = shared[tid + 16]
    al.syncthreads()
    if tid < 8:
        if shared[tid + 8] < shared[tid]:
            shared[tid] = shared[tid + 8]
    al.syncthreads()
    if tid < 4:
        if shared[tid + 4] < shared[tid]:
            shared[tid] = shared[tid + 4]
    al.syncthreads()
    if tid < 2:
        if shared[tid + 2] < shared[tid]:
            shared[tid] = shared[tid + 2]
    al.syncthreads()
    if tid < 1:
        if shared[tid + 1] < shared[tid]:
            shared[tid] = shared[tid + 1]
    al.syncthreads()

    if tid == 0:
        out = al.make_tensor(out_ptr, al.bf16, al.make_layout((M,), (1,)))
        out[row] = al.convert(shared[0], al.bf16)


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 2: Bias broadcast add  min_vals[m] + bias[c] → flat_out[c*M + m]
# ═══════════════════════════════════════════════════════════════════════════════
@avelang.jit
def bias_broadcast_kernel(
    min_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    C: al.i32,
):
    c_idx = al.block_id(0)
    m_idx = al.block_id(1)

    total = C * M

    min_vals = al.make_tensor(min_ptr, al.bf16, al.make_layout((M,), (1,)))
    bias_flat = al.make_tensor(bias_ptr, al.bf16, al.make_layout((C,), (1,)))
    out_flat = al.make_tensor(out_ptr, al.bf16, al.make_layout((total,), (1,)))

    idx = c_idx * M + m_idx
    result = al.convert(min_vals[m_idx], al.f32) + al.convert(bias_flat[c_idx], al.f32)
    out_flat[idx] = al.convert(result, al.bf16)


# ═══════════════════════════════════════════════════════════════════════════════
# ModelNew  entry point
# ═══════════════════════════════════════════════════════════════════════════════
class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        self.gemm_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gemm_bias = nn.Parameter(torch.empty(out_features))
        self.gn_weight = nn.Parameter(torch.empty(out_features))
        self.gn_bias = nn.Parameter(torch.empty(out_features))

        # Match reference model's RNG sequence exactly
        nn.init.kaiming_uniform_(self.gemm_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.gemm_weight)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.gemm_bias, -bound, bound)
        nn.init.ones_(self.gn_weight)
        nn.init.zeros_(self.gn_bias)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        assert x.is_cuda, "Input tensor must be on CUDA/HIP device."

        M = x.shape[0]
        N = self.out_features
        eps = 1e-5

        # ── 1. GEMM + 2. GroupNorm (PyTorch eager — precision-critical) ──
        x = F.linear(x, self.gemm_weight, self.gemm_bias)
        x = F.group_norm(x, self.num_groups, self.gn_weight, self.gn_bias, eps)

        # ── 3. Row-wise min reduction (AveLang kernel) ────────────────────
        min_vals = torch.empty(M, dtype=x.dtype, device=x.device)
        min_reduce_kernel[lambda: ((M, 1, 1), (REDUCE_BLOCK, 1, 1))](
            x.contiguous(), min_vals, M, N,
        )

        # ── 4. Bias broadcast add (AveLang kernel) ────────────────────────
        out = torch.empty(1, N, M, 1, dtype=x.dtype, device=x.device)
        bias_broadcast_kernel[lambda: ((N, M, 1), (1, 1, 1))](
            min_vals, self.bias.contiguous(), out, M, N,
        )

        return out
