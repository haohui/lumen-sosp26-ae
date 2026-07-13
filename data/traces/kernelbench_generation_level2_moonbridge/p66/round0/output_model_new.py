import math
import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── Tile constants ──────────────────────────────────────────────
TM = 128
TN = 128
TK = 16
BLOCK_SIZE = 256
VEC = 8
BF16_BYTES = 2


# ══════════════════════════════════════════════════════════════════
# Tiled GEMM kernel: C = A @ B.T + bias  (BF16, vectorized loads)
# ══════════════════════════════════════════════════════════════════

@avelang.jit
def linear_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    k: al.u32,
):
    tid = al.thread_id(0)
    bid_n = al.block_id(0)
    bid_m = al.block_id(1)

    a = al.make_tensor(a_ptr, al.bf16, al.make_layout((m, k), (k, 1)))
    b = al.make_tensor(b_ptr, al.bf16, al.make_layout((n, k), (k, 1)))
    g_bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((n,), (1,)))
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    # Shared memory tiles as flat u32 for vectorized access
    shm_a = al.make_shared((TM * TK // 2,), al.u32)
    shm_b = al.make_shared((TN * TK // 2,), al.u32)

    # Each thread owns one row of A tile × VEC columns of B tile
    tid_m = tid // (TN // VEC)
    tid_n = (tid % (TN // VEC)) * VEC

    acc = al.make_local((VEC,), al.f32)
    for v in al.range(VEC):
        acc[v] = al.convert(0.0, al.f32)

    a_row_base = bid_m * TM
    b_row_base = bid_n * TN

    # 1D views of shared memory for vectorized loads
    shm_a_bf16 = al.view(shm_a, al.Tensor((TM * TK,), al.bf16))
    shm_b_bf16 = al.view(shm_b, al.Tensor((TN * TK,), al.bf16))

    k_tiles = k // TK
    for kt in al.range(k_tiles):
        k_base = kt * TK

        # Cooperative vectorized load of A into shm
        for idx in al.range(tid, TM * TK // VEC, BLOCK_SIZE):
            r = (idx * VEC) // TK
            c = (idx * VEC) % TK
            g_r = a_row_base + r
            g_c = k_base + c
            if g_r < m:
                for v in al.range(VEC):
                    if (c + v) < TK:
                        shm_a_bf16[r * TK + c + v] = a[g_r, g_c + v]

        # Cooperative vectorized load of B into shm
        for idx in al.range(tid, TN * TK // VEC, BLOCK_SIZE):
            r = (idx * VEC) // TK
            c = (idx * VEC) % TK
            g_r = b_row_base + r
            g_c = k_base + c
            if g_r < n:
                for v in al.range(VEC):
                    if (c + v) < TK:
                        shm_b_bf16[r * TK + c + v] = b[g_r, g_c + v]

        al.syncthreads()

        # Dot product: each thread computes one row of A × VEC cols of B
        for ki in al.range(TK):
            a_val = al.convert(shm_a_bf16[tid_m * TK + ki], al.f32)
            for v in al.range(VEC):
                b_val = al.convert(shm_b_bf16[(tid_n + v) * TK + ki], al.f32)
                acc[v] = acc[v] + a_val * b_val

        al.syncthreads()

    # Writeback with bias
    out_row = a_row_base + tid_m
    out_col_base = b_row_base + tid_n
    if out_row < m:
        for v in al.range(VEC):
            out_col = out_col_base + v
            if out_col < n:
                bias_val = al.convert(g_bias[out_col], al.f32)
                result = acc[v] + bias_val
                g_out[out_row, out_col] = al.convert(result, al.bf16)


# ══════════════════════════════════════════════════════════════════
# Softmax kernel — one CTA per row, tree reduction for max & sum
# ══════════════════════════════════════════════════════════════════

@avelang.jit
def softmax_kernel(
    inp_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    rows: al.u32,
    cols: al.u32,
):
    tid = al.thread_id(0)
    row_idx = al.block_id(0)

    if row_idx < rows:
        smem = al.make_shared((BLOCK_SIZE,), al.f32)

        layout_in = al.make_layout((rows, cols), (cols, 1))
        inp = al.make_tensor(inp_ptr, al.bf16, layout_in)
        layout_out = al.make_layout((rows, cols), (cols, 1))
        out = al.make_tensor(out_ptr, al.bf16, layout_out)

        # Pass 1: find row max
        neg_inf = al.convert(-1.0e30, al.f32)
        local_max = neg_inf
        for i in al.range(tid, cols, BLOCK_SIZE):
            val = al.convert(inp[row_idx, i], al.f32)
            if val > local_max:
                local_max = val

        smem[tid] = local_max
        al.syncthreads()

        if tid < 128:
            a = smem[tid]
            b = smem[tid + 128]
            smem[tid] = a if a > b else b
        al.syncthreads()
        if tid < 64:
            a = smem[tid]
            b = smem[tid + 64]
            smem[tid] = a if a > b else b
        al.syncthreads()
        if tid < 32:
            a = smem[tid]
            b = smem[tid + 32]
            smem[tid] = a if a > b else b
        al.syncthreads()
        if tid < 16:
            a = smem[tid]
            b = smem[tid + 16]
            smem[tid] = a if a > b else b
        al.syncthreads()
        if tid < 8:
            a = smem[tid]
            b = smem[tid + 8]
            smem[tid] = a if a > b else b
        al.syncthreads()
        if tid < 4:
            a = smem[tid]
            b = smem[tid + 4]
            smem[tid] = a if a > b else b
        al.syncthreads()
        if tid < 2:
            a = smem[tid]
            b = smem[tid + 2]
            smem[tid] = a if a > b else b
        al.syncthreads()
        if tid < 1:
            a = smem[tid]
            b = smem[tid + 1]
            smem[tid] = a if a > b else b
        al.syncthreads()

        row_max = smem[0]

        # Pass 2: sum of exp(x - max)
        zero_f32 = al.convert(0.0, al.f32)
        local_sum = zero_f32
        for i in al.range(tid, cols, BLOCK_SIZE):
            val = al.convert(inp[row_idx, i], al.f32)
            diff = val - row_max
            local_sum = local_sum + al.exp(diff)

        smem[tid] = local_sum
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

        row_sum = smem[0]

        # Pass 3: write softmax values
        for i in al.range(tid, cols, BLOCK_SIZE):
            val = al.convert(inp[row_idx, i], al.f32)
            diff = val - row_max
            softmax_val = al.exp(diff) / row_sum
            out[row_idx, i] = al.convert(softmax_val, al.bf16)


# ══════════════════════════════════════════════════════════════════
# Host wrappers
# ══════════════════════════════════════════════════════════════════

def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight)
    bias_bf16 = _prepare_bf16_cuda_contiguous(bias)

    m_val, k_val = x_bf16.shape
    n_val, weight_k = weight_bf16.shape

    if m_val % TM != 0 or n_val % TN != 0 or k_val % TK != 0:
        raise ValueError(
            f"Shape constraints: m%{TM}==0 (m={m_val}), "
            f"n%{TN}==0 (n={n_val}), k%{TK}==0 (k={k_val})"
        )

    out = torch.empty((m_val, n_val), device=x_bf16.device, dtype=torch.bfloat16)
    grid = (n_val // TN, m_val // TM, 1)
    linear_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](
        x_bf16, weight_bf16, bias_bf16, out, m_val, n_val, k_val
    )
    return out


def avelang_softmax(x: torch.Tensor) -> torch.Tensor:
    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    rows, cols = x_bf16.shape

    out = torch.empty_like(x_bf16)
    softmax_kernel[lambda: ((rows, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, out, rows, cols
    )
    return out


# ══════════════════════════════════════════════════════════════════
# ModelNew
# ══════════════════════════════════════════════════════════════════

class ModelNew(nn.Module):
    """
    Optimized model: Linear -> Dropout -> Softmax via AveLang GPU kernels.
    GEMM and Softmax run on GPU; Dropout delegates to PyTorch.
    """

    def __init__(self, in_features: int, out_features: int, dropout_p: float):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gemm_out = avelang_linear(x, self.weight, self.bias)
        dropped = nn.functional.dropout(
            gemm_out, p=self.dropout_p, training=self.training
        )
        return avelang_softmax(dropped)
