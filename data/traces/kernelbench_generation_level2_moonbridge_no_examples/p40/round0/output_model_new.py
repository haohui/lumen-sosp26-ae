import torch
import torch.nn as nn
import math
import avelang
import avelang.language as al


@avelang.jit
def fused_matmul_scale_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
    combined_scale: al.constexpr,
    BM: al.constexpr,
    BN: al.constexpr,
    BK: al.constexpr,
):
    NUM_THREADS = 256
    THREADS_M = 16
    THREADS_N = 16

    block_m = al.block_id(0)
    block_n = al.block_id(1)

    tid = al.thread_id(0)
    thread_m = tid // THREADS_N
    thread_n = tid % THREADS_N

    smem_a = al.make_shared((BM, BK), al.bf16)
    smem_b = al.make_shared((BK, BN), al.bf16)

    a_layout = al.make_layout((M, K), (K, 1))
    a = al.make_tensor(a_ptr, al.bf16, a_layout)
    b_layout = al.make_layout((K, N), (N, 1))
    b = al.make_tensor(b_ptr, al.bf16, b_layout)
    c_layout = al.make_layout((M, N), (N, 1))
    c = al.make_tensor(c_ptr, al.bf16, c_layout)
    bias_layout = al.make_layout((N,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    m_start = block_m * BM
    n_start = block_n * BN

    acc = al.make_local((8, 4), al.f32)
    for i in al.range(8):
        for j in al.range(4):
            acc[i, j] = al.convert(0.0, al.f32)

    for k_blk in al.range(0, K, BK):
        a_elems_per_thread = (BM * BK) // NUM_THREADS
        for e in al.range(a_elems_per_thread):
            idx = tid * a_elems_per_thread + e
            row = idx // BK
            col = idx % BK
            g_row = m_start + row
            g_col = k_blk + col
            if (g_row < M) and (g_col < K):
                smem_a[row, col] = a[g_row, g_col]
            else:
                smem_a[row, col] = al.convert(0.0, al.bf16)

        b_elems_per_thread = (BK * BN) // NUM_THREADS
        for e in al.range(b_elems_per_thread):
            idx = tid * b_elems_per_thread + e
            row = idx // BN
            col = idx % BN
            g_row = k_blk + row
            g_col = n_start + col
            if (g_row < K) and (g_col < N):
                smem_b[row, col] = b[g_row, g_col]
            else:
                smem_b[row, col] = al.convert(0.0, al.bf16)

        al.syncthreads()

        for k in al.range(BK):
            for i in al.range(8):
                a_val = al.convert(smem_a[thread_m * 8 + i, k], al.f32)
                for j in al.range(4):
                    b_val = al.convert(smem_b[k, thread_n * 4 + j], al.f32)
                    acc[i, j] = acc[i, j] + a_val * b_val

        al.syncthreads()

    for i in al.range(8):
        g_row = m_start + thread_m * 8 + i
        for j in al.range(4):
            g_col = n_start + thread_n * 4 + j
            if (g_row < M) and (g_col < N):
                bias_val = al.convert(bias[g_col], al.f32)
                val = acc[i, j] * combined_scale
                val = val + bias_val * combined_scale
                c[g_row, g_col] = al.convert(val, al.bf16)


def _launch_fused_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scaling_factor: float,
) -> torch.Tensor:
    M, K = x.shape
    N = weight.shape[0]

    x_bf16 = x.to(torch.bfloat16).contiguous()
    weight_t_bf16 = weight.T.contiguous().to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16).contiguous()
    out_bf16 = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)

    combined_scale = scaling_factor + 1.0

    BM = 128
    BN = 64
    BK = 8

    grid_m = math.ceil(M / BM)
    grid_n = math.ceil(N / BN)

    fused_matmul_scale_kernel[lambda: ((grid_m, grid_n, 1), (256, 1, 1))](
        x_bf16,
        weight_t_bf16,
        bias_bf16,
        out_bf16,
        M,
        N,
        K,
        combined_scale,
        BM,
        BN,
        BK,
    )

    return out_bf16


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        return _launch_fused_matmul(
            x,
            self.matmul.weight,
            self.matmul.bias,
            self.scaling_factor,
        )
