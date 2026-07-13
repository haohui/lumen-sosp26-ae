import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_M = 16
TILE_N = 16
TILE_K = 16
REDUCE_BLOCK = 256


@avelang.jit
def matmul_sigmoid_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    tile_n = al.block_id(0)
    tile_m = al.block_id(1)
    tx = al.thread_id(0)
    ty = al.thread_id(1)

    m_start = tile_m * TILE_M
    n_start = tile_n * TILE_N

    m = m_start + ty
    n = n_start + tx

    x_layout = al.make_layout((M, K), (K, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    w_layout = al.make_layout((N, K), (K, 1))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    bias_layout = al.make_layout((N,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    out_layout = al.make_layout((M, N), (N, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    a_tile = al.make_shared((TILE_M, TILE_K), al.bf16)
    b_tile = al.make_shared((TILE_N, TILE_K), al.bf16)

    acc = al.convert(0.0, al.f32)

    for k_start in al.range(0, K, TILE_K):
        if m < M and k_start + tx < K:
            a_tile[ty, tx] = x[m, k_start + tx]

        if n < N and k_start + ty < K:
            b_tile[tx, ty] = w[n, k_start + ty]

        al.syncthreads()

        for kk in al.range(TILE_K):
            a_val = al.convert(a_tile[ty, kk], al.f32)
            b_val = al.convert(b_tile[tx, kk], al.f32)
            acc = acc + a_val * b_val

        al.syncthreads()

    if m < M and n < N:
        acc = acc + al.convert(bias[n], al.f32)
        neg_acc = -acc
        one = al.convert(1.0, al.f32)
        sig_val = one / (one + al.exp(neg_acc))
        out[m, n] = al.convert(sig_val, al.bf16)


@avelang.jit
def sum_reduce_kernel(
    inp_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
):
    row = al.block_id(0)
    tid = al.thread_id(0)

    inp_layout = al.make_layout((M, N), (N, 1))
    inp = al.make_tensor(inp_ptr, al.bf16, inp_layout)

    out_layout = al.make_layout((M, 1), (1, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    acc = al.convert(0.0, al.f32)
    for j in al.range(tid, N, REDUCE_BLOCK):
        if j < N:
            acc = acc + al.convert(inp[row, j], al.f32)

    sdata = al.make_shared((REDUCE_BLOCK,), al.f32)
    sdata[tid] = acc
    al.syncthreads()

    if tid < 128:
        sdata[tid] = sdata[tid] + sdata[tid + 128]
    al.syncthreads()

    if tid < 64:
        sdata[tid] = sdata[tid] + sdata[tid + 64]
    al.syncthreads()

    if tid < 64:
        val = sdata[tid]
        val = val + al.shuffle_down(val, 32, 64)
        val = val + al.shuffle_down(val, 16, 64)
        val = val + al.shuffle_down(val, 8, 64)
        val = val + al.shuffle_down(val, 4, 64)
        val = val + al.shuffle_down(val, 2, 64)
        val = val + al.shuffle_down(val, 1, 64)
        if tid == 0:
            out[row, 0] = al.convert(val, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        M = x.shape[0]
        K = self.input_size
        N = self.hidden_size

        w = self.linear.weight.data
        bias = self.linear.bias.data

        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = w.to(torch.bfloat16).contiguous()
        bias_bf16 = bias.to(torch.bfloat16).contiguous()

        inter = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)

        n_tiles = (N + TILE_N - 1) // TILE_N
        m_tiles = (M + TILE_M - 1) // TILE_M
        grid = (n_tiles, m_tiles, 1)
        block = (TILE_N, TILE_M, 1)

        matmul_sigmoid_kernel[lambda: (grid, block)](
            x_bf16, w_bf16, bias_bf16, inter, M, N, K
        )

        out = torch.empty(M, 1, dtype=torch.bfloat16, device=x.device)
        red_grid = (M, 1, 1)
        red_block = (REDUCE_BLOCK, 1, 1)

        sum_reduce_kernel[lambda: (red_grid, red_block)](
            inter, out, M, N
        )

        return out
