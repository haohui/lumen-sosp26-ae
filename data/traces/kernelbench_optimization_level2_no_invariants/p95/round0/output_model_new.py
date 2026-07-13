import torch
import torch.nn as nn
import avelang
import avelang.language as al

SQRT_2 = 1.4142135623730951

TILE_M = 64
TILE_N = 64
TILE_K = 8
WARP_PER_ROW = 2
WARP_PER_COL = 2
WARP_MAT_M = 32
WARP_MAT_N = 32
VEC_SIZE = 8
THREADS = 256
ACC_ROWS = 4
ACC_COLS = 4


@avelang.jit
def fused_kernel(
    X_ptr: al.Pointer(al.bf16),
    W_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    addv_ptr: al.Pointer(al.bf16),
    Y_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    k: al.u32,
):
    tid = al.thread_id(0)
    wid = tid // 64
    wtid = tid % 64
    warp_row = wid // WARP_PER_COL
    warp_col = wid % WARP_PER_COL

    n_groups = n // TILE_N
    block_id = al.block_id(0)
    group_m = block_id // n_groups
    group_n = block_id - group_m * n_groups

    X = al.make_tensor(X_ptr, al.bf16, al.make_layout((m, k), (k, 1)))
    W = al.make_tensor(W_ptr, al.bf16, al.make_layout((n, k), (k, 1)))
    bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((n,), (1,)))
    addv = al.make_tensor(addv_ptr, al.bf16, al.make_layout((n,), (1,)))
    Y = al.make_tensor(Y_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    x_rsrc = al.amdgpu.make_rsrc(X, m * k * 2)
    w_rsrc = al.amdgpu.make_rsrc(W, n * k * 2)

    shm_a = al.make_shared((TILE_M * TILE_K,), al.bf16)
    shm_b = al.make_shared((TILE_N * TILE_K,), al.bf16)

    acc = al.make_local((ACC_ROWS, ACC_COLS), al.f32)
    for ri in al.range(ACC_ROWS):
        for ci in al.range(ACC_COLS):
            acc[ri, ci] = al.convert(0.0, al.f32)

    zero = al.convert(0, al.u32)
    k_tiles = k // TILE_K

    for kt in al.range(k_tiles):
        if tid < TILE_M:
            g_row_a = group_m * TILE_M + tid
            g_off_a = (g_row_a * k + kt * TILE_K) * 2
            packed_a = al.amdgpu.raw_buffer_load_x4(x_rsrc, g_off_a, zero, 0)
            frag_a = al.view(packed_a, al.Tensor((VEC_SIZE,), al.bf16))
            lds_off_a = tid * TILE_K
            for v in al.range(VEC_SIZE):
                shm_a[lds_off_a + v] = frag_a[v]

        if tid < TILE_N:
            g_row_b = group_n * TILE_N + tid
            g_off_b = (g_row_b * k + kt * TILE_K) * 2
            packed_b = al.amdgpu.raw_buffer_load_x4(w_rsrc, g_off_b, zero, 0)
            frag_b = al.view(packed_b, al.Tensor((VEC_SIZE,), al.bf16))
            lds_off_b = tid * TILE_K
            for v in al.range(VEC_SIZE):
                shm_b[lds_off_b + v] = frag_b[v]

        al.syncthreads()

        row_group = wtid % 8
        col_group = wtid // 8

        for kk in al.range(TILE_K):
            for ri in al.range(ACC_ROWS):
                a_row = warp_row * WARP_MAT_M + row_group * ACC_ROWS + ri
                a_val = al.convert(shm_a[a_row * TILE_K + kk], al.f32)
                for ci in al.range(ACC_COLS):
                    b_row = warp_col * WARP_MAT_N + col_group * ACC_COLS + ci
                    b_val = al.convert(shm_b[b_row * TILE_K + kk], al.f32)
                    acc[ri, ci] = acc[ri, ci] + a_val * b_val

        al.syncthreads()

    one_f32 = al.convert(1.0, al.f32)
    neg_one = al.convert(-1.0, al.f32)
    sqrt2 = al.convert(SQRT_2, al.f32)
    half = al.convert(0.5, al.f32)

    row_group = wtid % 8
    col_group = wtid // 8

    for ri in al.range(ACC_ROWS):
        g_row = group_m * TILE_M + warp_row * WARP_MAT_M + row_group * ACC_ROWS + ri
        for ci in al.range(ACC_COLS):
            g_col = group_n * TILE_N + warp_col * WARP_MAT_N + col_group * ACC_COLS + ci

            val = acc[ri, ci]

            b_val = al.convert(bias[g_col], al.f32)
            a_val = al.convert(addv[g_col], al.f32)
            val = val + b_val + a_val

            val = val * (one_f32 / (one_f32 + al.exp(-val)))

            val = al.tanh(val)

            val = half * val * (one_f32 + al.erf(val / sqrt2))

            if val < neg_one:
                val = neg_one
            if val > one_f32:
                val = one_f32

            Y[g_row, g_col] = al.convert(val, al.bf16)


class ModelNew(nn.Module):

    def __init__(self, in_features, out_features, add_value_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))

    def forward(self, x):
        if x.dtype != torch.bfloat16:
            raise RuntimeError("This fused kernel only supports bf16 input.")
        m_val = x.shape[0]
        k_val = x.shape[1]
        n_val = self.matmul.out_features

        if m_val % TILE_M != 0 or n_val % TILE_N != 0 or k_val % TILE_K != 0:
            raise RuntimeError("Dimensions must be multiples of tile sizes.")

        w = self.matmul.weight.to(device=x.device, dtype=x.dtype).contiguous()
        bias = self.matmul.bias.to(device=x.device, dtype=x.dtype).contiguous()
        addv = self.add_value.to(device=x.device, dtype=x.dtype).contiguous()
        y = torch.empty((m_val, n_val), device=x.device, dtype=x.dtype)

        grid_0 = (m_val // TILE_M) * (n_val // TILE_N)
        fused_kernel[lambda: ((grid_0, 1, 1), (THREADS, 1, 1))](
            x.contiguous(), w, bias, addv, y,
            m_val, n_val, k_val,
        )
        return y
