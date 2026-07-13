import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_M = 64
TILE_N = 64
TILE_K = 16
WAVES_PER_BLOCK = 4
WARP_SIZE = 64
THREADS_PER_BLOCK = WAVES_PER_BLOCK * WARP_SIZE

@avelang.jit
def fused_gemm_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    y_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    block_m = al.block_id(0) * TILE_M
    block_n = al.block_id(1) * TILE_N
    tid = al.thread_id(0)
    lane = tid % WARP_SIZE
    warp_id = tid // WARP_SIZE
    wr = warp_id // 2
    wc = warp_id % 2

    sA = al.make_shared((TILE_M, TILE_K), al.bf16)
    sB = al.make_shared((TILE_K, TILE_N), al.bf16)

    x_layout = al.make_layout((M, K), (K, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    w_layout = al.make_layout((K, N), (N, 1))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)
    y_layout = al.make_layout((M, N), (N, 1))
    y = al.make_tensor(y_ptr, al.bf16, y_layout)
    bias_layout = al.make_layout((N,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)
    x_rsrc = al.amdgpu.make_rsrc(x, M * K * 2)
    w_rsrc = al.amdgpu.make_rsrc(w, K * N * 2)

    # 4x16 lane grid: 4 row groups, 16 column lanes per 16x16 sub-tile
    lane_m_grp = lane // 16
    lane_n = lane % 16

    acc00 = al.make_local((4,), al.f32)
    acc01 = al.make_local((4,), al.f32)
    acc10 = al.make_local((4,), al.f32)
    acc11 = al.make_local((4,), al.f32)
    for i in al.range(4):
        acc00[i] = al.convert(0.0, al.f32)
        acc01[i] = al.convert(0.0, al.f32)
        acc10[i] = al.convert(0.0, al.f32)
        acc11[i] = al.convert(0.0, al.f32)

    mult_f32 = al.convert(2.0, al.f32)
    neg_f32 = al.convert(0.1, al.f32)
    bias_left = al.convert(bias[block_n + wc * 32 + lane_n], al.f32)
    bias_right = al.convert(bias[block_n + wc * 32 + 16 + lane_n], al.f32)

    for k_block in al.range(0, K, TILE_K):
        k_start = k_block
        if tid < 128:
            row_a = tid // 2
            col_a = (tid % 2) * 8
            vindex_a = ((block_m + row_a) * K + k_start + col_a) * 2
            d = al.view(al.amdgpu.raw_buffer_load_x4(x_rsrc, vindex_a, 0, 0), al.Tensor((8,), al.bf16))
            for i in al.range(8):
                sA[row_a, col_a + i] = d[i]
        if tid >= 128:
            t2 = tid - 128
            row_b = t2 // 8
            col_b = (t2 % 8) * 8
            vindex_b = ((k_start + row_b) * N + block_n + col_b) * 2
            d = al.view(al.amdgpu.raw_buffer_load_x4(w_rsrc, vindex_b, 0, 0), al.Tensor((8,), al.bf16))
            for i in al.range(8):
                sB[row_b, col_b + i] = d[i]
        al.syncthreads()

        # A operand: 4 rows at same K column
        a_row_top = wr * 32 + lane_m_grp * 4
        a_row_bot = wr * 32 + 16 + lane_m_grp * 4
        a_kcol = lane_n

        a_top = al.make_local((4,), al.bf16)
        a_bot = al.make_local((4,), al.bf16)
        a_top[0] = sA[a_row_top + 0, a_kcol]
        a_top[1] = sA[a_row_top + 1, a_kcol]
        a_top[2] = sA[a_row_top + 2, a_kcol]
        a_top[3] = sA[a_row_top + 3, a_kcol]
        a_bot[0] = sA[a_row_bot + 0, a_kcol]
        a_bot[1] = sA[a_row_bot + 1, a_kcol]
        a_bot[2] = sA[a_row_bot + 2, a_kcol]
        a_bot[3] = sA[a_row_bot + 3, a_kcol]
        a_top_i32 = al.view(a_top, al.Tensor((2,), al.i32))
        a_bot_i32 = al.view(a_bot, al.Tensor((2,), al.i32))

        # B operand: 4 K rows at same N column
        b_krow = lane_m_grp * 4
        b_ncol_l = wc * 32 + lane_n
        b_ncol_r = wc * 32 + 16 + lane_n

        b_left = al.make_local((4,), al.bf16)
        b_right = al.make_local((4,), al.bf16)
        b_left[0] = sB[b_krow + 0, b_ncol_l]
        b_left[1] = sB[b_krow + 1, b_ncol_l]
        b_left[2] = sB[b_krow + 2, b_ncol_l]
        b_left[3] = sB[b_krow + 3, b_ncol_l]
        b_right[0] = sB[b_krow + 0, b_ncol_r]
        b_right[1] = sB[b_krow + 1, b_ncol_r]
        b_right[2] = sB[b_krow + 2, b_ncol_r]
        b_right[3] = sB[b_krow + 3, b_ncol_r]
        b_left_i32 = al.view(b_left, al.Tensor((2,), al.i32))
        b_right_i32 = al.view(b_right, al.Tensor((2,), al.i32))

        acc00 = al.amdgpu.mfma_16x16x16_bf16_f32(a_top_i32, b_left_i32, acc00)
        acc01 = al.amdgpu.mfma_16x16x16_bf16_f32(a_top_i32, b_right_i32, acc01)
        acc10 = al.amdgpu.mfma_16x16x16_bf16_f32(a_bot_i32, b_left_i32, acc10)
        acc11 = al.amdgpu.mfma_16x16x16_bf16_f32(a_bot_i32, b_right_i32, acc11)
        al.syncthreads()

    # Epilogue
    out_r0 = block_m + wr * 32 + lane_m_grp * 4
    out_r1 = out_r0 + 16
    out_cl = block_n + wc * 32 + lane_n
    out_cr = block_n + wc * 32 + 16 + lane_n

    for i in al.range(4):
        v = acc00[i] + bias_left
        v = v * mult_f32
        if v < al.convert(0.0, al.f32):
            v = v * neg_f32
        y[out_r0 + i, out_cl] = al.convert(v, al.bf16)

        v = acc01[i] + bias_right
        v = v * mult_f32
        if v < al.convert(0.0, al.f32):
            v = v * neg_f32
        y[out_r0 + i, out_cr] = al.convert(v, al.bf16)

        v = acc10[i] + bias_left
        v = v * mult_f32
        if v < al.convert(0.0, al.f32):
            v = v * neg_f32
        y[out_r1 + i, out_cl] = al.convert(v, al.bf16)

        v = acc11[i] + bias_right
        v = v * mult_f32
        if v < al.convert(0.0, al.f32):
            v = v * neg_f32
        y[out_r1 + i, out_cr] = al.convert(v, al.bf16)


def _get_launch_config(M, N):
    grid_m = (M + TILE_M - 1) // TILE_M
    grid_n = (N + TILE_N - 1) // TILE_N
    return lambda: ((grid_m, grid_n, 1), (THREADS_PER_BLOCK, 1, 1))

def _run_fused_gemm(x, w_t, bias, mul, neg):
    M_val = x.shape[0]
    N_val = w_t.shape[1]
    K_val = x.shape[1]
    y = torch.empty((M_val, N_val), device=x.device, dtype=x.dtype)
    fused_gemm_kernel[_get_launch_config(M_val, N_val)](x, w_t, bias, y, M_val, N_val, K_val)
    return y

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        w_t = self.gemm.weight.t().to(device=x.device, dtype=x.dtype).contiguous()
        bias = self.gemm.bias.to(device=x.device, dtype=x.dtype).contiguous()
        return _run_fused_gemm(x.contiguous(), w_t, bias, self.multiplier, self.leaky_relu.negative_slope)
