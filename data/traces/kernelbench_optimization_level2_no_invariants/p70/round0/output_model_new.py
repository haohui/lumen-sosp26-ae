import torch
import torch.nn as nn
import avelang
import avelang.language as al

BATCH = 1024
DIM = 8192
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64
WARP_M = 32
WARP_N = 32
THREADS_PER_BLOCK = 256
TILES_M = 2
TILES_N = 2


@avelang.jit
def fused_gemm_kernel(
    X: al.Tensor((BATCH, DIM), al.bf16),
    W: al.Tensor((DIM, DIM), al.bf16),
    Bias: al.Tensor((DIM,), al.bf16),
    Y: al.Tensor((BATCH, DIM), al.bf16),
    scale: al.constexpr,
):
    bx = al.block_id(0)
    by = al.block_id(1)
    tid = al.thread_id(0)

    warp_id = tid >> 6
    warp_m = warp_id >> 1
    warp_n = warp_id & 1
    lane_id = tid & 63

    m_base = by * BLOCK_M + warp_m * WARP_M
    n_base = bx * BLOCK_N + warp_n * WARP_N

    As = al.make_shared((BLOCK_M, BLOCK_K), al.bf16)
    Bs = al.make_shared((BLOCK_N, BLOCK_K), al.bf16)

    X_rsrc = al.amdgpu.make_rsrc(X, BATCH * DIM * 2)
    W_rsrc = al.amdgpu.make_rsrc(W, DIM * DIM * 2)

    acc = al.make_local((TILES_M, TILES_N, 4), al.f32)
    zero = al.convert(0.0, al.f32)
    for tm in al.range(TILES_M):
        for tn in al.range(TILES_N):
            for i in al.range(4):
                acc[tm, tn, i] = zero

    for k_block in al.range(0, DIM, BLOCK_K):
        for r in al.range(2):
            p = tid + r * THREADS_PER_BLOCK
            row = p >> 3
            col = (p & 7) << 3
            g_row = by * BLOCK_M + row
            g_col = k_block + col
            vindex = (g_row * DIM + g_col) << 1
            a_vec_u32 = al.amdgpu.raw_buffer_load_x4(X_rsrc, vindex, 0, 0)
            a_vec_bf16 = al.view(a_vec_u32, al.Tensor((8,), al.bf16))
            for i in al.range(8):
                As[row, col + i] = a_vec_bf16[i]

        for r in al.range(2):
            p = tid + r * THREADS_PER_BLOCK
            row = p >> 3
            col = (p & 7) << 3
            g_row = k_block + row
            g_col = bx * BLOCK_N + col
            vindex = (g_row * DIM + g_col) << 1
            b_vec_u32 = al.amdgpu.raw_buffer_load_x4(W_rsrc, vindex, 0, 0)
            b_vec_bf16 = al.view(b_vec_u32, al.Tensor((8,), al.bf16))
            for i in al.range(8):
                Bs[col + i, row] = b_vec_bf16[i]

        al.syncthreads()

        a_row_base = lane_id & 15
        b_col_base = lane_id & 15
        k_group = lane_id >> 4

        for batch in al.range(2):
            k_base = batch << 5

            a_data = al.make_local((TILES_M, 2, 4), al.bf16)
            for tm in al.range(TILES_M):
                a_row = warp_m * WARP_M + a_row_base + tm * 16
                for i in al.range(4):
                    a_data[tm, 0, i] = As[a_row, k_base + k_group + i * 4]
                    a_data[tm, 1, i] = As[a_row, k_base + 16 + k_group + i * 4]

            b_data = al.make_local((TILES_N, 2, 4), al.bf16)
            for tn in al.range(TILES_N):
                b_col = warp_n * WARP_N + b_col_base + tn * 16
                for i in al.range(4):
                    b_data[tn, 0, i] = Bs[b_col, k_base + k_group + i * 4]
                    b_data[tn, 1, i] = Bs[b_col, k_base + 16 + k_group + i * 4]

            for tm in al.range(TILES_M):
                for tn in al.range(TILES_N):
                    a0_u32 = al.view(a_data[tm, 0], al.Tensor((2,), al.u32))
                    a1_u32 = al.view(a_data[tm, 1], al.Tensor((2,), al.u32))
                    b0_u32 = al.view(b_data[tn, 0], al.Tensor((2,), al.u32))
                    b1_u32 = al.view(b_data[tn, 1], al.Tensor((2,), al.u32))
                    acc[tm, tn] = al.amdgpu.mfma_16x16x16_bf16_f32(
                        a0_u32, b0_u32, acc[tm, tn])
                    acc[tm, tn] = al.amdgpu.mfma_16x16x16_bf16_f32(
                        a1_u32, b1_u32, acc[tm, tn])

        al.syncthreads()

    row_group = lane_id >> 4
    thread_col = lane_id & 15

    one = al.convert(1.0, al.f32)
    scale_f32 = al.convert(scale, al.f32)

    for tm in al.range(TILES_M):
        for tn in al.range(TILES_N):
            g_col = n_base + tn * 16 + thread_col
            bias_val = al.convert(Bias[g_col], al.f32)
            for i in al.range(4):
                v = acc[tm, tn, i] + bias_val
                s = one / (one + al.exp(zero - v))
                acc[tm, tn, i] = v + s * scale_f32

    for tm in al.range(TILES_M):
        for tn in al.range(TILES_N):
            g_base_m = m_base + tm * 16
            g_base_n = n_base + tn * 16
            for i in al.range(4):
                g_row = g_base_m + row_group * 4 + i
                g_col = g_base_n + thread_col
                Y[g_row, g_col] = al.convert(acc[tm, tn, i], al.bf16)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        w = self.gemm.weight.t().to(device=x.device, dtype=torch.bfloat16).contiguous()
        bias = self.gemm.bias.to(device=x.device, dtype=torch.bfloat16).contiguous()
        y = torch.empty((BATCH, DIM), device=x.device, dtype=torch.bfloat16)

        def _launch():
            grid_m = (BATCH + BLOCK_M - 1) // BLOCK_M
            grid_n = (DIM + BLOCK_N - 1) // BLOCK_N
            return ((grid_n, grid_m, 1), (THREADS_PER_BLOCK, 1, 1))

        fused_gemm_kernel[_launch](
            x.contiguous(), w, bias, y, self.scaling_factor,
        )
        return y
