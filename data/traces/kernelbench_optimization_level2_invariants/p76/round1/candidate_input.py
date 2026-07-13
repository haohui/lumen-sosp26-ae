import torch
import torch.nn as nn
import avelang
import avelang.language as al

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WARP_M = 32
WARP_N = 32
THREADS_PER_WAVE = 64
WAVES_PER_BLOCK = 4
NUM_THREADS = WAVES_PER_BLOCK * THREADS_PER_WAVE

@avelang.jit
def fused_gemm_bias_relu(
    X_ptr: al.Pointer(al.bf16),
    W_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    Y_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    tid = al.thread_id(0)
    bid_m = al.block_id(0)
    bid_n = al.block_id(1)

    X = al.make_tensor(X_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    W = al.make_tensor(W_ptr, al.bf16, al.make_layout((K, N), (N, 1)))
    bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((N,), (1,)))
    Y = al.make_tensor(Y_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    m_block = bid_m * 64
    n_block = bid_n * 64

    warp_id = tid // 64
    warp_row = warp_id // 2
    warp_col = warp_id % 2
    lane_id = tid % 64

    m_warp = m_block + warp_row * 32
    n_warp = n_block + warp_col * 32

    A_smem = al.make_shared((64, 16), al.bf16)
    B_smem = al.make_shared((16, 64), al.bf16)

    A_u32 = al.view(A_smem, al.u32, al.make_layout((64, 8), (8, 1)))
    B_u32 = al.view(B_smem, al.u32, al.make_layout((16, 32), (32, 1)))

    rsrc_X = al.amdgpu.make_rsrc(X, 134217728)
    rsrc_W = al.amdgpu.make_rsrc(W, 134217728)

    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = al.convert(0.0, al.f32)

    a_frag = al.make_local((2,), al.u32)
    b_frag = al.make_local((2,), al.u32)

    for k_block in al.range(0, K, 16):
        if tid < 128:
            row_a = tid % 64
            col_group_a = tid // 64
            offset_el_a = (m_block + row_a) * K + (k_block + col_group_a * 8)
            offset_bytes_a = offset_el_a * 2
            data_a = al.amdgpu.raw_buffer_load_x4(rsrc_X, offset_bytes_a, 0, 0)
            bf16s_a = al.view(data_a, al.Tensor((8,), al.bf16))
            for j in al.range(8):
                A_smem[row_a, col_group_a * 8 + j] = bf16s_a[j]

        if tid >= 128:
            t = tid - 128
            row_b = t // 8
            col_group_b = t % 8
            offset_el_b = (k_block + row_b) * N + (n_block + col_group_b * 8)
            offset_bytes_b = offset_el_b * 2
            data_b = al.amdgpu.raw_buffer_load_x4(rsrc_W, offset_bytes_b, 0, 0)
            bf16s_b = al.view(data_b, al.Tensor((8,), al.bf16))
            for j in al.range(8):
                B_smem[row_b, col_group_b * 8 + j] = bf16s_b[j]

        al.syncthreads()

        local_row = lane_id % 32
        m_row = warp_row * 32 + local_row
        col_base = 4 * (lane_id // 32)
        a_col_u32 = col_base // 2
        b_col_off_u32 = warp_col * 16

        # Step 0
        a_frag[0] = A_u32[m_row, a_col_u32]
        a_frag[1] = A_u32[m_row, a_col_u32 + 1]

        if lane_id < 8:
            b_frag[0] = B_u32[lane_id, b_col_off_u32]
            b_frag[1] = B_u32[lane_id, b_col_off_u32 + 1]
        else:
            if lane_id >= 32:
                if lane_id < 40:
                    b_frag[0] = B_u32[lane_id - 32, b_col_off_u32 + 2]
                    b_frag[1] = B_u32[lane_id - 32, b_col_off_u32 + 3]
                else:
                    b_frag[0] = al.convert(0, al.u32)
                    b_frag[1] = al.convert(0, al.u32)
            else:
                b_frag[0] = al.convert(0, al.u32)
                b_frag[1] = al.convert(0, al.u32)

        a_vec = al.view(a_frag, al.Tensor((2,), al.u32))
        b_vec = al.view(b_frag, al.Tensor((2,), al.u32))
        acc_vec = al.view(acc, al.Tensor((16,), al.f32))
        acc_vec = al.amdgpu.mfma_32x32x8_bf16_f32(a_vec, b_vec, acc_vec)
        for i in al.range(16):
            acc[i] = acc_vec[i]

        # Step 1
        a_frag[0] = A_u32[m_row, a_col_u32 + 4]
        a_frag[1] = A_u32[m_row, a_col_u32 + 5]

        if lane_id < 8:
            b_frag[0] = B_u32[lane_id + 8, b_col_off_u32]
            b_frag[1] = B_u32[lane_id + 8, b_col_off_u32 + 1]
        else:
            if lane_id >= 32:
                if lane_id < 40:
                    b_frag[0] = B_u32[lane_id - 32 + 8, b_col_off_u32 + 2]
                    b_frag[1] = B_u32[lane_id - 32 + 8, b_col_off_u32 + 3]
                else:
                    b_frag[0] = al.convert(0, al.u32)
                    b_frag[1] = al.convert(0, al.u32)
            else:
                b_frag[0] = al.convert(0, al.u32)
                b_frag[1] = al.convert(0, al.u32)

        a_vec = al.view(a_frag, al.Tensor((2,), al.u32))
        b_vec = al.view(b_frag, al.Tensor((2,), al.u32))
        acc_vec = al.view(acc, al.Tensor((16,), al.f32))
        acc_vec = al.amdgpu.mfma_32x32x8_bf16_f32(a_vec, b_vec, acc_vec)
        for i in al.range(16):
            acc[i] = acc_vec[i]

        al.syncthreads()

    out_col = n_warp + lane_id % 32
    for ai in al.range(16):
        row_off = 8 * (ai // 4) + 4 * (lane_id // 32) + (ai % 4)
        out_row = m_warp + row_off
        val = acc[ai] + al.convert(bias[out_col], al.f32)
        if val < al.convert(0.0, al.f32):
            val = al.convert(0.0, al.f32)
        Y[out_row, out_col] = al.convert(val, al.bf16)


def _launch():
    grid_m = (BATCH_SIZE + BLOCK_M - 1) // BLOCK_M
    grid_n = (OUT_FEATURES + BLOCK_N - 1) // BLOCK_N
    return ((grid_m, grid_n, 1), (NUM_THREADS, 1, 1))


class ModelNew(nn.Module):

    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        M, K_in = x.shape
        N_out = self.bias.shape[0]

        w = self.gemm.weight.t().to(device=x.device, dtype=torch.bfloat16).contiguous()
        bias = self.bias.to(device=x.device, dtype=torch.bfloat16).contiguous()
        y = torch.empty((M, N_out), device=x.device, dtype=torch.bfloat16)

        fused_gemm_bias_relu[_launch](
            x.contiguous(), w, bias, y,
            M, N_out, K_in,
        )
        return y
