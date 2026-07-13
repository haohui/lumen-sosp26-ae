import torch
import torch.nn as nn
import avelang
import avelang.language as al

WARP_SIZE = 64
NUM_WARPS = 4
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
SUBTRACT_VALUE = 2.0
MULTIPLY_VALUE = 1.5


@avelang.jit
def fused_gemm_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    y_ptr: al.Pointer(al.bf16),
    M: al.u32,
    N: al.u32,
    K: al.u32,
):
    lane = al.thread_id(0)
    warp_id = lane // WARP_SIZE
    wtid = lane % WARP_SIZE
    warp_row = warp_id // 2
    warp_col = warp_id % 2
    lane_col = wtid & 31
    lane_group_warp = wtid >> 5

    block_m = al.block_id(1) * BLOCK_M
    block_n = al.block_id(0) * BLOCK_N

    k_vecs = K >> 3
    packed_row_stride = K >> 1

    X_bf16 = al.make_tensor(x_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    W_bf16 = al.make_tensor(w_ptr, al.bf16, al.make_layout((N, K), (K, 1)))
    Y_bf16 = al.make_tensor(y_ptr, al.bf16, al.make_layout((M, N), (N, 1)))
    bias_bf16 = al.make_tensor(bias_ptr, al.bf16, al.make_layout((N,), (1,)))

    X_vec = al.view(
        X_bf16, al.i32, al.make_layout((M, k_vecs, 4), (packed_row_stride, 4, 1))
    )
    W_vec = al.view(
        W_bf16, al.i32, al.make_layout((N, k_vecs, 4), (packed_row_stride, 4, 1))
    )

    smem_entries = (BLOCK_M // 2) * (BLOCK_K >> 3)
    a_smem = al.make_shared((NUM_WARPS * smem_entries, BLOCK_K >> 2), al.i32)
    b_smem = al.make_shared((NUM_WARPS * smem_entries, BLOCK_K >> 2), al.i32)
    c_smem = al.make_shared((BLOCK_M, BLOCK_N), al.f32)

    acc = al.full((16,), 0.0, al.f32)

    warp_lds_base = warp_id * smem_entries + wtid
    warp_a_row = block_m + warp_row * 32 + lane_col
    warp_b_row = block_n + warp_col * 32 + lane_col

    for kt in al.range(K // BLOCK_K):
        k_vec = kt * 2 + lane_group_warp

        a_smem[warp_lds_base] = X_vec[warp_a_row, k_vec]
        b_smem[warp_lds_base] = W_vec[warp_b_row, k_vec]

        al.syncthreads()

        a_words = a_smem[warp_lds_base]
        b_words = b_smem[warp_lds_base]
        a_frag = al.view(a_words, al.Tensor((2, 2, 1), al.u32))
        b_frag = al.view(b_words, al.Tensor((2, 2, 1), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)

        al.syncthreads()

    for r in al.range(16):
        row_offset = ((r >> 2) << 3) + lane_group_warp * 4 + (r & 3)
        c_smem[warp_row * 32 + row_offset, warp_col * 32 + lane_col] = acc[r]

    al.syncthreads()

    elements_per_thread = (BLOCK_M * BLOCK_N) // (WARP_SIZE * NUM_WARPS)
    for e in al.range(elements_per_thread):
        idx = lane + e * WARP_SIZE * NUM_WARPS
        r = idx // BLOCK_N
        c = idx - r * BLOCK_N

        val = c_smem[r, c]
        global_c = block_n + c
        bias_val = al.convert(bias_bf16[global_c], al.f32)
        val = val + bias_val
        val = (val - al.convert(SUBTRACT_VALUE, al.f32)) * al.convert(
            MULTIPLY_VALUE, al.f32
        )
        if val < al.convert(0.0, al.f32):
            val = al.convert(0.0, al.f32)

        global_r = block_m + r
        Y_bf16[global_r, global_c] = al.convert(val, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x):
        M_val = x.shape[0]
        K_val = x.shape[1]
        N_val = self.linear.out_features

        w = self.linear.weight.to(device=x.device, dtype=torch.bfloat16).contiguous()
        bias = (
            self.linear.bias.to(device=x.device, dtype=torch.bfloat16).contiguous()
        )
        x_bf16 = x.to(dtype=torch.bfloat16).contiguous()
        y = torch.empty((M_val, N_val), device=x.device, dtype=torch.bfloat16)

        grid_x = N_val // BLOCK_N
        grid_y = M_val // BLOCK_M

        fused_gemm_kernel[
            lambda: ((grid_x, grid_y, 1), (WARP_SIZE * NUM_WARPS, 1, 1))
        ](x_bf16, w, bias, y, M_val, N_val, K_val)
        return y
