import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 16
NUM_WARPS = 4
THREADS = NUM_WARPS * 64


def _launch():
    # Each block covers 64x64 (4 waves x 32x32)
    # Grid: N/64 blocks in X dim, M/64 blocks in Y dim
    return ((128, 16, 1), (THREADS, 1, 1))


@avelang.jit
def fused_gemm_relu_div_kernel(
    X_ptr: al.Pointer(al.bf16),
    W_ptr: al.Pointer(al.bf16),
    Bias_ptr: al.Pointer(al.bf16),
    Y_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
    LDX: al.i32,
    LDW: al.i32,
    LDY: al.i32,
    divisor: al.constexpr,
):
    bid_n = al.block_id(0)
    bid_m = al.block_id(1)
    tid = al.thread_id(0)
    lane = tid % 64
    lane_col = lane & 31
    lane_group = lane >> 5
    warp_id = tid >> 6
    warp_row = warp_id >> 1
    warp_col = warp_id & 1

    tile_m_base = bid_m * 64 + warp_row * BLOCK_M
    tile_n_base = bid_n * 64 + warp_col * BLOCK_N

    # Tensor views: X is (M, K), W is (N, K) like the tutorial's A(M,K) and B(N,K)
    X_bf16 = al.make_tensor(X_ptr, al.bf16, al.make_layout((M, K), (LDX, al.convert(1, al.i32))))
    W_bf16 = al.make_tensor(W_ptr, al.bf16, al.make_layout((N, K), (LDW, al.convert(1, al.i32))))
    Y_bf16 = al.make_tensor(Y_ptr, al.bf16, al.make_layout((M, N), (LDY, al.convert(1, al.i32))))
    bias_layout = al.make_layout((N,), (al.convert(1, al.i32),))
    bias_tensor = al.make_tensor(Bias_ptr, al.bf16, bias_layout)

    # Packed i32 vector views: each i32 packs 2 bf16 values
    k_groups = K >> 3
    x_packed_stride = K >> 1
    w_packed_stride = K >> 1

    X_vec = al.view(X_bf16, al.i32, al.make_layout((M, k_groups, 4), (x_packed_stride, 4, 1)))
    W_vec = al.view(W_bf16, al.i32, al.make_layout((N, k_groups, 4), (w_packed_stride, 4, 1)))

    # Accumulator: 16 f32 values per thread (vector type)
    acc = al.full((16,), al.convert(0.0, al.f32), al.f32)

    # LDS for X and W tiles: 1D with 4-i32 entries, 64 entries per wave
    lds_size = NUM_WARPS * 32 * 2
    x_smem = al.make_shared((lds_size, 4), al.i32)
    w_smem = al.make_shared((lds_size, 4), al.i32)
    warp_offset = warp_id * 64

    K_STEPS = K // 16
    for k_idx in al.range(K_STEPS):
        k_step = k_idx * 16
        k_vec = (k_step >> 3) + lane_group

        # Load X tile: each thread loads 8 bf16 (4 i32) from its M-row
        x_smem[warp_offset + lane] = X_vec[tile_m_base + lane_col, k_vec]

        # Load W tile: W is (N, K), load like B in tutorial
        w_smem[warp_offset + lane] = W_vec[tile_n_base + lane_col, k_vec]

        al.syncthreads()

        # Read back; split each 16-byte word into two MFMA fragments
        x_words = x_smem[warp_offset + lane]
        w_words = w_smem[warp_offset + lane]

        x_frag = al.view(x_words, al.Tensor((2, 2, 1), al.u32))
        w_frag = al.view(w_words, al.Tensor((2, 2, 1), al.u32))

        # Two MFMA ops covering K=16 (B operand first, then A)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(x_frag[0], w_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(x_frag[1], w_frag[1], acc)

        al.syncthreads()

    # Post-processing: bias + ReLU + divide, write to output
    out_col = tile_n_base + lane_col
    for a in al.range(16):
        row_offset = ((a >> 2) << 3) + lane_group * 4 + (a & 3)
        out_row = tile_m_base + row_offset
        val = acc[a]
        b = al.convert(bias_tensor[out_col], al.f32)
        val = val + b
        zero = al.convert(0.0, al.f32)
        if val < zero:
            val = zero
        val = al.convert(val / divisor, al.f32)
        Y_bf16[out_row, out_col] = al.convert(val, al.bf16)


class ModelNew(nn.Module):

    def __init__(self, in_features, out_features, divisor):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor

    def forward(self, x):
        M_val = x.shape[0]
        N_val = self.linear.out_features
        K_val = x.shape[1]

        x_dev = x.to(dtype=torch.bfloat16).contiguous()
        # Pass weight directly as (N, K) to match the tutorial's B(N, K) layout
        w_dev = self.linear.weight.to(device=x_dev.device, dtype=torch.bfloat16).contiguous()
        bias = self.linear.bias.to(device=x_dev.device, dtype=torch.bfloat16).contiguous()
        y = torch.empty((M_val, N_val), device=x_dev.device, dtype=torch.bfloat16)

        fused_gemm_relu_div_kernel[_launch](
            x_dev,
            w_dev,
            bias,
            y,
            M_val,
            N_val,
            K_val,
            x_dev.stride(0),
            w_dev.stride(0),
            y.stride(0),
            self.divisor,
        )
        return y
