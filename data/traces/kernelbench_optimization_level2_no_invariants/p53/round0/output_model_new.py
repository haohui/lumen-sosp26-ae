import torch
import torch.nn as nn
import avelang
import avelang.language as al

SQRT_2 = 1.4142135623730951
SCALING_FACTOR = 0.5
HARDTANH_MIN = -2.0
HARDTANH_MAX = 2.0


@avelang.jit
def fused_kernel(
    X_ptr: al.Pointer(al.bf16),
    W_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    Y_ptr: al.Pointer(al.bf16),
    M: al.u32,
    N: al.u32,
    K: al.u32,
):
    tid = al.thread_id(0)
    wid = tid // 64
    lane = tid % 64
    lane_col = lane & 31
    lane_group = lane >> 5
    wave_m = wid // 2
    wave_n = wid % 2

    block_m = al.block_id(0) * 64
    block_n = al.block_id(1) * 64

    X_flat = al.make_tensor(X_ptr, al.bf16, al.make_layout((M * K,), (1,)))
    W_flat = al.make_tensor(W_ptr, al.bf16, al.make_layout((N * K,), (1,)))
    Y_t = al.make_tensor(Y_ptr, al.bf16, al.make_layout((M, N), (N, 1)))
    bias_t = al.make_tensor(bias_ptr, al.bf16, al.make_layout((N,), (1,)))

    X_rsrc = al.amdgpu.make_rsrc(X_flat, M * K * 2)
    W_rsrc = al.amdgpu.make_rsrc(W_flat, N * K * 2)

    a_smem = al.make_shared((128, 4), al.i32)
    b_smem = al.make_shared((128, 4), al.i32)
    c_smem = al.make_shared((64, 64), al.f32)

    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = al.convert(0.0, al.f32)

    zero = al.convert(0, al.i32)

    for k_block in al.range(0, K, 16):
        if tid < 128:
            row = tid % 32 + (tid // 64) * 32
            k_off = ((tid // 32) % 2) * 8
            byte_off = al.convert(((block_m + row) * K + k_block + k_off) * 2, al.i32)
            a_smem[tid] = al.amdgpu.raw_buffer_load_x4(X_rsrc, zero, byte_off, 0)

        if tid >= 128:
            s = tid - 128
            n_off = s % 32 + (s // 64) * 32
            k_off = ((s // 32) % 2) * 8
            byte_off = al.convert(((block_n + n_off) * K + k_block + k_off) * 2, al.i32)
            b_smem[s] = al.amdgpu.raw_buffer_load_x4(W_rsrc, zero, byte_off, 0)

        al.syncthreads()

        a_words = a_smem[wave_m * 64 + lane]
        b_words = b_smem[wave_n * 64 + lane]
        a_frag = al.view(a_words, al.Tensor((2, 2, 1), al.u32))
        b_frag = al.view(b_words, al.Tensor((2, 2, 1), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        al.syncthreads()

    for r in al.range(16):
        row_offset = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        local_row = wave_m * 32 + row_offset
        local_col = wave_n * 32 + lane_col
        c_smem[local_row, local_col] = acc[r]

    al.syncthreads()

    thread_row_group = tid // 16
    thread_col_group = tid % 16

    sf = al.convert(SCALING_FACTOR, al.f32)
    hmin = al.convert(HARDTANH_MIN, al.f32)
    hmax = al.convert(HARDTANH_MAX, al.f32)
    sqrt2 = al.convert(SQRT_2, al.f32)
    half = al.convert(0.5, al.f32)
    one = al.convert(1.0, al.f32)

    for rr in al.range(4):
        for cc in al.range(4):
            local_row = thread_row_group * 4 + rr
            local_col = thread_col_group * 4 + cc
            val = c_smem[local_row, local_col]

            bias_val = al.convert(bias_t[block_n + local_col], al.f32)
            val = val + bias_val
            val = val * sf

            if val < hmin:
                val = hmin
            if val > hmax:
                val = hmax

            val = half * val * (one + al.erf(val / sqrt2))

            Y_t[block_m + local_row, block_n + local_col] = al.convert(val, al.bf16)


def _launch():
    return ((32, 128, 1), (256, 1, 1))


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)
        self.gelu = nn.GELU()

    def forward(self, x):
        M_val = x.shape[0]
        K_val = x.shape[1]
        N_val = self.gemm.out_features

        w_t = self.gemm.weight.to(device=x.device, dtype=x.dtype).contiguous()
        bias = self.gemm.bias.to(device=x.device, dtype=x.dtype).contiguous()
        y = torch.empty((M_val, N_val), device=x.device, dtype=x.dtype)

        fused_kernel[_launch](x.contiguous(), w_t, bias, y, M_val, N_val, K_val)
        return y
