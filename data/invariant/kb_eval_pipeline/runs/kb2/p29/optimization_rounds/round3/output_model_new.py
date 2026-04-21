import math

import torch
import torch.nn as nn

import substrate
import substrate.language as S


BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
PIPE_UNROLL = 2
K_STEP = BLOCK_K * PIPE_UNROLL
STAGES = 2
WAVE_SIZE = 64
WAVES_M = 2
WAVES_N = 2
NUM_WAVES = WAVES_M * WAVES_N
THREADS = NUM_WAVES * WAVE_SIZE
BF16_BYTES = 2
X_RSRC_RANGE = BATCH_SIZE * IN_FEATURES * BF16_BYTES
W_RSRC_RANGE = IN_FEATURES * OUT_FEATURES * BF16_BYTES


def _launch():
    return ((OUT_FEATURES // BLOCK_N, BATCH_SIZE // BLOCK_M, 1), (THREADS, 1, 1))


def _mish2(x):
    x = torch.nn.functional.mish(x)
    x = torch.nn.functional.mish(x)
    return x


@substrate.jit
def fused_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((IN_FEATURES, OUT_FEATURES), S.bf16),
    BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    wave = tid // WAVE_SIZE
    wave_row = wave // WAVES_N
    wave_col = wave % WAVES_N

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N
    wave_row_base = block_row + wave_row * 32
    wave_col_base = block_col + wave_col * 32

    x_rsrc = S.amdgpu.make_rsrc(X, X_RSRC_RANGE)
    w_rsrc = S.amdgpu.make_rsrc(W, W_RSRC_RANGE)

    a_shared = S.make_shared((STAGES, BLOCK_M, BLOCK_K), S.bf16)
    b_shared = S.make_shared((STAGES, BLOCK_K, BLOCK_N), S.bf16)
    a_frag_lo_shared = S.make_shared((THREADS, 4), S.bf16)
    a_frag_hi_shared = S.make_shared((THREADS, 4), S.bf16)
    b_frag_lo_shared = S.make_shared((THREADS, 4), S.bf16)
    b_frag_hi_shared = S.make_shared((THREADS, 4), S.bf16)
    acc = S.full((16,), 0.0, S.f32)

    zero_i32 = S.convert(0, S.i32)

    if tid < 128:
        a_slot = tid
        a_row = a_slot // 2
        a_chunk = a_slot % 2
        a_offset = ((block_row + a_row) * IN_FEATURES + a_chunk * 8) * BF16_BYTES
        a_vec_i32 = S.amdgpu.raw_buffer_load_x4(x_rsrc, zero_i32, a_offset, zero_i32)
        a_vec = S.view(a_vec_i32, S.Tensor((8,), S.bf16))
        for e in S.range(8):
            a_shared[0, a_row, a_chunk * 8 + e] = a_vec[e]
    else:
        b_slot = tid - 128
        b_row = b_slot // 8
        b_chunk = b_slot % 8
        b_offset = (b_row * OUT_FEATURES + block_col + b_chunk * 8) * BF16_BYTES
        b_vec_i32 = S.amdgpu.raw_buffer_load_x4(w_rsrc, zero_i32, b_offset, zero_i32)
        b_vec = S.view(b_vec_i32, S.Tensor((8,), S.bf16))
        for e in S.range(8):
            b_shared[0, b_row, b_chunk * 8 + e] = b_vec[e]

    if tid < 128:
        a_slot = tid
        a_row = a_slot // 2
        a_chunk = a_slot % 2
        a_offset = ((block_row + a_row) * IN_FEATURES + BLOCK_K + a_chunk * 8) * BF16_BYTES
        a_vec_i32 = S.amdgpu.raw_buffer_load_x4(x_rsrc, zero_i32, a_offset, zero_i32)
        a_vec = S.view(a_vec_i32, S.Tensor((8,), S.bf16))
        for e in S.range(8):
            a_shared[1, a_row, a_chunk * 8 + e] = a_vec[e]
    else:
        b_slot = tid - 128
        b_row = b_slot // 8
        b_chunk = b_slot % 8
        b_offset = ((BLOCK_K + b_row) * OUT_FEATURES + block_col + b_chunk * 8) * BF16_BYTES
        b_vec_i32 = S.amdgpu.raw_buffer_load_x4(w_rsrc, zero_i32, b_offset, zero_i32)
        b_vec = S.view(b_vec_i32, S.Tensor((8,), S.bf16))
        for e in S.range(8):
            b_shared[1, b_row, b_chunk * 8 + e] = b_vec[e]

    S.syncthreads()

    for k0 in S.range(0, IN_FEATURES, K_STEP):
        a_row = wave_row * 32 + (lane % 32)
        b_col = wave_col * 32 + (lane % 32)
        a_col0 = (lane // 32) * 4
        a_col1 = 8 + (lane // 32) * 4
        b_row0 = (lane // 32) * 4
        b_row1 = 8 + (lane // 32) * 4

        for e in S.range(4):
            a_frag_lo_shared[tid, e] = a_shared[0, a_row, a_col0 + e]
            a_frag_hi_shared[tid, e] = a_shared[0, a_row, a_col1 + e]
            b_frag_lo_shared[tid, e] = b_shared[0, b_row0 + e, b_col]
            b_frag_hi_shared[tid, e] = b_shared[0, b_row1 + e, b_col]
        S.syncthreads()
        a_frag_00 = S.view(a_frag_lo_shared[tid], S.Tensor((1, 4, 1), S.bf16))
        a_frag_01 = S.view(a_frag_hi_shared[tid], S.Tensor((1, 4, 1), S.bf16))
        b_frag_00 = S.view(b_frag_lo_shared[tid], S.Tensor((1, 4, 1), S.bf16))
        b_frag_01 = S.view(b_frag_hi_shared[tid], S.Tensor((1, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_00[0], b_frag_00[0], acc)

        next_k0 = k0 + K_STEP
        if tid < 128:
            a_slot = tid
            a_row_load = a_slot // 2
            a_chunk = a_slot % 2
            a_offset = ((block_row + a_row_load) * IN_FEATURES + next_k0 + a_chunk * 8) * BF16_BYTES
            a_vec_i32 = S.amdgpu.raw_buffer_load_x4(x_rsrc, zero_i32, a_offset, zero_i32)
            a_vec = S.view(a_vec_i32, S.Tensor((8,), S.bf16))
            for e in S.range(8):
                a_shared[0, a_row_load, a_chunk * 8 + e] = a_vec[e]
        else:
            b_slot = tid - 128
            b_row_load = b_slot // 8
            b_chunk = b_slot % 8
            b_offset = ((next_k0 + b_row_load) * OUT_FEATURES + block_col + b_chunk * 8) * BF16_BYTES
            b_vec_i32 = S.amdgpu.raw_buffer_load_x4(w_rsrc, zero_i32, b_offset, zero_i32)
            b_vec = S.view(b_vec_i32, S.Tensor((8,), S.bf16))
            for e in S.range(8):
                b_shared[0, b_row_load, b_chunk * 8 + e] = b_vec[e]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_01[0], b_frag_01[0], acc)

        for e in S.range(4):
            a_frag_lo_shared[tid, e] = a_shared[1, a_row, a_col0 + e]
            a_frag_hi_shared[tid, e] = a_shared[1, a_row, a_col1 + e]
            b_frag_lo_shared[tid, e] = b_shared[1, b_row0 + e, b_col]
            b_frag_hi_shared[tid, e] = b_shared[1, b_row1 + e, b_col]
        S.syncthreads()
        a_frag_10 = S.view(a_frag_lo_shared[tid], S.Tensor((1, 4, 1), S.bf16))
        a_frag_11 = S.view(a_frag_hi_shared[tid], S.Tensor((1, 4, 1), S.bf16))
        b_frag_10 = S.view(b_frag_lo_shared[tid], S.Tensor((1, 4, 1), S.bf16))
        b_frag_11 = S.view(b_frag_hi_shared[tid], S.Tensor((1, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_10[0], b_frag_10[0], acc)

        next_k1 = k0 + K_STEP + BLOCK_K
        if tid < 128:
            a_slot = tid
            a_row_load = a_slot // 2
            a_chunk = a_slot % 2
            a_offset = ((block_row + a_row_load) * IN_FEATURES + next_k1 + a_chunk * 8) * BF16_BYTES
            a_vec_i32 = S.amdgpu.raw_buffer_load_x4(x_rsrc, zero_i32, a_offset, zero_i32)
            a_vec = S.view(a_vec_i32, S.Tensor((8,), S.bf16))
            for e in S.range(8):
                a_shared[1, a_row_load, a_chunk * 8 + e] = a_vec[e]
        else:
            b_slot = tid - 128
            b_row_load = b_slot // 8
            b_chunk = b_slot % 8
            b_offset = ((next_k1 + b_row_load) * OUT_FEATURES + block_col + b_chunk * 8) * BF16_BYTES
            b_vec_i32 = S.amdgpu.raw_buffer_load_x4(w_rsrc, zero_i32, b_offset, zero_i32)
            b_vec = S.view(b_vec_i32, S.Tensor((8,), S.bf16))
            for e in S.range(8):
                b_shared[1, b_row_load, b_chunk * 8 + e] = b_vec[e]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_11[0], b_frag_11[0], acc)
        S.syncthreads()

    for acc_idx in S.range(16):
        out_col = wave_col_base + (lane % 32)
        out_row = wave_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        x = acc[acc_idx] + S.convert(BIAS[out_col], S.f32)
        Y[out_row, out_col] = S.convert(x, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        if in_features != IN_FEATURES or out_features != OUT_FEATURES:
            raise ValueError("This kernel is specialized for the benchmark shape.")
        ref_linear = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(ref_linear.weight.detach().t().contiguous())
        self.bias = nn.Parameter(ref_linear.bias.detach().contiguous())

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES) or x.dtype != torch.bfloat16:
            raise ValueError("This kernel only supports the benchmark input shape and bf16 dtype.")
        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=x.dtype)
        fused_kernel[_launch](x.contiguous(), self.weight, self.bias, y, num_warps=4)
        return _mish2(y)
