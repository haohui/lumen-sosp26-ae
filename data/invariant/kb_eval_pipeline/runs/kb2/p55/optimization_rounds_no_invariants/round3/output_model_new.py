import math

import torch
import torch.nn as nn

import substrate
import substrate.language as S


BATCH_SIZE = 128
IN_FEATURES = 32768
OUT_FEATURES = 32768
POOL_KERNEL_SIZE = 2
SCALE_FACTOR = 0.5

WAVE_SIZE = 64
NUM_WAVES = 4
THREADS = WAVE_SIZE * NUM_WAVES
BLOCK_M = 64
BLOCK_N = 64
K_STAGE = 16


def _kernel_launch():
    return ((BATCH_SIZE // BLOCK_M, 1, 1), (THREADS, 1, 1))


@substrate.jit
def fused_linear_pool_sum_kernel(
    x: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    weight: S.Tensor((OUT_FEATURES, IN_FEATURES), S.bf16),
    bias: S.Tensor((OUT_FEATURES,), S.bf16),
    out: S.Tensor((BATCH_SIZE,), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    wave = tid // WAVE_SIZE
    wave_m = wave // 2
    wave_n = wave % 2
    lane_mod_32 = lane % 32
    lane_hi = lane // 32
    block_m = S.block_id(0)
    m_base = block_m * BLOCK_M

    x_rsrc = S.amdgpu.make_rsrc(x, BATCH_SIZE * IN_FEATURES * 2)
    weight_rsrc = S.amdgpu.make_rsrc(weight, OUT_FEATURES * IN_FEATURES * 2)
    out_rsrc = S.amdgpu.make_rsrc(out, BATCH_SIZE * 2)

    a_shared = S.make_shared((2, 2, WAVE_SIZE, 8), S.bf16)
    b_shared = S.make_shared((2, 2, WAVE_SIZE, 8), S.bf16)
    accum_shared = S.make_shared((2, 2, WAVE_SIZE, 16), S.f32)
    row_sum = S.make_shared((2, 32), S.f32)

    c_lane = S.full((16,), 0.0, S.f32)

    row_owner = wave_n == 0 and lane < 32
    if row_owner:
        row_sum[wave_m, lane] = 0.0
    S.syncthreads()

    for n_base in S.range(0, OUT_FEATURES, BLOCK_N):
        for i in S.range(16):
            c_lane[i] = 0.0

        if wave_n == 0:
            row_local = 8 * (lane_mod_32 // 8) + 2 * (lane_mod_32 % 4) + ((lane_mod_32 // 4) % 2)
            row_global = m_base + wave_m * 32 + row_local
            a_offset = (row_global * IN_FEATURES + lane_hi * 8) * 2
            a_words = S.amdgpu.raw_buffer_load_x4(x_rsrc, a_offset, 0, 0)
            a_vals = S.view(a_words, S.Tensor((8,), S.bf16))
            for kk in S.range(8):
                a_shared[0, wave_m, lane, kk] = a_vals[kk]

        if wave_m == 0:
            col_global = n_base + wave_n * 32 + lane_mod_32
            b_offset = (col_global * IN_FEATURES + lane_hi * 8) * 2
            b_words = S.amdgpu.raw_buffer_load_x4(weight_rsrc, b_offset, 0, 0)
            b_vals = S.view(b_words, S.Tensor((8,), S.bf16))
            for kk in S.range(8):
                b_shared[0, wave_n, lane, kk] = b_vals[kk]

        S.syncthreads()

        for k_base in S.range(0, IN_FEATURES, K_STAGE):
            buf = (k_base // K_STAGE) % 2
            next_buf = 1 - buf

            a_frag = S.view(a_shared[buf, wave_m, lane], S.Tensor((2, 4, 1), S.bf16))
            b_frag = S.view(b_shared[buf, wave_n, lane], S.Tensor((2, 4, 1), S.bf16))

            next_k = k_base + K_STAGE
            if next_k < IN_FEATURES:
                if wave_n == 0:
                    row_local = 8 * (lane_mod_32 // 8) + 2 * (lane_mod_32 % 4) + ((lane_mod_32 // 4) % 2)
                    row_global = m_base + wave_m * 32 + row_local
                    a_offset = (row_global * IN_FEATURES + next_k + lane_hi * 8) * 2
                    a_words = S.amdgpu.raw_buffer_load_x4(x_rsrc, a_offset, 0, 0)
                    a_vals = S.view(a_words, S.Tensor((8,), S.bf16))
                    for kk in S.range(8):
                        a_shared[next_buf, wave_m, lane, kk] = a_vals[kk]

                if wave_m == 0:
                    col_global = n_base + wave_n * 32 + lane_mod_32
                    b_offset = (col_global * IN_FEATURES + next_k + lane_hi * 8) * 2
                    b_words = S.amdgpu.raw_buffer_load_x4(weight_rsrc, b_offset, 0, 0)
                    b_vals = S.view(b_words, S.Tensor((8,), S.bf16))
                    for kk in S.range(8):
                        b_shared[next_buf, wave_n, lane, kk] = b_vals[kk]

            c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], c_lane)
            c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], c_lane)
            S.syncthreads()

        for e in S.range(16):
            accum_shared[wave_m, wave_n, lane, e] = c_lane[e]

        S.syncthreads()

        if row_owner:
            tile_sum = row_sum[wave_m, lane]
            lane_group = lane % 2
            e = 4 * (lane // 8) + ((lane % 8) // 2)
            for pair in S.range(16):
                lane0 = 2 * pair + 32 * lane_group
                lane1 = lane0 + 1

                v00 = accum_shared[wave_m, 0, lane0, e] + S.convert(bias[n_base + 2 * pair], S.f32)
                v01 = accum_shared[wave_m, 0, lane1, e] + S.convert(bias[n_base + 2 * pair + 1], S.f32)
                tile_sum = tile_sum + (v00 if v00 > v01 else v01)

                v10 = accum_shared[wave_m, 1, lane0, e] + S.convert(bias[n_base + 32 + 2 * pair], S.f32)
                v11 = accum_shared[wave_m, 1, lane1, e] + S.convert(bias[n_base + 32 + 2 * pair + 1], S.f32)
                tile_sum = tile_sum + (v10 if v10 > v11 else v11)
            row_sum[wave_m, lane] = tile_sum

        S.syncthreads()

    if row_owner and (lane % 2 == 0):
        row_idx0 = m_base + wave_m * 32 + lane
        row_idx1 = row_idx0 + 1
        out0 = S.convert(row_sum[wave_m, lane] * SCALE_FACTOR, S.bf16)
        out1 = S.convert(row_sum[wave_m, lane + 1] * SCALE_FACTOR, S.bf16)
        out0_bits = S.convert(S.bitcast(out0, S.u16), S.u32)
        out1_bits = S.convert(S.bitcast(out1, S.u16), S.u32)
        packed = out0_bits | (out1_bits << 16)
        S.amdgpu.raw_buffer_store_x1(packed, out_rsrc, row_idx0 * 2, 0, 0)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        out = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
        fused_linear_pool_sum_kernel[_kernel_launch](x, self.weight, self.bias, out, num_warps=NUM_WAVES)
        return out
