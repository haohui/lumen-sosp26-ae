import torch
import torch.nn as nn

import substrate
import substrate.language as S


BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192

WAVE_SIZE = 64
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = WAVE_SIZE * WAVES_PER_BLOCK
WAVE_TILE_M = 32
WAVE_TILE_N = 32
BLOCK_TILE_M = 64
BLOCK_TILE_N = 64
BLOCK_TILE_K = 16
PIPELINE_UNROLL = 2
PIPELINE_STEP_K = BLOCK_TILE_K * PIPELINE_UNROLL


def _launch():
    return (
        (OUT_FEATURES // BLOCK_TILE_N, BATCH_SIZE // BLOCK_TILE_M, 1),
        (THREADS_PER_BLOCK, 1, 1),
    )


@substrate.jit
def fused_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((OUT_FEATURES, IN_FEATURES), S.bf16),
    BIAS0: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    tid = S.thread_id(0)
    wave = tid // WAVE_SIZE
    lane = tid % WAVE_SIZE
    warp_row = wave // 2
    warp_col = wave % 2

    tile_row_base = S.block_id(1) * BLOCK_TILE_M + warp_row * WAVE_TILE_M
    tile_col_base = S.block_id(0) * BLOCK_TILE_N + warp_col * WAVE_TILE_N

    row = tile_row_base + (lane % 32)
    col = tile_col_base + (lane % 32)
    k_lane_base = (lane // 32) * 8
    pack_lo = lane % 32
    pack_hi = pack_lo + 32
    dst_base = (lane // 32) * 2

    x_range_bytes = BATCH_SIZE * IN_FEATURES * 2
    w_range_bytes = OUT_FEATURES * IN_FEATURES * 2
    x_rsrc = S.amdgpu.make_rsrc(X, x_range_bytes)
    w_rsrc = S.amdgpu.make_rsrc(W, w_range_bytes)

    a_stage = S.make_shared((2, WAVES_PER_BLOCK, WAVE_SIZE, 4), S.u32)
    b_stage = S.make_shared((2, WAVES_PER_BLOCK, WAVE_SIZE, 4), S.u32)

    acc = S.full((16,), 0.0, S.f32)

    k_chunk = k_lane_base
    a_offset = (row * IN_FEATURES + k_chunk) * 2
    b_offset = (col * IN_FEATURES + k_chunk) * 2
    a_raw = S.amdgpu.raw_buffer_load_x4(x_rsrc, a_offset, 0, 0)
    b_raw = S.amdgpu.raw_buffer_load_x4(w_rsrc, b_offset, 0, 0)

    a_stage[0, wave, pack_lo, dst_base] = a_raw[0]
    a_stage[0, wave, pack_lo, dst_base + 1] = a_raw[1]
    a_stage[0, wave, pack_hi, dst_base] = a_raw[2]
    a_stage[0, wave, pack_hi, dst_base + 1] = a_raw[3]

    b_stage[0, wave, pack_lo, dst_base] = b_raw[0]
    b_stage[0, wave, pack_lo, dst_base + 1] = b_raw[1]
    b_stage[0, wave, pack_hi, dst_base] = b_raw[2]
    b_stage[0, wave, pack_hi, dst_base + 1] = b_raw[3]

    k_chunk = BLOCK_TILE_K + k_lane_base
    a_offset = (row * IN_FEATURES + k_chunk) * 2
    b_offset = (col * IN_FEATURES + k_chunk) * 2
    a_raw = S.amdgpu.raw_buffer_load_x4(x_rsrc, a_offset, 0, 0)
    b_raw = S.amdgpu.raw_buffer_load_x4(w_rsrc, b_offset, 0, 0)

    a_stage[1, wave, pack_lo, dst_base] = a_raw[0]
    a_stage[1, wave, pack_lo, dst_base + 1] = a_raw[1]
    a_stage[1, wave, pack_hi, dst_base] = a_raw[2]
    a_stage[1, wave, pack_hi, dst_base + 1] = a_raw[3]

    b_stage[1, wave, pack_lo, dst_base] = b_raw[0]
    b_stage[1, wave, pack_lo, dst_base + 1] = b_raw[1]
    b_stage[1, wave, pack_hi, dst_base] = b_raw[2]
    b_stage[1, wave, pack_hi, dst_base + 1] = b_raw[3]

    S.syncthreads()

    for k0 in S.range(0, IN_FEATURES, PIPELINE_STEP_K):
        a_lane0 = a_stage[0, wave, lane]
        b_lane0 = b_stage[0, wave, lane]
        a_frag0 = S.view(a_lane0, S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(b_lane0, S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)

        next_k0 = k0 + PIPELINE_STEP_K
        k_chunk = next_k0 + k_lane_base
        a_offset = (row * IN_FEATURES + k_chunk) * 2
        b_offset = (col * IN_FEATURES + k_chunk) * 2
        a_raw = S.amdgpu.raw_buffer_load_x4(x_rsrc, a_offset, 0, 0)
        b_raw = S.amdgpu.raw_buffer_load_x4(w_rsrc, b_offset, 0, 0)

        a_stage[0, wave, pack_lo, dst_base] = a_raw[0]
        a_stage[0, wave, pack_lo, dst_base + 1] = a_raw[1]
        a_stage[0, wave, pack_hi, dst_base] = a_raw[2]
        a_stage[0, wave, pack_hi, dst_base + 1] = a_raw[3]

        b_stage[0, wave, pack_lo, dst_base] = b_raw[0]
        b_stage[0, wave, pack_lo, dst_base + 1] = b_raw[1]
        b_stage[0, wave, pack_hi, dst_base] = b_raw[2]
        b_stage[0, wave, pack_hi, dst_base + 1] = b_raw[3]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

        a_lane1 = a_stage[1, wave, lane]
        b_lane1 = b_stage[1, wave, lane]
        a_frag1 = S.view(a_lane1, S.Tensor((2, 4, 1), S.bf16))
        b_frag1 = S.view(b_lane1, S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)

        next_k1 = k0 + PIPELINE_STEP_K + BLOCK_TILE_K
        k_chunk = next_k1 + k_lane_base
        a_offset = (row * IN_FEATURES + k_chunk) * 2
        b_offset = (col * IN_FEATURES + k_chunk) * 2
        a_raw = S.amdgpu.raw_buffer_load_x4(x_rsrc, a_offset, 0, 0)
        b_raw = S.amdgpu.raw_buffer_load_x4(w_rsrc, b_offset, 0, 0)

        a_stage[1, wave, pack_lo, dst_base] = a_raw[0]
        a_stage[1, wave, pack_lo, dst_base + 1] = a_raw[1]
        a_stage[1, wave, pack_hi, dst_base] = a_raw[2]
        a_stage[1, wave, pack_hi, dst_base + 1] = a_raw[3]

        b_stage[1, wave, pack_lo, dst_base] = b_raw[0]
        b_stage[1, wave, pack_lo, dst_base + 1] = b_raw[1]
        b_stage[1, wave, pack_hi, dst_base] = b_raw[2]
        b_stage[1, wave, pack_hi, dst_base + 1] = b_raw[3]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

        S.syncthreads()

    one = S.convert(1.0, S.f32)
    half = S.convert(0.5, S.f32)
    neg_one = S.convert(-1.0, S.f32)
    bias = S.convert(BIAS0[tile_col_base + (lane % 32)], S.f32)

    for acc_idx in S.range(16):
        out_col = tile_col_base + (lane % 32)
        out_row = (
            tile_row_base
            + 8 * (acc_idx // 4)
            + 4 * (lane // 32)
            + (acc_idx % 4)
        )

        x = acc[acc_idx] + bias
        x = x * (one / (one + S.exp(-x)))
        x = x * half
        if x < neg_one:
            x = neg_one
        if x > one:
            x = one
        x = S.tanh(x)
        if x < neg_one:
            x = neg_one
        if x > one:
            x = one
        Y[out_row, out_col] = S.convert(x, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES):
            raise RuntimeError(f"expected input shape {(BATCH_SIZE, IN_FEATURES)}, got {tuple(x.shape)}")
        if x.dtype != torch.bfloat16:
            raise RuntimeError(f"expected torch.bfloat16 input, got {x.dtype}")
        if self.gemm.bias is None:
            raise RuntimeError("bias=False is unsupported for this fused kernel")

        x_buf = x.contiguous()
        w_buf = self.gemm.weight.to(device=x.device, dtype=torch.bfloat16).contiguous()
        b_buf = self.gemm.bias.to(device=x.device, dtype=torch.bfloat16).contiguous()
        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=torch.bfloat16)
        fused_kernel[_launch](x_buf, w_buf, b_buf, y)
        return y
