import torch
import torch.nn as nn
import torch.nn.functional as F

import substrate
import substrate.language as S


BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192

WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS

WAVE_TILE_M = 32
WAVE_TILE_N = 32
BLOCK_TILE_M = 64
BLOCK_TILE_N = 64
BLOCK_TILE_K = 16
PIPE_STAGES = 2
K_UNROLL = 2
PAIR_TILE_K = BLOCK_TILE_K * K_UNROLL

GRID_M = BATCH_SIZE // BLOCK_TILE_M
GRID_N = OUT_FEATURES // BLOCK_TILE_N

X_RANGE_BYTES = BATCH_SIZE * IN_FEATURES * 2
W_RANGE_BYTES = IN_FEATURES * OUT_FEATURES * 2
Y_RANGE_BYTES = BATCH_SIZE * OUT_FEATURES * 2


def _launch():
    return ((GRID_N, GRID_M, 1), (THREADS, 1, 1))


@substrate.jit
def mfma_gemm_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((IN_FEATURES, OUT_FEATURES), S.bf16),
    BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WARP_SIZE
    warp = tid // WARP_SIZE
    warp_row = warp // 2
    warp_col = warp % 2

    block_row = S.block_id(1)
    block_col = S.block_id(0)
    tile_row_base = block_row * BLOCK_TILE_M + warp_row * WAVE_TILE_M
    tile_col_base = block_col * BLOCK_TILE_N + warp_col * WAVE_TILE_N

    lane_row = lane % 32
    lane_hi = lane // 32

    x_rsrc = S.amdgpu.make_rsrc(X, X_RANGE_BYTES)
    w_rsrc = S.amdgpu.make_rsrc(W, W_RANGE_BYTES)
    y_rsrc = S.amdgpu.make_rsrc(Y, Y_RANGE_BYTES)

    a_words = S.make_shared((PIPE_STAGES, NUM_WARPS, WARP_SIZE, 4), S.u32)
    b_words = S.make_shared((PIPE_STAGES, NUM_WARPS, WARP_SIZE, 4), S.u32)

    acc = S.full((16,), 0.0, S.f32)

    a_lane_base = lane % 16 + (lane // 32) * 16
    a_dest_lo = a_lane_base
    a_dest_hi = a_lane_base + 32
    a_half = (lane // 16) % 2
    a_word_base = a_half * 2

    b_col_group_base = (lane % 4) * 2
    b_k_half = lane // 32
    b_lane_row = (lane // 4) % 8
    b_dest0 = b_lane_row + b_col_group_base * 8
    b_dest1 = b_lane_row + (b_col_group_base + 1) * 8
    b_word_base = b_k_half * 2

    a_row = tile_row_base + lane % 16 + (lane // 32) * 16

    a_global_col = (lane % 32) // 16 * 8
    a_byte_index = (a_row * IN_FEATURES + a_global_col) * 2
    a_vec = S.amdgpu.raw_buffer_load_x4(
        x_rsrc,
        S.convert(a_byte_index, S.i32),
        0,
        0,
    )
    a_words[0, warp, a_dest_lo, a_word_base + 0] = a_vec[0]
    a_words[0, warp, a_dest_lo, a_word_base + 1] = a_vec[1]
    a_words[0, warp, a_dest_hi, a_word_base + 0] = a_vec[2]
    a_words[0, warp, a_dest_hi, a_word_base + 1] = a_vec[3]

    b_global_row = lane // 4
    b_global_col = tile_col_base + (lane % 4) * 8
    b_byte_index = (b_global_row * OUT_FEATURES + b_global_col) * 2
    b_vec = S.amdgpu.raw_buffer_load_x4(
        w_rsrc,
        S.convert(b_byte_index, S.i32),
        0,
        0,
    )
    b_words[0, warp, b_dest0, b_word_base + 0] = b_vec[0]
    b_words[0, warp, b_dest0, b_word_base + 1] = b_vec[1]
    b_words[0, warp, b_dest1, b_word_base + 0] = b_vec[2]
    b_words[0, warp, b_dest1, b_word_base + 1] = b_vec[3]

    a_global_col = BLOCK_TILE_K + (lane % 32) // 16 * 8
    a_byte_index = (a_row * IN_FEATURES + a_global_col) * 2
    a_vec = S.amdgpu.raw_buffer_load_x4(
        x_rsrc,
        S.convert(a_byte_index, S.i32),
        0,
        0,
    )
    a_words[1, warp, a_dest_lo, a_word_base + 0] = a_vec[0]
    a_words[1, warp, a_dest_lo, a_word_base + 1] = a_vec[1]
    a_words[1, warp, a_dest_hi, a_word_base + 0] = a_vec[2]
    a_words[1, warp, a_dest_hi, a_word_base + 1] = a_vec[3]

    b_global_row = BLOCK_TILE_K + lane // 4
    b_byte_index = (b_global_row * OUT_FEATURES + b_global_col) * 2
    b_vec = S.amdgpu.raw_buffer_load_x4(
        w_rsrc,
        S.convert(b_byte_index, S.i32),
        0,
        0,
    )
    b_words[1, warp, b_dest0, b_word_base + 0] = b_vec[0]
    b_words[1, warp, b_dest0, b_word_base + 1] = b_vec[1]
    b_words[1, warp, b_dest1, b_word_base + 0] = b_vec[2]
    b_words[1, warp, b_dest1, b_word_base + 1] = b_vec[3]

    S.syncthreads()

    for k_pair_base in S.range(0, IN_FEATURES - PAIR_TILE_K, PAIR_TILE_K):
        a_frag = S.view(a_words[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_words[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)

        a_global_col = k_pair_base + PAIR_TILE_K + (lane % 32) // 16 * 8
        a_byte_index = (a_row * IN_FEATURES + a_global_col) * 2
        a_vec = S.amdgpu.raw_buffer_load_x4(
            x_rsrc,
            S.convert(a_byte_index, S.i32),
            0,
            0,
        )
        b_global_row = k_pair_base + PAIR_TILE_K + lane // 4
        b_byte_index = (b_global_row * OUT_FEATURES + b_global_col) * 2
        b_vec = S.amdgpu.raw_buffer_load_x4(
            w_rsrc,
            S.convert(b_byte_index, S.i32),
            0,
            0,
        )

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        a_words[0, warp, a_dest_lo, a_word_base + 0] = a_vec[0]
        a_words[0, warp, a_dest_lo, a_word_base + 1] = a_vec[1]
        a_words[0, warp, a_dest_hi, a_word_base + 0] = a_vec[2]
        a_words[0, warp, a_dest_hi, a_word_base + 1] = a_vec[3]
        b_words[0, warp, b_dest0, b_word_base + 0] = b_vec[0]
        b_words[0, warp, b_dest0, b_word_base + 1] = b_vec[1]
        b_words[0, warp, b_dest1, b_word_base + 0] = b_vec[2]
        b_words[0, warp, b_dest1, b_word_base + 1] = b_vec[3]

        a_frag = S.view(a_words[1, warp, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_words[1, warp, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)

        a_global_col = k_pair_base + PAIR_TILE_K + BLOCK_TILE_K + (lane % 32) // 16 * 8
        a_byte_index = (a_row * IN_FEATURES + a_global_col) * 2
        a_vec = S.amdgpu.raw_buffer_load_x4(
            x_rsrc,
            S.convert(a_byte_index, S.i32),
            0,
            0,
        )
        b_global_row = k_pair_base + PAIR_TILE_K + BLOCK_TILE_K + lane // 4
        b_byte_index = (b_global_row * OUT_FEATURES + b_global_col) * 2
        b_vec = S.amdgpu.raw_buffer_load_x4(
            w_rsrc,
            S.convert(b_byte_index, S.i32),
            0,
            0,
        )

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        a_words[1, warp, a_dest_lo, a_word_base + 0] = a_vec[0]
        a_words[1, warp, a_dest_lo, a_word_base + 1] = a_vec[1]
        a_words[1, warp, a_dest_hi, a_word_base + 0] = a_vec[2]
        a_words[1, warp, a_dest_hi, a_word_base + 1] = a_vec[3]
        b_words[1, warp, b_dest0, b_word_base + 0] = b_vec[0]
        b_words[1, warp, b_dest0, b_word_base + 1] = b_vec[1]
        b_words[1, warp, b_dest1, b_word_base + 0] = b_vec[2]
        b_words[1, warp, b_dest1, b_word_base + 1] = b_vec[3]

        S.syncthreads()

    a_frag = S.view(a_words[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag = S.view(b_words[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

    a_frag = S.view(a_words[1, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag = S.view(b_words[1, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

    out_col = tile_col_base + lane % 32
    out_pair_col = tile_col_base + ((lane % 32) // 2) * 2
    pair_lane_base = ((lane % 32) // 2) * 2
    bias = S.convert(BIAS[out_col], S.f32)

    for acc_idx in S.range(16):
        out_row = tile_row_base + 8 * (acc_idx // 4) + 4 * lane_hi + (acc_idx % 4)
        out_byte_index = (out_row * OUT_FEATURES + out_pair_col) * 2
        out_value = S.convert(acc[acc_idx] + bias, S.bf16)
        even_value = S.shuffle(out_value, pair_lane_base + 0, 32)
        odd_value = S.shuffle(out_value, pair_lane_base + 1, 32)
        even_bits = S.bitcast(even_value, S.u16)
        odd_bits = S.bitcast(odd_value, S.u16)
        packed_bits = S.convert(even_bits, S.u32) | (S.convert(odd_bits, S.u32) << 16)
        S.amdgpu.raw_buffer_store_x1(
            S.bitcast(packed_bits, S.i32),
            y_rsrc,
            S.convert(out_byte_index, S.i32),
            0,
            0,
        )


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._cached_weight_ptr = None
        self._cached_bias_ptr = None
        self._cached_weight_t = None
        self._cached_bias = None

    def _refresh_params(self, device: torch.device, dtype: torch.dtype) -> None:
        weight = self.linear.weight
        bias = self.linear.bias
        weight_ptr = weight.untyped_storage().data_ptr()
        bias_ptr = bias.untyped_storage().data_ptr()

        if (
            self._cached_weight_t is None
            or self._cached_bias is None
            or self._cached_weight_ptr != weight_ptr
            or self._cached_bias_ptr != bias_ptr
            or self._cached_weight_t.device != device
            or self._cached_weight_t.dtype != dtype
        ):
            self._cached_weight_t = weight.t().to(device=device, dtype=dtype).contiguous()
            self._cached_bias = bias.to(device=device, dtype=dtype).contiguous()
            self._cached_weight_ptr = weight_ptr
            self._cached_bias_ptr = bias_ptr

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES) or x.dtype != torch.bfloat16:
            raise NotImplementedError(
                "ModelNew only supports the KernelBench bf16 input shape "
                f"({BATCH_SIZE}, {IN_FEATURES})."
            )

        x = x.contiguous()
        self._refresh_params(x.device, x.dtype)

        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=x.dtype)
        mfma_gemm_kernel[_launch](x, self._cached_weight_t, self._cached_bias, y, num_warps=NUM_WARPS)
        return y
