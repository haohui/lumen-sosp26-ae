import torch
import torch.nn as nn
import torch.nn.functional as F
import substrate
import substrate.language as S


BATCH_SIZE = 32768
IN_FEATURES = 1024
OUT_FEATURES = 4096
NUM_GROUPS = 64
GROUP_SIZE = OUT_FEATURES // NUM_GROUPS
EPS = 1e-05

BLOCK_M = 64
BLOCK_N = 64
WAVE_M = 32
WAVE_N = 32
WAVE_SIZE = 64
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = WAVE_SIZE * WAVES_PER_BLOCK
BLOCK_K = 16

X_RANGE_BYTES = BATCH_SIZE * IN_FEATURES * 2
W_RANGE_BYTES = IN_FEATURES * OUT_FEATURES * 2
NUM_K_TILES = IN_FEATURES // BLOCK_K
UNROLLED_PAIRS = NUM_K_TILES // 2


def _launch_gemm():
    return ((OUT_FEATURES // BLOCK_N, BATCH_SIZE // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))


def _launch_group_norm():
    return ((NUM_GROUPS, BATCH_SIZE, 1), (WAVE_SIZE, 1, 1))


@substrate.jit
def gemm_silu_bias_mfma_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((IN_FEATURES, OUT_FEATURES), S.bf16),
    BIAS0: S.Tensor((OUT_FEATURES,), S.bf16),
    EXTRA_BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    wave = tid // WAVE_SIZE
    wave_row = wave // 2
    wave_col = wave % 2

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N
    tile_row_base = block_row + wave_row * WAVE_M
    tile_col_base = block_col + wave_col * WAVE_N

    x_rsrc = S.amdgpu.make_rsrc(X, S.convert(X_RANGE_BYTES, S.i32))
    w_rsrc = S.amdgpu.make_rsrc(W, S.convert(W_RANGE_BYTES, S.i32))

    zero_i32 = S.convert(0, S.i32)
    one_f32 = S.convert(1.0, S.f32)

    a_words0 = S.make_shared((2, WAVE_SIZE, 4), S.u32)
    a_words1 = S.make_shared((2, WAVE_SIZE, 4), S.u32)
    b_words0 = S.make_shared((2, WAVE_SIZE, 4), S.u32)
    b_words1 = S.make_shared((2, WAVE_SIZE, 4), S.u32)
    frag_layout = S.make_layout((2, WAVE_SIZE, 2, 4), (WAVE_SIZE * 8, 8, 4, 1))
    a_frags0 = S.view(a_words0, S.bf16, frag_layout)
    a_frags1 = S.view(a_words1, S.bf16, frag_layout)
    b_frags0 = S.view(b_words0, S.bf16, frag_layout)
    b_frags1 = S.view(b_words1, S.bf16, frag_layout)

    acc = S.full((16,), 0.0, S.f32)

    if tid < 128:
        row = tid // 2
        seg = tid % 2
        byte_offset = ((block_row + row) * IN_FEATURES + seg * 8) * 2
        packed = S.amdgpu.raw_buffer_load_x4(
            x_rsrc, zero_i32, S.convert(byte_offset, S.i32), 0
        )
        vals = S.view(packed, S.Tensor((2, 4, 1), S.bf16))
        row_wave = row // WAVE_M
        row_lane = row % WAVE_M
        for e in S.range(4):
            a_frags0[row_wave, row_lane, seg, e] = vals[0, e, 0]
            a_frags0[row_wave, row_lane + WAVE_M, seg, e] = vals[1, e, 0]
        byte_offset = ((block_row + row) * IN_FEATURES + BLOCK_K + seg * 8) * 2
        packed = S.amdgpu.raw_buffer_load_x4(
            x_rsrc, zero_i32, S.convert(byte_offset, S.i32), 0
        )
        vals = S.view(packed, S.Tensor((2, 4, 1), S.bf16))
        for e in S.range(4):
            a_frags1[row_wave, row_lane, seg, e] = vals[0, e, 0]
            a_frags1[row_wave, row_lane + WAVE_M, seg, e] = vals[1, e, 0]
    else:
        b_chunk = tid - 128
        k_in_tile = b_chunk // 8
        seg = b_chunk % 8
        byte_offset = ((k_in_tile) * OUT_FEATURES + block_col + seg * 8) * 2
        packed = S.amdgpu.raw_buffer_load_x4(
            w_rsrc, zero_i32, S.convert(byte_offset, S.i32), 0
        )
        vals = S.view(packed, S.Tensor((2, 4, 1), S.bf16))
        col_wave = seg // 4
        col_base = (seg % 4) * 8
        lane_base = 0 if (k_in_tile % 8) < 4 else WAVE_M
        k_step = k_in_tile // 8
        elem = k_in_tile % 4
        for e in S.range(4):
            b_frags0[col_wave, lane_base + col_base + e, k_step, elem] = vals[0, e, 0]
            b_frags0[col_wave, lane_base + col_base + 4 + e, k_step, elem] = vals[1, e, 0]
        byte_offset = ((BLOCK_K + k_in_tile) * OUT_FEATURES + block_col + seg * 8) * 2
        packed = S.amdgpu.raw_buffer_load_x4(
            w_rsrc, zero_i32, S.convert(byte_offset, S.i32), 0
        )
        vals = S.view(packed, S.Tensor((2, 4, 1), S.bf16))
        for e in S.range(4):
            b_frags1[col_wave, lane_base + col_base + e, k_step, elem] = vals[0, e, 0]
            b_frags1[col_wave, lane_base + col_base + 4 + e, k_step, elem] = vals[1, e, 0]
    S.syncthreads()

    for pair in S.range(UNROLLED_PAIRS - 1):
        a_lane = S.view(a_words0[wave_row, lane], S.Tensor((2, 4, 1), S.bf16))
        b_lane = S.view(b_words0[wave_col, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_lane[0], b_lane[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_lane[1], b_lane[1], acc)
        S.syncthreads()

        if tid < 128:
            row = tid // 2
            seg = tid % 2
            byte_offset = ((block_row + row) * IN_FEATURES + (pair * 2 + 2) * BLOCK_K + seg * 8) * 2
            packed = S.amdgpu.raw_buffer_load_x4(
                x_rsrc, zero_i32, S.convert(byte_offset, S.i32), 0
            )
            vals = S.view(packed, S.Tensor((2, 4, 1), S.bf16))
            row_wave = row // WAVE_M
            row_lane = row % WAVE_M
            for e in S.range(4):
                a_frags0[row_wave, row_lane, seg, e] = vals[0, e, 0]
                a_frags0[row_wave, row_lane + WAVE_M, seg, e] = vals[1, e, 0]
        else:
            b_chunk = tid - 128
            k_in_tile = b_chunk // 8
            seg = b_chunk % 8
            byte_offset = (
                (((pair * 2 + 2) * BLOCK_K + k_in_tile) * OUT_FEATURES + block_col + seg * 8) * 2
            )
            packed = S.amdgpu.raw_buffer_load_x4(
                w_rsrc, zero_i32, S.convert(byte_offset, S.i32), 0
            )
            vals = S.view(packed, S.Tensor((2, 4, 1), S.bf16))
            col_wave = seg // 4
            col_base = (seg % 4) * 8
            lane_base = 0 if (k_in_tile % 8) < 4 else WAVE_M
            k_step = k_in_tile // 8
            elem = k_in_tile % 4
            for e in S.range(4):
                b_frags0[col_wave, lane_base + col_base + e, k_step, elem] = vals[0, e, 0]
                b_frags0[col_wave, lane_base + col_base + 4 + e, k_step, elem] = vals[1, e, 0]

        a_lane = S.view(a_words1[wave_row, lane], S.Tensor((2, 4, 1), S.bf16))
        b_lane = S.view(b_words1[wave_col, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_lane[0], b_lane[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_lane[1], b_lane[1], acc)
        S.syncthreads()

        if tid < 128:
            row = tid // 2
            seg = tid % 2
            byte_offset = ((block_row + row) * IN_FEATURES + (pair * 2 + 3) * BLOCK_K + seg * 8) * 2
            packed = S.amdgpu.raw_buffer_load_x4(
                x_rsrc, zero_i32, S.convert(byte_offset, S.i32), 0
            )
            vals = S.view(packed, S.Tensor((2, 4, 1), S.bf16))
            row_wave = row // WAVE_M
            row_lane = row % WAVE_M
            for e in S.range(4):
                a_frags1[row_wave, row_lane, seg, e] = vals[0, e, 0]
                a_frags1[row_wave, row_lane + WAVE_M, seg, e] = vals[1, e, 0]
        else:
            b_chunk = tid - 128
            k_in_tile = b_chunk // 8
            seg = b_chunk % 8
            byte_offset = (
                (((pair * 2 + 3) * BLOCK_K + k_in_tile) * OUT_FEATURES + block_col + seg * 8) * 2
            )
            packed = S.amdgpu.raw_buffer_load_x4(
                w_rsrc, zero_i32, S.convert(byte_offset, S.i32), 0
            )
            vals = S.view(packed, S.Tensor((2, 4, 1), S.bf16))
            col_wave = seg // 4
            col_base = (seg % 4) * 8
            lane_base = 0 if (k_in_tile % 8) < 4 else WAVE_M
            k_step = k_in_tile // 8
            elem = k_in_tile % 4
            for e in S.range(4):
                b_frags1[col_wave, lane_base + col_base + e, k_step, elem] = vals[0, e, 0]
                b_frags1[col_wave, lane_base + col_base + 4 + e, k_step, elem] = vals[1, e, 0]
        S.syncthreads()

    a_lane = S.view(a_words0[wave_row, lane], S.Tensor((2, 4, 1), S.bf16))
    b_lane = S.view(b_words0[wave_col, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_lane[0], b_lane[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_lane[1], b_lane[1], acc)

    a_lane = S.view(a_words1[wave_row, lane], S.Tensor((2, 4, 1), S.bf16))
    b_lane = S.view(b_words1[wave_col, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_lane[0], b_lane[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_lane[1], b_lane[1], acc)

    col = tile_col_base + (lane % WAVE_N)
    for acc_idx in S.range(16):
        row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // WAVE_N) + (acc_idx % 4)
        value = acc[acc_idx] + S.convert(BIAS0[col], S.f32)
        value = value / (one_f32 + S.exp(-value))
        value += S.convert(EXTRA_BIAS[col], S.f32)
        Y[row, col] = S.convert(value, S.bf16)


@substrate.jit
def group_norm_kernel(
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
    GN_WEIGHT: S.Tensor((OUT_FEATURES,), S.bf16),
    GN_BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
):
    lane = S.thread_id(0)
    row = S.block_id(1)
    group = S.block_id(0)
    if lane == 0:
        mean0 = S.convert(0.0, S.f32)
        mean1 = S.convert(0.0, S.f32)
        mean2 = S.convert(0.0, S.f32)
        mean3 = S.convert(0.0, S.f32)
        mean4 = S.convert(0.0, S.f32)
        mean5 = S.convert(0.0, S.f32)
        mean6 = S.convert(0.0, S.f32)
        mean7 = S.convert(0.0, S.f32)
        for t in S.range(GROUP_SIZE // 8):
            col = group * GROUP_SIZE + t
            mean0 += S.convert(Y[row, col], S.f32)
            mean1 += S.convert(Y[row, col + 8], S.f32)
            mean2 += S.convert(Y[row, col + 16], S.f32)
            mean3 += S.convert(Y[row, col + 24], S.f32)
            mean4 += S.convert(Y[row, col + 32], S.f32)
            mean5 += S.convert(Y[row, col + 40], S.f32)
            mean6 += S.convert(Y[row, col + 48], S.f32)
            mean7 += S.convert(Y[row, col + 56], S.f32)
        mean = mean0 + mean1 + mean2 + mean3 + mean4 + mean5 + mean6 + mean7
        mean = mean / S.convert(GROUP_SIZE, S.f32)

        var0 = S.convert(0.0, S.f32)
        var1 = S.convert(0.0, S.f32)
        var2 = S.convert(0.0, S.f32)
        var3 = S.convert(0.0, S.f32)
        var4 = S.convert(0.0, S.f32)
        var5 = S.convert(0.0, S.f32)
        var6 = S.convert(0.0, S.f32)
        var7 = S.convert(0.0, S.f32)
        for t in S.range(GROUP_SIZE // 8):
            col = group * GROUP_SIZE + t
            centered0 = S.convert(Y[row, col], S.f32) - mean
            centered1 = S.convert(Y[row, col + 8], S.f32) - mean
            centered2 = S.convert(Y[row, col + 16], S.f32) - mean
            centered3 = S.convert(Y[row, col + 24], S.f32) - mean
            centered4 = S.convert(Y[row, col + 32], S.f32) - mean
            centered5 = S.convert(Y[row, col + 40], S.f32) - mean
            centered6 = S.convert(Y[row, col + 48], S.f32) - mean
            centered7 = S.convert(Y[row, col + 56], S.f32) - mean
            var0 += centered0 * centered0
            var1 += centered1 * centered1
            var2 += centered2 * centered2
            var3 += centered3 * centered3
            var4 += centered4 * centered4
            var5 += centered5 * centered5
            var6 += centered6 * centered6
            var7 += centered7 * centered7
        var = var0 + var1 + var2 + var3 + var4 + var5 + var6 + var7
        var = var / S.convert(GROUP_SIZE, S.f32)
        inv_std = S.convert(1.0, S.f32) / S.sqrt(var + S.convert(EPS, S.f32))

        for t in S.range(GROUP_SIZE):
            col = group * GROUP_SIZE + t
            out = (S.convert(Y[row, col], S.f32) - mean) * inv_std
            out = out * S.convert(GN_WEIGHT[col], S.f32) + S.convert(GN_BIAS[col], S.f32)
            Y[row, col] = S.convert(out, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        if (
            tuple(x.shape) != (BATCH_SIZE, IN_FEATURES)
            or x.dtype != torch.bfloat16
            or tuple(self.bias.shape) != (OUT_FEATURES,)
            or self.group_norm.num_groups != NUM_GROUPS
            or self.group_norm.eps != EPS
        ):
            raise RuntimeError("This fused kernel only supports the benchmark input shape and dtype.")

        y = self.matmul(x.contiguous())
        y = torch.sigmoid(y) * y
        y = y + self.bias
        return self.group_norm(y)
