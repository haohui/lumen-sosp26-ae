import torch
import torch.nn as nn
import substrate
import substrate.language as S


BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
EPS = 1e-05

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WAVE_SIZE = 64
WAVES_M = 2
WAVES_N = 2
THREADS = WAVE_SIZE * WAVES_M * WAVES_N

X_RANGE_BYTES = BATCH_SIZE * IN_FEATURES * 2
W_RANGE_BYTES = IN_FEATURES * OUT_FEATURES * 2


def _launch_gemm():
    return ((OUT_FEATURES // BLOCK_N, BATCH_SIZE // BLOCK_M, 1), (THREADS, 1, 1))


def _launch_stats():
    return ((OUT_FEATURES // 128, 1, 1), (128, 1, 1))


def _launch_norm():
    return (((BATCH_SIZE * OUT_FEATURES) // 256, 1, 1), (256, 1, 1))


@substrate.jit
def fused_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((IN_FEATURES, OUT_FEATURES), S.bf16),
    BIAS0: S.Tensor((OUT_FEATURES,), S.f32),
    SCALE: S.Tensor((OUT_FEATURES,), S.f32),
    Y_TMP: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    block_m = S.block_id(1) * BLOCK_M
    block_n = S.block_id(0) * BLOCK_N

    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    wave = tid >> 6
    wave_m = wave >> 1
    wave_n = wave % 2
    lane_row = lane % 32
    lane_group = lane >> 5

    x_rsrc = S.amdgpu.make_rsrc(X, S.convert(X_RANGE_BYTES, S.i32))
    w_rsrc = S.amdgpu.make_rsrc(W, S.convert(W_RANGE_BYTES, S.i32))

    a_smem = S.make_shared((BLOCK_M, 2, 8), S.bf16)
    b_smem = S.make_shared((BLOCK_N, 2, 8), S.bf16)

    acc = S.full((16,), 0.0, S.f32)
    zero = S.convert(0, S.i32)

    for k0 in S.range(0, IN_FEATURES, BLOCK_K):
        a_row = wave_m * 32 + lane_row
        a_half = lane_group
        a_offset_elems = (block_m + a_row) * IN_FEATURES + k0 + a_half * 8
        a_packed = S.amdgpu.raw_buffer_load_x4(
            x_rsrc,
            zero,
            S.convert(a_offset_elems * 2, S.i32),
            0,
        )
        a_frag = S.view(a_packed, S.Tensor((2, 4, 1), S.bf16))
        for sub in S.range(2):
            for i in S.range(4):
                a_smem[a_row, a_half, sub * 4 + i] = a_frag[sub, i, 0]

        b_k = lane >> 2
        b_half = b_k >> 3
        b_chunk = lane % 4
        b_col = wave_n * 32 + b_chunk * 8
        b_offset_elems = (k0 + b_k) * OUT_FEATURES + block_n + b_col
        b_packed = S.amdgpu.raw_buffer_load_x4(
            w_rsrc,
            zero,
            S.convert(b_offset_elems * 2, S.i32),
            0,
        )
        b_frag = S.view(b_packed, S.Tensor((2, 4, 1), S.bf16))
        for sub in S.range(2):
            for i in S.range(4):
                b_smem[b_col + sub * 4 + i, b_half, b_k % 8] = b_frag[sub, i, 0]

        S.syncthreads()

        a_lane = S.view(
            a_smem[wave_m * 32 + lane_row, lane_group],
            S.Tensor((2, 4, 1), S.bf16),
        )
        b_lane = S.view(
            b_smem[wave_n * 32 + lane_row, lane_group],
            S.Tensor((2, 4, 1), S.bf16),
        )
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_lane[0], b_lane[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_lane[1], b_lane[1], acc)

        S.syncthreads()

    out_row = block_m + wave_m * 32 + lane_row
    out_col = block_n + wave_n * 32 + lane_group * 16

    for ni in S.range(16):
        col = out_col + ni
        value = acc[ni] + BIAS0[col]
        value = value * SCALE[col]
        Y_TMP[out_row, col] = S.convert(value, S.bf16)


@substrate.jit
def stats_kernel(
    Y_TMP: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
    MEAN: S.Tensor((OUT_FEATURES,), S.f32),
    VAR: S.Tensor((OUT_FEATURES,), S.f32),
):
    col = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    mean = S.convert(0.0, S.f32)
    for row in S.range(BATCH_SIZE):
        mean += S.convert(Y_TMP[row, col], S.f32)
    mean = mean / S.convert(BATCH_SIZE, S.f32)

    var = S.convert(0.0, S.f32)
    for row in S.range(BATCH_SIZE):
        delta = S.convert(Y_TMP[row, col], S.f32) - mean
        var += delta * delta
    var = var / S.convert(BATCH_SIZE, S.f32)

    MEAN[col] = mean
    VAR[col] = var


@substrate.jit
def norm_kernel(
    Y_TMP: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
    MEAN: S.Tensor((OUT_FEATURES,), S.f32),
    VAR: S.Tensor((OUT_FEATURES,), S.f32),
    BN_WEIGHT: S.Tensor((OUT_FEATURES,), S.f32),
    BN_BIAS: S.Tensor((OUT_FEATURES,), S.f32),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    row = idx // OUT_FEATURES
    col = idx % OUT_FEATURES
    value = S.convert(Y_TMP[row, col], S.f32)
    value = (value - MEAN[col]) / S.sqrt(VAR[col] + S.convert(EPS, S.f32))
    value = value * BN_WEIGHT[col] + BN_BIAS[col]
    Y[row, col] = S.convert(value, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-05, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.gemm(x)
        x = x * self.scale
        x = self.bn(x)
        return x
