import torch
import torch.nn as nn

import substrate
import substrate.language as S

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
EPS = 1.0e-5

BLOCK_THREADS = 256
WAVE_SIZE = 64
WAVE_TILE = 32
BLOCK_TILE = 64
K_TILE = 8
K_UNROLL = 16
TOTAL_ELEMENTS = BATCH_SIZE * OUT_FEATURES

BF16_BYTES = 2
F32_BYTES = 4
X_RANGE_BYTES = BATCH_SIZE * IN_FEATURES * BF16_BYTES
W_RANGE_BYTES = OUT_FEATURES * IN_FEATURES * BF16_BYTES
TMP_RANGE_BYTES = TOTAL_ELEMENTS * F32_BYTES
STATS_RANGE_BYTES = OUT_FEATURES * F32_BYTES
Y_RANGE_BYTES = TOTAL_ELEMENTS * BF16_BYTES


def _launch_gemm():
    return ((OUT_FEATURES // BLOCK_TILE, BATCH_SIZE // BLOCK_TILE, 1), (BLOCK_THREADS, 1, 1))


def _launch_stats():
    return ((OUT_FEATURES, 1, 1), (BLOCK_THREADS, 1, 1))


def _launch_total():
    return ((TOTAL_ELEMENTS // BLOCK_THREADS, 1, 1), (BLOCK_THREADS, 1, 1))


@substrate.jit
def gemm_pipelined_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((OUT_FEATURES, IN_FEATURES), S.bf16),
    TMP: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.f32),
):
    tid = S.thread_id(0)
    wave = tid // WAVE_SIZE
    lane = tid % WAVE_SIZE
    wave_m = wave // 2
    wave_n = wave % 2

    block_m = S.block_id(1) * BLOCK_TILE
    block_n = S.block_id(0) * BLOCK_TILE

    row_base = block_m + wave_m * WAVE_TILE
    col_base = block_n + wave_n * WAVE_TILE

    sh_idx = wave * WAVE_SIZE + lane
    sh_partner = sh_idx + WAVE_TILE

    a_sh = S.make_shared((2, BLOCK_THREADS, 2), S.u32)
    b_sh = S.make_shared((2, BLOCK_THREADS, 2), S.u32)

    x_rsrc = S.amdgpu.make_rsrc(X, X_RANGE_BYTES)
    w_rsrc = S.amdgpu.make_rsrc(W, W_RANGE_BYTES)

    if lane < WAVE_TILE:
        row_i32 = S.convert(row_base + lane, S.i32)
        col_i32 = S.convert(col_base + lane, S.i32)

        a_off0 = (row_i32 * IN_FEATURES) * BF16_BYTES
        b_off0 = (col_i32 * IN_FEATURES) * BF16_BYTES
        a_vec0 = S.amdgpu.raw_buffer_load_x4(x_rsrc, a_off0, 0, 0)
        b_vec0 = S.amdgpu.raw_buffer_load_x4(w_rsrc, b_off0, 0, 0)
        a_sh[0, sh_idx, 0] = a_vec0[0]
        a_sh[0, sh_idx, 1] = a_vec0[1]
        a_sh[0, sh_partner, 0] = a_vec0[2]
        a_sh[0, sh_partner, 1] = a_vec0[3]
        b_sh[0, sh_idx, 0] = b_vec0[0]
        b_sh[0, sh_idx, 1] = b_vec0[1]
        b_sh[0, sh_partner, 0] = b_vec0[2]
        b_sh[0, sh_partner, 1] = b_vec0[3]

        a_off1 = (row_i32 * IN_FEATURES + K_TILE) * BF16_BYTES
        b_off1 = (col_i32 * IN_FEATURES + K_TILE) * BF16_BYTES
        a_vec1 = S.amdgpu.raw_buffer_load_x4(x_rsrc, a_off1, 0, 0)
        b_vec1 = S.amdgpu.raw_buffer_load_x4(w_rsrc, b_off1, 0, 0)
        a_sh[1, sh_idx, 0] = a_vec1[0]
        a_sh[1, sh_idx, 1] = a_vec1[1]
        a_sh[1, sh_partner, 0] = a_vec1[2]
        a_sh[1, sh_partner, 1] = a_vec1[3]
        b_sh[1, sh_idx, 0] = b_vec1[0]
        b_sh[1, sh_idx, 1] = b_vec1[1]
        b_sh[1, sh_partner, 0] = b_vec1[2]
        b_sh[1, sh_partner, 1] = b_vec1[3]

    S.syncthreads()

    c = S.full((16,), 0.0, S.f32)

    for k_base in S.range(0, IN_FEATURES - K_UNROLL, K_UNROLL):
        a_frag0 = S.view(a_sh[0, sh_idx], S.Tensor((1, 4, 1), S.bf16))
        b_frag0 = S.view(b_sh[0, sh_idx], S.Tensor((1, 4, 1), S.bf16))
        c = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], c)

        if lane < WAVE_TILE:
            row_i32 = S.convert(row_base + lane, S.i32)
            col_i32 = S.convert(col_base + lane, S.i32)
            a_next0 = (row_i32 * IN_FEATURES + k_base + K_UNROLL) * BF16_BYTES
            b_next0 = (col_i32 * IN_FEATURES + k_base + K_UNROLL) * BF16_BYTES
            a_vec_next0 = S.amdgpu.raw_buffer_load_x4(x_rsrc, a_next0, 0, 0)
            b_vec_next0 = S.amdgpu.raw_buffer_load_x4(w_rsrc, b_next0, 0, 0)
            a_sh[0, sh_idx, 0] = a_vec_next0[0]
            a_sh[0, sh_idx, 1] = a_vec_next0[1]
            a_sh[0, sh_partner, 0] = a_vec_next0[2]
            a_sh[0, sh_partner, 1] = a_vec_next0[3]
            b_sh[0, sh_idx, 0] = b_vec_next0[0]
            b_sh[0, sh_idx, 1] = b_vec_next0[1]
            b_sh[0, sh_partner, 0] = b_vec_next0[2]
            b_sh[0, sh_partner, 1] = b_vec_next0[3]

        a_frag1 = S.view(a_sh[1, sh_idx], S.Tensor((1, 4, 1), S.bf16))
        b_frag1 = S.view(b_sh[1, sh_idx], S.Tensor((1, 4, 1), S.bf16))
        c = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], c)

        if lane < WAVE_TILE:
            row_i32 = S.convert(row_base + lane, S.i32)
            col_i32 = S.convert(col_base + lane, S.i32)
            a_next1 = (row_i32 * IN_FEATURES + k_base + K_UNROLL + K_TILE) * BF16_BYTES
            b_next1 = (col_i32 * IN_FEATURES + k_base + K_UNROLL + K_TILE) * BF16_BYTES
            a_vec_next1 = S.amdgpu.raw_buffer_load_x4(x_rsrc, a_next1, 0, 0)
            b_vec_next1 = S.amdgpu.raw_buffer_load_x4(w_rsrc, b_next1, 0, 0)
            a_sh[1, sh_idx, 0] = a_vec_next1[0]
            a_sh[1, sh_idx, 1] = a_vec_next1[1]
            a_sh[1, sh_partner, 0] = a_vec_next1[2]
            a_sh[1, sh_partner, 1] = a_vec_next1[3]
            b_sh[1, sh_idx, 0] = b_vec_next1[0]
            b_sh[1, sh_idx, 1] = b_vec_next1[1]
            b_sh[1, sh_partner, 0] = b_vec_next1[2]
            b_sh[1, sh_partner, 1] = b_vec_next1[3]

        S.syncthreads()

    a_frag0 = S.view(a_sh[0, sh_idx], S.Tensor((1, 4, 1), S.bf16))
    b_frag0 = S.view(b_sh[0, sh_idx], S.Tensor((1, 4, 1), S.bf16))
    c = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], c)

    a_frag1 = S.view(a_sh[1, sh_idx], S.Tensor((1, 4, 1), S.bf16))
    b_frag1 = S.view(b_sh[1, sh_idx], S.Tensor((1, 4, 1), S.bf16))
    c = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], c)

    col = col_base + (lane % WAVE_TILE)
    row_group = lane // WAVE_TILE
    for i in S.range(16):
        row = row_base + (i // 4) * 8 + row_group * 4 + (i % 4)
        TMP[row, col] = c[i]


@substrate.jit
def bias_scale_kernel(
    TMP: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.f32),
    BIAS0: S.Tensor((OUT_FEATURES,), S.bf16),
    SCALE: S.Tensor((OUT_FEATURES,), S.bf16),
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    row = idx // OUT_FEATURES
    col = idx % OUT_FEATURES
    value = TMP[row, col] + S.convert(BIAS0[col], S.f32)
    TMP[row, col] = value * S.convert(SCALE[col], S.f32)


@substrate.jit
def batchnorm_stats_kernel(
    TMP: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.f32),
    MEAN: S.Tensor((OUT_FEATURES,), S.f32),
    INVSTD: S.Tensor((OUT_FEATURES,), S.f32),
):
    tid = S.thread_id(0)
    col = S.block_id(0)

    mean_sh = S.make_shared((BLOCK_THREADS,), S.f32)
    m2_sh = S.make_shared((BLOCK_THREADS,), S.f32)
    count_sh = S.make_shared((BLOCK_THREADS,), S.i32)

    mean = S.convert(0.0, S.f32)
    m2 = S.convert(0.0, S.f32)
    count = S.convert(0, S.i32)
    for row in S.range(tid, BATCH_SIZE, BLOCK_THREADS):
        value = TMP[row, col]
        new_count = count + 1
        delta = value - mean
        mean = mean + delta / S.convert(new_count, S.f32)
        delta2 = value - mean
        m2 = m2 + delta * delta2
        count = new_count

    mean_sh[tid] = mean
    m2_sh[tid] = m2
    count_sh[tid] = count
    S.syncthreads()

    if tid < 128:
        count_a = count_sh[tid]
        count_b = count_sh[tid + 128]
        total = count_a + count_b
        if count_b > 0:
            delta = mean_sh[tid + 128] - mean_sh[tid]
            mean_sh[tid] = mean_sh[tid] + delta * S.convert(count_b, S.f32) / S.convert(total, S.f32)
            m2_sh[tid] = (
                m2_sh[tid]
                + m2_sh[tid + 128]
                + delta * delta * S.convert(count_a * count_b, S.f32) / S.convert(total, S.f32)
            )
            count_sh[tid] = total
    S.syncthreads()

    if tid < 64:
        count_a = count_sh[tid]
        count_b = count_sh[tid + 64]
        total = count_a + count_b
        if count_b > 0:
            delta = mean_sh[tid + 64] - mean_sh[tid]
            mean_sh[tid] = mean_sh[tid] + delta * S.convert(count_b, S.f32) / S.convert(total, S.f32)
            m2_sh[tid] = (
                m2_sh[tid]
                + m2_sh[tid + 64]
                + delta * delta * S.convert(count_a * count_b, S.f32) / S.convert(total, S.f32)
            )
            count_sh[tid] = total
    S.syncthreads()

    if tid < 32:
        count_a = count_sh[tid]
        count_b = count_sh[tid + 32]
        total = count_a + count_b
        if count_b > 0:
            delta = mean_sh[tid + 32] - mean_sh[tid]
            mean_sh[tid] = mean_sh[tid] + delta * S.convert(count_b, S.f32) / S.convert(total, S.f32)
            m2_sh[tid] = (
                m2_sh[tid]
                + m2_sh[tid + 32]
                + delta * delta * S.convert(count_a * count_b, S.f32) / S.convert(total, S.f32)
            )
            count_sh[tid] = total
    S.syncthreads()

    if tid < 16:
        count_a = count_sh[tid]
        count_b = count_sh[tid + 16]
        total = count_a + count_b
        if count_b > 0:
            delta = mean_sh[tid + 16] - mean_sh[tid]
            mean_sh[tid] = mean_sh[tid] + delta * S.convert(count_b, S.f32) / S.convert(total, S.f32)
            m2_sh[tid] = (
                m2_sh[tid]
                + m2_sh[tid + 16]
                + delta * delta * S.convert(count_a * count_b, S.f32) / S.convert(total, S.f32)
            )
            count_sh[tid] = total
    S.syncthreads()

    if tid < 8:
        count_a = count_sh[tid]
        count_b = count_sh[tid + 8]
        total = count_a + count_b
        if count_b > 0:
            delta = mean_sh[tid + 8] - mean_sh[tid]
            mean_sh[tid] = mean_sh[tid] + delta * S.convert(count_b, S.f32) / S.convert(total, S.f32)
            m2_sh[tid] = (
                m2_sh[tid]
                + m2_sh[tid + 8]
                + delta * delta * S.convert(count_a * count_b, S.f32) / S.convert(total, S.f32)
            )
            count_sh[tid] = total
    S.syncthreads()

    if tid < 4:
        count_a = count_sh[tid]
        count_b = count_sh[tid + 4]
        total = count_a + count_b
        if count_b > 0:
            delta = mean_sh[tid + 4] - mean_sh[tid]
            mean_sh[tid] = mean_sh[tid] + delta * S.convert(count_b, S.f32) / S.convert(total, S.f32)
            m2_sh[tid] = (
                m2_sh[tid]
                + m2_sh[tid + 4]
                + delta * delta * S.convert(count_a * count_b, S.f32) / S.convert(total, S.f32)
            )
            count_sh[tid] = total
    S.syncthreads()

    if tid < 2:
        count_a = count_sh[tid]
        count_b = count_sh[tid + 2]
        total = count_a + count_b
        if count_b > 0:
            delta = mean_sh[tid + 2] - mean_sh[tid]
            mean_sh[tid] = mean_sh[tid] + delta * S.convert(count_b, S.f32) / S.convert(total, S.f32)
            m2_sh[tid] = (
                m2_sh[tid]
                + m2_sh[tid + 2]
                + delta * delta * S.convert(count_a * count_b, S.f32) / S.convert(total, S.f32)
            )
            count_sh[tid] = total
    S.syncthreads()

    if tid < 1:
        count_a = count_sh[tid]
        count_b = count_sh[tid + 1]
        total = count_a + count_b
        if count_b > 0:
            delta = mean_sh[tid + 1] - mean_sh[tid]
            mean_sh[tid] = mean_sh[tid] + delta * S.convert(count_b, S.f32) / S.convert(total, S.f32)
            m2_sh[tid] = (
                m2_sh[tid]
                + m2_sh[tid + 1]
                + delta * delta * S.convert(count_a * count_b, S.f32) / S.convert(total, S.f32)
            )
            count_sh[tid] = total
    S.syncthreads()

    if tid == 0:
        mean = mean_sh[0]
        var = m2_sh[0] / S.convert(BATCH_SIZE, S.f32)
        if var < 0.0:
            var = 0.0
        MEAN[col] = mean
        INVSTD[col] = S.convert(1.0 / S.sqrt(var + S.convert(EPS, S.f32)), S.f32)


@substrate.jit
def batchnorm_apply_kernel(
    TMP: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.f32),
    MEAN: S.Tensor((OUT_FEATURES,), S.f32),
    INVSTD: S.Tensor((OUT_FEATURES,), S.f32),
    BN_WEIGHT: S.Tensor((OUT_FEATURES,), S.bf16),
    BN_BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    row = idx // OUT_FEATURES
    col = idx % OUT_FEATURES

    value = (TMP[row, col] - MEAN[col]) * INVSTD[col]
    value = value * S.convert(BN_WEIGHT[col], S.f32) + S.convert(BN_BIAS[col], S.f32)
    Y[row, col] = S.convert(value, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

        self._tmp = None
        self._mean = None
        self._invstd = None
        self._out = None

    def _ensure_workspace(self, x: torch.Tensor):
        if self._tmp is None or self._tmp.device != x.device:
            self._tmp = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=torch.float32)
            self._mean = torch.empty((OUT_FEATURES,), device=x.device, dtype=torch.float32)
            self._invstd = torch.empty((OUT_FEATURES,), device=x.device, dtype=torch.float32)
            self._out = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=torch.bfloat16)

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES):
            raise RuntimeError("ModelNew expects the fixed KernelBench batch/input shape")
        if x.dtype != torch.bfloat16:
            raise RuntimeError("ModelNew expects bfloat16 inputs")
        if tuple(self.scale.shape) != (OUT_FEATURES,):
            raise RuntimeError("ModelNew expects a 1D scale tensor with OUT_FEATURES entries")
        if self.bn.eps != EPS:
            raise RuntimeError("ModelNew expects the fixed BatchNorm epsilon")

        self._ensure_workspace(x)

        x = x.contiguous()
        w = self.gemm.weight.to(device=x.device, dtype=x.dtype).contiguous()
        bias = self.gemm.bias.to(device=x.device, dtype=x.dtype).contiguous()
        scale = self.scale.to(device=x.device, dtype=x.dtype).contiguous()
        bn_w = self.bn.weight.to(device=x.device, dtype=x.dtype).contiguous()
        bn_b = self.bn.bias.to(device=x.device, dtype=x.dtype).contiguous()

        gemm_pipelined_kernel[_launch_gemm](x, w, self._tmp)
        bias_scale_kernel[_launch_total](self._tmp, bias, scale)
        batchnorm_stats_kernel[_launch_stats](self._tmp, self._mean, self._invstd)
        batchnorm_apply_kernel[_launch_total](self._tmp, self._mean, self._invstd, bn_w, bn_b, self._out)
        return self._out
