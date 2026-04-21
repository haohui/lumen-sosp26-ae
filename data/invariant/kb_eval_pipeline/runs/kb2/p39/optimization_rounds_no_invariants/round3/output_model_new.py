import torch
import torch.nn as nn

import substrate
import substrate.language as S

BATCH_SIZE = 16384
IN_FEATURES = 4096
OUT_FEATURES = 4096
EPS = 1.0e-5

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = 64 * WAVES_PER_BLOCK
BN_THREADS = 256


def _gemm_launch():
    return ((OUT_FEATURES // BLOCK_N, BATCH_SIZE // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))


def _bn_stats_launch():
    return ((OUT_FEATURES // BN_THREADS, 1, 1), (BN_THREADS, 1, 1))


def _bn_apply_launch():
    return ((OUT_FEATURES // BN_THREADS, BATCH_SIZE, 1), (BN_THREADS, 1, 1))


def _a_lane_for_row(row):
    return (row % 4) + 8 * ((row % 16) // 4) + 4 * (row // 16)


@substrate.jit
def gemm_bias_scale_mfma(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((OUT_FEATURES, IN_FEATURES), S.bf16),
    BIAS0: S.Tensor((OUT_FEATURES,), S.bf16),
    SCALE: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    tid = S.thread_id(0)
    wave = tid // 64
    lane = tid % 64
    wave_row = wave // 2
    wave_col = wave % 2

    block_m = S.block_id(1)
    block_n = S.block_id(0)
    m0 = block_m * BLOCK_M
    n0 = block_n * BLOCK_N

    x_rsrc = S.amdgpu.make_rsrc(X, BATCH_SIZE * IN_FEATURES * 2)
    w_rsrc = S.amdgpu.make_rsrc(W, OUT_FEATURES * IN_FEATURES * 2)

    a_lds0 = S.make_shared((128, 4), S.u32)
    a_lds1 = S.make_shared((128, 4), S.u32)
    b_lds0 = S.make_shared((128, 4), S.u32)
    b_lds1 = S.make_shared((128, 4), S.u32)

    acc = S.full((16,), 0.0, S.f32)

    if tid < 128:
        a_idx = tid
        a_wave = a_idx // 64
        a_lane = a_idx % 64
        a_row_in_wave = (a_lane // 8) * 4 + (a_lane % 4) + ((a_lane // 4) % 2) * 16
        a_sel = a_lane // 32
        a_row = m0 + a_wave * 32 + a_row_in_wave
        a_k0 = a_sel * 4
        a_lo_off = (a_row * IN_FEATURES + a_k0) * 2
        a_hi_off = (a_row * IN_FEATURES + a_k0 + 8) * 2
        a_lo = S.amdgpu.raw_buffer_load_x4(x_rsrc, 0, a_lo_off, 0)
        a_hi = S.amdgpu.raw_buffer_load_x4(x_rsrc, 0, a_hi_off, 0)
        a_pack = S.full((4,), 0, S.u32)
        a_pack[0] = a_lo[0]
        a_pack[1] = a_lo[1]
        a_pack[2] = a_hi[0]
        a_pack[3] = a_hi[1]
        a_lds0[a_idx] = a_pack
    else:
        b_idx = tid - 128
        b_wave = b_idx // 64
        b_lane = b_idx % 64
        b_col_in_wave = b_lane % 32
        b_sel = b_lane // 32
        b_col = n0 + b_wave * 32 + b_col_in_wave
        b_k0 = b_sel * 4
        b_lo_off = (b_col * IN_FEATURES + b_k0) * 2
        b_hi_off = (b_col * IN_FEATURES + b_k0 + 8) * 2
        b_lo = S.amdgpu.raw_buffer_load_x4(w_rsrc, 0, b_lo_off, 0)
        b_hi = S.amdgpu.raw_buffer_load_x4(w_rsrc, 0, b_hi_off, 0)
        b_pack = S.full((4,), 0, S.u32)
        b_pack[0] = b_lo[0]
        b_pack[1] = b_lo[1]
        b_pack[2] = b_hi[0]
        b_pack[3] = b_hi[1]
        b_lds0[b_idx] = b_pack

    if tid < 128:
        a_idx = tid
        a_wave = a_idx // 64
        a_lane = a_idx % 64
        a_row_in_wave = (a_lane // 8) * 4 + (a_lane % 4) + ((a_lane // 4) % 2) * 16
        a_sel = a_lane // 32
        a_row = m0 + a_wave * 32 + a_row_in_wave
        a_k0 = BLOCK_K + a_sel * 4
        a_lo_off = (a_row * IN_FEATURES + a_k0) * 2
        a_hi_off = (a_row * IN_FEATURES + a_k0 + 8) * 2
        a_lo = S.amdgpu.raw_buffer_load_x4(x_rsrc, 0, a_lo_off, 0)
        a_hi = S.amdgpu.raw_buffer_load_x4(x_rsrc, 0, a_hi_off, 0)
        a_pack = S.full((4,), 0, S.u32)
        a_pack[0] = a_lo[0]
        a_pack[1] = a_lo[1]
        a_pack[2] = a_hi[0]
        a_pack[3] = a_hi[1]
        a_lds1[a_idx] = a_pack
    else:
        b_idx = tid - 128
        b_wave = b_idx // 64
        b_lane = b_idx % 64
        b_col_in_wave = b_lane % 32
        b_sel = b_lane // 32
        b_col = n0 + b_wave * 32 + b_col_in_wave
        b_k0 = BLOCK_K + b_sel * 4
        b_lo_off = (b_col * IN_FEATURES + b_k0) * 2
        b_hi_off = (b_col * IN_FEATURES + b_k0 + 8) * 2
        b_lo = S.amdgpu.raw_buffer_load_x4(w_rsrc, 0, b_lo_off, 0)
        b_hi = S.amdgpu.raw_buffer_load_x4(w_rsrc, 0, b_hi_off, 0)
        b_pack = S.full((4,), 0, S.u32)
        b_pack[0] = b_lo[0]
        b_pack[1] = b_lo[1]
        b_pack[2] = b_hi[0]
        b_pack[3] = b_hi[1]
        b_lds1[b_idx] = b_pack

    S.syncthreads()

    for k0 in S.range(0, IN_FEATURES, 2 * BLOCK_K):
        a_frag0 = S.view(a_lds0[wave_row * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(b_lds0[wave_col * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
        a_frag1 = S.view(a_lds1[wave_row * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag1 = S.view(b_lds1[wave_col * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
        S.syncthreads()

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

        next_k0 = k0 + 2 * BLOCK_K
        if tid < 128:
            a_idx = tid
            a_wave = a_idx // 64
            a_lane = a_idx % 64
            a_row_in_wave = (a_lane // 8) * 4 + (a_lane % 4) + ((a_lane // 4) % 2) * 16
            a_sel = a_lane // 32
            a_row = m0 + a_wave * 32 + a_row_in_wave
            a_k0 = next_k0 + a_sel * 4
            a_lo_off = (a_row * IN_FEATURES + a_k0) * 2
            a_hi_off = (a_row * IN_FEATURES + a_k0 + 8) * 2
            a_lo = S.amdgpu.raw_buffer_load_x4(x_rsrc, 0, a_lo_off, 0)
            a_hi = S.amdgpu.raw_buffer_load_x4(x_rsrc, 0, a_hi_off, 0)
            a_pack = S.full((4,), 0, S.u32)
            a_pack[0] = a_lo[0]
            a_pack[1] = a_lo[1]
            a_pack[2] = a_hi[0]
            a_pack[3] = a_hi[1]
            a_lds0[a_idx] = a_pack
        else:
            b_idx = tid - 128
            b_wave = b_idx // 64
            b_lane = b_idx % 64
            b_col_in_wave = b_lane % 32
            b_sel = b_lane // 32
            b_col = n0 + b_wave * 32 + b_col_in_wave
            b_k0 = next_k0 + b_sel * 4
            b_lo_off = (b_col * IN_FEATURES + b_k0) * 2
            b_hi_off = (b_col * IN_FEATURES + b_k0 + 8) * 2
            b_lo = S.amdgpu.raw_buffer_load_x4(w_rsrc, 0, b_lo_off, 0)
            b_hi = S.amdgpu.raw_buffer_load_x4(w_rsrc, 0, b_hi_off, 0)
            b_pack = S.full((4,), 0, S.u32)
            b_pack[0] = b_lo[0]
            b_pack[1] = b_lo[1]
            b_pack[2] = b_hi[0]
            b_pack[3] = b_hi[1]
            b_lds0[b_idx] = b_pack

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

        next_k1 = k0 + 3 * BLOCK_K
        if tid < 128:
            a_idx = tid
            a_wave = a_idx // 64
            a_lane = a_idx % 64
            a_row_in_wave = (a_lane // 8) * 4 + (a_lane % 4) + ((a_lane // 4) % 2) * 16
            a_sel = a_lane // 32
            a_row = m0 + a_wave * 32 + a_row_in_wave
            a_k0 = next_k1 + a_sel * 4
            a_lo_off = (a_row * IN_FEATURES + a_k0) * 2
            a_hi_off = (a_row * IN_FEATURES + a_k0 + 8) * 2
            a_lo = S.amdgpu.raw_buffer_load_x4(x_rsrc, 0, a_lo_off, 0)
            a_hi = S.amdgpu.raw_buffer_load_x4(x_rsrc, 0, a_hi_off, 0)
            a_pack = S.full((4,), 0, S.u32)
            a_pack[0] = a_lo[0]
            a_pack[1] = a_lo[1]
            a_pack[2] = a_hi[0]
            a_pack[3] = a_hi[1]
            a_lds1[a_idx] = a_pack
        else:
            b_idx = tid - 128
            b_wave = b_idx // 64
            b_lane = b_idx % 64
            b_col_in_wave = b_lane % 32
            b_sel = b_lane // 32
            b_col = n0 + b_wave * 32 + b_col_in_wave
            b_k0 = next_k1 + b_sel * 4
            b_lo_off = (b_col * IN_FEATURES + b_k0) * 2
            b_hi_off = (b_col * IN_FEATURES + b_k0 + 8) * 2
            b_lo = S.amdgpu.raw_buffer_load_x4(w_rsrc, 0, b_lo_off, 0)
            b_hi = S.amdgpu.raw_buffer_load_x4(w_rsrc, 0, b_hi_off, 0)
            b_pack = S.full((4,), 0, S.u32)
            b_pack[0] = b_lo[0]
            b_pack[1] = b_lo[1]
            b_pack[2] = b_hi[0]
            b_pack[3] = b_hi[1]
            b_lds1[b_idx] = b_pack

        S.syncthreads()

    out_col = n0 + wave_col * 32 + (lane % 32)
    bias = S.convert(BIAS0[out_col], S.f32)
    scale = S.convert(SCALE[out_col], S.f32)
    row_block = lane // 32
    base_row = m0 + wave_row * 32 + row_block * 16
    for i in S.range(16):
        out_row = base_row + i
        y = (acc[i] + bias) * scale
        Y[out_row, out_col] = S.convert(y, S.bf16)


@substrate.jit
def bn_stats_kernel(
    X: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
    MEAN: S.Tensor((OUT_FEATURES,), S.f32),
    INVSTD: S.Tensor((OUT_FEATURES,), S.f32),
):
    col = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = S.convert(0.0, S.f32)
    for row in S.range(BATCH_SIZE):
        total += S.convert(X[row, col], S.f32)
    mean = total / S.convert(BATCH_SIZE, S.f32)
    var = S.convert(0.0, S.f32)
    for row in S.range(BATCH_SIZE):
        d = S.convert(X[row, col], S.f32) - mean
        var += d * d
    var = var / S.convert(BATCH_SIZE, S.f32)
    MEAN[col] = mean
    INVSTD[col] = S.convert(1.0, S.f32) / S.sqrt(var + S.convert(EPS, S.f32))


@substrate.jit
def bn_apply_kernel(
    X: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
    MEAN: S.Tensor((OUT_FEATURES,), S.f32),
    INVSTD: S.Tensor((OUT_FEATURES,), S.f32),
    BN_WEIGHT: S.Tensor((OUT_FEATURES,), S.bf16),
    BN_BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    row = S.block_id(1)
    col = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    v = (S.convert(X[row, col], S.f32) - MEAN[col]) * INVSTD[col]
    v = v * S.convert(BN_WEIGHT[col], S.f32) + S.convert(BN_BIAS[col], S.f32)
    Y[row, col] = S.convert(v, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self._cache_key = None
        self._w_bf16 = None
        self._bias_bf16 = None
        self._scale_bf16 = None
        self._bn_w_bf16 = None
        self._bn_b_bf16 = None

    def _refresh_param_cache(self, device):
        key = (
            device,
            self.gemm.weight._version,
            self.gemm.bias._version,
            self.scale._version,
            self.bn.weight._version,
            self.bn.bias._version,
        )
        if key == self._cache_key:
            return
        self._w_bf16 = self.gemm.weight.detach().to(device=device, dtype=torch.bfloat16).contiguous()
        self._bias_bf16 = self.gemm.bias.detach().to(device=device, dtype=torch.bfloat16).contiguous()
        self._scale_bf16 = self.scale.detach().to(device=device, dtype=torch.bfloat16).contiguous()
        self._bn_w_bf16 = self.bn.weight.detach().to(device=device, dtype=torch.bfloat16).contiguous()
        self._bn_b_bf16 = self.bn.bias.detach().to(device=device, dtype=torch.bfloat16).contiguous()
        self._cache_key = key

    def forward(self, x):
        if (
            tuple(x.shape) != (BATCH_SIZE, IN_FEATURES)
            or x.dtype != torch.bfloat16
            or tuple(self.scale.shape) != (OUT_FEATURES,)
            or self.bn.eps != EPS
            or x.device.type != "cuda"
        ):
            raise RuntimeError("This optimized kernel only supports the fixed KernelBench bf16 CUDA shape.")

        self._refresh_param_cache(x.device)

        gemm_out = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=torch.bfloat16)
        mean = torch.empty((OUT_FEATURES,), device=x.device, dtype=torch.float32)
        invstd = torch.empty((OUT_FEATURES,), device=x.device, dtype=torch.float32)
        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=torch.bfloat16)

        gemm_bias_scale_mfma[_gemm_launch](x.contiguous(), self._w_bf16, self._bias_bf16, self._scale_bf16, gemm_out)
        bn_stats_kernel[_bn_stats_launch](gemm_out, mean, invstd)
        bn_apply_kernel[_bn_apply_launch](gemm_out, mean, invstd, self._bn_w_bf16, self._bn_b_bf16, y)
        return y
