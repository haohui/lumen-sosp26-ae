import torch
import torch.nn as nn

import substrate
import substrate.language as S


BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
EPS = 1.0e-5
WARP_SIZE = 64
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = WARP_SIZE * WAVES_PER_BLOCK
WAVE_TILE_M = 32
WAVE_TILE_N = 32
BLOCK_TILE_M = 64
BLOCK_TILE_N = 64
K_TILE = 16
M_TILES = BATCH_SIZE // WAVE_TILE_M
N_TILES = OUT_FEATURES // WAVE_TILE_N
K_TILES = IN_FEATURES // K_TILE
GRID_N = OUT_FEATURES // BLOCK_TILE_N
TOTAL_ELEMENTS = BATCH_SIZE * OUT_FEATURES
A_PACKED_BYTES = M_TILES * K_TILES * WARP_SIZE * 16
B_PACKED_BYTES = N_TILES * K_TILES * WARP_SIZE * 16
RUNNING_VAR_SCALE = BATCH_SIZE / (BATCH_SIZE - 1)


def _gemm_launch():
    grid = ((BATCH_SIZE // BLOCK_TILE_M) * (OUT_FEATURES // BLOCK_TILE_N), 1, 1)
    return grid, (THREADS_PER_BLOCK, 1, 1)


def _col_reduce_launch():
    return ((OUT_FEATURES, 1, 1), (1, 1, 1))


def _elem_launch():
    grid = ((TOTAL_ELEMENTS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, 1, 1)
    return grid, (THREADS_PER_BLOCK, 1, 1)


def _row_launch():
    return ((BATCH_SIZE, 1, 1), (1, 1, 1))


@substrate.jit
def gemm_mfma_kernel(
    A_PACKED: S.Tensor((M_TILES, K_TILES, WARP_SIZE, 8), S.u16),
    B_PACKED: S.Tensor((N_TILES, K_TILES, WARP_SIZE, 8), S.u16),
    BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
    C: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.f32),
):
    tid = S.thread_id(0)
    lane = tid % WARP_SIZE
    warp = tid // WARP_SIZE
    warp_row = warp // 2
    warp_col = warp % 2

    block = S.block_id(0)
    block_m = block // GRID_N
    block_n = block % GRID_N
    tile_m = block_m * 2 + warp_row
    tile_n = block_n * 2 + warp_col

    a_rsrc = S.amdgpu.make_rsrc(A_PACKED, A_PACKED_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B_PACKED, B_PACKED_BYTES)

    a_shared = S.make_shared((2, WAVES_PER_BLOCK, WARP_SIZE, 4), S.u32)
    b_shared = S.make_shared((2, WAVES_PER_BLOCK, WARP_SIZE, 4), S.u32)
    acc = S.full((16,), 0.0, S.f32)

    a_words_0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, ((tile_m * K_TILES) * WARP_SIZE + lane) * 16, 0)
    b_words_0 = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, ((tile_n * K_TILES) * WARP_SIZE + lane) * 16, 0)
    a_shared[0, warp, lane] = a_words_0
    b_shared[0, warp, lane] = b_words_0

    a_words_1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, ((tile_m * K_TILES + 1) * WARP_SIZE + lane) * 16, 0)
    b_words_1 = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, ((tile_n * K_TILES + 1) * WARP_SIZE + lane) * 16, 0)
    a_shared[1, warp, lane] = a_words_1
    b_shared[1, warp, lane] = b_words_1
    S.syncthreads()

    for kk in S.range(0, K_TILES - 2, 2):
        curr_a_words_0 = a_shared[0, warp, lane]
        curr_b_words_0 = b_shared[0, warp, lane]
        a_frag_0 = S.view(curr_a_words_0, S.Tensor((2, 4, 1), S.bf16))
        b_frag_0 = S.view(curr_b_words_0, S.Tensor((2, 4, 1), S.bf16))
        next_a_words_0 = S.amdgpu.raw_buffer_load_x4(
            a_rsrc, 0, ((tile_m * K_TILES + kk + 2) * WARP_SIZE + lane) * 16, 0
        )
        next_b_words_0 = S.amdgpu.raw_buffer_load_x4(
            b_rsrc, 0, ((tile_n * K_TILES + kk + 2) * WARP_SIZE + lane) * 16, 0
        )
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_0[0], b_frag_0[0], acc)
        a_shared[0, warp, lane] = next_a_words_0
        b_shared[0, warp, lane] = next_b_words_0
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_0[1], b_frag_0[1], acc)

        curr_a_words_1 = a_shared[1, warp, lane]
        curr_b_words_1 = b_shared[1, warp, lane]
        a_frag_1 = S.view(curr_a_words_1, S.Tensor((2, 4, 1), S.bf16))
        b_frag_1 = S.view(curr_b_words_1, S.Tensor((2, 4, 1), S.bf16))
        next_a_words_1 = S.amdgpu.raw_buffer_load_x4(
            a_rsrc, 0, ((tile_m * K_TILES + kk + 3) * WARP_SIZE + lane) * 16, 0
        )
        next_b_words_1 = S.amdgpu.raw_buffer_load_x4(
            b_rsrc, 0, ((tile_n * K_TILES + kk + 3) * WARP_SIZE + lane) * 16, 0
        )
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_1[0], b_frag_1[0], acc)
        a_shared[1, warp, lane] = next_a_words_1
        b_shared[1, warp, lane] = next_b_words_1
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_1[1], b_frag_1[1], acc)
        S.syncthreads()

    a_frag_last0 = S.view(a_shared[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag_last0 = S.view(b_shared[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_last0[0], b_frag_last0[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_last0[1], b_frag_last0[1], acc)

    a_frag_last1 = S.view(a_shared[1, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag_last1 = S.view(b_shared[1, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_last1[0], b_frag_last1[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_last1[1], b_frag_last1[1], acc)

    col = tile_n * WAVE_TILE_N + (lane % 32)
    bias = S.convert(BIAS[col], S.f32)
    for acc_idx in S.range(16):
        row = tile_m * WAVE_TILE_M + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        C[row, col] = acc[acc_idx] + bias


@substrate.jit
def batchnorm_stats_kernel(
    X: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.f32),
    MEAN: S.Tensor((OUT_FEATURES,), S.f32),
    INVSTD: S.Tensor((OUT_FEATURES,), S.f32),
    RUNNING_MEAN: S.Tensor((OUT_FEATURES,), S.bf16),
    RUNNING_VAR: S.Tensor((OUT_FEATURES,), S.bf16),
    MOMENTUM: S.Tensor((1,), S.f32),
):
    col = S.block_id(0)
    mean = S.convert(0.0, S.f32)
    for row in S.range(BATCH_SIZE):
        mean += X[row, col]
    mean = mean / S.convert(BATCH_SIZE, S.f32)

    var = S.convert(0.0, S.f32)
    for row in S.range(BATCH_SIZE):
        diff = X[row, col] - mean
        var += diff * diff
    var = var / S.convert(BATCH_SIZE, S.f32)

    momentum = MOMENTUM[0]
    keep = S.convert(1.0, S.f32) - momentum
    running_mean = keep * S.convert(RUNNING_MEAN[col], S.f32) + momentum * mean
    running_var = keep * S.convert(RUNNING_VAR[col], S.f32) + momentum * (var * S.convert(RUNNING_VAR_SCALE, S.f32))

    MEAN[col] = mean
    INVSTD[col] = S.convert(1.0, S.f32) / S.sqrt(var + S.convert(EPS, S.f32))
    RUNNING_MEAN[col] = S.convert(running_mean, S.bf16)
    RUNNING_VAR[col] = S.convert(running_var, S.bf16)


@substrate.jit
def batchnorm_apply_kernel(
    X: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.f32),
    MEAN: S.Tensor((OUT_FEATURES,), S.f32),
    INVSTD: S.Tensor((OUT_FEATURES,), S.f32),
    WEIGHT: S.Tensor((OUT_FEATURES,), S.bf16),
    BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
    SCALE: S.Tensor((1,), S.bf16),
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    if idx < TOTAL_ELEMENTS:
        row = idx // OUT_FEATURES
        col = idx % OUT_FEATURES
        val = (X[row, col] - MEAN[col]) * INVSTD[col]
        val = val * S.convert(WEIGHT[col], S.f32) + S.convert(BIAS[col], S.f32)
        X[row, col] = val * S.convert(SCALE[0], S.f32)


@substrate.jit
def softmax_kernel(
    X: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.f32),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    row = S.block_id(0)
    max_val = X[row, 0]
    for col in S.range(1, OUT_FEATURES):
        val = X[row, col]
        if val > max_val:
            max_val = val

    sum_exp = S.convert(0.0, S.f32)
    for col in S.range(OUT_FEATURES):
        sum_exp += S.exp(X[row, col] - max_val)

    inv_sum = S.convert(1.0, S.f32) / sum_exp
    for col in S.range(OUT_FEATURES):
        Y[row, col] = S.convert(S.exp(X[row, col] - max_val) * inv_sum, S.bf16)


def _pack_a(x: torch.Tensor) -> torch.Tensor:
    x_tiles = x.contiguous().view(M_TILES, WAVE_TILE_M, K_TILES, K_TILE).permute(0, 2, 1, 3).contiguous()
    lo = torch.cat((x_tiles[..., 0:4], x_tiles[..., 8:12]), dim=-1)
    hi = torch.cat((x_tiles[..., 4:8], x_tiles[..., 12:16]), dim=-1)
    packed = torch.cat((lo, hi), dim=2).contiguous()
    return packed.view(torch.uint16).contiguous()


def _pack_b(w_t: torch.Tensor) -> torch.Tensor:
    w_tiles = w_t.contiguous().view(K_TILES, K_TILE, N_TILES, WAVE_TILE_N).permute(2, 0, 1, 3).contiguous()
    half0 = w_tiles[:, :, 0:8, :].contiguous().view(N_TILES, K_TILES, 8, 8, 4).permute(0, 1, 3, 2, 4)
    half1 = w_tiles[:, :, 8:16, :].contiguous().view(N_TILES, K_TILES, 8, 8, 4).permute(0, 1, 3, 2, 4)
    packed = torch.cat(
        (
            half0.contiguous().view(N_TILES, K_TILES, WARP_SIZE, 4),
            half1.contiguous().view(N_TILES, K_TILES, WARP_SIZE, 4),
        ),
        dim=-1,
    ).contiguous()
    return packed.view(torch.uint16).contiguous()


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self._bn_momentum = float(bn_momentum)
        self._cached_weight_key = None
        self._cached_weight_packed = None
        self._workspace_key = None
        self._gemm_out = None
        self._mean = None
        self._invstd = None
        self._momentum_buf = None
        self._softmax_out = None

    def _get_packed_weight(self, device: torch.device) -> torch.Tensor:
        weight = self.gemm.weight
        key = (device.type, device.index, weight.data_ptr(), weight.dtype)
        if self._cached_weight_key != key:
            w_t = weight.t().contiguous()
            self._cached_weight_packed = _pack_b(w_t)
            self._cached_weight_key = key
        return self._cached_weight_packed

    def _ensure_workspace(self, device: torch.device):
        key = (device.type, device.index)
        if self._workspace_key != key:
            self._gemm_out = torch.empty((BATCH_SIZE, OUT_FEATURES), device=device, dtype=torch.float32)
            self._mean = torch.empty((OUT_FEATURES,), device=device, dtype=torch.float32)
            self._invstd = torch.empty((OUT_FEATURES,), device=device, dtype=torch.float32)
            self._momentum_buf = torch.tensor([self._bn_momentum], device=device, dtype=torch.float32)
            self._softmax_out = torch.empty((BATCH_SIZE, OUT_FEATURES), device=device, dtype=torch.bfloat16)
            self._workspace_key = key

    def forward(self, x):
        device = x.device
        self._ensure_workspace(device)

        x_packed = _pack_a(x)
        w_packed = self._get_packed_weight(device)

        gemm_grid, gemm_block = _gemm_launch()
        gemm_mfma_kernel[lambda: (gemm_grid, gemm_block)](
            x_packed,
            w_packed,
            self.gemm.bias,
            self._gemm_out,
        )

        stats_grid, stats_block = _col_reduce_launch()
        batchnorm_stats_kernel[lambda: (stats_grid, stats_block)](
            self._gemm_out,
            self._mean,
            self._invstd,
            self.bn.running_mean,
            self.bn.running_var,
            self._momentum_buf,
        )

        elem_grid, elem_block = _elem_launch()
        batchnorm_apply_kernel[lambda: (elem_grid, elem_block)](
            self._gemm_out,
            self._mean,
            self._invstd,
            self.bn.weight,
            self.bn.bias,
            self.scale,
        )

        row_grid, row_block = _row_launch()
        softmax_kernel[lambda: (row_grid, row_block)](
            self._gemm_out,
            self._softmax_out,
        )
        return self._softmax_out
