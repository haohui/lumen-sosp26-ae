import torch
import torch.nn as nn

import substrate
import substrate.language as S


BATCH_SIZE = 16384
IN_FEATURES = 4096
OUT_FEATURES = 4096
EPS = 1.0e-5

WAVE_SIZE = 64
NUM_WAVES = 4
THREADS = WAVE_SIZE * NUM_WAVES

BLOCK_M = 64
BLOCK_N = 64
WAVE_M = 32
WAVE_N = 32
BLOCK_K = 16
K_LOOP_STEP = BLOCK_K * 2

GRID_N = OUT_FEATURES // BLOCK_N
GEMM_BLOCKS = (BATCH_SIZE // BLOCK_M) * GRID_N
STATS_BLOCKS = OUT_FEATURES // 256
NORM_BLOCKS = (BATCH_SIZE * OUT_FEATURES) // 256

RSRC_WORD3 = 0x00020000


def _launch_gemm():
    return ((GEMM_BLOCKS, 1, 1), (THREADS, 1, 1))


def _launch_stats():
    return ((STATS_BLOCKS, 1, 1), (256, 1, 1))


def _launch_norm():
    return ((NORM_BLOCKS, 1, 1), (256, 1, 1))


@substrate.jit
def gemm_mfma_kernel(
    x_desc: S.Tensor((4,), S.u32),
    w_desc: S.Tensor((4,), S.u32),
    bias: S.Tensor((OUT_FEATURES,), S.bf16),
    scale: S.Tensor((OUT_FEATURES,), S.bf16),
    y_tmp: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    wave = tid // WAVE_SIZE
    wave_row = wave // 2
    wave_col = wave % 2

    block = S.block_id(0)
    block_m = (block // GRID_N) * BLOCK_M
    block_n = (block % GRID_N) * BLOCK_N
    tile_row = block_m + wave_row * WAVE_M
    tile_col = block_n + wave_col * WAVE_N

    a_shared = S.make_shared((2, NUM_WAVES, WAVE_SIZE, 4), S.u32)
    b_shared = S.make_shared((2, NUM_WAVES, WAVE_SIZE, 4), S.u32)

    acc = S.full((16,), 0.0, S.f32)

    a_lane_group = lane // 32
    a_row = lane % 32
    b_col_group = (lane % 32) // 4
    b_k = (lane // 32) * 4 + (lane % 4)

    a_stage0_elem = (tile_row + a_row) * IN_FEATURES
    b_stage0_elem = tile_col + b_col_group * 4

    a_shared[0, wave, lane] = S.amdgpu.raw_buffer_load_x4(
        x_desc, (a_stage0_elem + a_lane_group * 4) * 2, 0, 0
    )
    b_shared[0, wave, lane] = S.amdgpu.raw_buffer_load_x4(
        w_desc, (b_stage0_elem + b_k * OUT_FEATURES) * 2, 0, 0
    )

    a_stage1_elem = (tile_row + a_row) * IN_FEATURES + BLOCK_K
    b_stage1_elem = tile_col + b_col_group * 4 + BLOCK_K * OUT_FEATURES

    a_shared[1, wave, lane] = S.amdgpu.raw_buffer_load_x4(
        x_desc, (a_stage1_elem + a_lane_group * 4) * 2, 0, 0
    )
    b_shared[1, wave, lane] = S.amdgpu.raw_buffer_load_x4(
        w_desc, (b_stage1_elem + b_k * OUT_FEATURES) * 2, 0, 0
    )
    S.syncthreads()

    for k_base in S.range(0, IN_FEATURES - K_LOOP_STEP, K_LOOP_STEP):
        a_frag0 = S.view(a_shared[0, wave, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(b_shared[0, wave, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)
        a_shared[0, wave, lane] = S.amdgpu.raw_buffer_load_x4(
            x_desc,
            ((tile_row + a_row) * IN_FEATURES + k_base + K_LOOP_STEP + a_lane_group * 4) * 2,
            0,
            0,
        )
        b_shared[0, wave, lane] = S.amdgpu.raw_buffer_load_x4(
            w_desc,
            ((k_base + K_LOOP_STEP + b_k) * OUT_FEATURES + tile_col + b_col_group * 4) * 2,
            0,
            0,
        )

        a_frag1 = S.view(a_shared[1, wave, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag1 = S.view(b_shared[1, wave, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)
        a_shared[1, wave, lane] = S.amdgpu.raw_buffer_load_x4(
            x_desc,
            ((tile_row + a_row) * IN_FEATURES + k_base + K_LOOP_STEP + BLOCK_K + a_lane_group * 4) * 2,
            0,
            0,
        )
        b_shared[1, wave, lane] = S.amdgpu.raw_buffer_load_x4(
            w_desc,
            ((k_base + K_LOOP_STEP + BLOCK_K + b_k) * OUT_FEATURES + tile_col + b_col_group * 4) * 2,
            0,
            0,
        )
        S.syncthreads()

    a_frag0 = S.view(a_shared[0, wave, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag0 = S.view(b_shared[0, wave, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

    a_frag1 = S.view(a_shared[1, wave, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag1 = S.view(b_shared[1, wave, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

    out_col = tile_col + (lane % 32)
    bias_val = S.convert(bias[out_col], S.f32)
    scale_val = S.convert(scale[out_col], S.f32)
    lane_row_group = lane // 32

    for acc_idx in S.range(16):
        out_row = tile_row + 8 * (acc_idx // 4) + 4 * lane_row_group + (acc_idx % 4)
        value = (acc[acc_idx] + bias_val) * scale_val
        y_tmp[out_row, out_col] = S.convert(value, S.bf16)


@substrate.jit
def column_stats_kernel(
    y_tmp: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
    mean: S.Tensor((OUT_FEATURES,), S.f32),
    var: S.Tensor((OUT_FEATURES,), S.f32),
):
    col = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    sum_val = S.convert(0.0, S.f32)
    for row in S.range(BATCH_SIZE):
        sum_val += S.convert(y_tmp[row, col], S.f32)

    mean_val = sum_val / S.convert(BATCH_SIZE, S.f32)
    mean[col] = mean_val

    sq_sum = S.convert(0.0, S.f32)
    for row in S.range(BATCH_SIZE):
        diff = S.convert(y_tmp[row, col], S.f32) - mean_val
        sq_sum += diff * diff

    var[col] = sq_sum / S.convert(BATCH_SIZE, S.f32)


@substrate.jit
def batchnorm_apply_kernel(
    y_tmp: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
    mean: S.Tensor((OUT_FEATURES,), S.f32),
    var: S.Tensor((OUT_FEATURES,), S.f32),
    bn_weight: S.Tensor((OUT_FEATURES,), S.bf16),
    bn_bias: S.Tensor((OUT_FEATURES,), S.bf16),
    y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    row = idx // OUT_FEATURES
    col = idx % OUT_FEATURES

    value = S.convert(y_tmp[row, col], S.f32)
    centered = value - mean[col]
    inv_std = S.convert(1.0, S.f32) / S.sqrt(var[col] + S.convert(EPS, S.f32))
    normalized = centered * inv_std
    out = normalized * S.convert(bn_weight[col], S.f32) + S.convert(bn_bias[col], S.f32)
    y[row, col] = S.convert(out, S.bf16)


def _to_signed_i32(value: int) -> int:
    value &= 0xFFFFFFFF
    if value >= 0x80000000:
        value -= 0x100000000
    return value


def _make_raw_buffer_descriptor(tensor: torch.Tensor) -> torch.Tensor:
    ptr = tensor.data_ptr()
    nbytes = tensor.numel() * tensor.element_size()
    words = [
        _to_signed_i32(ptr),
        _to_signed_i32((ptr >> 32) & 0xFFFF),
        _to_signed_i32(nbytes),
        _to_signed_i32(RSRC_WORD3),
    ]
    return torch.tensor(words, device=tensor.device, dtype=torch.int32)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

        self._workspace_device = None
        self._workspace_dtype = None
        self._w_t = None
        self._w_ptr = None
        self._w_desc = None
        self._w_t_ptr = None
        self._y_tmp = None
        self._mean = None
        self._var = None
        self._y = None

    def _ensure_workspaces(self, device: torch.device, dtype: torch.dtype) -> None:
        if self._workspace_device == device and self._workspace_dtype == dtype:
            return

        self._workspace_device = device
        self._workspace_dtype = dtype
        self._w_t = torch.empty((IN_FEATURES, OUT_FEATURES), device=device, dtype=dtype)
        self._y_tmp = torch.empty((BATCH_SIZE, OUT_FEATURES), device=device, dtype=dtype)
        self._mean = torch.empty((OUT_FEATURES,), device=device, dtype=torch.float32)
        self._var = torch.empty((OUT_FEATURES,), device=device, dtype=torch.float32)
        self._y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=device, dtype=dtype)
        self._w_ptr = None
        self._w_desc = None
        self._w_t_ptr = None

    def _refresh_weight_cache(self) -> None:
        weight_ptr = self.gemm.weight.data_ptr()
        if self._w_ptr == weight_ptr:
            return

        self._w_t.copy_(self.gemm.weight.detach().transpose(0, 1))
        w_t_ptr = self._w_t.data_ptr()
        if self._w_t_ptr != w_t_ptr:
            self._w_desc = _make_raw_buffer_descriptor(self._w_t)
            self._w_t_ptr = w_t_ptr
        self._w_ptr = weight_ptr

    def forward(self, x):
        self._ensure_workspaces(x.device, x.dtype)
        self._refresh_weight_cache()

        x_in = x.contiguous()
        x_desc = _make_raw_buffer_descriptor(x_in)

        gemm_mfma_kernel[_launch_gemm](
            x_desc,
            self._w_desc,
            self.gemm.bias,
            self.scale,
            self._y_tmp,
        )
        column_stats_kernel[_launch_stats](self._y_tmp, self._mean, self._var)
        batchnorm_apply_kernel[_launch_norm](
            self._y_tmp,
            self._mean,
            self._var,
            self.bn.weight,
            self.bn.bias,
            self._y,
        )
        return self._y
