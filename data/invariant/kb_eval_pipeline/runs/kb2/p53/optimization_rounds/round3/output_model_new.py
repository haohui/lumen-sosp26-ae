import torch
import torch.nn as nn

import substrate
import substrate.language as S

SQRT_2 = 1.4142135623730951

BATCH_SIZE = 2048
IN_FEATURES = 8192
OUT_FEATURES = 8192
SCALING_FACTOR = 0.5
HARDTANH_MIN = -2.0
HARDTANH_MAX = 2.0

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WAVES_PER_BLOCK = 4
WAVE_SIZE = 64
K_TILES = IN_FEATURES // BLOCK_K

X_RANGE_BYTES = BATCH_SIZE * IN_FEATURES * 2
W_PACKED_RANGE_BYTES = IN_FEATURES * OUT_FEATURES * 2


def _launch():
    return ((OUT_FEATURES // BLOCK_N, BATCH_SIZE // BLOCK_M, 1), (WAVES_PER_BLOCK * WAVE_SIZE, 1, 1))


def _pack_weight(weight: torch.Tensor) -> torch.Tensor:
    w_t = weight.t().contiguous().view(IN_FEATURES // BLOCK_K, BLOCK_K, OUT_FEATURES // 32, 32)
    w_t = w_t.permute(2, 0, 3, 1).contiguous()
    packed = torch.empty((OUT_FEATURES // 32, IN_FEATURES // BLOCK_K, WAVE_SIZE, 8), device=weight.device, dtype=weight.dtype)
    packed[:, :, 0:32, 0:4] = w_t[:, :, :, 0:4]
    packed[:, :, 0:32, 4:8] = w_t[:, :, :, 8:12]
    packed[:, :, 32:64, 0:4] = w_t[:, :, :, 4:8]
    packed[:, :, 32:64, 4:8] = w_t[:, :, :, 12:16]
    return packed.contiguous()


def _pack_input(x: torch.Tensor) -> torch.Tensor:
    x_t = x.view(BATCH_SIZE // 32, 32, IN_FEATURES // BLOCK_K, BLOCK_K)
    x_t = x_t.permute(0, 2, 1, 3).contiguous()
    packed = torch.empty((BATCH_SIZE // 32, IN_FEATURES // BLOCK_K, WAVE_SIZE, 8), device=x.device, dtype=x.dtype)
    packed[:, :, 0:32, 0:4] = x_t[:, :, :, 0:4]
    packed[:, :, 0:32, 4:8] = x_t[:, :, :, 8:12]
    packed[:, :, 32:64, 0:4] = x_t[:, :, :, 4:8]
    packed[:, :, 32:64, 4:8] = x_t[:, :, :, 12:16]
    return packed.contiguous()


@substrate.jit
def fused_kernel(
    X_PACKED: S.Tensor((BATCH_SIZE // 32, IN_FEATURES // BLOCK_K, WAVE_SIZE, 8), S.bf16),
    W_PACKED: S.Tensor((OUT_FEATURES // 32, IN_FEATURES // BLOCK_K, WAVE_SIZE, 8), S.bf16),
    BIAS0: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    warp = tid // WAVE_SIZE
    warp_row = warp // 2
    warp_col = warp % 2

    block_row = S.block_id(1)
    block_col = S.block_id(0)

    tile_row_base = block_row * BLOCK_M + warp_row * 32
    tile_col_base = block_col * BLOCK_N + warp_col * 32

    x_rsrc = S.amdgpu.make_rsrc(X_PACKED, X_RANGE_BYTES)
    w_rsrc = S.amdgpu.make_rsrc(W_PACKED, W_PACKED_RANGE_BYTES)

    a_shared = S.make_shared((2, WAVES_PER_BLOCK, WAVE_SIZE, 4), S.u32)
    b_shared = S.make_shared((2, WAVES_PER_BLOCK, WAVE_SIZE, 4), S.u32)

    acc = S.full((16,), 0.0, S.f32)

    a_tile = tile_row_base // 32
    b_tile = tile_col_base // 32

    a_src_elem = ((a_tile * K_TILES + 0) * WAVE_SIZE + lane) * 8
    a_src_byte = S.convert(a_src_elem * 2, S.i32)
    a_words = S.amdgpu.raw_buffer_load_x4(x_rsrc, a_src_byte, 0, 0)
    a_shared[0, warp, lane, 0] = a_words[0]
    a_shared[0, warp, lane, 1] = a_words[1]
    a_shared[0, warp, lane, 2] = a_words[2]
    a_shared[0, warp, lane, 3] = a_words[3]

    b_src_elem = ((b_tile * K_TILES + 0) * WAVE_SIZE + lane) * 8
    b_src_byte = S.convert(b_src_elem * 2, S.i32)
    b_words = S.amdgpu.raw_buffer_load_x4(w_rsrc, b_src_byte, 0, 0)
    b_shared[0, warp, lane, 0] = b_words[0]
    b_shared[0, warp, lane, 1] = b_words[1]
    b_shared[0, warp, lane, 2] = b_words[2]
    b_shared[0, warp, lane, 3] = b_words[3]

    a_src_elem = ((a_tile * K_TILES + 1) * WAVE_SIZE + lane) * 8
    a_src_byte = S.convert(a_src_elem * 2, S.i32)
    a_words = S.amdgpu.raw_buffer_load_x4(x_rsrc, a_src_byte, 0, 0)
    a_shared[1, warp, lane, 0] = a_words[0]
    a_shared[1, warp, lane, 1] = a_words[1]
    a_shared[1, warp, lane, 2] = a_words[2]
    a_shared[1, warp, lane, 3] = a_words[3]

    b_src_elem = ((b_tile * K_TILES + 1) * WAVE_SIZE + lane) * 8
    b_src_byte = S.convert(b_src_elem * 2, S.i32)
    b_words = S.amdgpu.raw_buffer_load_x4(w_rsrc, b_src_byte, 0, 0)
    b_shared[1, warp, lane, 0] = b_words[0]
    b_shared[1, warp, lane, 1] = b_words[1]
    b_shared[1, warp, lane, 2] = b_words[2]
    b_shared[1, warp, lane, 3] = b_words[3]

    S.syncthreads()

    for k_tile in S.range(0, K_TILES, 2):
        a_words_0 = a_shared[0, warp, lane]
        b_words_0 = b_shared[0, warp, lane]
        a_frag_0 = S.view(a_words_0, S.Tensor((2, 4, 1), S.bf16))
        b_frag_0 = S.view(b_words_0, S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_0[0], b_frag_0[0], acc)

        a_src_elem = ((a_tile * K_TILES + (k_tile + 2)) * WAVE_SIZE + lane) * 8
        a_src_byte = S.convert(a_src_elem * 2, S.i32)
        a_words = S.amdgpu.raw_buffer_load_x4(x_rsrc, a_src_byte, 0, 0)
        a_shared[0, warp, lane, 0] = a_words[0]
        a_shared[0, warp, lane, 1] = a_words[1]
        a_shared[0, warp, lane, 2] = a_words[2]
        a_shared[0, warp, lane, 3] = a_words[3]

        b_src_elem = ((b_tile * K_TILES + (k_tile + 2)) * WAVE_SIZE + lane) * 8
        b_src_byte = S.convert(b_src_elem * 2, S.i32)
        b_words = S.amdgpu.raw_buffer_load_x4(w_rsrc, b_src_byte, 0, 0)
        b_shared[0, warp, lane, 0] = b_words[0]
        b_shared[0, warp, lane, 1] = b_words[1]
        b_shared[0, warp, lane, 2] = b_words[2]
        b_shared[0, warp, lane, 3] = b_words[3]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_0[1], b_frag_0[1], acc)

        a_words_1 = a_shared[1, warp, lane]
        b_words_1 = b_shared[1, warp, lane]
        a_frag_1 = S.view(a_words_1, S.Tensor((2, 4, 1), S.bf16))
        b_frag_1 = S.view(b_words_1, S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_1[0], b_frag_1[0], acc)

        a_src_elem = ((a_tile * K_TILES + (k_tile + 3)) * WAVE_SIZE + lane) * 8
        a_src_byte = S.convert(a_src_elem * 2, S.i32)
        a_words = S.amdgpu.raw_buffer_load_x4(x_rsrc, a_src_byte, 0, 0)
        a_shared[1, warp, lane, 0] = a_words[0]
        a_shared[1, warp, lane, 1] = a_words[1]
        a_shared[1, warp, lane, 2] = a_words[2]
        a_shared[1, warp, lane, 3] = a_words[3]

        b_src_elem = ((b_tile * K_TILES + (k_tile + 3)) * WAVE_SIZE + lane) * 8
        b_src_byte = S.convert(b_src_elem * 2, S.i32)
        b_words = S.amdgpu.raw_buffer_load_x4(w_rsrc, b_src_byte, 0, 0)
        b_shared[1, warp, lane, 0] = b_words[0]
        b_shared[1, warp, lane, 1] = b_words[1]
        b_shared[1, warp, lane, 2] = b_words[2]
        b_shared[1, warp, lane, 3] = b_words[3]

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_1[1], b_frag_1[1], acc)

        S.syncthreads()

    col = tile_col_base + (lane % 32)
    bias = S.convert(BIAS0[col], S.f32)

    for acc_idx in S.range(16):
        row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        x = S.convert(S.convert(acc[acc_idx] + bias, S.bf16), S.f32)
        x = S.convert(S.convert(x * S.convert(SCALING_FACTOR, S.f32), S.bf16), S.f32)
        if x < S.convert(HARDTANH_MIN, S.f32):
            x = S.convert(HARDTANH_MIN, S.f32)
        if x > S.convert(HARDTANH_MAX, S.f32):
            x = S.convert(HARDTANH_MAX, S.f32)
        x = S.convert(S.convert(x, S.bf16), S.f32)
        x = S.convert(0.5, S.f32) * x * (S.convert(1.0, S.f32) + S.erf(x / S.convert(SQRT_2, S.f32)))
        Y[row, col] = S.convert(x, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self._packed_weight = None
        self._packed_weight_ptr = None
        self._packed_weight_device = None
        self._packed_weight_dtype = None
        self._bias_cache = None
        self._bias_ptr = None
        self._bias_device = None
        self._bias_dtype = None

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES):
            raise ValueError(f"expected input shape {(BATCH_SIZE, IN_FEATURES)}, got {tuple(x.shape)}")
        if x.dtype != torch.bfloat16:
            raise TypeError(f"expected torch.bfloat16 input, got {x.dtype}")
        if self.scaling_factor != SCALING_FACTOR:
            raise ValueError(f"expected scaling_factor={SCALING_FACTOR}, got {self.scaling_factor}")
        if self.hardtanh_min != HARDTANH_MIN or self.hardtanh_max != HARDTANH_MAX:
            raise ValueError(
                f"expected hardtanh range ({HARDTANH_MIN}, {HARDTANH_MAX}), "
                f"got ({self.hardtanh_min}, {self.hardtanh_max})"
            )

        x_in = _pack_input(x.contiguous())

        weight = self.gemm.weight
        if (
            self._packed_weight is None
            or self._packed_weight_ptr != weight.data_ptr()
            or self._packed_weight_device != x.device
            or self._packed_weight_dtype != torch.bfloat16
        ):
            weight_bf16 = weight.to(device=x.device, dtype=torch.bfloat16).contiguous()
            self._packed_weight = _pack_weight(weight_bf16)
            self._packed_weight_ptr = weight.data_ptr()
            self._packed_weight_device = x.device
            self._packed_weight_dtype = torch.bfloat16

        bias_param = self.gemm.bias
        if (
            self._bias_cache is None
            or self._bias_ptr != bias_param.data_ptr()
            or self._bias_device != x.device
            or self._bias_dtype != torch.bfloat16
        ):
            self._bias_cache = bias_param.to(device=x.device, dtype=torch.bfloat16).contiguous()
            self._bias_ptr = bias_param.data_ptr()
            self._bias_device = x.device
            self._bias_dtype = torch.bfloat16

        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=torch.bfloat16)
        fused_kernel[_launch](x_in, self._packed_weight, self._bias_cache, y)
        return y
