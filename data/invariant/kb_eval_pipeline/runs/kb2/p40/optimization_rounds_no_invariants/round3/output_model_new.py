import torch
import torch.nn as nn

import substrate
import substrate.language as S


BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WAVES_M = 2
WAVES_N = 2
THREADS = 256

BATCH_SIZE = 16384
IN_FEATURES = 4096
OUT_FEATURES = 4096
NUM_K_TILES = IN_FEATURES // BLOCK_K
SCALING_FACTOR = 0.5
FUSED_SCALE = 1.0 + SCALING_FACTOR


def _launch():
    return ((OUT_FEATURES // BLOCK_N, BATCH_SIZE // BLOCK_M, 1), (THREADS, 1, 1))


@substrate.jit
def fused_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((OUT_FEATURES, IN_FEATURES), S.bf16),
    BIAS0: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % 64
    wave = tid // 64
    wave_m = wave // WAVES_N
    wave_n = wave % WAVES_N

    block_m = S.block_id(1) * BLOCK_M
    block_n = S.block_id(0) * BLOCK_N

    x_flat = S.view(X, S.Tensor((BATCH_SIZE * IN_FEATURES,), S.bf16))
    w_flat = S.view(W, S.Tensor((OUT_FEATURES * IN_FEATURES,), S.bf16))
    x_rsrc = S.amdgpu.make_rsrc(x_flat, BATCH_SIZE * IN_FEATURES * 2)
    w_rsrc = S.amdgpu.make_rsrc(w_flat, OUT_FEATURES * IN_FEATURES * 2)

    a_shared = S.make_shared((2, BLOCK_M, 2, 4), S.u32)
    b_shared = S.make_shared((2, BLOCK_N, 2, 4), S.u32)

    c_lane = S.full((16,), 0.0, S.f32)

    if tid < 128:
        row = tid // 2
        k8 = tid % 2

        x_elem0 = (block_m + row) * IN_FEATURES + k8 * 8
        x_offset0 = S.convert(x_elem0 * 2, S.i32)
        packed0 = S.amdgpu.raw_buffer_load_x4(x_rsrc, 0, x_offset0, 0)
        halves0 = S.view(packed0, S.Tensor((2, 2), S.u32))
        a_shared[0, row, 0, 2 * k8] = halves0[0][0]
        a_shared[0, row, 0, 2 * k8 + 1] = halves0[0][1]
        a_shared[0, row, 1, 2 * k8] = halves0[1][0]
        a_shared[0, row, 1, 2 * k8 + 1] = halves0[1][1]

        x_elem1 = (block_m + row) * IN_FEATURES + BLOCK_K + k8 * 8
        x_offset1 = S.convert(x_elem1 * 2, S.i32)
        packed1 = S.amdgpu.raw_buffer_load_x4(x_rsrc, 0, x_offset1, 0)
        halves1 = S.view(packed1, S.Tensor((2, 2), S.u32))
        a_shared[1, row, 0, 2 * k8] = halves1[0][0]
        a_shared[1, row, 0, 2 * k8 + 1] = halves1[0][1]
        a_shared[1, row, 1, 2 * k8] = halves1[1][0]
        a_shared[1, row, 1, 2 * k8 + 1] = halves1[1][1]
    else:
        load_id = tid - 128
        col = load_id // 2
        k8 = load_id % 2

        w_elem0 = (block_n + col) * IN_FEATURES + k8 * 8
        w_offset0 = S.convert(w_elem0 * 2, S.i32)
        packed0 = S.amdgpu.raw_buffer_load_x4(w_rsrc, 0, w_offset0, 0)
        halves0 = S.view(packed0, S.Tensor((2, 2), S.u32))
        b_shared[0, col, 0, 2 * k8] = halves0[0][0]
        b_shared[0, col, 0, 2 * k8 + 1] = halves0[0][1]
        b_shared[0, col, 1, 2 * k8] = halves0[1][0]
        b_shared[0, col, 1, 2 * k8 + 1] = halves0[1][1]

        w_elem1 = (block_n + col) * IN_FEATURES + BLOCK_K + k8 * 8
        w_offset1 = S.convert(w_elem1 * 2, S.i32)
        packed1 = S.amdgpu.raw_buffer_load_x4(w_rsrc, 0, w_offset1, 0)
        halves1 = S.view(packed1, S.Tensor((2, 2), S.u32))
        b_shared[1, col, 0, 2 * k8] = halves1[0][0]
        b_shared[1, col, 0, 2 * k8 + 1] = halves1[0][1]
        b_shared[1, col, 1, 2 * k8] = halves1[1][0]
        b_shared[1, col, 1, 2 * k8 + 1] = halves1[1][1]

    S.syncthreads()

    for k_pair in S.range(NUM_K_TILES // 2):
        k_base = k_pair * (2 * BLOCK_K)

        a_u32_0 = a_shared[0, wave_m * 32 + lane // 2, lane % 2]
        b_u32_0 = b_shared[0, wave_n * 32 + lane // 2, lane % 2]
        a_frag_0 = S.view(a_u32_0, S.Tensor((2, 4, 1), S.bf16))
        b_frag_0 = S.view(b_u32_0, S.Tensor((2, 4, 1), S.bf16))
        S.syncthreads()

        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_0[0], b_frag_0[0], c_lane)

        if tid < 128:
            row = tid // 2
            k8 = tid % 2
            x_elem = (block_m + row) * IN_FEATURES + k_base + 2 * BLOCK_K + k8 * 8
            x_offset = S.convert(x_elem * 2, S.i32)
            packed = S.amdgpu.raw_buffer_load_x4(x_rsrc, 0, x_offset, 0)
            halves = S.view(packed, S.Tensor((2, 2), S.u32))
            a_shared[0, row, 0, 2 * k8] = halves[0][0]
            a_shared[0, row, 0, 2 * k8 + 1] = halves[0][1]
            a_shared[0, row, 1, 2 * k8] = halves[1][0]
            a_shared[0, row, 1, 2 * k8 + 1] = halves[1][1]
        else:
            load_id = tid - 128
            col = load_id // 2
            k8 = load_id % 2
            w_elem = (block_n + col) * IN_FEATURES + k_base + 2 * BLOCK_K + k8 * 8
            w_offset = S.convert(w_elem * 2, S.i32)
            packed = S.amdgpu.raw_buffer_load_x4(w_rsrc, 0, w_offset, 0)
            halves = S.view(packed, S.Tensor((2, 2), S.u32))
            b_shared[0, col, 0, 2 * k8] = halves[0][0]
            b_shared[0, col, 0, 2 * k8 + 1] = halves[0][1]
            b_shared[0, col, 1, 2 * k8] = halves[1][0]
            b_shared[0, col, 1, 2 * k8 + 1] = halves[1][1]

        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_0[1], b_frag_0[1], c_lane)

        a_u32_1 = a_shared[1, wave_m * 32 + lane // 2, lane % 2]
        b_u32_1 = b_shared[1, wave_n * 32 + lane // 2, lane % 2]
        a_frag_1 = S.view(a_u32_1, S.Tensor((2, 4, 1), S.bf16))
        b_frag_1 = S.view(b_u32_1, S.Tensor((2, 4, 1), S.bf16))
        S.syncthreads()

        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_1[0], b_frag_1[0], c_lane)

        if tid < 128:
            row = tid // 2
            k8 = tid % 2
            x_elem = (block_m + row) * IN_FEATURES + k_base + 3 * BLOCK_K + k8 * 8
            x_offset = S.convert(x_elem * 2, S.i32)
            packed = S.amdgpu.raw_buffer_load_x4(x_rsrc, 0, x_offset, 0)
            halves = S.view(packed, S.Tensor((2, 2), S.u32))
            a_shared[1, row, 0, 2 * k8] = halves[0][0]
            a_shared[1, row, 0, 2 * k8 + 1] = halves[0][1]
            a_shared[1, row, 1, 2 * k8] = halves[1][0]
            a_shared[1, row, 1, 2 * k8 + 1] = halves[1][1]
        else:
            load_id = tid - 128
            col = load_id // 2
            k8 = load_id % 2
            w_elem = (block_n + col) * IN_FEATURES + k_base + 3 * BLOCK_K + k8 * 8
            w_offset = S.convert(w_elem * 2, S.i32)
            packed = S.amdgpu.raw_buffer_load_x4(w_rsrc, 0, w_offset, 0)
            halves = S.view(packed, S.Tensor((2, 2), S.u32))
            b_shared[1, col, 0, 2 * k8] = halves[0][0]
            b_shared[1, col, 0, 2 * k8 + 1] = halves[0][1]
            b_shared[1, col, 1, 2 * k8] = halves[1][0]
            b_shared[1, col, 1, 2 * k8 + 1] = halves[1][1]

        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_1[1], b_frag_1[1], c_lane)
        S.syncthreads()

    out_row = block_m + wave_m * 32 + lane // 2
    out_col = block_n + wave_n * 32 + (lane % 2) * 16
    scale = S.convert(FUSED_SCALE, S.f32)

    for j in S.range(16):
        acc = c_lane[j] + S.convert(BIAS0[out_col + j], S.f32)
        Y[out_row, out_col + j] = S.convert(acc * scale, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self._cached_weight = None
        self._cached_bias = None
        self._cache_key = None

    def _get_kernel_tensors(self, x: torch.Tensor):
        weight = self.matmul.weight
        bias = self.matmul.bias
        key = (
            x.device,
            x.dtype,
            weight.data_ptr(),
            bias.data_ptr(),
            weight.stride(),
            bias.stride(),
        )
        if key != self._cache_key:
            self._cached_weight = weight.to(device=x.device, dtype=x.dtype).contiguous()
            self._cached_bias = bias.to(device=x.device, dtype=x.dtype).contiguous()
            self._cache_key = key
        return self._cached_weight, self._cached_bias

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES):
            raise ValueError(f"expected input shape {(BATCH_SIZE, IN_FEATURES)}, got {tuple(x.shape)}")
        if x.dtype != torch.bfloat16:
            raise TypeError(f"expected torch.bfloat16 input, got {x.dtype}")
        if self.scaling_factor != SCALING_FACTOR:
            raise ValueError(f"expected scaling_factor={SCALING_FACTOR}, got {self.scaling_factor}")

        weight, bias = self._get_kernel_tensors(x)
        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=x.dtype)
        fused_kernel[_launch](x.contiguous(), weight, bias, y)
        return y
