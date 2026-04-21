import torch
import torch.nn as nn

import substrate
import substrate.language as S

SQRT_2 = 1.4142135623730951

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
POOL_KERNEL_SIZE = 16
POOLED_SIZE = OUT_FEATURES // POOL_KERNEL_SIZE
SCALE_FACTOR = 2.0

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
THREADS_PER_BLOCK = 128
WAVE_SIZE = 64
PIPE_STAGES = 2
K_UNROLL = 2

X_RANGE_BYTES = BATCH_SIZE * IN_FEATURES * 2
W_RANGE_BYTES = IN_FEATURES * OUT_FEATURES * 2


def _launch_gemm():
    return ((OUT_FEATURES // BLOCK_N, BATCH_SIZE // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))


def _launch_reduce():
    return ((BATCH_SIZE, 1, 1), (WAVE_SIZE, 1, 1))


@substrate.jit
def gemm_mfma_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((IN_FEATURES, OUT_FEATURES), S.bf16),
    OUT: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    tid = S.thread_id(0)
    wave = tid // WAVE_SIZE
    lane = tid % WAVE_SIZE
    warp_row = wave // 2
    warp_col = wave % 2

    block_m = S.block_id(1)
    block_n = S.block_id(0)
    tile_m = block_m * BLOCK_M
    tile_n = block_n * BLOCK_N

    x_rsrc = S.amdgpu.make_rsrc(X, X_RANGE_BYTES)
    w_rsrc = S.amdgpu.make_rsrc(W, W_RANGE_BYTES)

    a_shared = S.make_shared((PIPE_STAGES, THREADS_PER_BLOCK, 4), S.u32)
    b_shared = S.make_shared((PIPE_STAGES, THREADS_PER_BLOCK, 8), S.bf16)
    acc = S.full((16,), 0.0, S.f32)

    a_row = tid // 2
    a_half = tid % 2
    a_group = a_row // 32
    a_row_in_group = a_row % 32
    a_lane_base = a_group * 64

    b_k = tid // 8
    b_seg = tid % 8
    b_warp = b_seg // 4
    b_local_col = (b_seg % 4) * 8
    b_half = b_k // 8
    b_lane_group = ((b_k % 8) // 4) * 32
    b_elem = b_k % 4
    b_slot = b_half * 4 + b_elem
    b_lane_base = b_warp * 64 + b_lane_group + b_local_col

    x_row = tile_m + a_row
    w_col = tile_n + b_seg * 8

    k0 = 0
    x_col0 = k0 + a_half * 8
    x_byte_offset0 = (x_row * IN_FEATURES + x_col0) * 2
    a_vec0 = S.amdgpu.raw_buffer_load_x4(x_rsrc, x_byte_offset0, 0, 0)
    if a_half == 0:
        a_shared[0, a_lane_base + a_row_in_group, 0] = a_vec0[0]
        a_shared[0, a_lane_base + a_row_in_group, 1] = a_vec0[1]
        a_shared[0, a_lane_base + 32 + a_row_in_group, 0] = a_vec0[2]
        a_shared[0, a_lane_base + 32 + a_row_in_group, 1] = a_vec0[3]
    else:
        a_shared[0, a_lane_base + a_row_in_group, 2] = a_vec0[0]
        a_shared[0, a_lane_base + a_row_in_group, 3] = a_vec0[1]
        a_shared[0, a_lane_base + 32 + a_row_in_group, 2] = a_vec0[2]
        a_shared[0, a_lane_base + 32 + a_row_in_group, 3] = a_vec0[3]

    w_row0 = k0 + b_k
    w_byte_offset0 = (w_row0 * OUT_FEATURES + w_col) * 2
    b_vec0 = S.amdgpu.raw_buffer_load_x4(w_rsrc, w_byte_offset0, 0, 0)
    b_vals0 = S.view(b_vec0, S.Tensor((8,), S.bf16))
    for c in S.range(8):
        b_shared[0, b_lane_base + c, b_slot] = b_vals0[c]

    S.syncthreads()

    for k_base in S.range(0, IN_FEATURES, K_UNROLL * BLOCK_K):
        k1 = k_base + BLOCK_K

        x_col1 = k1 + a_half * 8
        x_byte_offset1 = (x_row * IN_FEATURES + x_col1) * 2
        a_vec1 = S.amdgpu.raw_buffer_load_x4(x_rsrc, x_byte_offset1, 0, 0)
        if a_half == 0:
            a_shared[1, a_lane_base + a_row_in_group, 0] = a_vec1[0]
            a_shared[1, a_lane_base + a_row_in_group, 1] = a_vec1[1]
            a_shared[1, a_lane_base + 32 + a_row_in_group, 0] = a_vec1[2]
            a_shared[1, a_lane_base + 32 + a_row_in_group, 1] = a_vec1[3]
        else:
            a_shared[1, a_lane_base + a_row_in_group, 2] = a_vec1[0]
            a_shared[1, a_lane_base + a_row_in_group, 3] = a_vec1[1]
            a_shared[1, a_lane_base + 32 + a_row_in_group, 2] = a_vec1[2]
            a_shared[1, a_lane_base + 32 + a_row_in_group, 3] = a_vec1[3]

        w_row1 = k1 + b_k
        w_byte_offset1 = (w_row1 * OUT_FEATURES + w_col) * 2
        b_vec1 = S.amdgpu.raw_buffer_load_x4(w_rsrc, w_byte_offset1, 0, 0)
        b_vals1 = S.view(b_vec1, S.Tensor((8,), S.bf16))
        for c in S.range(8):
            b_shared[1, b_lane_base + c, b_slot] = b_vals1[c]

        S.syncthreads()

        a_frag0 = S.view(a_shared[0, warp_row * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(b_shared[0, warp_col * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

        S.syncthreads()

        k2 = k_base + K_UNROLL * BLOCK_K
        x_col2 = k2 + a_half * 8
        x_byte_offset2 = (x_row * IN_FEATURES + x_col2) * 2
        a_vec2 = S.amdgpu.raw_buffer_load_x4(x_rsrc, x_byte_offset2, 0, 0)
        if a_half == 0:
            a_shared[0, a_lane_base + a_row_in_group, 0] = a_vec2[0]
            a_shared[0, a_lane_base + a_row_in_group, 1] = a_vec2[1]
            a_shared[0, a_lane_base + 32 + a_row_in_group, 0] = a_vec2[2]
            a_shared[0, a_lane_base + 32 + a_row_in_group, 1] = a_vec2[3]
        else:
            a_shared[0, a_lane_base + a_row_in_group, 2] = a_vec2[0]
            a_shared[0, a_lane_base + a_row_in_group, 3] = a_vec2[1]
            a_shared[0, a_lane_base + 32 + a_row_in_group, 2] = a_vec2[2]
            a_shared[0, a_lane_base + 32 + a_row_in_group, 3] = a_vec2[3]

        w_row2 = k2 + b_k
        w_byte_offset2 = (w_row2 * OUT_FEATURES + w_col) * 2
        b_vec2 = S.amdgpu.raw_buffer_load_x4(w_rsrc, w_byte_offset2, 0, 0)
        b_vals2 = S.view(b_vec2, S.Tensor((8,), S.bf16))
        for c in S.range(8):
            b_shared[0, b_lane_base + c, b_slot] = b_vals2[c]

        a_frag1 = S.view(a_shared[1, warp_row * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag1 = S.view(b_shared[1, warp_col * 64 + lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

        S.syncthreads()

    tile_row_base = tile_m + warp_row * 32
    tile_col_base = tile_n + warp_col * 32
    out_col = tile_col_base + (lane % 32)
    lane_row_base = 4 * (lane // 32)
    for acc_idx in S.range(16):
        out_row = tile_row_base + 8 * (acc_idx // 4) + lane_row_base + (acc_idx % 4)
        OUT[out_row, out_col] = S.convert(acc[acc_idx], S.bf16)


@substrate.jit
def reduction_kernel(
    MAT: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
    BIAS0: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE,), S.bf16),
):
    row = S.block_id(0)
    lane = S.thread_id(0)

    local_max = S.convert(-1.0e30, S.f32)
    for p in S.range(lane, POOLED_SIZE, WAVE_SIZE):
        total = S.convert(0.0, S.f32)
        base_col = p * POOL_KERNEL_SIZE
        for t in S.range(POOL_KERNEL_SIZE):
            j = base_col + t
            total += S.convert(MAT[row, j], S.f32) + S.convert(BIAS0[j], S.f32)
        v = total / S.convert(POOL_KERNEL_SIZE, S.f32)
        v = S.convert(0.5, S.f32) * v * (S.convert(1.0, S.f32) + S.erf(v / S.convert(SQRT_2, S.f32)))
        v = v * S.convert(SCALE_FACTOR, S.f32)
        if v > local_max:
            local_max = v

    other = S.shuffle_xor(local_max, 32, WAVE_SIZE)
    if other > local_max:
        local_max = other
    other = S.shuffle_xor(local_max, 16, WAVE_SIZE)
    if other > local_max:
        local_max = other
    other = S.shuffle_xor(local_max, 8, WAVE_SIZE)
    if other > local_max:
        local_max = other
    other = S.shuffle_xor(local_max, 4, WAVE_SIZE)
    if other > local_max:
        local_max = other
    other = S.shuffle_xor(local_max, 2, WAVE_SIZE)
    if other > local_max:
        local_max = other
    other = S.shuffle_xor(local_max, 1, WAVE_SIZE)
    if other > local_max:
        local_max = other

    if lane == 0:
        Y[row] = S.convert(local_max, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        self._cache_key = None
        self._cached_weight_t = None
        self._cached_bias = None
        self._cached_mat = None
        self._cached_out = None

    def _refresh_caches(self, device, dtype):
        weight = self.matmul.weight
        bias = self.matmul.bias
        key = (
            weight.data_ptr(),
            int(weight._version),
            bias.data_ptr(),
            int(bias._version),
            device.type,
            device.index,
            dtype,
        )
        if key != self._cache_key:
            self._cached_weight_t = weight.detach().transpose(0, 1).to(device=device, dtype=dtype).contiguous()
            self._cached_bias = bias.detach().to(device=device, dtype=dtype).contiguous()
            self._cache_key = key

        if self._cached_mat is None or self._cached_mat.device != device:
            self._cached_mat = torch.empty((BATCH_SIZE, OUT_FEATURES), device=device, dtype=dtype)
        if self._cached_out is None or self._cached_out.device != device or self._cached_out.dtype != dtype:
            self._cached_out = torch.empty((BATCH_SIZE,), device=device, dtype=dtype)

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES):
            raise ValueError(f"expected input shape {(BATCH_SIZE, IN_FEATURES)}, got {tuple(x.shape)}")
        if x.dtype != torch.bfloat16:
            raise ValueError(f"expected torch.bfloat16 input, got {x.dtype}")
        if self.pool_kernel_size != POOL_KERNEL_SIZE:
            raise ValueError(f"expected pool kernel size {POOL_KERNEL_SIZE}, got {self.pool_kernel_size}")
        if self.scale_factor != SCALE_FACTOR:
            raise ValueError(f"expected scale factor {SCALE_FACTOR}, got {self.scale_factor}")

        x_in = x.contiguous()
        self._refresh_caches(x_in.device, x_in.dtype)
        gemm_mfma_kernel[_launch_gemm](x_in, self._cached_weight_t, self._cached_mat)
        reduction_kernel[_launch_reduce](self._cached_mat, self._cached_bias, self._cached_out)
        return self._cached_out
