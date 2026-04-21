import torch
import torch.nn as nn
import torch.nn.functional as F

import substrate
import substrate.language as S


BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
NUM_GROUPS = 256
GROUP_SIZE = OUT_FEATURES // NUM_GROUPS
EPS = 1.0e-5

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WAVE_SIZE = 64
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = WAVE_SIZE * WAVES_PER_BLOCK
GRID_N = OUT_FEATURES // BLOCK_N
NUMEL = BATCH_SIZE * OUT_FEATURES
ACT_NUM_BYTES = NUMEL * 2
MUL_NUM_BYTES = OUT_FEATURES * 2
POINTWISE_VEC_ELEMS = 8

X_NUM_BYTES = BATCH_SIZE * IN_FEATURES * 2
W_NUM_BYTES = IN_FEATURES * OUT_FEATURES * 2


def _gemm_launch():
    return ((BATCH_SIZE // BLOCK_M * OUT_FEATURES // BLOCK_N, 1, 1), (THREADS_PER_BLOCK, 1, 1))


def _epilogue_launch():
    return ((BATCH_SIZE * NUM_GROUPS, 1, 1), (GROUP_SIZE, 1, 1))


def _pointwise_launch():
    threads = 256
    elems_per_block = threads * POINTWISE_VEC_ELEMS
    blocks = (NUMEL + elems_per_block - 1) // elems_per_block
    return ((blocks, 1, 1), (threads, 1, 1))


@substrate.jit
def gemm_bias_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((IN_FEATURES, OUT_FEATURES), S.bf16),
    BIAS0: S.Tensor((OUT_FEATURES,), S.bf16),
    OUT: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.f32),
):
    tid = S.thread_id(0)
    wave = tid // WAVE_SIZE
    lane = tid % WAVE_SIZE
    wave_m = wave // 2
    wave_n = wave % 2

    block_id = S.block_id(0)
    tile_m = block_id // GRID_N
    tile_n = block_id % GRID_N

    x_rsrc = S.amdgpu.make_rsrc(X, X_NUM_BYTES)
    w_rsrc = S.amdgpu.make_rsrc(W, W_NUM_BYTES)

    packed_a = S.make_shared((2, 2, WAVE_SIZE, 8), S.bf16)
    packed_a_u32 = S.view(packed_a, S.Tensor((2, 2, WAVE_SIZE, 4), S.u32))

    packed_b = S.make_shared((2, 2, WAVE_SIZE, 8), S.bf16)
    packed_b_u32 = S.view(packed_b, S.Tensor((2, 2, WAVE_SIZE, 4), S.u32))

    acc = S.full((16,), 0.0, S.f32)

    if tid < 128:
        row = tid // 2
        chunk = tid % 2
        x_offset = ((tile_m * BLOCK_M + row) * IN_FEATURES + chunk * 8) * 2
        a_vec = S.amdgpu.raw_buffer_load_x4(x_rsrc, x_offset, 0, 0)
        a_frag = S.view(a_vec, S.Tensor((2, 4, 1), S.bf16))

        packed_row = row % 32
        packed_group = row // 32
        dst_lane0 = packed_row
        dst_lane1 = packed_row + 32
        dst_col = chunk * 4
        for i in S.range(4):
            packed_a[0, packed_group, dst_lane0, dst_col + i] = a_frag[0, i, 0]
            packed_a[0, packed_group, dst_lane1, dst_col + i] = a_frag[1, i, 0]
    else:
        b_idx = tid - 128
        b_row_idx = b_idx // (BLOCK_N // 8)
        b_col_chunk = b_idx % (BLOCK_N // 8)
        w_offset = (b_row_idx * OUT_FEATURES + tile_n * BLOCK_N + b_col_chunk * 8) * 2
        b_vec = S.amdgpu.raw_buffer_load_x4(w_rsrc, w_offset, 0, 0)
        b_frag = S.view(b_vec, S.Tensor((2, 4, 1), S.bf16))

        lane_row_offset = (b_row_idx % 8 // 4) * 32
        dst_slot = (b_row_idx % 4) + (4 if b_row_idx >= 8 else 0)
        col_base = b_col_chunk * 8
        for part in S.range(2):
            dst_group = (col_base + part * 4) // 32
            for i in S.range(4):
                dst_lane = ((col_base + part * 4 + i) % 32) + lane_row_offset
                packed_b[0, dst_group, dst_lane, dst_slot] = b_frag[part, i, 0]

    S.syncthreads()

    for k0 in S.range(0, IN_FEATURES, 2 * BLOCK_K):
        if k0 + BLOCK_K < IN_FEATURES:
            if tid < 128:
                row = tid // 2
                chunk = tid % 2
                x_offset = ((tile_m * BLOCK_M + row) * IN_FEATURES + k0 + BLOCK_K + chunk * 8) * 2
                a_vec = S.amdgpu.raw_buffer_load_x4(x_rsrc, x_offset, 0, 0)
                a_frag = S.view(a_vec, S.Tensor((2, 4, 1), S.bf16))

                packed_row = row % 32
                packed_group = row // 32
                dst_lane0 = packed_row
                dst_lane1 = packed_row + 32
                dst_col = chunk * 4
                for i in S.range(4):
                    packed_a[1, packed_group, dst_lane0, dst_col + i] = a_frag[0, i, 0]
                    packed_a[1, packed_group, dst_lane1, dst_col + i] = a_frag[1, i, 0]
            else:
                b_idx = tid - 128
                b_row_idx = b_idx // (BLOCK_N // 8)
                b_col_chunk = b_idx % (BLOCK_N // 8)
                w_offset = ((k0 + BLOCK_K + b_row_idx) * OUT_FEATURES + tile_n * BLOCK_N + b_col_chunk * 8) * 2
                b_vec = S.amdgpu.raw_buffer_load_x4(w_rsrc, w_offset, 0, 0)
                b_frag = S.view(b_vec, S.Tensor((2, 4, 1), S.bf16))

                lane_row_offset = (b_row_idx % 8 // 4) * 32
                dst_slot = (b_row_idx % 4) + (4 if b_row_idx >= 8 else 0)
                col_base = b_col_chunk * 8
                for part in S.range(2):
                    dst_group = (col_base + part * 4) // 32
                    for i in S.range(4):
                        dst_lane = ((col_base + part * 4 + i) % 32) + lane_row_offset
                        packed_b[1, dst_group, dst_lane, dst_slot] = b_frag[part, i, 0]

        a_u32 = packed_a_u32[0, wave_m, lane]
        b_u32 = packed_b_u32[0, wave_n, lane]
        a_frag = S.view(a_u32, S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_u32, S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        if k0 + BLOCK_K < IN_FEATURES:
            S.syncthreads()

            if k0 + 2 * BLOCK_K < IN_FEATURES:
                if tid < 128:
                    row = tid // 2
                    chunk = tid % 2
                    x_offset = ((tile_m * BLOCK_M + row) * IN_FEATURES + k0 + 2 * BLOCK_K + chunk * 8) * 2
                    a_vec = S.amdgpu.raw_buffer_load_x4(x_rsrc, x_offset, 0, 0)
                    a_frag = S.view(a_vec, S.Tensor((2, 4, 1), S.bf16))

                    packed_row = row % 32
                    packed_group = row // 32
                    dst_lane0 = packed_row
                    dst_lane1 = packed_row + 32
                    dst_col = chunk * 4
                    for i in S.range(4):
                        packed_a[0, packed_group, dst_lane0, dst_col + i] = a_frag[0, i, 0]
                        packed_a[0, packed_group, dst_lane1, dst_col + i] = a_frag[1, i, 0]
                else:
                    b_idx = tid - 128
                    b_row_idx = b_idx // (BLOCK_N // 8)
                    b_col_chunk = b_idx % (BLOCK_N // 8)
                    w_offset = ((k0 + 2 * BLOCK_K + b_row_idx) * OUT_FEATURES + tile_n * BLOCK_N + b_col_chunk * 8) * 2
                    b_vec = S.amdgpu.raw_buffer_load_x4(w_rsrc, w_offset, 0, 0)
                    b_frag = S.view(b_vec, S.Tensor((2, 4, 1), S.bf16))

                    lane_row_offset = (b_row_idx % 8 // 4) * 32
                    dst_slot = (b_row_idx % 4) + (4 if b_row_idx >= 8 else 0)
                    col_base = b_col_chunk * 8
                    for part in S.range(2):
                        dst_group = (col_base + part * 4) // 32
                        for i in S.range(4):
                            dst_lane = ((col_base + part * 4 + i) % 32) + lane_row_offset
                            packed_b[0, dst_group, dst_lane, dst_slot] = b_frag[part, i, 0]

            a_u32 = packed_a_u32[1, wave_m, lane]
            b_u32 = packed_b_u32[1, wave_n, lane]
            a_frag = S.view(a_u32, S.Tensor((2, 4, 1), S.bf16))
            b_frag = S.view(b_u32, S.Tensor((2, 4, 1), S.bf16))

            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

            if k0 + 2 * BLOCK_K < IN_FEATURES:
                S.syncthreads()

    col = tile_n * BLOCK_N + wave_n * 32 + (lane % 32)
    row_base = tile_m * BLOCK_M + wave_m * 32 + (lane // 32) * 4
    bias = S.convert(BIAS0[col], S.f32)
    for i in S.range(16):
        row = row_base + (i % 4) + 8 * (i // 4)
        OUT[row, col] = acc[i] + bias


@substrate.jit
def pointwise_mul_kernel(
    A: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
    B: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    idx = (S.block_id(0) * S.block_dim(0) + S.thread_id(0)) * POINTWISE_VEC_ELEMS
    byte_offset = idx * 2

    a_rsrc = S.amdgpu.make_rsrc(A, ACT_NUM_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, ACT_NUM_BYTES)
    y_rsrc = S.amdgpu.make_rsrc(Y, ACT_NUM_BYTES)

    a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, byte_offset, 0, 0)
    b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, byte_offset, 0, 0)
    a_vals = S.view(a_vec, S.Tensor((POINTWISE_VEC_ELEMS,), S.bf16))
    b_vals = S.view(b_vec, S.Tensor((POINTWISE_VEC_ELEMS,), S.bf16))
    y_vals = S.full((POINTWISE_VEC_ELEMS,), 0.0, S.bf16)

    for i in S.range(POINTWISE_VEC_ELEMS):
        y_vals[i] = S.convert(
            S.convert(a_vals[i], S.f32) * S.convert(b_vals[i], S.f32),
            S.bf16,
        )

    y_vec = S.view(y_vals, S.Tensor((4,), S.u32))
    S.amdgpu.raw_buffer_store_x4(y_vec, y_rsrc, byte_offset, 0, 0)


@substrate.jit
def weight_mul_kernel(
    A: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
    W: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    idx = (S.block_id(0) * S.block_dim(0) + S.thread_id(0)) * POINTWISE_VEC_ELEMS
    byte_offset = idx * 2
    weight_offset = (idx % OUT_FEATURES) * 2

    a_rsrc = S.amdgpu.make_rsrc(A, ACT_NUM_BYTES)
    w_rsrc = S.amdgpu.make_rsrc(W, MUL_NUM_BYTES)
    y_rsrc = S.amdgpu.make_rsrc(Y, ACT_NUM_BYTES)

    a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, byte_offset, 0, 0)
    w_vec = S.amdgpu.raw_buffer_load_x4(w_rsrc, weight_offset, 0, 0)
    a_vals = S.view(a_vec, S.Tensor((POINTWISE_VEC_ELEMS,), S.bf16))
    w_vals = S.view(w_vec, S.Tensor((POINTWISE_VEC_ELEMS,), S.bf16))
    y_vals = S.full((POINTWISE_VEC_ELEMS,), 0.0, S.bf16)

    for i in S.range(POINTWISE_VEC_ELEMS):
        y_vals[i] = S.convert(
            S.convert(a_vals[i], S.f32) * S.convert(w_vals[i], S.f32),
            S.bf16,
        )

    y_vec = S.view(y_vals, S.Tensor((4,), S.u32))
    S.amdgpu.raw_buffer_store_x4(y_vec, y_rsrc, byte_offset, 0, 0)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

        self._cached_weight_t = None
        self._cached_bias = None
        self._cached_gn_weight = None
        self._cached_gn_bias = None
        self._cached_mul = None
        self._cache_key = None

    def _refresh_kernel_tensors(self, device):
        key = (
            self.gemm.weight.data_ptr(),
            self.gemm.bias.data_ptr(),
            self.group_norm.weight.data_ptr(),
            self.group_norm.bias.data_ptr(),
            self.multiply_weight.data_ptr(),
            device.type,
            device.index,
        )
        if self._cache_key == key:
            return

        dtype = torch.bfloat16
        self._cached_weight_t = self.gemm.weight.detach().t().to(device=device, dtype=dtype).contiguous()
        self._cached_bias = self.gemm.bias.detach().to(device=device, dtype=dtype).contiguous()
        self._cached_gn_weight = self.group_norm.weight.detach().to(device=device, dtype=dtype).contiguous()
        self._cached_gn_bias = self.group_norm.bias.detach().to(device=device, dtype=dtype).contiguous()
        self._cached_mul = self.multiply_weight.detach().to(device=device, dtype=dtype).contiguous()
        self._cache_key = key

    def forward(self, x):
        if (
            tuple(x.shape) != (BATCH_SIZE, IN_FEATURES)
            or x.dtype != torch.bfloat16
            or self.group_norm.num_groups != NUM_GROUPS
            or self.group_norm.eps != EPS
            or tuple(self.multiply_weight.shape) != (OUT_FEATURES,)
        ):
            raise RuntimeError("ModelNew only supports the benchmark configuration.")

        x = x.contiguous()
        self._refresh_kernel_tensors(x.device)

        gemm_out_fp32 = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=torch.float32)
        tmp0 = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=torch.bfloat16)
        tmp1 = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=torch.bfloat16)
        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=torch.bfloat16)

        gemm_bias_kernel[_gemm_launch](
            x,
            self._cached_weight_t,
            self._cached_bias,
            gemm_out_fp32,
            num_warps=4,
        )
        gemm_out = gemm_out_fp32.to(torch.bfloat16)

        norm = F.group_norm(
            gemm_out,
            NUM_GROUPS,
            self._cached_gn_weight,
            self._cached_gn_bias,
            EPS,
        )
        sig0 = torch.sigmoid(norm)
        pointwise_mul_kernel[_pointwise_launch](norm, sig0, tmp0)
        weight_mul_kernel[_pointwise_launch](tmp0, self._cached_mul, tmp1)
        sig1 = torch.sigmoid(tmp1)
        pointwise_mul_kernel[_pointwise_launch](tmp1, sig1, y)
        return y
