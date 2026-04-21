import torch
import torch.nn as nn

import substrate
import substrate.language as S


BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
NUM_GROUPS = 512
GROUP_SIZE = OUT_FEATURES // NUM_GROUPS
EPS = 1.0e-5

BF16_BYTES = 2
X_NUM_BYTES = BATCH_SIZE * IN_FEATURES * BF16_BYTES
W_NUM_BYTES = OUT_FEATURES * IN_FEATURES * BF16_BYTES
ROW_BYTES = IN_FEATURES * BF16_BYTES

GEMM_THREADS = 256
GEMM_TILE_M = 64
GEMM_TILE_N = 64
WAVE_TILE_M = 32
WAVE_TILE_N = 32
GEMM_TILE_K = 16
GEMM_PIPE_PAIRS = IN_FEATURES // 32
POST_THREADS = 1
MFMA_TOUCH_THREADS = 256


def _gemm_launch():
    grid_x = (OUT_FEATURES + GEMM_TILE_N - 1) // GEMM_TILE_N
    grid_y = (BATCH_SIZE + GEMM_TILE_M - 1) // GEMM_TILE_M
    return ((grid_x, grid_y, 1), (GEMM_THREADS, 1, 1))


def _post_launch():
    return ((BATCH_SIZE, 1, 1), (POST_THREADS, 1, 1))


def _mfma_touch_launch():
    return ((1, 1, 1), (MFMA_TOUCH_THREADS, 1, 1))


@substrate.jit
def mfma_touch_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((OUT_FEATURES, IN_FEATURES), S.bf16),
    TOUCH: S.Tensor((MFMA_TOUCH_THREADS,), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % 64
    warp = tid // 64
    warp_m = warp // 2
    warp_n = warp % 2

    x_desc = S.amdgpu.make_rsrc(X, ROW_BYTES)
    w_desc = S.amdgpu.make_rsrc(W, ROW_BYTES)

    a_shm = S.make_shared((MFMA_TOUCH_THREADS, 4), S.u32)
    b_shm = S.make_shared((MFMA_TOUCH_THREADS, 4), S.u32)

    a_row = warp_m * 32 + (lane % 32)
    b_row = warp_n * 32 + (lane % 32)
    a_row_offset = S.convert((a_row * IN_FEATURES) * BF16_BYTES, S.i32)
    b_row_offset = S.convert((b_row * IN_FEATURES) * BF16_BYTES, S.i32)

    a_vec = S.amdgpu.raw_buffer_load_x4(x_desc, 0, a_row_offset, 0)
    b_vec = S.amdgpu.raw_buffer_load_x4(w_desc, 0, b_row_offset, 0)

    a_shm[tid] = a_vec
    b_shm[tid] = b_vec
    S.syncthreads()

    a_frag = S.view(a_shm[tid], S.Tensor((2, 4, 1), S.bf16))
    b_frag = S.view(b_shm[tid], S.Tensor((2, 4, 1), S.bf16))

    acc = S.full((16,), 0.0, S.f32)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

    TOUCH[tid] = S.convert(acc[0], S.bf16)


@substrate.jit
def gemm_vectorized_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((OUT_FEATURES, IN_FEATURES), S.bf16),
    BIAS0: S.Tensor((OUT_FEATURES,), S.bf16),
    Y0: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % 64
    warp = tid // 64
    warp_m = warp // 2
    warp_n = warp % 2

    block_row = S.block_id(1) * GEMM_TILE_M
    block_col = S.block_id(0) * GEMM_TILE_N
    wave_row = block_row + warp_m * WAVE_TILE_M
    wave_col = block_col + warp_n * WAVE_TILE_N

    x_desc = S.amdgpu.make_rsrc(X, ROW_BYTES)
    w_desc = S.amdgpu.make_rsrc(W, ROW_BYTES)

    a_shm = S.make_shared((2, GEMM_THREADS, 4), S.u32)
    b_shm = S.make_shared((2, GEMM_THREADS, 4), S.u32)

    c_row = wave_row + (lane % 32)
    c_col_base = wave_col + (lane // 32) * 16

    a_row_offset = S.convert((c_row * IN_FEATURES) * BF16_BYTES, S.i32)
    b_row = wave_col + (lane % 32)
    b_row_offset = S.convert((b_row * IN_FEATURES) * BF16_BYTES, S.i32)
    a_shm[0, tid] = S.amdgpu.raw_buffer_load_x4(x_desc, 0, a_row_offset, 0)
    b_shm[0, tid] = S.amdgpu.raw_buffer_load_x4(w_desc, 0, b_row_offset, 0)

    acc = S.full((16,), 0.0, S.f32)

    for k_pair in S.range(GEMM_PIPE_PAIRS - 1):
        k_second = k_pair * 32 + GEMM_TILE_K
        k_second_offset = S.convert(k_second * BF16_BYTES, S.i32)
        a_shm[1, tid] = S.amdgpu.raw_buffer_load_x4(x_desc, k_second_offset, a_row_offset, 0)
        b_shm[1, tid] = S.amdgpu.raw_buffer_load_x4(w_desc, k_second_offset, b_row_offset, 0)

        a_frag0 = S.view(a_shm[0, tid], S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(b_shm[0, tid], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

        k_next = k_pair * 32 + 32
        k_next_offset = S.convert(k_next * BF16_BYTES, S.i32)
        a_shm[0, tid] = S.amdgpu.raw_buffer_load_x4(x_desc, k_next_offset, a_row_offset, 0)
        b_shm[0, tid] = S.amdgpu.raw_buffer_load_x4(w_desc, k_next_offset, b_row_offset, 0)

        a_frag1 = S.view(a_shm[1, tid], S.Tensor((2, 4, 1), S.bf16))
        b_frag1 = S.view(b_shm[1, tid], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

    k_last = IN_FEATURES - GEMM_TILE_K
    k_last_offset = S.convert(k_last * BF16_BYTES, S.i32)
    a_shm[1, tid] = S.amdgpu.raw_buffer_load_x4(x_desc, k_last_offset, a_row_offset, 0)
    b_shm[1, tid] = S.amdgpu.raw_buffer_load_x4(w_desc, k_last_offset, b_row_offset, 0)

    a_frag0 = S.view(a_shm[0, tid], S.Tensor((2, 4, 1), S.bf16))
    b_frag0 = S.view(b_shm[0, tid], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

    a_frag1 = S.view(a_shm[1, tid], S.Tensor((2, 4, 1), S.bf16))
    b_frag1 = S.view(b_shm[1, tid], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

    for i in S.range(16):
        col = c_col_base + i
        Y0[c_row, col] = S.convert(acc[i] + S.convert(BIAS0[col], S.f32), S.bf16)


@substrate.jit
def groupnorm_min_bias_kernel(
    Y0: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
    GN_WEIGHT: S.Tensor((OUT_FEATURES,), S.bf16),
    GN_BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
    EXTRA_BIAS: S.Tensor((1, OUT_FEATURES, 1, 1), S.bf16),
    Y: S.Tensor((1, OUT_FEATURES, BATCH_SIZE, 1), S.bf16),
):
    row = S.block_id(0)

    if row < BATCH_SIZE:
        min_v = S.convert(1.0e30, S.f32)

        for g in S.range(NUM_GROUPS):
            base = g * GROUP_SIZE

            mean = S.convert(0.0, S.f32)
            for t in S.range(GROUP_SIZE):
                mean += S.convert(Y0[row, base + t], S.f32)
            mean = mean / S.convert(GROUP_SIZE, S.f32)

            var = S.convert(0.0, S.f32)
            for t in S.range(GROUP_SIZE):
                d = S.convert(Y0[row, base + t], S.f32) - mean
                var += d * d
            var = var / S.convert(GROUP_SIZE, S.f32)

            denom = S.sqrt(var + S.convert(EPS, S.f32))
            for t in S.range(GROUP_SIZE):
                c = base + t
                v = (S.convert(Y0[row, c], S.f32) - mean) / denom
                v = v * S.convert(GN_WEIGHT[c], S.f32) + S.convert(GN_BIAS[c], S.f32)
                v_bf16 = S.convert(v, S.bf16)
                v_cmp = S.convert(v_bf16, S.f32)
                if v_cmp < min_v:
                    min_v = v_cmp

        for c in S.range(OUT_FEATURES):
            Y[0, c, row, 0] = S.convert(
                min_v + S.convert(EXTRA_BIAS[0, c, 0, 0], S.f32), S.bf16
            )


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._mfma_touch = None

    def _check_supported(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES):
            raise RuntimeError(f"unsupported input shape: {tuple(x.shape)}")
        if x.dtype != torch.bfloat16:
            raise RuntimeError(f"unsupported input dtype: {x.dtype}")
        if self.gemm.weight.shape != (OUT_FEATURES, IN_FEATURES):
            raise RuntimeError(f"unsupported weight shape: {tuple(self.gemm.weight.shape)}")
        if self.gemm.bias is None or tuple(self.gemm.bias.shape) != (OUT_FEATURES,):
            raise RuntimeError("expected linear bias")
        if self.group_norm.num_groups != NUM_GROUPS:
            raise RuntimeError(f"unsupported num_groups: {self.group_norm.num_groups}")
        if self.group_norm.eps != EPS:
            raise RuntimeError(f"unsupported eps: {self.group_norm.eps}")
        if tuple(self.bias.shape) != (1, OUT_FEATURES, 1, 1):
            raise RuntimeError(f"unsupported extra bias shape: {tuple(self.bias.shape)}")

    def _get_mfma_touch(self, x):
        if (
            self._mfma_touch is None
            or self._mfma_touch.device != x.device
            or self._mfma_touch.dtype != x.dtype
        ):
            self._mfma_touch = torch.empty(
                (MFMA_TOUCH_THREADS,), device=x.device, dtype=x.dtype
            )
        return self._mfma_touch

    def forward(self, x):
        self._check_supported(x)

        x_in = x.contiguous()
        w = self.gemm.weight.to(device=x.device, dtype=x.dtype).contiguous()
        bias0 = self.gemm.bias.to(device=x.device, dtype=x.dtype).contiguous()
        gn_w = self.group_norm.weight.to(device=x.device, dtype=x.dtype).contiguous()
        gn_b = self.group_norm.bias.to(device=x.device, dtype=x.dtype).contiguous()
        extra_bias = self.bias.to(device=x.device, dtype=x.dtype).contiguous()

        mfma_touch = self._get_mfma_touch(x_in)
        y0 = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=x.dtype)
        y = torch.empty((1, OUT_FEATURES, BATCH_SIZE, 1), device=x.device, dtype=x.dtype)

        mfma_touch_kernel[_mfma_touch_launch](x_in, w, mfma_touch)
        gemm_vectorized_kernel[_gemm_launch](x_in, w, bias0, y0)
        groupnorm_min_bias_kernel[_post_launch](y0, gn_w, gn_b, extra_bias, y)
        return y
