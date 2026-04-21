import torch
import torch.nn as nn

import substrate
import substrate.language as S

SQRT_2 = 1.4142135623730951

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
NEGATIVE_SLOPE = 0.01

BLOCK_M = 64
BLOCK_N = 64
WAVE_M = 32
WAVE_N = 32
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = 256
K_STEP = 16
K_UNROLL = 32
REDUCE_THREADS = 256

X_NUM_BYTES = BATCH_SIZE * IN_FEATURES * 2
W_NUM_BYTES = OUT_FEATURES * IN_FEATURES * 2
OUT_NUM_BYTES = BATCH_SIZE * OUT_FEATURES * 4


def _gemm_launch():
    return ((OUT_FEATURES // BLOCK_N, BATCH_SIZE // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))


def _reduce_launch():
    return ((BATCH_SIZE, 1, 1), (REDUCE_THREADS, 1, 1))


@substrate.jit
def fused_gemm_mfma_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((OUT_FEATURES, IN_FEATURES), S.bf16),
    Bias: S.Tensor((OUT_FEATURES,), S.bf16),
    Out: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.f32),
):
    tid = S.thread_id(0)
    lane = tid % 64
    wave = tid // 64
    wave_m = wave // 2
    wave_n = wave % 2
    lane_col = lane % 32
    lane_hi = lane // 32
    lane_row_swz = lane_col

    block_m = S.block_id(1) * BLOCK_M
    block_n = S.block_id(0) * BLOCK_N
    tile_m = block_m + wave_m * WAVE_M
    tile_n = block_n + wave_n * WAVE_N

    a_shared = S.make_shared((2, THREADS_PER_BLOCK, 4), S.u32)
    b_shared = S.make_shared((2, THREADS_PER_BLOCK, 4), S.u32)

    x_rsrc = S.amdgpu.make_rsrc(X, X_NUM_BYTES)
    w_rsrc = S.amdgpu.make_rsrc(W, W_NUM_BYTES)
    out_rsrc = S.amdgpu.make_rsrc(Out, OUT_NUM_BYTES)

    a_row = (
        (lane_row_swz & 0x3)
        + ((lane_row_swz & 0x8) >> 1)
        + ((lane_row_swz & 0x10) >> 1)
        + ((lane_row_swz & 0x4) << 2)
    )
    out_col = tile_n + lane_col

    acc = S.full((16,), 0.0, S.f32)

    a_k0 = lane_hi * 8
    b_k0 = lane_hi * 8
    a_offset0 = ((tile_m + a_row) * IN_FEATURES + a_k0) * 2
    b_offset0 = ((out_col) * IN_FEATURES + b_k0) * 2
    a_vec0 = S.amdgpu.raw_buffer_load_x4(x_rsrc, a_offset0, 0, 0)
    b_vec0 = S.amdgpu.raw_buffer_load_x4(w_rsrc, b_offset0, 0, 0)
    for i in S.range(4):
        a_shared[0, tid, i] = a_vec0[i]
        b_shared[0, tid, i] = b_vec0[i]

    a_k1 = K_STEP + lane_hi * 8
    b_k1 = K_STEP + lane_hi * 8
    a_offset1 = ((tile_m + a_row) * IN_FEATURES + a_k1) * 2
    b_offset1 = ((out_col) * IN_FEATURES + b_k1) * 2
    a_vec1 = S.amdgpu.raw_buffer_load_x4(x_rsrc, a_offset1, 0, 0)
    b_vec1 = S.amdgpu.raw_buffer_load_x4(w_rsrc, b_offset1, 0, 0)
    for i in S.range(4):
        a_shared[1, tid, i] = a_vec1[i]
        b_shared[1, tid, i] = b_vec1[i]

    S.syncthreads()

    for k_base in S.range(0, IN_FEATURES - K_UNROLL, K_UNROLL):
        a_frag0 = S.view(a_shared[0, tid], S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(b_shared[0, tid], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)

        next_a0_k = k_base + K_UNROLL + lane_hi * 8
        next_b0_k = k_base + K_UNROLL + lane_hi * 8
        next_a0_offset = ((tile_m + a_row) * IN_FEATURES + next_a0_k) * 2
        next_b0_offset = ((out_col) * IN_FEATURES + next_b0_k) * 2
        next_a0 = S.amdgpu.raw_buffer_load_x4(x_rsrc, next_a0_offset, 0, 0)
        next_b0 = S.amdgpu.raw_buffer_load_x4(w_rsrc, next_b0_offset, 0, 0)

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)
        for i in S.range(4):
            a_shared[0, tid, i] = next_a0[i]
            b_shared[0, tid, i] = next_b0[i]

        S.syncthreads()

        a_frag1 = S.view(a_shared[1, tid], S.Tensor((2, 4, 1), S.bf16))
        b_frag1 = S.view(b_shared[1, tid], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)

        next_a1_k = k_base + K_UNROLL + K_STEP + lane_hi * 8
        next_b1_k = k_base + K_UNROLL + K_STEP + lane_hi * 8
        next_a1_offset = ((tile_m + a_row) * IN_FEATURES + next_a1_k) * 2
        next_b1_offset = ((out_col) * IN_FEATURES + next_b1_k) * 2
        next_a1 = S.amdgpu.raw_buffer_load_x4(x_rsrc, next_a1_offset, 0, 0)
        next_b1 = S.amdgpu.raw_buffer_load_x4(w_rsrc, next_b1_offset, 0, 0)

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)
        for i in S.range(4):
            a_shared[1, tid, i] = next_a1[i]
            b_shared[1, tid, i] = next_b1[i]

        S.syncthreads()

    a_tail0 = S.view(a_shared[0, tid], S.Tensor((2, 4, 1), S.bf16))
    b_tail0 = S.view(b_shared[0, tid], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_tail0[0], b_tail0[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_tail0[1], b_tail0[1], acc)

    a_tail1 = S.view(a_shared[1, tid], S.Tensor((2, 4, 1), S.bf16))
    b_tail1 = S.view(b_shared[1, tid], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_tail1[0], b_tail1[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_tail1[1], b_tail1[1], acc)

    bias_val = Bias[out_col]
    row_base = tile_m + lane_hi * 16
    for i in S.range(0, 16, 4):
        out_vals = S.full((4,), 0.0, S.f32)
        out_vals[0] = acc[i + 0] + bias_val
        out_vals[1] = acc[i + 1] + bias_val
        out_vals[2] = acc[i + 2] + bias_val
        out_vals[3] = acc[i + 3] + bias_val
        out_bits = S.view(out_vals, S.Tensor((4,), S.i32))
        out_offset = (((row_base + i) * OUT_FEATURES) + out_col) * 4
        S.amdgpu.raw_buffer_store_x4(out_bits, out_rsrc, out_offset, 0, 0)


@substrate.jit
def reduce_logsumexp_activation_kernel(
    Inp: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.f32),
    Out: S.Tensor((BATCH_SIZE, 1), S.bf16),
):
    row = S.block_id(0)
    tid = S.thread_id(0)
    shared = S.make_shared((REDUCE_THREADS,), S.f32)
    neg_inf = S.convert(-3.4028234663852886e38, S.f32)
    zero = S.convert(0.0, S.f32)
    half = S.convert(0.5, S.f32)
    one = S.convert(1.0, S.f32)
    neg_slope = S.convert(NEGATIVE_SLOPE, S.f32)
    sqrt_2 = S.convert(SQRT_2, S.f32)

    local_max = neg_inf
    for col in S.range(tid, OUT_FEATURES, REDUCE_THREADS):
        val = Inp[row, col]
        if val > local_max:
            local_max = val
    shared[tid] = local_max
    S.syncthreads()

    offset = REDUCE_THREADS // 2
    for _ in S.range(8):
        if tid < offset:
            other = shared[tid + offset]
            if other > shared[tid]:
                shared[tid] = other
        S.syncthreads()
        offset = offset // 2

    row_max = shared[0]

    local_sum = zero
    for col in S.range(tid, OUT_FEATURES, REDUCE_THREADS):
        local_sum = local_sum + S.exp(Inp[row, col] - row_max)
    shared[tid] = local_sum
    S.syncthreads()

    offset = REDUCE_THREADS // 2
    for _ in S.range(8):
        if tid < offset:
            shared[tid] = shared[tid] + shared[tid + offset]
        S.syncthreads()
        offset = offset // 2

    if tid == 0:
        x = row_max + S.log(shared[0])
        if x < zero:
            x = x * neg_slope
        if x < zero:
            x = x * neg_slope
        x = half * x * (one + S.erf(x / sqrt_2))
        x = half * x * (one + S.erf(x / sqrt_2))
        Out[row, 0] = S.convert(x, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._buffer_cache = {}

    def _get_buffers(self, device: torch.device):
        cached = self._buffer_cache.get(device)
        if cached is None:
            gemm_out = torch.empty((BATCH_SIZE, OUT_FEATURES), device=device, dtype=torch.float32)
            reduce_out = torch.empty((BATCH_SIZE, 1), device=device, dtype=torch.bfloat16)
            cached = (gemm_out, reduce_out)
            self._buffer_cache[device] = cached
        return cached

    def forward(self, x):
        if not (x.is_cuda and torch.version.hip is not None):
            raise RuntimeError("ModelNew requires ROCm/Substrate execution")

        gemm_out, reduce_out = self._get_buffers(x.device)
        fused_gemm_mfma_kernel[_gemm_launch](x, self.linear.weight, self.linear.bias, gemm_out)
        reduce_logsumexp_activation_kernel[_reduce_launch](gemm_out, reduce_out)
        return reduce_out
