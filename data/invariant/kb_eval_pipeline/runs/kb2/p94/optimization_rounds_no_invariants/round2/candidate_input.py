import torch
import torch.nn as nn
import substrate
import substrate.language as S

SQRT_2 = 1.4142135623730951

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
NUM_GROUPS = 256
GROUP_SIZE = OUT_FEATURES // NUM_GROUPS
EPS = 1e-05

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = 64 * WAVES_PER_BLOCK


def _launch_gemm():
    return ((OUT_FEATURES // BLOCK_N, BATCH_SIZE // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))


def _launch_groupnorm():
    return ((NUM_GROUPS, BATCH_SIZE, 1), (GROUP_SIZE, 1, 1))


def _launch_cast():
    return (((BATCH_SIZE * OUT_FEATURES) // 256, 1, 1), (256, 1, 1))


@substrate.jit
def gemm_bias_act_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((OUT_FEATURES, IN_FEATURES), S.bf16),
    BIAS0: S.Tensor((OUT_FEATURES,), S.f32),
    EXTRA_BIAS: S.Tensor((OUT_FEATURES,), S.f32),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.f32),
):
    tid = S.thread_id(0)
    lane = tid % 64
    wave = tid // 64
    wave_m = wave // 2
    wave_n = wave % 2
    block_m = S.block_id(1) * BLOCK_M
    block_n = S.block_id(0) * BLOCK_N

    zero = S.convert(0, S.i32)
    x_range = S.convert(BATCH_SIZE * IN_FEATURES * 2, S.i32)
    w_range = S.convert(OUT_FEATURES * IN_FEATURES * 2, S.i32)
    x_rsrc = S.amdgpu.make_rsrc(X, x_range)
    w_rsrc = S.amdgpu.make_rsrc(W, w_range)

    a_words = S.make_shared((BLOCK_M * 8,), S.u32)
    b_words = S.make_shared((BLOCK_N * 8,), S.u32)

    acc = S.full((16,), 0.0, S.f32)

    for k0 in S.range(IN_FEATURES // BLOCK_K):
        k_base = k0 * BLOCK_K

        if tid < 128:
            row = block_m + (tid % BLOCK_M)
            a_half = tid // BLOCK_M
            a_offset = S.convert((row * IN_FEATURES + k_base + a_half * 8) * 2, S.i32)
            packed_a = S.amdgpu.raw_buffer_load_x4(x_rsrc, zero, a_offset, 0)
            a_idx = tid * 4
            a_words[a_idx + 0] = packed_a[0]
            a_words[a_idx + 1] = packed_a[1]
            a_words[a_idx + 2] = packed_a[2]
            a_words[a_idx + 3] = packed_a[3]

        if tid >= 128:
            b_tid = tid - 128
            col = block_n + (b_tid % BLOCK_N)
            b_half = b_tid // BLOCK_N
            b_offset = S.convert((col * IN_FEATURES + k_base + b_half * 8) * 2, S.i32)
            packed_b = S.amdgpu.raw_buffer_load_x4(w_rsrc, zero, b_offset, 0)
            b_idx = b_tid * 4
            b_words[b_idx + 0] = packed_b[0]
            b_words[b_idx + 1] = packed_b[1]
            b_words[b_idx + 2] = packed_b[2]
            b_words[b_idx + 3] = packed_b[3]

        S.syncthreads()

        a_row = block_m + wave_m * 32 + (lane % 32)
        b_col = block_n + wave_n * 32 + (lane % 32)
        lane_half = lane // 32

        a_lane_offset = S.convert((a_row * IN_FEATURES + k_base + lane_half * 8) * 2, S.i32)
        b_lane_offset = S.convert((b_col * IN_FEATURES + k_base + lane_half * 8) * 2, S.i32)
        a_packed = S.amdgpu.raw_buffer_load_x4(x_rsrc, zero, a_lane_offset, 0)
        b_packed = S.amdgpu.raw_buffer_load_x4(w_rsrc, zero, b_lane_offset, 0)
        a_frag = S.view(a_packed, S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_packed, S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        S.syncthreads()

    out_col = block_n + wave_n * 32 + (lane % 32)
    row_base = block_m + wave_m * 32 + (lane // 32) * 4

    for i in S.range(16):
        out_row = row_base + (i // 4) * 8 + (i % 4)
        x = acc[i] + BIAS0[out_col] + EXTRA_BIAS[out_col]
        if x < S.convert(-1.0, S.f32):
            x = S.convert(-1.0, S.f32)
        if x > S.convert(1.0, S.f32):
            x = S.convert(1.0, S.f32)
        x = x * S.tanh(S.log(S.convert(1.0, S.f32) + S.exp(x)))
        Y[out_row, out_col] = x


@substrate.jit
def groupnorm_kernel(
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.f32),
    GN_WEIGHT: S.Tensor((OUT_FEATURES,), S.f32),
    GN_BIAS: S.Tensor((OUT_FEATURES,), S.f32),
):
    tid = S.thread_id(0)
    row = S.block_id(1)
    group = S.block_id(0)
    col = group * GROUP_SIZE + tid

    stats = S.make_shared((2,), S.f32)

    if tid == 0:
        mean = S.convert(0.0, S.f32)
        for t in S.range(GROUP_SIZE):
            mean += Y[row, group * GROUP_SIZE + t]
        mean = mean / S.convert(GROUP_SIZE, S.f32)

        var = S.convert(0.0, S.f32)
        for t in S.range(GROUP_SIZE):
            d = Y[row, group * GROUP_SIZE + t] - mean
            var += d * d
        var = var / S.convert(GROUP_SIZE, S.f32)

        stats[0] = mean
        stats[1] = var

    S.syncthreads()

    v = Y[row, col]
    mean = stats[0]
    var = stats[1]
    inv_std = S.convert(1.0, S.f32) / S.sqrt(var + S.convert(EPS, S.f32))
    out = (v - mean) * inv_std
    out = out * GN_WEIGHT[col] + GN_BIAS[col]
    Y[row, col] = out


@substrate.jit
def cast_output_kernel(
    X: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.f32),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    row = idx // OUT_FEATURES
    col = idx % OUT_FEATURES
    Y[row, col] = S.convert(X[row, col], S.bf16)


class ModelNew(nn.Module):

    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)

    def forward(self, x):
        if (
            tuple(x.shape) != (BATCH_SIZE, IN_FEATURES)
            or x.dtype != torch.bfloat16
            or tuple(self.bias.shape) != (OUT_FEATURES,)
            or self.groupnorm.num_groups != NUM_GROUPS
            or self.groupnorm.eps != EPS
        ):
            raise RuntimeError("This fused kernel only supports the benchmark input shape and dtype.")

        x_in = x.contiguous()
        w = self.gemm.weight.to(device=x.device, dtype=x.dtype).contiguous()
        bias0 = self.gemm.bias.to(device=x.device, dtype=torch.float32).contiguous()
        extra_bias = self.bias.to(device=x.device, dtype=torch.float32).contiguous()
        gn_w = self.groupnorm.weight.to(device=x.device, dtype=torch.float32).contiguous()
        gn_b = self.groupnorm.bias.to(device=x.device, dtype=torch.float32).contiguous()

        y_acc = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=torch.float32)
        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=torch.bfloat16)
        gemm_bias_act_kernel[_launch_gemm](x_in, w, bias0, extra_bias, y_acc)
        groupnorm_kernel[_launch_groupnorm](y_acc, gn_w, gn_b)
        cast_output_kernel[_launch_cast](y_acc, y)
        return y
