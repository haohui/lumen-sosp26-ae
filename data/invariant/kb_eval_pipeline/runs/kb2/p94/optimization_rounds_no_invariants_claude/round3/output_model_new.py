import torch
import torch.nn as nn
import substrate
import substrate.language as S

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
NUM_GROUPS = 256
GROUP_SIZE = OUT_FEATURES // NUM_GROUPS
EPS = 1e-5

TILE_M = 64
TILE_N = 64
TILE_K = 8
WAVE_SIZE = 64
WG_SIZE = 256


def _launch():
    return ((OUT_FEATURES // TILE_N, BATCH_SIZE // TILE_M, 1), (WG_SIZE, 1, 1))


@substrate.jit
def fused_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((OUT_FEATURES, IN_FEATURES), S.bf16),
    BIAS0: S.Tensor((OUT_FEATURES,), S.bf16),
    EXTRA_BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
    GN_WEIGHT: S.Tensor((OUT_FEATURES,), S.bf16),
    GN_BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    bx = S.block_id(0)
    by = S.block_id(1)
    tid = S.thread_id(0)

    wave_id = tid // WAVE_SIZE
    lane = tid % WAVE_SIZE

    warp_row = wave_id // 2
    warp_col = wave_id % 2

    tile_row_base = by * TILE_M + warp_row * 32
    tile_col_base = bx * TILE_N + warp_col * 32

    # Resource descriptors for raw buffer loads with range (in bytes)
    rsrc_X = S.amdgpu.make_rsrc(X, BATCH_SIZE * IN_FEATURES * 2)
    rsrc_W = S.amdgpu.make_rsrc(W, OUT_FEATURES * IN_FEATURES * 2)
    rsrc_Y = S.amdgpu.make_rsrc(Y, BATCH_SIZE * OUT_FEATURES * 2)

    # Accumulator: 16 f32 per lane for 32x32x8 MFMA
    acc = S.full((16,), 0.0, S.f32)

    num_k_steps = IN_FEATURES // TILE_K
    num_unrolled = num_k_steps // 2

    for ui in S.range(num_unrolled):
        # === First K step in this unrolled iteration ===
        k_step_0 = ui * 2
        k_base_0 = k_step_0 * TILE_K

        # A fragment: X[global_row, k_base + col_group*4 + j]
        a_row_in_tile = lane % 32
        a_col_group = lane // 32
        global_a_row = tile_row_base + a_row_in_tile
        a_byte_offset_0 = (global_a_row * IN_FEATURES + k_base_0 + a_col_group * 4) * 2
        a_data_0 = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_byte_offset_0, 0, 0)
        a_frag_0 = S.view(a_data_0, S.Tensor((4,), S.bf16))

        # b fragment: W[global_col, k_base + k_group*4 + j]
        b_col_in_tile = lane % 32
        b_k_group = lane // 32
        global_b_col = tile_col_base + b_col_in_tile
        b_byte_offset_0 = (global_b_col * IN_FEATURES + k_base_0 + b_k_group * 4) * 2
        b_data_0 = S.amdgpu.raw_buffer_load_x2(rsrc_W, b_byte_offset_0, 0, 0)
        b_frag_0 = S.view(b_data_0, S.Tensor((4,), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_0, b_frag_0, acc)

        # === Second K step in this unrolled iteration ===
        k_step_1 = ui * 2 + 1
        k_base_1 = k_step_1 * TILE_K

        a_byte_offset_1 = (global_a_row * IN_FEATURES + k_base_1 + a_col_group * 4) * 2
        a_data_1 = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_byte_offset_1, 0, 0)
        a_frag_1 = S.view(a_data_1, S.Tensor((4,), S.bf16))

        b_byte_offset_1 = (global_b_col * IN_FEATURES + k_base_1 + b_k_group * 4) * 2
        b_data_1 = S.amdgpu.raw_buffer_load_x2(rsrc_W, b_byte_offset_1, 0, 0)
        b_frag_1 = S.view(b_data_1, S.Tensor((4,), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_1, b_frag_1, acc)

    # Write output using correct C accumulator mapping:
    # col = tile_col_base + (lane % 32)
    # row = tile_row_base + 8*(acc_idx//4) + 4*(lane//32) + (acc_idx%4)
    # RemovedOob branches - range in rsrc handles OOB access
    for acc_idx in S.range(16):
        col = tile_col_base + (lane % 32)
        row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)

        val = acc[acc_idx]
        val = val + S.convert(BIAS0[col], S.f32)
        # Round to bf16 to match reference's nn.Linear output precision
        val = S.convert(S.convert(val, S.bf16), S.f32)
        val = val + S.convert(EXTRA_BIAS[col], S.f32)
        # Round to bf16 to match reference's bf16 bias addition
        val = S.convert(S.convert(val, S.bf16), S.f32)

        # Hardtanh: clamp to [-1, 1]
        if val < S.convert(-1.0, S.f32):
            val = S.convert(-1.0, S.f32)
        if val > S.convert(1.0, S.f32):
            val = S.convert(1.0, S.f32)

        # Mish: x * tanh(log(1 + exp(x)))
        sp = S.log(S.convert(1.0, S.f32) + S.exp(val))
        val = val * S.tanh(sp)

        Y[row, col] = S.convert(val, S.bf16)

    S.syncthreads()

    # GroupNorm: each warp handles 32 consecutive columns = 1 group
    # Removedoob分支 - 使用raw缓冲加载X4读取
    for row_local in S.range(32):
        global_m = tile_row_base + row_local

        mean = S.convert(0.0, S.f32)
        # Use raw_buffer_load_x4 一次加载8个bf16值
        for vec_idx in S.range(4):
            byte_offset = (global_m * OUT_FEATURES + tile_col_base + vec_idx * 8) * 2
            data = S.amdgpu.raw_buffer_load_x4(rsrc_Y, byte_offset, 0, 0)
            data_bf16 = S.view(data, S.Tensor((8,), S.bf16))
            for j in S.range(8):
                mean = mean + S.convert(data_bf16[j], S.f32)
        mean = mean / S.convert(GROUP_SIZE, S.f32)

        var = S.convert(0.0, S.f32)
        # 计算方差
        for vec_idx in S.range(4):
            byte_offset = (global_m * OUT_FEATURES + tile_col_base + vec_idx * 8) * 2
            data = S.amdgpu.raw_buffer_load_x4(rsrc_Y, byte_offset, 0, 0)
            data_bf16 = S.view(data, S.Tensor((8,), S.bf16))
            for j in S.range(8):
                diff = S.convert(data_bf16[j], S.f32) - mean
                var = var + diff * diff
        var = var / S.convert(GROUP_SIZE, S.f32)

        denom = S.sqrt(var + S.convert(EPS, S.f32))
        # 写入归一化值
        for col_local in S.range(32):
            global_n = tile_col_base + col_local
            byte_offset = (global_m * OUT_FEATURES + global_n) * 2
            data = S.amdgpu.raw_buffer_load_x4(rsrc_Y, byte_offset, 0, 0)
            data_bf16 = S.view(data, S.Tensor((8,), S.bf16))
            v = (S.convert(data_bf16[0], S.f32) - mean) / denom
            v = v * S.convert(GN_WEIGHT[global_n], S.f32) + S.convert(GN_BIAS[global_n], S.f32)
            Y[global_m, global_n] = S.convert(v, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES) or x.dtype != torch.bfloat16 \
                or tuple(self.bias.shape) != (OUT_FEATURES,) \
                or (self.groupnorm.num_groups != NUM_GROUPS) \
                or (self.groupnorm.eps != EPS):
            raise RuntimeError('This fused kernel only supports the benchmark input shape and dtype.')

        # Pass weight directly (NOT transposed) - kernel expects (N, K) = (OUT, IN) layout
        w = self.gemm.weight.to(device=x.device, dtype=x.dtype).contiguous()
        bias0 = self.gemm.bias.to(device=x.device, dtype=x.dtype).contiguous()
        extra_bias = self.bias.to(device=x.device, dtype=x.dtype).contiguous()
        gn_w = self.groupnorm.weight.to(device=x.device, dtype=x.dtype).contiguous()
        gn_b = self.groupnorm.bias.to(device=x.device, dtype=x.dtype).contiguous()
        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=x.dtype)

        fused_kernel[_launch](x.contiguous(), w, bias0, extra_bias, gn_w, gn_b, y)
        return y
