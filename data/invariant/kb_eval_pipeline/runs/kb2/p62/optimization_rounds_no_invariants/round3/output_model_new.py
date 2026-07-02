import torch
import torch.nn as nn
import substrate
import substrate.language as S

BATCH_SIZE = 1024
INPUT_SIZE = 8192
HIDDEN_SIZE = 8192
NUM_GROUPS = 512
GROUP_SIZE = HIDDEN_SIZE // NUM_GROUPS
NEGATIVE_SLOPE = 0.01
EPS = 1e-05

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
PIPE_STAGES = 2
WAVE_SIZE = 64
WAVES_M = 2
WAVES_N = 2
WAVES_PER_BLOCK = WAVES_M * WAVES_N
THREADS_PER_BLOCK = WAVE_SIZE * WAVES_PER_BLOCK

X_RANGE_BYTES = BATCH_SIZE * INPUT_SIZE * 2
W_RANGE_BYTES = HIDDEN_SIZE * INPUT_SIZE * 2

def _launch_gemm():
    return ((HIDDEN_SIZE // BLOCK_N, BATCH_SIZE // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))


def _launch_post():
    return ((BATCH_SIZE, NUM_GROUPS, 1), (GROUP_SIZE, 1, 1))


@substrate.jit
def gemm_bias_mfma_kernel(
    X: S.Tensor((BATCH_SIZE, INPUT_SIZE), S.bf16),
    W_T: S.Tensor((HIDDEN_SIZE, INPUT_SIZE), S.bf16),
    BIAS0: S.Tensor((HIDDEN_SIZE,), S.bf16),
    TMP: S.Tensor((BATCH_SIZE, HIDDEN_SIZE), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    wave = tid // WAVE_SIZE
    wave_m = wave // WAVES_N
    wave_n = wave % WAVES_N

    block_m = S.block_id(1) * BLOCK_M
    block_n = S.block_id(0) * BLOCK_N
    warp_row = block_m + wave_m * 32
    warp_col = block_n + wave_n * 32

    a_words = S.make_shared((PIPE_STAGES * WAVES_M * 64 * 4,), S.u32)
    b_words = S.make_shared((PIPE_STAGES * WAVES_N * 64 * 4,), S.u32)

    x_rsrc = S.amdgpu.make_rsrc(X, X_RANGE_BYTES)
    w_rsrc = S.amdgpu.make_rsrc(W_T, W_RANGE_BYTES)
    zero = S.convert(0, S.i32)
    a_stage_words = WAVES_M * 64 * 4
    b_stage_words = WAVES_N * 64 * 4
    k_tiles = INPUT_SIZE // BLOCK_K
    a_layout = S.make_layout((PIPE_STAGES, WAVES_M, 64, 4), (a_stage_words, 256, 4, 1))
    b_layout = S.make_layout((PIPE_STAGES, WAVES_N, 64, 4), (b_stage_words, 256, 4, 1))
    a_packed = S.view(a_words, S.u32, a_layout)
    b_packed = S.view(b_words, S.u32, b_layout)

    c_lane = S.full((16,), 0.0, S.f32)

    for preload_stage in S.range(PIPE_STAGES):
        k_base = preload_stage * BLOCK_K
        if tid < 128:
            load_row = tid >> 1
            load_part = tid & 1
            row_bytes = ((block_m + load_row) * INPUT_SIZE + k_base + load_part * 8) * 2
            vec = S.amdgpu.raw_buffer_load_x4(
                x_rsrc, zero, S.convert(row_bytes, S.i32), 0
            )
            wave_sel = load_row >> 5
            row_in_wave = load_row & 31
            lane_swz = row_in_wave + load_part * 32
            base = preload_stage * a_stage_words + wave_sel * 256 + lane_swz * 4
            a_words[base + 0] = vec[0]
            a_words[base + 1] = vec[1]
            a_words[base + 2] = vec[2]
            a_words[base + 3] = vec[3]
        else:
            t = tid - 128
            load_row = t >> 1
            load_part = t & 1
            row_bytes = ((block_n + load_row) * INPUT_SIZE + k_base + load_part * 8) * 2
            vec = S.amdgpu.raw_buffer_load_x4(
                w_rsrc, zero, S.convert(row_bytes, S.i32), 0
            )
            wave_sel = load_row >> 5
            row_in_wave = load_row & 31
            lane_swz = row_in_wave + load_part * 32
            base = preload_stage * b_stage_words + wave_sel * 256 + lane_swz * 4
            b_words[base + 0] = vec[0]
            b_words[base + 1] = vec[1]
            b_words[base + 2] = vec[2]
            b_words[base + 3] = vec[3]
    S.syncthreads()

    for k_pair in S.range(k_tiles // 2 - 1):
        stage0 = 0
        stage1 = 1

        m_a0 = S.view(a_packed[stage0, wave_m, lane], S.Tensor((2, 4, 1), S.bf16))
        m_b0 = S.view(b_packed[stage0, wave_n, lane], S.Tensor((2, 4, 1), S.bf16))
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(m_a0[0], m_b0[0], c_lane)
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(m_a0[1], m_b0[1], c_lane)

        next_k0 = (k_pair + 1) * 2 * BLOCK_K
        if tid < 128:
            load_row = tid >> 1
            load_part = tid & 1
            row_bytes = ((block_m + load_row) * INPUT_SIZE + next_k0 + load_part * 8) * 2
            vec = S.amdgpu.raw_buffer_load_x4(
                x_rsrc, zero, S.convert(row_bytes, S.i32), 0
            )
            wave_sel = load_row >> 5
            row_in_wave = load_row & 31
            lane_swz = row_in_wave + load_part * 32
            base = stage0 * a_stage_words + wave_sel * 256 + lane_swz * 4
            a_words[base + 0] = vec[0]
            a_words[base + 1] = vec[1]
            a_words[base + 2] = vec[2]
            a_words[base + 3] = vec[3]
        else:
            t = tid - 128
            load_row = t >> 1
            load_part = t & 1
            row_bytes = ((block_n + load_row) * INPUT_SIZE + next_k0 + load_part * 8) * 2
            vec = S.amdgpu.raw_buffer_load_x4(
                w_rsrc, zero, S.convert(row_bytes, S.i32), 0
            )
            wave_sel = load_row >> 5
            row_in_wave = load_row & 31
            lane_swz = row_in_wave + load_part * 32
            base = stage0 * b_stage_words + wave_sel * 256 + lane_swz * 4
            b_words[base + 0] = vec[0]
            b_words[base + 1] = vec[1]
            b_words[base + 2] = vec[2]
            b_words[base + 3] = vec[3]

        m_a1 = S.view(a_packed[stage1, wave_m, lane], S.Tensor((2, 4, 1), S.bf16))
        m_b1 = S.view(b_packed[stage1, wave_n, lane], S.Tensor((2, 4, 1), S.bf16))
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(m_a1[0], m_b1[0], c_lane)
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(m_a1[1], m_b1[1], c_lane)

        next_k1 = ((k_pair + 1) * 2 + 1) * BLOCK_K
        if tid < 128:
            load_row = tid >> 1
            load_part = tid & 1
            row_bytes = ((block_m + load_row) * INPUT_SIZE + next_k1 + load_part * 8) * 2
            vec = S.amdgpu.raw_buffer_load_x4(
                x_rsrc, zero, S.convert(row_bytes, S.i32), 0
            )
            wave_sel = load_row >> 5
            row_in_wave = load_row & 31
            lane_swz = row_in_wave + load_part * 32
            base = stage1 * a_stage_words + wave_sel * 256 + lane_swz * 4
            a_words[base + 0] = vec[0]
            a_words[base + 1] = vec[1]
            a_words[base + 2] = vec[2]
            a_words[base + 3] = vec[3]
        else:
            t = tid - 128
            load_row = t >> 1
            load_part = t & 1
            row_bytes = ((block_n + load_row) * INPUT_SIZE + next_k1 + load_part * 8) * 2
            vec = S.amdgpu.raw_buffer_load_x4(
                w_rsrc, zero, S.convert(row_bytes, S.i32), 0
            )
            wave_sel = load_row >> 5
            row_in_wave = load_row & 31
            lane_swz = row_in_wave + load_part * 32
            base = stage1 * b_stage_words + wave_sel * 256 + lane_swz * 4
            b_words[base + 0] = vec[0]
            b_words[base + 1] = vec[1]
            b_words[base + 2] = vec[2]
            b_words[base + 3] = vec[3]

        S.syncthreads()

    m_a0 = S.view(a_packed[0, wave_m, lane], S.Tensor((2, 4, 1), S.bf16))
    m_b0 = S.view(b_packed[0, wave_n, lane], S.Tensor((2, 4, 1), S.bf16))
    c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(m_a0[0], m_b0[0], c_lane)
    c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(m_a0[1], m_b0[1], c_lane)

    m_a1 = S.view(a_packed[1, wave_m, lane], S.Tensor((2, 4, 1), S.bf16))
    m_b1 = S.view(b_packed[1, wave_n, lane], S.Tensor((2, 4, 1), S.bf16))
    c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(m_a1[0], m_b1[0], c_lane)
    c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(m_a1[1], m_b1[1], c_lane)

    lane_row = lane & 15
    lane_col_group = lane >> 4
    out_col = warp_col + lane_row + (lane_col_group & 1) * 16
    bias = S.convert(BIAS0[out_col], S.f32)

    for row_quad in S.range(2):
        row_base = warp_row + row_quad * 16 + (lane_col_group >> 1) * 4
        idx_base = row_quad * 8
        for i in S.range(4):
            TMP[row_base + i, out_col] = S.convert(c_lane[idx_base + i] + bias, S.bf16)
            TMP[row_base + 8 + i, out_col] = S.convert(c_lane[idx_base + 4 + i] + bias, S.bf16)


@substrate.jit
def groupnorm_leakyrelu_mul2_kernel(
    TMP: S.Tensor((BATCH_SIZE, HIDDEN_SIZE), S.bf16),
    GN_WEIGHT: S.Tensor((HIDDEN_SIZE,), S.bf16),
    GN_BIAS: S.Tensor((HIDDEN_SIZE,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, HIDDEN_SIZE), S.bf16),
):
    row = S.block_id(0)
    group = S.block_id(1)
    lane = S.thread_id(0)
    channel = group * GROUP_SIZE + lane

    shared_vals = S.make_shared((GROUP_SIZE,), S.f32)
    v = S.convert(TMP[row, channel], S.f32)
    shared_vals[lane] = v
    S.syncthreads()

    if lane == 0:
        mean = S.convert(0.0, S.f32)
        for i in S.range(GROUP_SIZE):
            mean += shared_vals[i]
        mean = mean / S.convert(GROUP_SIZE, S.f32)

        var = S.convert(0.0, S.f32)
        for i in S.range(GROUP_SIZE):
            d = shared_vals[i] - mean
            var += d * d
        var = var / S.convert(GROUP_SIZE, S.f32)

        shared_vals[0] = mean
        shared_vals[1] = S.convert(1.0, S.f32) / S.sqrt(var + S.convert(EPS, S.f32))

    S.syncthreads()

    mean = shared_vals[0]
    inv_std = shared_vals[1]
    out = (v - mean) * inv_std
    out = out * S.convert(GN_WEIGHT[channel], S.f32) + S.convert(GN_BIAS[channel], S.f32)
    if out < S.convert(0.0, S.f32):
        out = out * S.convert(NEGATIVE_SLOPE, S.f32)
    Y[row, channel] = S.convert(out + out, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-05, negative_slope=0.01):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        if (
            tuple(x.shape) != (BATCH_SIZE, INPUT_SIZE)
            or x.dtype != torch.bfloat16
            or self.gn.num_groups != NUM_GROUPS
            or self.gn.eps != EPS
            or self.leaky_relu.negative_slope != NEGATIVE_SLOPE
        ):
            raise RuntimeError("This fused kernel only supports the benchmark input shape and dtype.")

        x_in = x.contiguous()
        w_t = self.fc.weight.to(device=x.device, dtype=x.dtype).contiguous()
        bias = self.fc.bias.to(device=x.device, dtype=x.dtype).contiguous()
        gn_w = self.gn.weight.to(device=x.device, dtype=x.dtype).contiguous()
        gn_b = self.gn.bias.to(device=x.device, dtype=x.dtype).contiguous()

        tmp = torch.empty((BATCH_SIZE, HIDDEN_SIZE), device=x.device, dtype=x.dtype)
        y = torch.empty((BATCH_SIZE, HIDDEN_SIZE), device=x.device, dtype=x.dtype)

        gemm_bias_mfma_kernel[_launch_gemm](x_in, w_t, bias, tmp)
        groupnorm_leakyrelu_mul2_kernel[_launch_post](tmp, gn_w, gn_b, y)
        return y
