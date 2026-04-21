import torch
import torch.nn as nn

import substrate
import substrate.language as S

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WAVE_SIZE = 64
NUM_WAVES = 4
THREADS = WAVE_SIZE * NUM_WAVES
NUM_K_TILES = IN_FEATURES // BLOCK_K
NUM_K_PAIRS = NUM_K_TILES // 2

X_NUM_BYTES = BATCH_SIZE * IN_FEATURES * 2
W_NUM_BYTES = OUT_FEATURES * IN_FEATURES * 2


def _launch():
    return ((OUT_FEATURES // BLOCK_N, BATCH_SIZE // BLOCK_M, 1), (THREADS, 1, 1))


@substrate.jit
def fused_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((OUT_FEATURES, IN_FEATURES), S.bf16),
    BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    wave = tid // WAVE_SIZE
    wave_m = wave // 2
    wave_n = wave % 2

    block_n = S.block_id(0)
    block_m = S.block_id(1)
    tile_m = block_m * BLOCK_M
    tile_n = block_n * BLOCK_N

    x_rsrc = S.amdgpu.make_rsrc(X, X_NUM_BYTES)
    w_rsrc = S.amdgpu.make_rsrc(W, W_NUM_BYTES)

    a_lds = S.make_shared((2, 2, 64, 4), S.u32)
    b_lds = S.make_shared((2, 2, 64, 4), S.u32)

    acc = S.full((16,), 0.0, S.f32)

    if tid < 128:
        load_id = tid
        row_half = load_id // 64
        row_chunk = load_id % 64
        row_local = row_chunk % 32
        seg = row_chunk // 32
        row_global = tile_m + row_half * 32 + row_local
        x_offset = (row_global * IN_FEATURES + seg * 8) * 2
        x_vec = S.amdgpu.raw_buffer_load_x4(x_rsrc, 0, x_offset, 0)

        a_lds[0, row_half, row_local, seg * 2 + 0] = x_vec[0]
        a_lds[0, row_half, row_local, seg * 2 + 1] = x_vec[1]
        a_lds[0, row_half, row_local + 32, seg * 2 + 0] = x_vec[2]
        a_lds[0, row_half, row_local + 32, seg * 2 + 1] = x_vec[3]
    else:
        load_id = tid - 128
        col_half = load_id // 64
        col_chunk = load_id % 64
        col_local = col_chunk % 32
        seg = col_chunk // 32
        col_global = tile_n + col_half * 32 + col_local
        w_offset = (col_global * IN_FEATURES + seg * 8) * 2
        w_vec = S.amdgpu.raw_buffer_load_x4(w_rsrc, 0, w_offset, 0)

        b_lds[0, col_half, col_local, seg * 2 + 0] = w_vec[0]
        b_lds[0, col_half, col_local, seg * 2 + 1] = w_vec[1]
        b_lds[0, col_half, col_local + 32, seg * 2 + 0] = w_vec[2]
        b_lds[0, col_half, col_local + 32, seg * 2 + 1] = w_vec[3]

    S.syncthreads()

    for pair in S.range(NUM_K_PAIRS):
        k1_base = (pair * 2 + 1) * BLOCK_K

        if tid < 128:
            load_id = tid
            row_half = load_id // 64
            row_chunk = load_id % 64
            row_local = row_chunk % 32
            seg = row_chunk // 32
            row_global = tile_m + row_half * 32 + row_local
            x_offset = (row_global * IN_FEATURES + k1_base + seg * 8) * 2
            x_vec = S.amdgpu.raw_buffer_load_x4(x_rsrc, 0, x_offset, 0)

            a_lds[1, row_half, row_local, seg * 2 + 0] = x_vec[0]
            a_lds[1, row_half, row_local, seg * 2 + 1] = x_vec[1]
            a_lds[1, row_half, row_local + 32, seg * 2 + 0] = x_vec[2]
            a_lds[1, row_half, row_local + 32, seg * 2 + 1] = x_vec[3]
        else:
            load_id = tid - 128
            col_half = load_id // 64
            col_chunk = load_id % 64
            col_local = col_chunk % 32
            seg = col_chunk // 32
            col_global = tile_n + col_half * 32 + col_local
            w_offset = (col_global * IN_FEATURES + k1_base + seg * 8) * 2
            w_vec = S.amdgpu.raw_buffer_load_x4(w_rsrc, 0, w_offset, 0)

            b_lds[1, col_half, col_local, seg * 2 + 0] = w_vec[0]
            b_lds[1, col_half, col_local, seg * 2 + 1] = w_vec[1]
            b_lds[1, col_half, col_local + 32, seg * 2 + 0] = w_vec[2]
            b_lds[1, col_half, col_local + 32, seg * 2 + 1] = w_vec[3]

        a_frag0 = S.view(a_lds[0, wave_m, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(b_lds[0, wave_n, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

        S.syncthreads()

        a_frag1 = S.view(a_lds[1, wave_m, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag1 = S.view(b_lds[1, wave_n, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

        if pair + 1 < NUM_K_PAIRS:
            k2_base = (pair * 2 + 2) * BLOCK_K

            if tid < 128:
                load_id = tid
                row_half = load_id // 64
                row_chunk = load_id % 64
                row_local = row_chunk % 32
                seg = row_chunk // 32
                row_global = tile_m + row_half * 32 + row_local
                x_offset = (row_global * IN_FEATURES + k2_base + seg * 8) * 2
                x_vec = S.amdgpu.raw_buffer_load_x4(x_rsrc, 0, x_offset, 0)

                a_lds[0, row_half, row_local, seg * 2 + 0] = x_vec[0]
                a_lds[0, row_half, row_local, seg * 2 + 1] = x_vec[1]
                a_lds[0, row_half, row_local + 32, seg * 2 + 0] = x_vec[2]
                a_lds[0, row_half, row_local + 32, seg * 2 + 1] = x_vec[3]
            else:
                load_id = tid - 128
                col_half = load_id // 64
                col_chunk = load_id % 64
                col_local = col_chunk % 32
                seg = col_chunk // 32
                col_global = tile_n + col_half * 32 + col_local
                w_offset = (col_global * IN_FEATURES + k2_base + seg * 8) * 2
                w_vec = S.amdgpu.raw_buffer_load_x4(w_rsrc, 0, w_offset, 0)

                b_lds[0, col_half, col_local, seg * 2 + 0] = w_vec[0]
                b_lds[0, col_half, col_local, seg * 2 + 1] = w_vec[1]
                b_lds[0, col_half, col_local + 32, seg * 2 + 0] = w_vec[2]
                b_lds[0, col_half, col_local + 32, seg * 2 + 1] = w_vec[3]

        S.syncthreads()

    col_local = lane % 32
    col_global = tile_n + wave_n * 32 + col_local
    bias = S.convert(BIAS[col_global], S.f32)
    row_base = tile_m + wave_m * 32
    row_lane_group = lane // 32

    for i in S.range(16):
        row_local = (i % 4) + row_lane_group * 4 + (i // 4) * 8
        row_global = row_base + row_local
        x = acc[i] + bias
        s1 = S.log(S.convert(1.0, S.f32) + S.exp(x))
        x = x * S.tanh(s1)
        s2 = S.log(S.convert(1.0, S.f32) + S.exp(x))
        x = x * S.tanh(s2)
        Y[row_global, col_global] = S.convert(x, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES) or x.dtype != torch.bfloat16:
            raise RuntimeError("ModelNew only supports the fixed bf16 KernelBench shape")

        x = x.contiguous()
        w = self.linear.weight.to(device=x.device, dtype=x.dtype).contiguous()
        bias = self.linear.bias.to(device=x.device, dtype=x.dtype).contiguous()
        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=x.dtype)
        fused_kernel[_launch](x, w, bias, y, num_warps=NUM_WAVES)
        return y
