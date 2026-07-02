import torch
import torch.nn as nn
import substrate
import substrate.language as S

BATCH_SIZE = 128
INPUT_SIZE = 32768
HIDDEN_SIZE = 32768
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
THREADS = 256
WAVE_SIZE = 64


def _launch():
    return ((BATCH_SIZE // BLOCK_M, 1, 1), (THREADS, 1, 1))


@substrate.jit
def fused_kernel(
    X: S.Tensor((BATCH_SIZE, INPUT_SIZE), S.bf16),
    W: S.Tensor((INPUT_SIZE, HIDDEN_SIZE), S.bf16),
    BIAS0: S.Tensor((HIDDEN_SIZE,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, 1), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    wave = tid // WAVE_SIZE
    warp_row = wave // 2
    warp_col = wave % 2
    block_row = S.block_id(0) * BLOCK_M

    zero_i32 = S.convert(0, S.i32)
    sixteen_i32 = S.convert(16, S.i32)
    one = S.convert(1.0, S.f32)
    zero_f32 = S.convert(0.0, S.f32)

    a_lane_words = S.make_shared((THREADS, 4), S.u32)
    b_lane_words = S.make_shared((THREADS, 4), S.u32)
    act_tile = S.make_shared((BLOCK_M, BLOCK_N), S.f32)
    row_totals = S.make_shared((BLOCK_M,), S.f32)

    if tid < BLOCK_M:
        row_totals[tid] = zero_f32
    S.syncthreads()

    for n_iter in S.range(HIDDEN_SIZE // BLOCK_N):
        n_base = n_iter * BLOCK_N
        acc = S.full((16,), 0.0, S.f32)

        for k_iter in S.range(INPUT_SIZE // BLOCK_K):
            kk = k_iter * BLOCK_K
            x_row = block_row + warp_row * 32 + (lane % 32)
            x_pack_0_view = S.subview(X, (x_row, kk), (1, 8), (1, 1))
            x_pack_1_view = S.subview(X, (x_row, kk + 8), (1, 8), (1, 1))
            x_pack_0 = S.amdgpu.raw_buffer_load_x4(
                S.amdgpu.make_rsrc(x_pack_0_view, sixteen_i32), zero_i32, zero_i32, 0
            )
            x_pack_1 = S.amdgpu.raw_buffer_load_x4(
                S.amdgpu.make_rsrc(x_pack_1_view, sixteen_i32), zero_i32, zero_i32, 0
            )
            a_word_base = 2 * (lane // 32)
            a_lane_words[tid, 0] = x_pack_0[a_word_base]
            a_lane_words[tid, 1] = x_pack_0[a_word_base + 1]
            a_lane_words[tid, 2] = x_pack_1[a_word_base]
            a_lane_words[tid, 3] = x_pack_1[a_word_base + 1]

            b_col_base = n_base + warp_col * 32 + 4 * ((lane % 32) // 8)
            b_k = lane % 8
            w_pack_0_view = S.subview(W, (kk + b_k, b_col_base), (1, 8), (1, 1))
            w_pack_1_view = S.subview(W, (kk + 8 + b_k, b_col_base), (1, 8), (1, 1))
            w_pack_0 = S.amdgpu.raw_buffer_load_x4(
                S.amdgpu.make_rsrc(w_pack_0_view, sixteen_i32), zero_i32, zero_i32, 0
            )
            w_pack_1 = S.amdgpu.raw_buffer_load_x4(
                S.amdgpu.make_rsrc(w_pack_1_view, sixteen_i32), zero_i32, zero_i32, 0
            )
            b_lane_words[tid, 0] = w_pack_0[0]
            b_lane_words[tid, 1] = w_pack_0[1]
            b_lane_words[tid, 2] = w_pack_1[0]
            b_lane_words[tid, 3] = w_pack_1[1]

            S.syncthreads()

            a_frag = S.view(a_lane_words[tid], S.Tensor((2, 4, 1), S.bf16))
            b_frag = S.view(b_lane_words[tid], S.Tensor((2, 4, 1), S.bf16))

            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

            S.syncthreads()

        col = n_base + warp_col * 32 + (lane % 32)
        row_lane_base = warp_row * 32 + 4 * (lane // 32)
        for acc_idx in S.range(16):
            row = block_row + row_lane_base + 8 * (acc_idx // 4) + (acc_idx % 4)
            row_local = row_lane_base + 8 * (acc_idx // 4) + (acc_idx % 4)
            z = acc[acc_idx] + S.convert(BIAS0[col], S.f32)
            value = one / (one + S.exp(-z))
            act_tile[row_local, warp_col * 32 + (lane % 32)] = value

        S.syncthreads()

        if tid < BLOCK_M:
            total = row_totals[tid]
            for c in S.range(BLOCK_N):
                total += act_tile[tid, c]
            row_totals[tid] = total
        S.syncthreads()

    if tid < BLOCK_M:
        Y[block_row + tid, 0] = S.convert(row_totals[tid], S.bf16)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, INPUT_SIZE) or x.dtype != torch.bfloat16:
            raise RuntimeError("This fused kernel only supports the benchmark input shape and dtype.")
        w_t = self.linear.weight.t().to(device=x.device, dtype=x.dtype).contiguous()
        bias = self.linear.bias.to(device=x.device, dtype=x.dtype).contiguous()
        y = torch.empty((BATCH_SIZE, 1), device=x.device, dtype=x.dtype)
        fused_kernel[_launch](x.contiguous(), w_t, bias, y)
        return y
