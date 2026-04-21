import torch
import torch.nn as nn
import substrate
import substrate.language as S

BATCH_SIZE = 128
INPUT_SIZE = 32768
HIDDEN_SIZE = 32768

WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS

TILE_M = 32
TILE_N = 32
TILE_K = 16  # 2 x mfma_32x32x8

BLOCKS_M = 2
M_PER_BLOCK = BATCH_SIZE // BLOCKS_M  # 64

K_TILES = INPUT_SIZE // TILE_K  # 2048
NUM_UNROLLED = K_TILES // 2  # 1024


def _launch():
    return ((BLOCKS_M, 1, 1), (THREADS, 1, 1))


@substrate.jit
def fused_kernel(
    X: S.Tensor((BATCH_SIZE, INPUT_SIZE), S.bf16),
    W: S.Tensor((INPUT_SIZE, HIDDEN_SIZE), S.bf16),
    BIAS0: S.Tensor((HIDDEN_SIZE,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, 1), S.bf16),
):
    tid = S.thread_id(0)
    warp_id = tid // WARP_SIZE
    lane = tid - warp_id * WARP_SIZE

    warp_row = warp_id // 2
    warp_col = warp_id - warp_row * 2

    block_m = S.block_id(0)
    m_base = block_m * M_PER_BLOCK + warp_row * TILE_M

    # range in make_rsrc: OOB loads return 0, OOB stores discarded
    # This eliminates need for explicit OOB branches in the loop
    rsrc_X = S.amdgpu.make_rsrc(X, BATCH_SIZE * INPUT_SIZE * 2)
    rsrc_W = S.amdgpu.make_rsrc(W, INPUT_SIZE * HIDDEN_SIZE * 2)

    # Double-buffered LDS for A and B
    lds_A0 = S.make_shared((NUM_WARPS, TILE_M, TILE_K), S.bf16)
    lds_A1 = S.make_shared((NUM_WARPS, TILE_M, TILE_K), S.bf16)
    lds_B0 = S.make_shared((NUM_WARPS, TILE_K, TILE_N), S.bf16)
    lds_B1 = S.make_shared((NUM_WARPS, TILE_K, TILE_N), S.bf16)
    lds_C = S.make_shared((NUM_WARPS, 32, 32), S.f32)
    lds_red = S.make_shared((2, 2, TILE_M), S.f32)

    one = S.convert(1.0, S.f32)
    sigmoid_sum = S.convert(0.0, S.f32)

    is_even_lane = lane - (lane // 2) * 2
    my_row = lane // 2

    # LDS load indices
    a_row_l = lane // 2
    a_col_l = (lane - a_row_l * 2) * 8
    b_row_l = lane // 4
    b_col_l = (lane - b_row_l * 4) * 8

    # MFMA fragment read indices
    a_frag_row = lane // 2
    a_frag_col = (lane - a_frag_row * 2) * 4
    b_frag_row = lane // 4
    b_frag_col = (lane - b_frag_row * 4) * 8

    N_TILES = HIDDEN_SIZE // (2 * TILE_N)

    for nt in S.range(N_TILES):
        n_base = nt * (2 * TILE_N) + warp_col * TILE_N
        c_lane = S.full((16,), 0.0, S.f32)

        # ---- Prologue: load kt=0 into buf0 ----
        # No OOB branch needed: range in rsrc_X returns 0 for OOB
        a_byte = (m_base + a_row_l) * INPUT_SIZE * 2 + (0 + a_col_l) * 2
        a_v = S.amdgpu.raw_buffer_load_x4(rsrc_X, a_byte, 0, 0)
        a_bf16 = S.view(a_v, S.Tensor((1, 8, 1), S.bf16))
        for ci in S.range(8):
            lds_A0[warp_id, a_row_l, a_col_l + ci] = a_bf16[0, ci, 0]

        b_byte = (0 + b_row_l) * HIDDEN_SIZE * 2 + (n_base + b_col_l) * 2
        b_v = S.amdgpu.raw_buffer_load_x4(rsrc_W, b_byte, 0, 0)
        b_bf16 = S.view(b_v, S.Tensor((1, 8, 1), S.bf16))
        for ci in S.range(8):
            lds_B0[warp_id, b_row_l, b_col_l + ci] = b_bf16[0, ci, 0]

        S.syncthreads()

        # ---- Software-pipelined K-loop, unrolled by 2 ----
        for ukt in S.range(NUM_UNROLLED):
            kt = ukt * 2

            # == Sub A: consume buf0, load kt+1 into buf1 ==
            # Read MFMA fragments directly via raw_buffer_load — no OOB branch
            # needed because range in rsrc handles out-of-bounds (returns 0)
            k_base_a = kt * TILE_K
            a_lo_byte = (m_base + a_frag_row) * INPUT_SIZE * 2 + (k_base_a + a_frag_col) * 2
            a_lo_raw = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_lo_byte, 0, 0)
            a_lo_vec = S.view(a_lo_raw, S.Tensor((1, 4, 1), S.bf16))[0]

            a_hi_byte = (m_base + a_frag_row) * INPUT_SIZE * 2 + (k_base_a + a_frag_col + 8) * 2
            a_hi_raw = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_hi_byte, 0, 0)
            a_hi_vec = S.view(a_hi_raw, S.Tensor((1, 4, 1), S.bf16))[0]

            b_lo_byte_g = (k_base_a + b_frag_row) * HIDDEN_SIZE * 2 + (n_base + b_frag_col) * 2
            b_lo_raw = S.amdgpu.raw_buffer_load_x2(rsrc_W, b_lo_byte_g, 0, 0)
            b_lo_vec = S.view(b_lo_raw, S.Tensor((1, 4, 1), S.bf16))[0]

            b_hi_row_g = b_frag_row + 8
            b_hi_byte_g = (k_base_a + b_hi_row_g) * HIDDEN_SIZE * 2 + (n_base + b_frag_col) * 2
            b_hi_raw = S.amdgpu.raw_buffer_load_x2(rsrc_W, b_hi_byte_g, 0, 0)
            b_hi_vec = S.view(b_hi_raw, S.Tensor((1, 4, 1), S.bf16))[0]

            # Issue MFMA
            c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_lo_vec, b_lo_vec, c_lane)
            c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_hi_vec, b_hi_vec, c_lane)

            # Load kt+1 into buf1 (overlaps with MFMA)
            kt1_k_base = (kt + 1) * TILE_K
            a_byte1 = (m_base + a_row_l) * INPUT_SIZE * 2 + (kt1_k_base + a_col_l) * 2
            a_v1 = S.amdgpu.raw_buffer_load_x4(rsrc_X, a_byte1, 0, 0)
            a_bf16_1 = S.view(a_v1, S.Tensor((1, 8, 1), S.bf16))
            for ci in S.range(8):
                lds_A1[warp_id, a_row_l, a_col_l + ci] = a_bf16_1[0, ci, 0]

            b_byte1 = (kt1_k_base + b_row_l) * HIDDEN_SIZE * 2 + (n_base + b_col_l) * 2
            b_v1 = S.amdgpu.raw_buffer_load_x4(rsrc_W, b_byte1, 0, 0)
            b_bf16_1 = S.view(b_v1, S.Tensor((1, 8, 1), S.bf16))
            for ci in S.range(8):
                lds_B1[warp_id, b_row_l, b_col_l + ci] = b_bf16_1[0, ci, 0]

            S.syncthreads()

            # == Sub B: consume buf1, load kt+2 into buf0 ==
            k_base_b = (kt + 1) * TILE_K
            a_lo_byte2 = (m_base + a_frag_row) * INPUT_SIZE * 2 + (k_base_b + a_frag_col) * 2
            a_lo_raw2 = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_lo_byte2, 0, 0)
            a_lo_vec2 = S.view(a_lo_raw2, S.Tensor((1, 4, 1), S.bf16))[0]

            a_hi_byte2 = (m_base + a_frag_row) * INPUT_SIZE * 2 + (k_base_b + a_frag_col + 8) * 2
            a_hi_raw2 = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_hi_byte2, 0, 0)
            a_hi_vec2 = S.view(a_hi_raw2, S.Tensor((1, 4, 1), S.bf16))[0]

            b_lo_byte2 = (k_base_b + b_frag_row) * HIDDEN_SIZE * 2 + (n_base + b_frag_col) * 2
            b_lo_raw2 = S.amdgpu.raw_buffer_load_x2(rsrc_W, b_lo_byte2, 0, 0)
            b_lo_vec2 = S.view(b_lo_raw2, S.Tensor((1, 4, 1), S.bf16))[0]

            b_hi_byte2 = (k_base_b + b_frag_row + 8) * HIDDEN_SIZE * 2 + (n_base + b_frag_col) * 2
            b_hi_raw2 = S.amdgpu.raw_buffer_load_x2(rsrc_W, b_hi_byte2, 0, 0)
            b_hi_vec2 = S.view(b_hi_raw2, S.Tensor((1, 4, 1), S.bf16))[0]

            # Issue MFMA
            c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_lo_vec2, b_lo_vec2, c_lane)
            c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_hi_vec2, b_hi_vec2, c_lane)

            # Load kt+2 into buf0 for next iteration (overlaps with MFMA)
            # At last iteration, kt+2 goes OOB but range returns 0 — no branch needed
            kt2_k_base = (kt + 2) * TILE_K
            a_byte2n = (m_base + a_row_l) * INPUT_SIZE * 2 + (kt2_k_base + a_col_l) * 2
            a_v2n = S.amdgpu.raw_buffer_load_x4(rsrc_X, a_byte2n, 0, 0)
            a_bf16_2n = S.view(a_v2n, S.Tensor((1, 8, 1), S.bf16))
            for ci in S.range(8):
                lds_A0[warp_id, a_row_l, a_col_l + ci] = a_bf16_2n[0, ci, 0]

            b_byte2n = (kt2_k_base + b_row_l) * HIDDEN_SIZE * 2 + (n_base + b_col_l) * 2
            b_v2n = S.amdgpu.raw_buffer_load_x4(rsrc_W, b_byte2n, 0, 0)
            b_bf16_2n = S.view(b_v2n, S.Tensor((1, 8, 1), S.bf16))
            for ci in S.range(8):
                lds_B0[warp_id, b_row_l, b_col_l + ci] = b_bf16_2n[0, ci, 0]

            S.syncthreads()

        # Store MFMA output to LDS
        l_group = lane // 8
        l_sub = lane - l_group * 8
        for e in S.range(16):
            oi = l_group * 4 + e // 4
            oj = l_sub * 4 + e - (e // 4) * 4
            lds_C[warp_id, oi, oj] = c_lane[e]

        S.syncthreads()

        # Add bias, sigmoid, accumulate per-row
        my_col_start = is_even_lane * 16
        row_partial = S.convert(0.0, S.f32)
        for cc in S.range(16):
            col_idx = my_col_start + cc
            val = lds_C[warp_id, my_row, col_idx]
            bias_val = BIAS0[n_base + col_idx]
            val = val + S.convert(bias_val, S.f32)
            sig = one / (one + S.exp(-val))
            row_partial = row_partial + sig

        other_half = S.shuffle_xor(row_partial, 1, WARP_SIZE)
        row_total = row_partial + other_half

        if is_even_lane == 0:
            sigmoid_sum = sigmoid_sum + row_total

    # Cross-warp-col reduction
    if is_even_lane == 0:
        lds_red[warp_row, warp_col, my_row] = sigmoid_sum

    S.syncthreads()

    if is_even_lane == 0:
        if warp_col == 0:
            total = lds_red[warp_row, 0, my_row] + lds_red[warp_row, 1, my_row]
            batch_idx = m_base + my_row
            Y[batch_idx, 0] = S.convert(total, S.bf16)


class ModelNew(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, INPUT_SIZE) or x.dtype != torch.bfloat16:
            raise RuntimeError('This fused kernel only supports the benchmark input shape and dtype.')
        w_t = self.linear.weight.t().to(device=x.device, dtype=x.dtype).contiguous()
        bias = self.linear.bias.to(device=x.device, dtype=x.dtype).contiguous()
        y = torch.empty((BATCH_SIZE, 1), device=x.device, dtype=x.dtype)
        fused_kernel[_launch](x.contiguous(), w_t, bias, y)
        return y
