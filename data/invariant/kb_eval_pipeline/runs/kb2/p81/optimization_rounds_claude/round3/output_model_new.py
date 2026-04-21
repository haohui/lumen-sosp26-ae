import torch
import torch.nn as nn
import substrate
import substrate.language as S

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192

# MFMA tile dimensions
TM = 32
TN = 32
TK_MFMA = 8
TK_TILE = 16
WAVES_M = 2
WAVES_N = 2
WAVES = WAVES_M * WAVES_N
LANES = 64
THREADS = WAVES * LANES
WG_M = BATCH_SIZE // (WAVES_M * TM)
WG_N = OUT_FEATURES // (WAVES_N * TN)

# Packed dimensions: each u32 = 2 bf16, so (N//4, 2) u32 = N bf16
IN_U32_PAIRS = IN_FEATURES // 4  # pairs of u32 per row
N_K = IN_FEATURES // TK_TILE  # 512

# Full-tile LDS: 4 u32 per thread per buffer (both MFMA halves)
LDS_ELEMS = THREADS * 4  # u32 per buffer
LDS_DBUF = 2 * LDS_ELEMS  # double-buffer total

# Byte sizes for range in make_rsrc
X_TOTAL_BYTES = BATCH_SIZE * IN_U32_PAIRS * 2 * 4
W_TOTAL_BYTES = OUT_FEATURES * IN_U32_PAIRS * 2 * 4

# Byte strides
ROW_BYTE_STRIDE = IN_U32_PAIRS * 8  # bytes per row of X or W


def _launch():
    return ((WG_M * WG_N, 1, 1), (THREADS, 1, 1))


@substrate.jit
def fused_kernel(
    X: S.Tensor((BATCH_SIZE, IN_U32_PAIRS, 2), S.u32),
    W: S.Tensor((OUT_FEATURES, IN_U32_PAIRS, 2), S.u32),
    BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % LANES
    warp_id = tid // LANES
    warp_row = warp_id // WAVES_N
    warp_col = warp_id % WAVES_N

    wg = S.block_id(0)
    wg_m = wg // WG_N
    wg_n = wg % WG_N

    m_base = wg_m * (WAVES_M * TM) + warp_row * TM
    n_base = wg_n * (WAVES_N * TN) + warp_col * TN

    # Accumulator: 16 f32 per lane
    acc = S.full((16,), 0.0, S.f32)

    # MFMA swizzle: lane L holds row L%32, K-group L//32
    lane_m = lane % 32
    lane_kg = lane // 32

    # Create buffer resource descriptors with range for OOB handling
    rsrc_X = S.amdgpu.make_rsrc(X, X_TOTAL_BYTES)
    rsrc_W = S.amdgpu.make_rsrc(W, W_TOTAL_BYTES)

    # Double-buffered LDS: viewed as (2, THREADS, 4) for slice access
    lds_A_raw = S.make_shared((LDS_DBUF,), S.u32)
    lds_B_raw = S.make_shared((LDS_DBUF,), S.u32)
    lds_A = S.view(lds_A_raw, S.Tensor((2, THREADS, 4), S.u32))
    lds_B = S.view(lds_B_raw, S.Tensor((2, THREADS, 4), S.u32))

    # Precompute base byte offsets for this thread's row
    a_row_offset = (m_base + lane_m) * ROW_BYTE_STRIDE
    b_row_offset = (n_base + lane_m) * ROW_BYTE_STRIDE

    # --- Prologue: load K=0 full tile into buf0 using raw_buffer_load_x2 ---
    kh0 = lane_kg
    kh1 = lane_kg + 2
    val_a0 = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_row_offset + kh0 * 8, 0, 0)
    val_a1 = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_row_offset + kh1 * 8, 0, 0)
    val_b0 = S.amdgpu.raw_buffer_load_x2(rsrc_W, b_row_offset + kh0 * 8, 0, 0)
    val_b1 = S.amdgpu.raw_buffer_load_x2(rsrc_W, b_row_offset + kh1 * 8, 0, 0)
    lds_A[0, tid, 0] = val_a0[0]
    lds_A[0, tid, 1] = val_a0[1]
    lds_A[0, tid, 2] = val_a1[0]
    lds_A[0, tid, 3] = val_a1[1]
    lds_B[0, tid, 0] = val_b0[0]
    lds_B[0, tid, 1] = val_b0[1]
    lds_B[0, tid, 2] = val_b1[0]
    lds_B[0, tid, 3] = val_b1[1]
    S.syncthreads()

    # --- Main loop: unrolled by 2 K steps per iteration ---
    for k_outer in S.range(N_K // 2):
        k_step0 = 2 * k_outer
        k_step1 = 2 * k_outer + 1

        # == Read buf0 -> 2 MFMA (loaded in previous iteration) ==
        a_full = lds_A[0, tid]
        b_full = lds_B[0, tid]
        a_v = S.view(a_full, S.Tensor((2, 4, 1), S.bf16))
        b_v = S.view(b_full, S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_v[0], b_v[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_v[1], b_v[1], acc)

        # Load k_step1 full tile into buf1 using raw_buffer_load_x2
        kh0_1 = k_step1 * (TK_TILE // 4) + lane_kg
        kh1_1 = k_step1 * (TK_TILE // 4) + lane_kg + 2
        val_a0 = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_row_offset + kh0_1 * 8, 0, 0)
        val_a1 = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_row_offset + kh1_1 * 8, 0, 0)
        val_b0 = S.amdgpu.raw_buffer_load_x2(rsrc_W, b_row_offset + kh0_1 * 8, 0, 0)
        val_b1 = S.amdgpu.raw_buffer_load_x2(rsrc_W, b_row_offset + kh1_1 * 8, 0, 0)
        lds_A[1, tid, 0] = val_a0[0]
        lds_A[1, tid, 1] = val_a0[1]
        lds_A[1, tid, 2] = val_a1[0]
        lds_A[1, tid, 3] = val_a1[1]
        lds_B[1, tid, 0] = val_b0[0]
        lds_B[1, tid, 1] = val_b0[1]
        lds_B[1, tid, 2] = val_b1[0]
        lds_B[1, tid, 3] = val_b1[1]

        # == Read buf1 -> 2 MFMA ==
        a_full = lds_A[1, tid]
        b_full = lds_B[1, tid]
        a_v = S.view(a_full, S.Tensor((2, 4, 1), S.bf16))
        b_v = S.view(b_full, S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_v[0], b_v[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_v[1], b_v[1], acc)

        # Prefetch next outer iteration into buf0
        # No modular wrap needed: raw_buffer_load returns 0 for OOB via range
        next_k = k_step1 + 1
        nkh0 = next_k * (TK_TILE // 4) + lane_kg
        nkh1 = next_k * (TK_TILE // 4) + lane_kg + 2
        val_a0 = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_row_offset + nkh0 * 8, 0, 0)
        val_a1 = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_row_offset + nkh1 * 8, 0, 0)
        val_b0 = S.amdgpu.raw_buffer_load_x2(rsrc_W, b_row_offset + nkh0 * 8, 0, 0)
        val_b1 = S.amdgpu.raw_buffer_load_x2(rsrc_W, b_row_offset + nkh1 * 8, 0, 0)
        lds_A[0, tid, 0] = val_a0[0]
        lds_A[0, tid, 1] = val_a0[1]
        lds_A[0, tid, 2] = val_a1[0]
        lds_A[0, tid, 3] = val_a1[1]
        lds_B[0, tid, 0] = val_b0[0]
        lds_B[0, tid, 1] = val_b0[1]
        lds_B[0, tid, 2] = val_b1[0]
        lds_B[0, tid, 3] = val_b1[1]

    # Unpack accumulator using prescribed invariant:
    #   col = n_base + (lane % 32)
    #   row = m_base + 8*(ai//4) + 4*(lane//32) + (ai%4)
    n_col = n_base + (lane % 32)
    one = S.convert(1.0, S.f32)
    two = S.convert(2.0, S.f32)
    neg_one = S.convert(-1.0, S.f32)

    for ai in S.range(16):
        m_row = m_base + 8 * (ai // 4) + 4 * (lane // 32) + (ai % 4)
        x = acc[ai]
        x = x + S.convert(BIAS[n_col], S.f32)
        x = x * (one / (one + S.exp(-x)))
        x = x / two
        if x < neg_one:
            x = neg_one
        if x > one:
            x = one
        x = S.tanh(x)
        if x < neg_one:
            x = neg_one
        if x > one:
            x = one
        Y[m_row, n_col] = S.convert(x, S.bf16)


class ModelNew(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self._w_u32 = None
        self._w_ptr = None
        self._bias = None
        self._bias_ptr = None

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES) or x.dtype != torch.bfloat16:
            raise RuntimeError(
                "This fused kernel only supports the benchmark input shape and dtype."
            )
        x = x.contiguous()
        # Reinterpret bf16 bytes as i32, reshape to (B, IN_FEATURES//4, 2)
        x_u32 = x.view(torch.int32).reshape(BATCH_SIZE, IN_U32_PAIRS, 2)

        # Weight in (N, K) layout for contiguous B-fragment access
        w = self.gemm.weight.to(device=x.device, dtype=torch.bfloat16).contiguous()
        if self._w_u32 is None or self._w_ptr != w.data_ptr():
            self._w_u32 = w.view(torch.int32).reshape(OUT_FEATURES, IN_U32_PAIRS, 2)
            self._w_ptr = w.data_ptr()

        bias = self.gemm.bias.to(device=x.device, dtype=torch.bfloat16).contiguous()
        if self._bias is None or self._bias_ptr != bias.data_ptr():
            self._bias = bias
            self._bias_ptr = bias.data_ptr()

        y = torch.empty(
            (BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=torch.bfloat16
        )
        fused_kernel[_launch](x_u32, self._w_u32, self._bias, y)
        return y
