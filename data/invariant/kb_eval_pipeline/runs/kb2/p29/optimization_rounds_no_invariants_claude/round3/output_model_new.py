import torch
import torch.nn as nn
import substrate
import substrate.language as S

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192

MFMA_M = 32
MFMA_N = 32
MFMA_K = 8
WAVE_SIZE = 64

BLOCK_M = 64
BLOCK_N = 64
THREADS = 256  # 4 waves x 64 lanes

K_TILE = 16  # 2 x MFMA_K
EFFECTIVE_K = 32  # 2 x K_TILE (unroll K-loop by 2)
NUM_ITERS = IN_FEATURES // EFFECTIVE_K  # 256

# Byte sizes for range calculation
X_U32_COLS = IN_FEATURES // 2  # 4096
X_U32_ROW_STRIDE = X_U32_COLS * 4  # 16384 bytes per row
X_U32_RANGE = BATCH_SIZE * X_U32_ROW_STRIDE  # 16777216 bytes total

W_COLS = OUT_FEATURES  # 8192
W_ROW_STRIDE = W_COLS * 4  # 32768 bytes per row
W_RANGE = (IN_FEATURES // 2) * W_ROW_STRIDE  # 134217728 bytes total

Y_COLS = OUT_FEATURES  # 8192
Y_ROW_STRIDE = Y_COLS * 2  # 16384 bytes per row (bf16)
Y_RANGE = BATCH_SIZE * Y_ROW_STRIDE  # 16777216 bytes total


@substrate.jit
def mfma_gemm_mish_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((IN_FEATURES // 2, OUT_FEATURES), S.u32),
    BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    bx = S.block_id(0)
    by = S.block_id(1)
    tid = S.thread_id(0)

    wave_id = tid // WAVE_SIZE
    wave_row = wave_id // 2
    wave_col = wave_id % 2
    lane = tid % WAVE_SIZE

    wave_m = by * BLOCK_M + wave_row * MFMA_M
    wave_n = bx * BLOCK_N + wave_col * MFMA_N

    # View X bf16 as u32 (packs adjacent bf16 columns into u32)
    X_u32 = S.view(X, S.Tensor((BATCH_SIZE, IN_FEATURES // 2), S.u32))

    # Create resource descriptors with range for OOB handling
    rsrc_X = S.amdgpu.make_rsrc(X_u32, X_U32_RANGE)
    rsrc_W = S.amdgpu.make_rsrc(W, W_RANGE)
    rsrc_Y = S.amdgpu.make_rsrc(Y, Y_RANGE)

    # Double-buffered shared memory for A and B (split access)
    A_s0 = S.make_shared((THREADS, 2), S.u32)
    A_s1 = S.make_shared((THREADS, 2), S.u32)
    B_s0 = S.make_shared((THREADS, 2), S.u32)
    B_s1 = S.make_shared((THREADS, 2), S.u32)

    # Accumulator
    c_acc = S.full((16,), 0.0, S.f32)

    # Precompute constant addressing offsets
    a_row = wave_m + lane % 32
    b_col = wave_n + lane % 32
    lane_k_off = (lane // 32) * 4  # 0 or 4

    # Row-stride byte offsets
    a_row_byte_off = a_row * X_U32_ROW_STRIDE
    b_col_byte_off = b_col * 4  # Each column is 4 bytes (u32)

    # ---- Prologue: prefetch k_half=0 of first k_tile into buf 0 ----
    k_off = lane_k_off
    a_byte_off = a_row_byte_off + k_off * 2  # k_off // 2 * 4 = k_off * 2
    b_byte_off = k_off * 2 + b_col_byte_off
    loaded_a = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_byte_off, 0, 0)
    loaded_b = S.amdgpu.raw_buffer_load_x2(rsrc_W, b_byte_off, 0, 0)
    A_s0[tid, 0] = loaded_a[0]
    A_s0[tid, 1] = loaded_a[1]
    B_s0[tid, 0] = loaded_b[0]
    B_s0[tid, 1] = loaded_b[1]
    S.syncthreads()

    # ---- Main pipeline loop: 2 k_tiles (4 MFMA) per iteration ----
    for k_iter in S.range(NUM_ITERS):
        k_base = k_iter * EFFECTIVE_K

        # ==== Step 0: MFMA on buf 0 (data from prologue / prev prefetch) ====
        m_a = S.view(A_s0[tid], S.Tensor((1, 4, 1), S.bf16))
        m_b = S.view(B_s0[tid], S.Tensor((1, 4, 1), S.bf16))
        c_acc = S.amdgpu.mfma_32x32x8_bf16_f32(m_a[0], m_b[0], c_acc)

        # ==== Load k_half=1 of k_tile 0 into buf 1 (overlaps with MFMA 0) ====
        k_off = k_base + MFMA_K + lane_k_off
        a_byte_off = a_row_byte_off + k_off * 2
        b_byte_off = k_off * 2 + b_col_byte_off
        loaded_a = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_byte_off, 0, 0)
        loaded_b = S.amdgpu.raw_buffer_load_x2(rsrc_W, b_byte_off, 0, 0)
        A_s1[tid, 0] = loaded_a[0]
        A_s1[tid, 1] = loaded_a[1]
        B_s1[tid, 0] = loaded_b[0]
        B_s1[tid, 1] = loaded_b[1]
        S.syncthreads()

        # ==== Step 1: MFMA on buf 1 ====
        m_a = S.view(A_s1[tid], S.Tensor((1, 4, 1), S.bf16))
        m_b = S.view(B_s1[tid], S.Tensor((1, 4, 1), S.bf16))
        c_acc = S.amdgpu.mfma_32x32x8_bf16_f32(m_a[0], m_b[0], c_acc)

        # ==== Load k_half=0 of k_tile 1 into buf 0 (overlaps with MFMA 1) ====
        k_off = k_base + K_TILE + lane_k_off
        a_byte_off = a_row_byte_off + k_off * 2
        b_byte_off = k_off * 2 + b_col_byte_off
        loaded_a = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_byte_off, 0, 0)
        loaded_b = S.amdgpu.raw_buffer_load_x2(rsrc_W, b_byte_off, 0, 0)
        A_s0[tid, 0] = loaded_a[0]
        A_s0[tid, 1] = loaded_a[1]
        B_s0[tid, 0] = loaded_b[0]
        B_s0[tid, 1] = loaded_b[1]
        S.syncthreads()

        # ==== Step 2: MFMA on buf 0 ====
        m_a = S.view(A_s0[tid], S.Tensor((1, 4, 1), S.bf16))
        m_b = S.view(B_s0[tid], S.Tensor((1, 4, 1), S.bf16))
        c_acc = S.amdgpu.mfma_32x32x8_bf16_f32(m_a[0], m_b[0], c_acc)

        # ==== Load k_half=1 of k_tile 1 into buf 1 (overlaps with MFMA 2) ====
        k_off = k_base + K_TILE + MFMA_K + lane_k_off
        a_byte_off = a_row_byte_off + k_off * 2
        b_byte_off = k_off * 2 + b_col_byte_off
        loaded_a = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_byte_off, 0, 0)
        loaded_b = S.amdgpu.raw_buffer_load_x2(rsrc_W, b_byte_off, 0, 0)
        A_s1[tid, 0] = loaded_a[0]
        A_s1[tid, 1] = loaded_a[1]
        B_s1[tid, 0] = loaded_b[0]
        B_s1[tid, 1] = loaded_b[1]
        S.syncthreads()

        # ==== Step 3: MFMA on buf 1 ====
        m_a = S.view(A_s1[tid], S.Tensor((1, 4, 1), S.bf16))
        m_b = S.view(B_s1[tid], S.Tensor((1, 4, 1), S.bf16))
        c_acc = S.amdgpu.mfma_32x32x8_bf16_f32(m_a[0], m_b[0], c_acc)

        # ==== Prefetch k_half=0 of next iteration into buf 0 ====
        # No OOB branch needed - raw_buffer_load_x2 returns 0 for OOB
        k_off = (k_iter + 1) * EFFECTIVE_K + lane_k_off
        a_byte_off = a_row_byte_off + k_off * 2
        b_byte_off = k_off * 2 + b_col_byte_off
        loaded_a = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_byte_off, 0, 0)
        loaded_b = S.amdgpu.raw_buffer_load_x2(rsrc_W, b_byte_off, 0, 0)
        A_s0[tid, 0] = loaded_a[0]
        A_s0[tid, 1] = loaded_a[1]
        B_s0[tid, 0] = loaded_b[0]
        B_s0[tid, 1] = loaded_b[1]
        S.syncthreads()

    # Store output with bias and double Mish
    for i in S.range(16):
        out_col = wave_n + lane % 32
        out_row = wave_m + 8 * (i // 4) + 4 * (lane // 32) + (i % 4)

        # Compute byte offset for Y store (bf16 = 2 bytes)
        y_byte_off = out_row * Y_ROW_STRIDE + out_col * 2

        val = c_acc[i] + S.convert(BIAS[out_col], S.f32)
        s1 = S.log(S.convert(1.0, S.f32) + S.exp(val))
        val = val * S.tanh(s1)
        s2 = S.log(S.convert(1.0, S.f32) + S.exp(val))
        val = val * S.tanh(s2)
        val_bf16 = S.convert(val, S.bf16)
        # Use raw_buffer_store_x1 with range - OOB writes are discarded
        S.amdgpu.raw_buffer_store_x1(S.bitcast(val_bf16, S.u16), rsrc_Y, y_byte_off, 0, 0)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._w_ptr = None
        self._b_ptr = None
        self._w_packed = None
        self._cached_bias = None

    def _pack_w_along_k(self, w_bf16):
        """Pack bf16 weight along K (row) dimension: (K, N) bf16 -> (K//2, N) u32."""
        w_u16 = w_bf16.view(torch.int16)
        lo = w_u16[0::2, :].to(torch.int32) & 0xFFFF
        hi = (w_u16[1::2, :].to(torch.int32) & 0xFFFF) << 16
        return (lo | hi).to(torch.int32).contiguous()

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES) or x.dtype != torch.bfloat16:
            raise RuntimeError('This kernel only supports (1024, 8192) bf16 input.')

        w_t = self.linear.weight.t().contiguous().to(x.dtype)
        bias = self.linear.bias.contiguous().to(x.dtype)

        if self._w_ptr != w_t.data_ptr() or self._b_ptr != bias.data_ptr():
            self._w_packed = self._pack_w_along_k(w_t)
            self._cached_bias = bias
            self._w_ptr = w_t.data_ptr()
            self._b_ptr = bias.data_ptr()

        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=x.dtype)

        grid_n = OUT_FEATURES // BLOCK_N
        grid_m = BATCH_SIZE // BLOCK_M

        def launch():
            return ((grid_n, grid_m, 1), (THREADS, 1, 1))

        mfma_gemm_mish_kernel[launch](x.contiguous(), self._w_packed, self._cached_bias, y)
        return y
