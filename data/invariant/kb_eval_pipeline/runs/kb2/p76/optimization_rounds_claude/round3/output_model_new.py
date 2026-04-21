import torch
import torch.nn as nn
import substrate
import substrate.language as S

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192

TILE_M = 32
TILE_N = 32
WAVES_M = 2
WAVES_N = 2
BLOCK_M = TILE_M * WAVES_M  # 64
BLOCK_N = TILE_N * WAVES_N  # 64
NUM_BLOCKS_M = BATCH_SIZE // BLOCK_M  # 16
NUM_BLOCKS_N = OUT_FEATURES // BLOCK_N  # 128

K_TILES = IN_FEATURES // 16  # 512
K_SUPER = K_TILES // 2  # 256 (unrolled by 2)

def _launch():
    return ((NUM_BLOCKS_M * NUM_BLOCKS_N, 1, 1), (WAVES_M * WAVES_N * 64, 1, 1))

@substrate.jit
def fused_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES // 4, 2), S.u32),
    W: S.Tensor((OUT_FEATURES, IN_FEATURES // 4, 2), S.u32),
    EXTRA_BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    lane = tid % 64
    warp_id = tid // 64
    warp_row = warp_id // 2
    warp_col = warp_id % 2

    block_m = bid // NUM_BLOCKS_N
    block_n = bid % NUM_BLOCKS_N

    m0 = block_m * BLOCK_M + warp_row * TILE_M
    n0 = block_n * BLOCK_N + warp_col * TILE_N

    acc = S.full((16,), 0.0, S.f32)

    x_total_bytes = BATCH_SIZE * IN_FEATURES * 2
    w_total_bytes = OUT_FEATURES * IN_FEATURES * 2
    rsrc_X = S.amdgpu.make_rsrc(X, x_total_bytes)
    rsrc_W = S.amdgpu.make_rsrc(W, w_total_bytes)

    # Double-buffered LDS: (buf, warp, row, k_blk, u32)
    A_lds = S.make_shared((2, 4, 32, 4, 2), S.u32)
    B_lds = S.make_shared((2, 4, 32, 4, 2), S.u32)

    a_row = lane % 32
    a_blk = lane // 32
    b_row = lane % 32
    b_blk = lane // 32

    x_row_stride = IN_FEATURES * 2
    w_row_stride = IN_FEATURES * 2

    ld_row_a = lane // 2
    ld_kpair = (lane % 2) * 2

    # ---- Prologue: load kt=0 into buf 0 ----
    a_off_p = (m0 + ld_row_a) * x_row_stride + (0 + ld_kpair) * 8
    a_data_p = S.amdgpu.raw_buffer_load_x4(rsrc_X, a_off_p, 0, x_total_bytes)
    A_lds[0, warp_id, ld_row_a, ld_kpair + 0, 0] = a_data_p[0]
    A_lds[0, warp_id, ld_row_a, ld_kpair + 0, 1] = a_data_p[1]
    A_lds[0, warp_id, ld_row_a, ld_kpair + 1, 0] = a_data_p[2]
    A_lds[0, warp_id, ld_row_a, ld_kpair + 1, 1] = a_data_p[3]

    b_off_p = (n0 + ld_row_a) * w_row_stride + (0 + ld_kpair) * 8
    b_data_p = S.amdgpu.raw_buffer_load_x4(rsrc_W, b_off_p, 0, w_total_bytes)
    B_lds[0, warp_id, ld_row_a, ld_kpair + 0, 0] = b_data_p[0]
    B_lds[0, warp_id, ld_row_a, ld_kpair + 0, 1] = b_data_p[1]
    B_lds[0, warp_id, ld_row_a, ld_kpair + 1, 0] = b_data_p[2]
    B_lds[0, warp_id, ld_row_a, ld_kpair + 1, 1] = b_data_p[3]

    S.syncthreads()

    # ---- Main loop: K-unrolled by 2, double-buffered, software pipelined ----
    # Full K_SUPER iterations; last iteration's OOB loads return 0 (range set)
    for ks in S.range(K_SUPER):
        kt0 = ks * 2
        kt1 = kt0 + 1

        # == Stage 1: Issue loads for kt1, then compute kt0 from buf 0 ==

        # Issue global loads for kt1 first (latency hidden behind MFMA)
        k_blk_1 = kt1 * 4
        a_off1 = (m0 + ld_row_a) * x_row_stride + (k_blk_1 + ld_kpair) * 8
        a_data1 = S.amdgpu.raw_buffer_load_x4(rsrc_X, a_off1, 0, x_total_bytes)
        b_off1 = (n0 + ld_row_a) * w_row_stride + (k_blk_1 + ld_kpair) * 8
        b_data1 = S.amdgpu.raw_buffer_load_x4(rsrc_W, b_off1, 0, w_total_bytes)

        # Split LDS read 0 + MFMA 0 (overlap with global load)
        a_s0a = A_lds[0, warp_id, a_row, a_blk]
        b_s0a = B_lds[0, warp_id, b_row, b_blk]
        a_v0a = S.view(a_s0a, S.Tensor((1, 4, 1), S.bf16))
        b_v0a = S.view(b_s0a, S.Tensor((1, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_v0a[0], b_v0a[0], acc)

        # Split LDS read 1 + MFMA 1
        a_s1a = A_lds[0, warp_id, a_row, 2 + a_blk]
        b_s1a = B_lds[0, warp_id, b_row, 2 + b_blk]
        a_v1a = S.view(a_s1a, S.Tensor((1, 4, 1), S.bf16))
        b_v1a = S.view(b_s1a, S.Tensor((1, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_v1a[0], b_v1a[0], acc)

        # Store kt1 data into buf 1
        A_lds[1, warp_id, ld_row_a, ld_kpair + 0, 0] = a_data1[0]
        A_lds[1, warp_id, ld_row_a, ld_kpair + 0, 1] = a_data1[1]
        A_lds[1, warp_id, ld_row_a, ld_kpair + 1, 0] = a_data1[2]
        A_lds[1, warp_id, ld_row_a, ld_kpair + 1, 1] = a_data1[3]
        B_lds[1, warp_id, ld_row_a, ld_kpair + 0, 0] = b_data1[0]
        B_lds[1, warp_id, ld_row_a, ld_kpair + 0, 1] = b_data1[1]
        B_lds[1, warp_id, ld_row_a, ld_kpair + 1, 0] = b_data1[2]
        B_lds[1, warp_id, ld_row_a, ld_kpair + 1, 1] = b_data1[3]

        S.syncthreads()

        # == Stage 2: Issue loads for next kt0, then compute kt1 from buf 1 ==

        k_blk_n = (kt0 + 2) * 4
        a_off_n = (m0 + ld_row_a) * x_row_stride + (k_blk_n + ld_kpair) * 8
        a_data_n = S.amdgpu.raw_buffer_load_x4(rsrc_X, a_off_n, 0, x_total_bytes)
        b_off_n = (n0 + ld_row_a) * w_row_stride + (k_blk_n + ld_kpair) * 8
        b_data_n = S.amdgpu.raw_buffer_load_x4(rsrc_W, b_off_n, 0, w_total_bytes)

        # Split LDS read 0 + MFMA 0 for kt1
        a_s0b = A_lds[1, warp_id, a_row, a_blk]
        b_s0b = B_lds[1, warp_id, b_row, b_blk]
        a_v0b = S.view(a_s0b, S.Tensor((1, 4, 1), S.bf16))
        b_v0b = S.view(b_s0b, S.Tensor((1, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_v0b[0], b_v0b[0], acc)

        # Split LDS read 1 + MFMA 1 for kt1
        a_s1b = A_lds[1, warp_id, a_row, 2 + a_blk]
        b_s1b = B_lds[1, warp_id, b_row, 2 + b_blk]
        a_v1b = S.view(a_s1b, S.Tensor((1, 4, 1), S.bf16))
        b_v1b = S.view(b_s1b, S.Tensor((1, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_v1b[0], b_v1b[0], acc)

        # Store next kt0 data into buf 0
        A_lds[0, warp_id, ld_row_a, ld_kpair + 0, 0] = a_data_n[0]
        A_lds[0, warp_id, ld_row_a, ld_kpair + 0, 1] = a_data_n[1]
        A_lds[0, warp_id, ld_row_a, ld_kpair + 1, 0] = a_data_n[2]
        A_lds[0, warp_id, ld_row_a, ld_kpair + 1, 1] = a_data_n[3]
        B_lds[0, warp_id, ld_row_a, ld_kpair + 0, 0] = b_data_n[0]
        B_lds[0, warp_id, ld_row_a, ld_kpair + 0, 1] = b_data_n[1]
        B_lds[0, warp_id, ld_row_a, ld_kpair + 1, 0] = b_data_n[2]
        B_lds[0, warp_id, ld_row_a, ld_kpair + 1, 1] = b_data_n[3]

        S.syncthreads()

    # Write results using accumulator invariant
    for ai in S.range(16):
        col = n0 + lane % 32
        row = m0 + 8 * (ai // 4) + 4 * (lane // 32) + ai % 4

        val = acc[ai]
        val = val + S.convert(EXTRA_BIAS[col], S.f32)
        neg = val < S.convert(0.0, S.f32)
        if neg:
            val = S.convert(0.0, S.f32)
        Y[row, col] = S.convert(val, S.bf16)


class ModelNew(nn.Module):

    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._x_u32 = None
        self._x_ptr = None
        self._w_u32 = None
        self._w_ptr = None

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES) or x.dtype != torch.bfloat16 or tuple(self.bias.shape) != (OUT_FEATURES,):
            raise RuntimeError('This fused kernel only supports the benchmark input shape and dtype.')
        w = self.gemm.weight.to(device=x.device, dtype=x.dtype).contiguous()
        extra_bias = self.bias.to(device=x.device, dtype=x.dtype).contiguous()

        xc = x.contiguous()
        wc = w.contiguous()

        x_ptr = xc.data_ptr()
        if self._x_u32 is None or self._x_ptr != x_ptr:
            self._x_u32 = xc.flatten().view(torch.int32).reshape(BATCH_SIZE, IN_FEATURES // 4, 2)
            self._x_ptr = x_ptr

        w_ptr = wc.data_ptr()
        if self._w_u32 is None or self._w_ptr != w_ptr:
            self._w_u32 = wc.flatten().view(torch.int32).reshape(OUT_FEATURES, IN_FEATURES // 4, 2)
            self._w_ptr = w_ptr

        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=x.dtype)
        fused_kernel[_launch](self._x_u32, self._w_u32, extra_bias, y)
        return y
