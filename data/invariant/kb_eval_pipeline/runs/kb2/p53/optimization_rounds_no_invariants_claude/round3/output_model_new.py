import torch
import torch.nn as nn
import substrate
import substrate.language as S
import math

SQRT_2 = 1.4142135623730951

BATCH_SIZE = 2048
IN_FEATURES = 8192
OUT_FEATURES = 8192
SCALING_FACTOR = 0.5
HARDTANH_MIN = -2.0
HARDTANH_MAX = 2.0

# Tile sizes
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16  # Double buffering: 2 x 8, enables K-loop unroll by 2
THREADS = 256

BF16_SIZE = 2  # bytes per bf16 element


def _launch():
    grid_m = (BATCH_SIZE + BLOCK_M - 1) // BLOCK_M
    grid_n = (OUT_FEATURES + BLOCK_N - 1) // BLOCK_N
    return ((grid_m * grid_n, 1, 1), (THREADS, 1, 1))


@substrate.jit
def fused_kernel(X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
                 W: S.Tensor((IN_FEATURES, OUT_FEATURES), S.bf16),
                 BIAS0: S.Tensor((OUT_FEATURES,), S.bf16),
                 Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16)):
    tid = S.thread_id(0)
    block_idx = S.block_id(0)

    grid_n = (OUT_FEATURES + BLOCK_N - 1) // BLOCK_N
    block_m = block_idx // grid_n
    block_n = block_idx % grid_n

    # Each thread computes 4x4 = 16 output elements
    thread_m = tid // 16
    thread_n = tid % 16

    # Local accumulators
    accum = S.make_local((4, 4), S.f32)
    for i in S.range(4):
        for j in S.range(4):
            accum[i, j] = S.convert(0.0, S.f32)

    # Double-buffered LDS for A and B to enable software pipelining
    # Buffer 0: current computation, Buffer 1: next tile load
    A_shared = S.make_shared((2, 64, BLOCK_K), S.bf16)
    B_shared = S.make_shared((2, BLOCK_K, 64), S.bf16)

    num_k_tiles = IN_FEATURES // BLOCK_K

    # Create resource descriptors with range (in bytes) for OOB handling
    # Range allows raw_buffer_load_x4 to return 0 for OOB accesses
    rsrc_x = S.amdgpu.make_rsrc(X, BATCH_SIZE * IN_FEATURES * BF16_SIZE)
    rsrc_w = S.amdgpu.make_rsrc(W, IN_FEATURES * OUT_FEATURES * BF16_SIZE)

    # ========== Software Pipelining Setup ==========
    # Prefetch first tile into buffer 0 using raw_buffer_load_x4 with range
    # No explicit OOB checks - raw_buffer_load_x4 returns 0 for OOB
    cur_buf = 0

    # Load A tile using raw_buffer_load_x4
    # 64*16=1024 elements, each load gets 8 elements, need 128 loads
    # Use threads 0-127 for loading (each does 1 load of 8 elements)
    # Threads 128-255 do redundant loads that may be OOB (returns 0)
    if tid < 128:
        elem_id = tid * 8  # 8 elements per thread
        row = elem_id // BLOCK_K
        col = elem_id % BLOCK_K
        global_row = block_m * 64 + row
        global_col = col
        vindex_x = (global_row * IN_FEATURES + global_col) * BF16_SIZE
        raw_data = S.amdgpu.raw_buffer_load_x4(rsrc_x, vindex_x, 0, 0)
        data_bf16 = S.view(raw_data, S.Tensor((8,), S.bf16))
        # Store to shared memory - col is always 0-7 for valid threads
        for e in S.range(8):
            actual_col = col + e
            if actual_col < BLOCK_K:
                A_shared[cur_buf, row, actual_col] = data_bf16[e]

    # Load B tile using raw_buffer_load_x4
    # 16*64=1024 elements, each load gets 8 elements, need 128 loads
    if tid < 128:
        elem_id = tid * 8
        row = elem_id // 64
        col = elem_id % 64
        global_row = row
        global_col = block_n * 64 + col
        vindex_w = (global_row * OUT_FEATURES + global_col) * BF16_SIZE
        raw_data = S.amdgpu.raw_buffer_load_x4(rsrc_w, vindex_w, 0, 0)
        data_bf16 = S.view(raw_data, S.Tensor((8,), S.bf16))
        for e in S.range(8):
            actual_col = col + e
            if actual_col < 64:
                B_shared[cur_buf, row, actual_col] = data_bf16[e]

    S.syncthreads()

    # ========== Main K-loop with Double Buffering and Unrolling ==========
    for k_tile in S.range(num_k_tiles):
        next_buf = 1 - cur_buf
        k_start = k_tile * BLOCK_K

        # ---- Phase 1: Load next tile (overlapped with computation) ----
        # Removed the `if k_tile + 1 < num_k_tiles:` check
        # raw_buffer_load_x4 with range handles OOB by returning 0
        next_k_start = (k_tile + 1) * BLOCK_K

        # Load A tile for next iteration into next_buf
        if tid < 128:
            elem_id = tid * 8
            row = elem_id // BLOCK_K
            col = elem_id % BLOCK_K
            global_row = block_m * 64 + row
            global_col = next_k_start + col
            vindex_x = (global_row * IN_FEATURES + global_col) * BF16_SIZE
            raw_data = S.amdgpu.raw_buffer_load_x4(rsrc_x, vindex_x, 0, 0)
            data_bf16 = S.view(raw_data, S.Tensor((8,), S.bf16))
            for e in S.range(8):
                actual_col = col + e
                if actual_col < BLOCK_K:
                    A_shared[next_buf, row, actual_col] = data_bf16[e]

        # Load B tile for next iteration into next_buf
        if tid < 128:
            elem_id = tid * 8
            row = elem_id // 64
            col = elem_id % 64
            global_row = next_k_start + row
            global_col = block_n * 64 + col
            vindex_w = (global_row * OUT_FEATURES + global_col) * BF16_SIZE
            raw_data = S.amdgpu.raw_buffer_load_x4(rsrc_w, vindex_w, 0, 0)
            data_bf16 = S.view(raw_data, S.Tensor((8,), S.bf16))
            for e in S.range(8):
                actual_col = col + e
                if actual_col < 64:
                    B_shared[next_buf, row, actual_col] = data_bf16[e]

        # ---- Phase 2: Compute matmul with K-loop unrolled by 2 ----
        row_start = thread_m * 4
        col_start = thread_n * 4

        # K-loop unroll: First half (K = 0..7 within BLOCK_K = 16)
        for k in S.range(8):
            for i in S.range(4):
                for j in S.range(4):
                    a_val = S.convert(A_shared[cur_buf, row_start + i, k], S.f32)
                    b_val = S.convert(B_shared[cur_buf, k, col_start + j], S.f32)
                    accum[i, j] = accum[i, j] + a_val * b_val

        # K-loop unroll: Second half (K = 8..15 within BLOCK_K = 16)
        for k in S.range(8):
            for i in S.range(4):
                for j in S.range(4):
                    a_val = S.convert(A_shared[cur_buf, row_start + i, 8 + k], S.f32)
                    b_val = S.convert(B_shared[cur_buf, 8 + k, col_start + j], S.f32)
                    accum[i, j] = accum[i, j] + a_val * b_val

        # ---- Phase 3: Switch buffers and synchronize ----
        cur_buf = next_buf
        S.syncthreads()

    # ========== Store results with post-processing ==========
    row_start = thread_m * 4
    col_start = thread_n * 4

    for i in S.range(4):
        for j in S.range(4):
            val = accum[i, j]
            out_row = block_m * 64 + row_start + i
            out_col = block_n * 64 + col_start + j

            if out_row < BATCH_SIZE and out_col < OUT_FEATURES:
                bias_val = S.convert(BIAS0[out_col], S.f32)
                val = val + bias_val
                val = val * S.convert(SCALING_FACTOR, S.f32)

                val_min = S.convert(HARDTANH_MIN, S.f32)
                val_max = S.convert(HARDTANH_MAX, S.f32)
                if val < val_min:
                    val = val_min
                if val > val_max:
                    val = val_max

                gelu_val = S.convert(0.5, S.f32) * val * (S.convert(1.0, S.f32) + S.erf(val / S.convert(SQRT_2, S.f32)))
                Y[out_row, out_col] = S.convert(gelu_val, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)
        self.gelu = nn.GELU()

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES) or x.dtype != torch.bfloat16 or self.scaling_factor != SCALING_FACTOR or (self.hardtanh.min_val != HARDTANH_MIN) or (self.hardtanh.max_val != HARDTANH_MAX):
            raise RuntimeError('This fused kernel only supports the benchmark input shape and dtype.')
        w_t = self.gemm.weight.t().to(device=x.device, dtype=x.dtype).contiguous()
        bias = self.gemm.bias.to(device=x.device, dtype=x.dtype).contiguous()
        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=x.dtype)
        fused_kernel[_launch](x.contiguous(), w_t, bias, y)
        return y
