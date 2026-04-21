import torch
import torch.nn as nn
import substrate
import substrate.language as S

BATCH_SIZE = 16384
INPUT_SIZE = 2048
HIDDEN_SIZE = 4096
OUTPUT_SIZE = 1024

# MFMA tile sizes
MFMA_M = 32
MFMA_N = 32
MFMA_K = 8

# Use 2x2 warp grid
WARP_GRID_M = 2
WARP_GRID_N = 2
NUM_WARPS = WARP_GRID_M * WARP_GRID_N
WAVE_SIZE = 64
BLOCK_SIZE = NUM_WARPS * WAVE_SIZE  # 256 threads

BLOCK_TILE_M = WARP_GRID_M * MFMA_M  # 64
BLOCK_TILE_N = WARP_GRID_N * MFMA_N  # 64

# Double buffering
NUM_BUFFERS = 2


@substrate.jit
def gemm1_mfma(
    X: S.Tensor((BATCH_SIZE, INPUT_SIZE), S.bf16),
    W1_T: S.Tensor((HIDDEN_SIZE, INPUT_SIZE), S.bf16),
    B1: S.Tensor((HIDDEN_SIZE,), S.bf16),
    H: S.Tensor((BATCH_SIZE, HIDDEN_SIZE), S.bf16),
):
    """First GEMM: H = sigmoid(X @ W1 + B1) using MFMA with software pipelining."""

    tid = S.thread_id(0)
    warp_id = tid // WAVE_SIZE
    lane_id = tid % WAVE_SIZE

    warp_m = warp_id // WARP_GRID_N
    warp_n = warp_id % WARP_GRID_N

    block_m = S.block_id(0)
    block_n = S.block_id(1)

    batch_base = block_m * BLOCK_TILE_M + warp_m * MFMA_M
    col_base = block_n * BLOCK_TILE_N + warp_n * MFMA_N

    acc = S.full((16,), 0.0, S.f32)

    rsrc_X = S.amdgpu.make_rsrc(X, BATCH_SIZE * INPUT_SIZE * 2)
    rsrc_W1_T = S.amdgpu.make_rsrc(W1_T, HIDDEN_SIZE * INPUT_SIZE * 2)

    # LDS for double buffering A and B
    # Each thread needs 2 u32s (8 bytes = 4 bf16s) per K-step
    # Double buffer: 2 buffers * BLOCK_SIZE threads * 2 u32s = BLOCK_SIZE * 4 u32s
    lds_A_raw = S.make_shared((NUM_BUFFERS * BLOCK_SIZE * 2,), S.u32)
    lds_B_raw = S.make_shared((NUM_BUFFERS * BLOCK_SIZE * 2,), S.u32)

    # View LDS as [buffer][tid][2] for easier indexing
    lds_A = S.view(lds_A_raw, S.Tensor((NUM_BUFFERS, BLOCK_SIZE, 2), S.u32))
    lds_B = S.view(lds_B_raw, S.Tensor((NUM_BUFFERS, BLOCK_SIZE, 2), S.u32))

    # MFMA swizzle calculations
    a_row = lane_id % 32
    a_col_group = lane_id // 32
    b_col_idx = lane_id % 32
    b_k_group = lane_id // 32

    num_k_steps = INPUT_SIZE // MFMA_K

    # ========== Software Pipelining with Double Buffering ==========
    # Unroll K-loop by 2: process 2 MFMA_K steps per iteration
    # This allows overlapping global loads with MFMA computation

    # Prefetch first k_step into buffer 0
    k_base = 0
    a_offset = (batch_base + a_row) * INPUT_SIZE + k_base + a_col_group * 4
    a_data = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_offset * 2, 0, 0)
    lds_A[0, tid, 0] = a_data[0]
    lds_A[0, tid, 1] = a_data[1]

    b_offset = (col_base + b_col_idx) * INPUT_SIZE + k_base + b_k_group * 4
    b_data = S.amdgpu.raw_buffer_load_x2(rsrc_W1_T, b_offset * 2, 0, 0)
    lds_B[0, tid, 0] = b_data[0]
    lds_B[0, tid, 1] = b_data[1]

    # Wait for first load to complete
    S.amdgpu.s_waitcnt(0, 0, 0)
    S.syncthreads()

    buf = 0
    half_k_steps = num_k_steps // 2

    for k_half in S.range(half_k_steps):
        k_step0 = k_half * 2
        k_step1 = k_half * 2 + 1
        next_buf = 1 - buf

        # ===== Compute from current buffer (k_step0) =====
        # View LDS slice as bf16 for MFMA
        a_frag0 = S.view(lds_A[buf, tid], S.Tensor((4,), S.bf16))
        b_frag0 = S.view(lds_B[buf, tid], S.Tensor((4,), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0, b_frag0, acc)

        # ===== Prefetch k_step1 into next buffer (overlapped with MFMA) =====
        k_base1 = k_step1 * MFMA_K
        a_offset1 = (batch_base + a_row) * INPUT_SIZE + k_base1 + a_col_group * 4
        a_data1 = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_offset1 * 2, 0, 0)
        lds_A[next_buf, tid, 0] = a_data1[0]
        lds_A[next_buf, tid, 1] = a_data1[1]

        b_offset1 = (col_base + b_col_idx) * INPUT_SIZE + k_base1 + b_k_group * 4
        b_data1 = S.amdgpu.raw_buffer_load_x2(rsrc_W1_T, b_offset1 * 2, 0, 0)
        lds_B[next_buf, tid, 0] = b_data1[0]
        lds_B[next_buf, tid, 1] = b_data1[1]

        # Wait for prefetch to complete
        S.amdgpu.s_waitcnt(0, 0, 0)
        S.syncthreads()

        # ===== Compute from next buffer (k_step1) =====
        a_frag1 = S.view(lds_A[next_buf, tid], S.Tensor((4,), S.bf16))
        b_frag1 = S.view(lds_B[next_buf, tid], S.Tensor((4,), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1, b_frag1, acc)

        # ===== Prefetch next k_step0 into buf =====
        # OOB access handled by range in make_rsrc: raw_buffer_load returns 0 for OOB
        # Removing branch reduces divergence overhead
        next_k_step0 = (k_half + 1) * 2
        k_base_next = next_k_step0 * MFMA_K
        a_offset_next = (batch_base + a_row) * INPUT_SIZE + k_base_next + a_col_group * 4
        a_data_next = S.amdgpu.raw_buffer_load_x2(rsrc_X, a_offset_next * 2, 0, 0)
        lds_A[buf, tid, 0] = a_data_next[0]
        lds_A[buf, tid, 1] = a_data_next[1]

        b_offset_next = (col_base + b_col_idx) * INPUT_SIZE + k_base_next + b_k_group * 4
        b_data_next = S.amdgpu.raw_buffer_load_x2(rsrc_W1_T, b_offset_next * 2, 0, 0)
        lds_B[buf, tid, 0] = b_data_next[0]
        lds_B[buf, tid, 1] = b_data_next[1]

        S.amdgpu.s_waitcnt(0, 0, 0)
        S.syncthreads()

    # Write results with sigmoid activation
    one = S.convert(1.0, S.f32)
    for acc_idx in S.range(16):
        out_col = lane_id % 32
        out_row = 8 * (acc_idx // 4) + 4 * (lane_id // 32) + (acc_idx % 4)
        val = acc[acc_idx] + S.convert(B1[col_base + out_col], S.f32)
        val = one / (one + S.exp(-val))
        H[batch_base + out_row, col_base + out_col] = S.convert(val, S.bf16)


@substrate.jit
def gemm2_mfma(
    H: S.Tensor((BATCH_SIZE, HIDDEN_SIZE), S.bf16),
    W2_T: S.Tensor((OUTPUT_SIZE, HIDDEN_SIZE), S.bf16),
    B2: S.Tensor((OUTPUT_SIZE,), S.bf16),
    HiddenOut: S.Tensor((BATCH_SIZE, OUTPUT_SIZE), S.bf16),
):
    """Second GEMM: HiddenOut = H @ W2 + B2 using MFMA with software pipelining."""

    tid = S.thread_id(0)
    warp_id = tid // WAVE_SIZE
    lane_id = tid % WAVE_SIZE

    warp_m = warp_id // WARP_GRID_N
    warp_n = warp_id % WARP_GRID_N

    block_m = S.block_id(0)
    block_n = S.block_id(1)

    batch_base = block_m * BLOCK_TILE_M + warp_m * MFMA_M
    col_base = block_n * BLOCK_TILE_N + warp_n * MFMA_N

    acc = S.full((16,), 0.0, S.f32)

    rsrc_H = S.amdgpu.make_rsrc(H, BATCH_SIZE * HIDDEN_SIZE * 2)
    rsrc_W2_T = S.amdgpu.make_rsrc(W2_T, OUTPUT_SIZE * HIDDEN_SIZE * 2)

    # LDS for double buffering
    lds_A_raw = S.make_shared((NUM_BUFFERS * BLOCK_SIZE * 2,), S.u32)
    lds_B_raw = S.make_shared((NUM_BUFFERS * BLOCK_SIZE * 2,), S.u32)
    lds_A = S.view(lds_A_raw, S.Tensor((NUM_BUFFERS, BLOCK_SIZE, 2), S.u32))
    lds_B = S.view(lds_B_raw, S.Tensor((NUM_BUFFERS, BLOCK_SIZE, 2), S.u32))

    a_row = lane_id % 32
    a_col_group = lane_id // 32
    b_col_idx = lane_id % 32
    b_k_group = lane_id // 32

    num_k_steps = HIDDEN_SIZE // MFMA_K

    # Prefetch first k_step into buffer 0
    k_base = 0
    a_offset = (batch_base + a_row) * HIDDEN_SIZE + k_base + a_col_group * 4
    a_data = S.amdgpu.raw_buffer_load_x2(rsrc_H, a_offset * 2, 0, 0)
    lds_A[0, tid, 0] = a_data[0]
    lds_A[0, tid, 1] = a_data[1]

    b_offset = (col_base + b_col_idx) * HIDDEN_SIZE + k_base + b_k_group * 4
    b_data = S.amdgpu.raw_buffer_load_x2(rsrc_W2_T, b_offset * 2, 0, 0)
    lds_B[0, tid, 0] = b_data[0]
    lds_B[0, tid, 1] = b_data[1]

    S.amdgpu.s_waitcnt(0, 0, 0)
    S.syncthreads()

    buf = 0
    half_k_steps = num_k_steps // 2

    for k_half in S.range(half_k_steps):
        k_step0 = k_half * 2
        k_step1 = k_half * 2 + 1
        next_buf = 1 - buf

        # Compute from current buffer (k_step0)
        a_frag0 = S.view(lds_A[buf, tid], S.Tensor((4,), S.bf16))
        b_frag0 = S.view(lds_B[buf, tid], S.Tensor((4,), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0, b_frag0, acc)

        # Prefetch k_step1
        k_base1 = k_step1 * MFMA_K
        a_offset1 = (batch_base + a_row) * HIDDEN_SIZE + k_base1 + a_col_group * 4
        a_data1 = S.amdgpu.raw_buffer_load_x2(rsrc_H, a_offset1 * 2, 0, 0)
        lds_A[next_buf, tid, 0] = a_data1[0]
        lds_A[next_buf, tid, 1] = a_data1[1]

        b_offset1 = (col_base + b_col_idx) * HIDDEN_SIZE + k_base1 + b_k_group * 4
        b_data1 = S.amdgpu.raw_buffer_load_x2(rsrc_W2_T, b_offset1 * 2, 0, 0)
        lds_B[next_buf, tid, 0] = b_data1[0]
        lds_B[next_buf, tid, 1] = b_data1[1]

        S.amdgpu.s_waitcnt(0, 0, 0)
        S.syncthreads()

        # Compute from next buffer (k_step1)
        a_frag1 = S.view(lds_A[next_buf, tid], S.Tensor((4,), S.bf16))
        b_frag1 = S.view(lds_B[next_buf, tid], S.Tensor((4,), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1, b_frag1, acc)

        # Prefetch next k_step0
        # OOB access handled by range in make_rsrc: raw_buffer_load returns 0 for OOB
        # Removing branch reduces divergence overhead
        next_k_step0 = (k_half + 1) * 2
        k_base_next = next_k_step0 * MFMA_K
        a_offset_next = (batch_base + a_row) * HIDDEN_SIZE + k_base_next + a_col_group * 4
        a_data_next = S.amdgpu.raw_buffer_load_x2(rsrc_H, a_offset_next * 2, 0, 0)
        lds_A[buf, tid, 0] = a_data_next[0]
        lds_A[buf, tid, 1] = a_data_next[1]

        b_offset_next = (col_base + b_col_idx) * HIDDEN_SIZE + k_base_next + b_k_group * 4
        b_data_next = S.amdgpu.raw_buffer_load_x2(rsrc_W2_T, b_offset_next * 2, 0, 0)
        lds_B[buf, tid, 0] = b_data_next[0]
        lds_B[buf, tid, 1] = b_data_next[1]

        S.amdgpu.s_waitcnt(0, 0, 0)
        S.syncthreads()

    for acc_idx in S.range(16):
        out_col = lane_id % 32
        out_row = 8 * (acc_idx // 4) + 4 * (lane_id // 32) + (acc_idx % 4)
        val = acc[acc_idx] + S.convert(B2[col_base + out_col], S.f32)
        HiddenOut[batch_base + out_row, col_base + out_col] = S.convert(val, S.bf16)


@substrate.jit
def logsumexp_parallel(
    HiddenOut: S.Tensor((BATCH_SIZE, OUTPUT_SIZE), S.bf16),
    Y: S.Tensor((BATCH_SIZE,), S.bf16),
):
    """Parallel logsumexp kernel - each block handles one batch, 256 threads per block."""

    tid = S.thread_id(0)
    batch_idx = S.block_id(0)

    # Each thread handles 4 elements (1024 / 256 = 4)
    chunk_size = 4

    # Find local max
    local_max = S.convert(-1e+30, S.f32)
    for i in S.range(chunk_size):
        idx = tid * chunk_size + i
        val = S.convert(HiddenOut[batch_idx, idx], S.f32)
        if val > local_max:
            local_max = val

    # LDS for reduction
    lds_max = S.make_shared((BLOCK_SIZE,), S.f32)
    lds_sum = S.make_shared((BLOCK_SIZE,), S.f32)

    lds_max[tid] = local_max
    S.syncthreads()

    # Parallel reduction to find global max
    step = 128
    for s in S.range(8):
        if tid < step:
            other = lds_max[tid + step]
            if other > lds_max[tid]:
                lds_max[tid] = other
        step = step // 2
        S.syncthreads()

    global_max = lds_max[0]

    # Compute local sum of exp(x - global_max)
    local_sum = S.convert(0.0, S.f32)
    for i in S.range(chunk_size):
        idx = tid * chunk_size + i
        val = S.convert(HiddenOut[batch_idx, idx], S.f32)
        local_sum += S.exp(val - global_max)

    lds_sum[tid] = local_sum
    S.syncthreads()

    # Parallel reduction for sum
    step = 128
    for s in S.range(8):
        if tid < step:
            lds_sum[tid] += lds_sum[tid + step]
        step = step // 2
        S.syncthreads()

    if tid == 0:
        Y[batch_idx] = S.convert(global_max + S.log(lds_sum[0]), S.bf16)


def _launch_gemm1():
    return ((BATCH_SIZE // BLOCK_TILE_M, HIDDEN_SIZE // BLOCK_TILE_N, 1), (BLOCK_SIZE, 1, 1))


def _launch_gemm2():
    return ((BATCH_SIZE // BLOCK_TILE_M, OUTPUT_SIZE // BLOCK_TILE_N, 1), (BLOCK_SIZE, 1, 1))


def _launch_logsumexp():
    return ((BATCH_SIZE, 1, 1), (BLOCK_SIZE, 1, 1))


class ModelNew(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, INPUT_SIZE) or x.dtype != torch.bfloat16:
            raise RuntimeError('This fused kernel only supports the benchmark input shape and dtype.')

        w1_t = self.linear1.weight.to(device=x.device, dtype=x.dtype).contiguous()
        b1 = self.linear1.bias.to(device=x.device, dtype=x.dtype).contiguous()
        w2_t = self.linear2.weight.to(device=x.device, dtype=x.dtype).contiguous()
        b2 = self.linear2.bias.to(device=x.device, dtype=x.dtype).contiguous()

        h = torch.empty((BATCH_SIZE, HIDDEN_SIZE), device=x.device, dtype=x.dtype)
        hidden_out = torch.empty((BATCH_SIZE, OUTPUT_SIZE), device=x.device, dtype=x.dtype)
        y = torch.empty((BATCH_SIZE,), device=x.device, dtype=x.dtype)

        gemm1_mfma[_launch_gemm1](x.contiguous(), w1_t, b1, h)
        gemm2_mfma[_launch_gemm2](h, w2_t, b2, hidden_out)
        logsumexp_parallel[_launch_logsumexp](hidden_out, y)

        return y
