import torch
import torch.nn as nn

import substrate
import substrate.language as S


BATCH_SIZE = 16384
INPUT_SIZE = 2048
HIDDEN_SIZE = 4096
OUTPUT_SIZE = 1024

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
THREADS_PER_BLOCK = 256
WAVE_SIZE = 64

X_RANGE_BYTES = BATCH_SIZE * INPUT_SIZE * 2
W1_RANGE_BYTES = INPUT_SIZE * HIDDEN_SIZE * 2
H_RANGE_BYTES = BATCH_SIZE * HIDDEN_SIZE * 2
W2_RANGE_BYTES = HIDDEN_SIZE * OUTPUT_SIZE * 2
LOGITS_RANGE_BYTES = BATCH_SIZE * OUTPUT_SIZE * 4


def _gemm1_launch():
    return ((HIDDEN_SIZE // BLOCK_N, BATCH_SIZE // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))


def _gemm2_launch():
    return ((OUTPUT_SIZE // BLOCK_N, BATCH_SIZE // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))


def _reduce_launch():
    return ((BATCH_SIZE, 1, 1), (256, 1, 1))


@substrate.jit
def gemm1_sigmoid_kernel(
    X: S.Tensor((BATCH_SIZE, INPUT_SIZE), S.bf16),
    W1: S.Tensor((INPUT_SIZE, HIDDEN_SIZE), S.bf16),
    B1: S.Tensor((HIDDEN_SIZE,), S.bf16),
    H: S.Tensor((BATCH_SIZE, HIDDEN_SIZE), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    wave = tid // WAVE_SIZE
    wave_row = wave // 2
    wave_col = wave % 2
    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N
    one = S.convert(1.0, S.f32)
    num_k_pairs = INPUT_SIZE // (BLOCK_K * 2)

    a_shared = S.make_shared((2, 2, 64, 4), S.u32)
    b_shared = S.make_shared((2, 2, 64, 4), S.u32)
    acc = S.full((16,), 0.0, S.f32)

    x_rsrc = S.amdgpu.make_rsrc(X, X_RANGE_BYTES)
    w1_rsrc = S.amdgpu.make_rsrc(W1, W1_RANGE_BYTES)

    if tid < 128:
        frag = tid
        row_in_block = frag // 2
        half = frag % 2
        row = block_row + row_in_block
        global_offset = (row * INPUT_SIZE + half * 8) * 2
        vec = S.amdgpu.raw_buffer_load_x4(x_rsrc, 0, global_offset, 0)

        owner = row_in_block // 32
        local_row = row_in_block % 32
        dst = half * 2

        a_shared[0, owner, local_row, dst + 0] = vec[0]
        a_shared[0, owner, local_row, dst + 1] = vec[1]
        a_shared[0, owner, local_row + 32, dst + 0] = vec[2]
        a_shared[0, owner, local_row + 32, dst + 1] = vec[3]
    else:
        frag = tid - 128
        k_row = frag // 8
        col_chunk = frag % 8
        col = block_col + col_chunk * 8
        global_offset = (k_row * HIDDEN_SIZE + col) * 2
        vec = S.amdgpu.raw_buffer_load_x4(w1_rsrc, 0, global_offset, 0)

        owner = col_chunk // 4
        local_chunk = col_chunk % 4
        group0 = local_chunk * 2
        j = k_row % 8
        dst = (k_row // 8) * 2
        lane0 = j + 8 * group0
        lane1 = lane0 + 8

        b_shared[0, owner, lane0, dst + 0] = vec[0]
        b_shared[0, owner, lane0, dst + 1] = vec[1]
        b_shared[0, owner, lane1, dst + 0] = vec[2]
        b_shared[0, owner, lane1, dst + 1] = vec[3]

    S.syncthreads()

    for k_pair in S.range(num_k_pairs - 1):
        odd_k_base = (k_pair * 2 + 1) * BLOCK_K
        if tid < 128:
            frag = tid
            row_in_block = frag // 2
            half = frag % 2
            row = block_row + row_in_block
            global_offset = (row * INPUT_SIZE + odd_k_base + half * 8) * 2
            vec = S.amdgpu.raw_buffer_load_x4(x_rsrc, 0, global_offset, 0)
        else:
            frag = tid - 128
            k_row = frag // 8
            col_chunk = frag % 8
            col = block_col + col_chunk * 8
            global_offset = ((odd_k_base + k_row) * HIDDEN_SIZE + col) * 2
            vec = S.amdgpu.raw_buffer_load_x4(w1_rsrc, 0, global_offset, 0)

        a_frag = S.view(a_shared[0, wave_row, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_shared[0, wave_col, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        if tid < 128:
            owner = row_in_block // 32
            local_row = row_in_block % 32
            dst = half * 2

            a_shared[1, owner, local_row, dst + 0] = vec[0]
            a_shared[1, owner, local_row, dst + 1] = vec[1]
            a_shared[1, owner, local_row + 32, dst + 0] = vec[2]
            a_shared[1, owner, local_row + 32, dst + 1] = vec[3]
        else:
            owner = col_chunk // 4
            local_chunk = col_chunk % 4
            group0 = local_chunk * 2
            j = k_row % 8
            dst = (k_row // 8) * 2
            lane0 = j + 8 * group0
            lane1 = lane0 + 8

            b_shared[1, owner, lane0, dst + 0] = vec[0]
            b_shared[1, owner, lane0, dst + 1] = vec[1]
            b_shared[1, owner, lane1, dst + 0] = vec[2]
            b_shared[1, owner, lane1, dst + 1] = vec[3]

        S.syncthreads()

        even_k_base = (k_pair * 2 + 2) * BLOCK_K
        if tid < 128:
            global_offset = (row * INPUT_SIZE + even_k_base + half * 8) * 2
            vec = S.amdgpu.raw_buffer_load_x4(x_rsrc, 0, global_offset, 0)
        else:
            global_offset = ((even_k_base + k_row) * HIDDEN_SIZE + col) * 2
            vec = S.amdgpu.raw_buffer_load_x4(w1_rsrc, 0, global_offset, 0)

        a_frag = S.view(a_shared[1, wave_row, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_shared[1, wave_col, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        if tid < 128:
            a_shared[0, owner, local_row, dst + 0] = vec[0]
            a_shared[0, owner, local_row, dst + 1] = vec[1]
            a_shared[0, owner, local_row + 32, dst + 0] = vec[2]
            a_shared[0, owner, local_row + 32, dst + 1] = vec[3]
        else:
            b_shared[0, owner, lane0, dst + 0] = vec[0]
            b_shared[0, owner, lane0, dst + 1] = vec[1]
            b_shared[0, owner, lane1, dst + 0] = vec[2]
            b_shared[0, owner, lane1, dst + 1] = vec[3]

        S.syncthreads()

    final_odd_k_base = (num_k_pairs * 2 - 1) * BLOCK_K
    if tid < 128:
        frag = tid
        row_in_block = frag // 2
        half = frag % 2
        row = block_row + row_in_block
        global_offset = (row * INPUT_SIZE + final_odd_k_base + half * 8) * 2
        vec = S.amdgpu.raw_buffer_load_x4(x_rsrc, 0, global_offset, 0)
        owner = row_in_block // 32
        local_row = row_in_block % 32
        dst = half * 2
    else:
        frag = tid - 128
        k_row = frag // 8
        col_chunk = frag % 8
        col = block_col + col_chunk * 8
        global_offset = ((final_odd_k_base + k_row) * HIDDEN_SIZE + col) * 2
        vec = S.amdgpu.raw_buffer_load_x4(w1_rsrc, 0, global_offset, 0)
        owner = col_chunk // 4
        local_chunk = col_chunk % 4
        group0 = local_chunk * 2
        j = k_row % 8
        dst = (k_row // 8) * 2
        lane0 = j + 8 * group0
        lane1 = lane0 + 8

    a_frag = S.view(a_shared[0, wave_row, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag = S.view(b_shared[0, wave_col, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

    if tid < 128:
        a_shared[1, owner, local_row, dst + 0] = vec[0]
        a_shared[1, owner, local_row, dst + 1] = vec[1]
        a_shared[1, owner, local_row + 32, dst + 0] = vec[2]
        a_shared[1, owner, local_row + 32, dst + 1] = vec[3]
    else:
        b_shared[1, owner, lane0, dst + 0] = vec[0]
        b_shared[1, owner, lane0, dst + 1] = vec[1]
        b_shared[1, owner, lane1, dst + 0] = vec[2]
        b_shared[1, owner, lane1, dst + 1] = vec[3]

    S.syncthreads()

    a_frag = S.view(a_shared[1, wave_row, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag = S.view(b_shared[1, wave_col, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

    col = block_col + wave_col * 32 + (lane % 32)
    bias = S.convert(B1[col], S.f32)
    for acc_idx in S.range(16):
        row = block_row + wave_row * 32 + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        val = acc[acc_idx] + bias
        val = one / (one + S.exp(-val))
        H[row, col] = S.convert(val, S.bf16)


@substrate.jit
def gemm2_bias_kernel(
    H: S.Tensor((BATCH_SIZE, HIDDEN_SIZE), S.bf16),
    W2: S.Tensor((HIDDEN_SIZE, OUTPUT_SIZE), S.bf16),
    B2: S.Tensor((OUTPUT_SIZE,), S.bf16),
    Logits: S.Tensor((BATCH_SIZE, OUTPUT_SIZE), S.f32),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    wave = tid // WAVE_SIZE
    wave_row = wave // 2
    wave_col = wave % 2
    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N
    num_k_pairs = HIDDEN_SIZE // (BLOCK_K * 2)

    a_shared = S.make_shared((2, 2, 64, 4), S.u32)
    b_shared = S.make_shared((2, 2, 64, 4), S.u32)
    acc = S.full((16,), 0.0, S.f32)

    h_rsrc = S.amdgpu.make_rsrc(H, H_RANGE_BYTES)
    w2_rsrc = S.amdgpu.make_rsrc(W2, W2_RANGE_BYTES)
    logits_rsrc = S.amdgpu.make_rsrc(Logits, LOGITS_RANGE_BYTES)

    if tid < 128:
        frag = tid
        row_in_block = frag // 2
        half = frag % 2
        row = block_row + row_in_block
        global_offset = (row * HIDDEN_SIZE + half * 8) * 2
        vec = S.amdgpu.raw_buffer_load_x4(h_rsrc, 0, global_offset, 0)

        owner = row_in_block // 32
        local_row = row_in_block % 32
        dst = half * 2

        a_shared[0, owner, local_row, dst + 0] = vec[0]
        a_shared[0, owner, local_row, dst + 1] = vec[1]
        a_shared[0, owner, local_row + 32, dst + 0] = vec[2]
        a_shared[0, owner, local_row + 32, dst + 1] = vec[3]
    else:
        frag = tid - 128
        k_row = frag // 8
        col_chunk = frag % 8
        col = block_col + col_chunk * 8
        global_offset = (k_row * OUTPUT_SIZE + col) * 2
        vec = S.amdgpu.raw_buffer_load_x4(w2_rsrc, 0, global_offset, 0)

        owner = col_chunk // 4
        local_chunk = col_chunk % 4
        group0 = local_chunk * 2
        j = k_row % 8
        dst = (k_row // 8) * 2
        lane0 = j + 8 * group0
        lane1 = lane0 + 8

        b_shared[0, owner, lane0, dst + 0] = vec[0]
        b_shared[0, owner, lane0, dst + 1] = vec[1]
        b_shared[0, owner, lane1, dst + 0] = vec[2]
        b_shared[0, owner, lane1, dst + 1] = vec[3]

    S.syncthreads()

    for k_pair in S.range(num_k_pairs - 1):
        odd_k_base = (k_pair * 2 + 1) * BLOCK_K
        if tid < 128:
            frag = tid
            row_in_block = frag // 2
            half = frag % 2
            row = block_row + row_in_block
            global_offset = (row * HIDDEN_SIZE + odd_k_base + half * 8) * 2
            vec = S.amdgpu.raw_buffer_load_x4(h_rsrc, 0, global_offset, 0)
        else:
            frag = tid - 128
            k_row = frag // 8
            col_chunk = frag % 8
            col = block_col + col_chunk * 8
            global_offset = ((odd_k_base + k_row) * OUTPUT_SIZE + col) * 2
            vec = S.amdgpu.raw_buffer_load_x4(w2_rsrc, 0, global_offset, 0)

        a_frag = S.view(a_shared[0, wave_row, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_shared[0, wave_col, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        if tid < 128:
            owner = row_in_block // 32
            local_row = row_in_block % 32
            dst = half * 2

            a_shared[1, owner, local_row, dst + 0] = vec[0]
            a_shared[1, owner, local_row, dst + 1] = vec[1]
            a_shared[1, owner, local_row + 32, dst + 0] = vec[2]
            a_shared[1, owner, local_row + 32, dst + 1] = vec[3]
        else:
            owner = col_chunk // 4
            local_chunk = col_chunk % 4
            group0 = local_chunk * 2
            j = k_row % 8
            dst = (k_row // 8) * 2
            lane0 = j + 8 * group0
            lane1 = lane0 + 8

            b_shared[1, owner, lane0, dst + 0] = vec[0]
            b_shared[1, owner, lane0, dst + 1] = vec[1]
            b_shared[1, owner, lane1, dst + 0] = vec[2]
            b_shared[1, owner, lane1, dst + 1] = vec[3]

        S.syncthreads()

        even_k_base = (k_pair * 2 + 2) * BLOCK_K
        if tid < 128:
            global_offset = (row * HIDDEN_SIZE + even_k_base + half * 8) * 2
            vec = S.amdgpu.raw_buffer_load_x4(h_rsrc, 0, global_offset, 0)
        else:
            global_offset = ((even_k_base + k_row) * OUTPUT_SIZE + col) * 2
            vec = S.amdgpu.raw_buffer_load_x4(w2_rsrc, 0, global_offset, 0)

        a_frag = S.view(a_shared[1, wave_row, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_shared[1, wave_col, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        if tid < 128:
            a_shared[0, owner, local_row, dst + 0] = vec[0]
            a_shared[0, owner, local_row, dst + 1] = vec[1]
            a_shared[0, owner, local_row + 32, dst + 0] = vec[2]
            a_shared[0, owner, local_row + 32, dst + 1] = vec[3]
        else:
            b_shared[0, owner, lane0, dst + 0] = vec[0]
            b_shared[0, owner, lane0, dst + 1] = vec[1]
            b_shared[0, owner, lane1, dst + 0] = vec[2]
            b_shared[0, owner, lane1, dst + 1] = vec[3]

        S.syncthreads()

    final_odd_k_base = (num_k_pairs * 2 - 1) * BLOCK_K
    if tid < 128:
        frag = tid
        row_in_block = frag // 2
        half = frag % 2
        row = block_row + row_in_block
        global_offset = (row * HIDDEN_SIZE + final_odd_k_base + half * 8) * 2
        vec = S.amdgpu.raw_buffer_load_x4(h_rsrc, 0, global_offset, 0)
        owner = row_in_block // 32
        local_row = row_in_block % 32
        dst = half * 2
    else:
        frag = tid - 128
        k_row = frag // 8
        col_chunk = frag % 8
        col = block_col + col_chunk * 8
        global_offset = ((final_odd_k_base + k_row) * OUTPUT_SIZE + col) * 2
        vec = S.amdgpu.raw_buffer_load_x4(w2_rsrc, 0, global_offset, 0)
        owner = col_chunk // 4
        local_chunk = col_chunk % 4
        group0 = local_chunk * 2
        j = k_row % 8
        dst = (k_row // 8) * 2
        lane0 = j + 8 * group0
        lane1 = lane0 + 8

    a_frag = S.view(a_shared[0, wave_row, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag = S.view(b_shared[0, wave_col, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

    if tid < 128:
        a_shared[1, owner, local_row, dst + 0] = vec[0]
        a_shared[1, owner, local_row, dst + 1] = vec[1]
        a_shared[1, owner, local_row + 32, dst + 0] = vec[2]
        a_shared[1, owner, local_row + 32, dst + 1] = vec[3]
    else:
        b_shared[1, owner, lane0, dst + 0] = vec[0]
        b_shared[1, owner, lane0, dst + 1] = vec[1]
        b_shared[1, owner, lane1, dst + 0] = vec[2]
        b_shared[1, owner, lane1, dst + 1] = vec[3]

    S.syncthreads()

    a_frag = S.view(a_shared[1, wave_row, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag = S.view(b_shared[1, wave_col, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

    col = block_col + wave_col * 32 + (lane % 32)
    bias = S.convert(B2[col], S.f32)
    for acc_idx in S.range(16):
        row = block_row + wave_row * 32 + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        out = acc[acc_idx] + bias
        global_offset = (row * OUTPUT_SIZE + col) * 4
        S.amdgpu.raw_buffer_store_x1(S.bitcast(out, S.i32), logits_rsrc, 0, global_offset, 0)


@substrate.jit
def logsumexp_kernel(
    Logits: S.Tensor((BATCH_SIZE, OUTPUT_SIZE), S.f32),
    Y: S.Tensor((BATCH_SIZE,), S.bf16),
):
    tid = S.thread_id(0)
    row = S.block_id(0)
    scratch = S.make_shared((256,), S.f32)

    max_val = S.convert(-1.0e30, S.f32)
    for tile in S.range(OUTPUT_SIZE // 256):
        col = tile * 256 + tid
        val = Logits[row, col]
        if val > max_val:
            max_val = val

    scratch[tid] = max_val
    S.syncthreads()

    if tid < 128:
        rhs = scratch[tid + 128]
        if rhs > scratch[tid]:
            scratch[tid] = rhs
    S.syncthreads()
    if tid < 64:
        rhs = scratch[tid + 64]
        if rhs > scratch[tid]:
            scratch[tid] = rhs
    S.syncthreads()
    if tid < 32:
        rhs = scratch[tid + 32]
        if rhs > scratch[tid]:
            scratch[tid] = rhs
    S.syncthreads()
    if tid < 16:
        rhs = scratch[tid + 16]
        if rhs > scratch[tid]:
            scratch[tid] = rhs
    S.syncthreads()
    if tid < 8:
        rhs = scratch[tid + 8]
        if rhs > scratch[tid]:
            scratch[tid] = rhs
    S.syncthreads()
    if tid < 4:
        rhs = scratch[tid + 4]
        if rhs > scratch[tid]:
            scratch[tid] = rhs
    S.syncthreads()
    if tid < 2:
        rhs = scratch[tid + 2]
        if rhs > scratch[tid]:
            scratch[tid] = rhs
    S.syncthreads()
    if tid < 1:
        rhs = scratch[tid + 1]
        if rhs > scratch[tid]:
            scratch[tid] = rhs
    S.syncthreads()

    row_max = scratch[0]
    sum_val = S.convert(0.0, S.f32)
    for tile in S.range(OUTPUT_SIZE // 256):
        col = tile * 256 + tid
        sum_val += S.exp(Logits[row, col] - row_max)

    scratch[tid] = sum_val
    S.syncthreads()

    if tid < 128:
        scratch[tid] = scratch[tid] + scratch[tid + 128]
    S.syncthreads()
    if tid < 64:
        scratch[tid] = scratch[tid] + scratch[tid + 64]
    S.syncthreads()
    if tid < 32:
        scratch[tid] = scratch[tid] + scratch[tid + 32]
    S.syncthreads()
    if tid < 16:
        scratch[tid] = scratch[tid] + scratch[tid + 16]
    S.syncthreads()
    if tid < 8:
        scratch[tid] = scratch[tid] + scratch[tid + 8]
    S.syncthreads()
    if tid < 4:
        scratch[tid] = scratch[tid] + scratch[tid + 4]
    S.syncthreads()
    if tid < 2:
        scratch[tid] = scratch[tid] + scratch[tid + 2]
    S.syncthreads()
    if tid < 1:
        scratch[tid] = scratch[tid] + scratch[tid + 1]

    if tid == 0:
        Y[row] = S.convert(row_max + S.log(scratch[0]), S.bf16)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self._cached_operands = {}

    def _get_operands(self, x: torch.Tensor):
        device = x.device
        cache = self._cached_operands.get(device)
        ptrs = (
            self.linear1.weight.data_ptr(),
            self.linear1.bias.data_ptr(),
            self.linear2.weight.data_ptr(),
            self.linear2.bias.data_ptr(),
        )
        if cache is None or cache["ptrs"] != ptrs:
            cache = {
                "ptrs": ptrs,
                "w1": self.linear1.weight.detach().t().to(device=device, dtype=torch.bfloat16).contiguous(),
                "b1": self.linear1.bias.detach().to(device=device, dtype=torch.bfloat16).contiguous(),
                "w2": self.linear2.weight.detach().t().to(device=device, dtype=torch.bfloat16).contiguous(),
                "b2": self.linear2.bias.detach().to(device=device, dtype=torch.bfloat16).contiguous(),
            }
            self._cached_operands[device] = cache
        return cache["w1"], cache["b1"], cache["w2"], cache["b2"]

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, INPUT_SIZE) or x.dtype != torch.bfloat16:
            raise RuntimeError("ModelNew only supports the benchmark input shape in bfloat16")

        x = x.contiguous()
        w1, b1, w2, b2 = self._get_operands(x)

        h = torch.empty((BATCH_SIZE, HIDDEN_SIZE), device=x.device, dtype=torch.bfloat16)
        logits = torch.empty((BATCH_SIZE, OUTPUT_SIZE), device=x.device, dtype=torch.float32)
        y = torch.empty((BATCH_SIZE,), device=x.device, dtype=torch.bfloat16)

        gemm1_sigmoid_kernel[_gemm1_launch](x, w1, b1, h)
        gemm2_bias_kernel[_gemm2_launch](h, w2, b2, logits)
        logsumexp_kernel[_reduce_launch](logits, y)
        return y
