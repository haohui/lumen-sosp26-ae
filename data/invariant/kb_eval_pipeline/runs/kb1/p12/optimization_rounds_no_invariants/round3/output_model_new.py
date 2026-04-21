import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 4096
N = 4096
BLOCK_M = 64
BLOCK_N = 64
WAVE_M = 32
WAVE_N = 32
THREADS = 256
WARP_SIZE = 64
VEC_ELEMS = 8
# Raw-buffer resource ranges provide OOB zero-fill on loads and discard OOB stores.
A_RANGE_BYTES = M * 2
B_RANGE_BYTES = M * N * 2
C_RANGE_BYTES = M * N * 2


@substrate.jit
def diag_left_kernel(
    A: S.Tensor((M,), S.bf16),
    B: S.Tensor((M, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    pid_n = S.block_id(0)
    pid_m = S.block_id(1)
    tid = S.thread_id(0)

    wave = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    wave_m = wave // 2
    wave_n = wave % 2

    block_row = pid_m * BLOCK_M
    block_col = pid_n * BLOCK_N

    a_rsrc = S.amdgpu.make_rsrc(A, A_RANGE_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, B_RANGE_BYTES)
    c_rsrc = S.amdgpu.make_rsrc(C, C_RANGE_BYTES)

    a_stage = S.make_shared((8, 4), S.u32)
    b_stage = S.make_shared((2, 4, 64, 4), S.u32)

    if tid < 8:
        a_offset = (block_row + tid * VEC_ELEMS) * 2
        a_stage[tid] = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, a_offset, 0)

    chunk0 = lane
    chunk1 = lane + WARP_SIZE

    row0 = chunk0 // 4
    col_chunk0 = chunk0 % 4
    row1 = chunk1 // 4
    col_chunk1 = chunk1 % 4

    global_row0 = block_row + wave_m * WAVE_M + row0
    global_row1 = block_row + wave_m * WAVE_M + row1
    global_col0 = block_col + wave_n * WAVE_N + col_chunk0 * VEC_ELEMS
    global_col1 = block_col + wave_n * WAVE_N + col_chunk1 * VEC_ELEMS

    b_offset0 = (global_row0 * N + global_col0) * 2
    b_stage[0, wave, lane] = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, b_offset0, 0)

    S.syncthreads()

    a_vals = S.view(a_stage, S.Tensor((64,), S.bf16))
    a_scale0 = a_vals[wave_m * WAVE_M + row0]
    a_scale1 = a_vals[wave_m * WAVE_M + row1]

    out_pack = S.make_local((2, 4), S.u32)
    out_vals = S.view(out_pack, S.Tensor((2, 8), S.bf16))

    b_vals0 = S.view(b_stage[0, wave, lane], S.Tensor((8,), S.bf16))
    for i in S.range(8):
        out_vals[0, i] = a_scale0 * b_vals0[i]

    b_offset1 = (global_row1 * N + global_col1) * 2
    b_stage[1, wave, lane] = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, b_offset1, 0)

    c_offset0 = (global_row0 * N + global_col0) * 2
    S.amdgpu.raw_buffer_store_x4(out_pack[0], c_rsrc, 0, c_offset0, 0)

    S.syncthreads()

    b_vals1 = S.view(b_stage[1, wave, lane], S.Tensor((8,), S.bf16))
    for i in S.range(8):
        out_vals[1, i] = a_scale1 * b_vals1[i]

    c_offset1 = (global_row1 * N + global_col1) * 2
    S.amdgpu.raw_buffer_store_x4(out_pack[1], c_rsrc, 0, c_offset1, 0)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M,) or tuple(B.shape) != (M, N):
            return A.reshape(-1, 1) * B
        if A.device.type != "cuda" or B.device.type != "cuda":
            return A.reshape(M, 1) * B
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=B.device, dtype=B.dtype)
        diag_left_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS, 1, 1))](A, B, C)
        return C
