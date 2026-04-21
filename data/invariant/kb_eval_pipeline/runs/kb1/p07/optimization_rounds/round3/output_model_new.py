import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 32768
K = 64
N = 32768

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
THREADS = 256
WAVE_SIZE = 64

A_RANGE_BYTES = M * K * 2
B_RANGE_BYTES = K * N * 2


@substrate.jit
def gemm_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    tid = S.thread_id(0)
    warp_id = tid // WAVE_SIZE
    lane = tid % WAVE_SIZE
    warp_row = warp_id // 2
    warp_col = warp_id % 2

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N

    # Keep the raw buffer byte ranges explicit at resource creation so out-of-range
    # accesses can be handled by the hardware descriptor path during lowering.
    a_rsrc = S.amdgpu.make_rsrc(A, A_RANGE_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(B, B_RANGE_BYTES)

    # Ping-pong buffers for 16x64 operand tiles. B is staged and repacked in 8-row slices
    # so each MFMA step touches a smaller LDS working set.
    a_shared = S.make_shared((2, 2, 2, 64, 2), S.u32)
    b_stage = S.make_shared((2, 2, 8, 64), S.bf16)
    b_shared = S.make_shared((2, 2, 2, 64, 2), S.u32)

    acc = S.full((16,), 0.0, S.f32)
    zero = S.convert(0, S.i32)

    # Prefetch k_block 0 into LDS buffer 0.
    if tid < 128:
        a_row = tid // 2
        a_k8 = tid % 2
        a_elem = (block_row + a_row) * K + a_k8 * 8
        a_byte = S.convert(a_elem * 2, S.i32)
        a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_byte, zero, zero)
        a_pack = S.view(a_vec, S.Tensor((2, 2), S.u32))
        a_warp = a_row // 32
        a_lane = a_row % 32
        a_shared[0, a_k8, a_warp, a_lane] = a_pack[0]
        a_shared[0, a_k8, a_warp, a_lane + 32] = a_pack[1]
    else:
        b_tid = tid - 128
        b_k = b_tid // 8
        b_n8 = b_tid % 8
        b_step = b_k // 8
        b_row = b_k % 8
        b_elem = b_k * N + block_col + b_n8 * 8
        b_byte = S.convert(b_elem * 2, S.i32)
        b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_byte, zero, zero)
        b_pack = S.view(b_vec, S.Tensor((2, 4, 1), S.bf16))
        for ii in S.range(4):
            b_stage[0, b_step, b_row, b_n8 * 8 + ii] = b_pack[0, ii, 0]
            b_stage[0, b_step, b_row, b_n8 * 8 + 4 + ii] = b_pack[1, ii, 0]

    S.syncthreads()

    # Prefetch k_block 1 into LDS buffer 1.
    if tid < 128:
        a_row = tid // 2
        a_k8 = tid % 2
        a_elem = (block_row + a_row) * K + BLOCK_K + a_k8 * 8
        a_byte = S.convert(a_elem * 2, S.i32)
        a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_byte, zero, zero)
        a_pack = S.view(a_vec, S.Tensor((2, 2), S.u32))
        a_warp = a_row // 32
        a_lane = a_row % 32
        a_shared[1, a_k8, a_warp, a_lane] = a_pack[0]
        a_shared[1, a_k8, a_warp, a_lane + 32] = a_pack[1]
    else:
        b_tid = tid - 128
        b_k = b_tid // 8
        b_n8 = b_tid % 8
        b_step = b_k // 8
        b_row = b_k % 8
        b_elem = (BLOCK_K + b_k) * N + block_col + b_n8 * 8
        b_byte = S.convert(b_elem * 2, S.i32)
        b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_byte, zero, zero)
        b_pack = S.view(b_vec, S.Tensor((2, 4, 1), S.bf16))
        for ii in S.range(4):
            b_stage[1, b_step, b_row, b_n8 * 8 + ii] = b_pack[0, ii, 0]
            b_stage[1, b_step, b_row, b_n8 * 8 + 4 + ii] = b_pack[1, ii, 0]

    S.syncthreads()

    # Compute k_block 0 from buffer 0, 8 rows at a time.
    for k_step in S.range(2):
        b_repack_lane = tid % 64
        b_repack_warp = (tid % 128) // 64
        b_col = b_repack_warp * 32 + (b_repack_lane % 32)
        b_quad = 4 * (b_repack_lane // 32)
        b0 = S.convert(S.bitcast(b_stage[0, k_step, b_quad + 0, b_col], S.u16), S.u32)
        b1 = S.convert(S.bitcast(b_stage[0, k_step, b_quad + 1, b_col], S.u16), S.u32)
        b2 = S.convert(S.bitcast(b_stage[0, k_step, b_quad + 2, b_col], S.u16), S.u32)
        b3 = S.convert(S.bitcast(b_stage[0, k_step, b_quad + 3, b_col], S.u16), S.u32)
        b_shared[0, k_step, b_repack_warp, b_repack_lane, 0] = (
            b0 | (b1 << S.convert(16, S.u32))
        )
        b_shared[0, k_step, b_repack_warp, b_repack_lane, 1] = (
            b2 | (b3 << S.convert(16, S.u32))
        )

        S.syncthreads()

        a_frag = S.view(a_shared[0, k_step, warp_row, lane], S.Tensor((1, 4, 1), S.bf16))
        b_frag = S.view(b_shared[0, k_step, warp_col, lane], S.Tensor((1, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)

    S.syncthreads()

    # While buffer 1 is consumed next, prefetch k_block 2 into buffer 0.
    if tid < 128:
        a_row = tid // 2
        a_k8 = tid % 2
        a_elem = (block_row + a_row) * K + 2 * BLOCK_K + a_k8 * 8
        a_byte = S.convert(a_elem * 2, S.i32)
        a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_byte, zero, zero)
        a_pack = S.view(a_vec, S.Tensor((2, 2), S.u32))
        a_warp = a_row // 32
        a_lane = a_row % 32
        a_shared[0, a_k8, a_warp, a_lane] = a_pack[0]
        a_shared[0, a_k8, a_warp, a_lane + 32] = a_pack[1]
    else:
        b_tid = tid - 128
        b_k = b_tid // 8
        b_n8 = b_tid % 8
        b_step = b_k // 8
        b_row = b_k % 8
        b_elem = (2 * BLOCK_K + b_k) * N + block_col + b_n8 * 8
        b_byte = S.convert(b_elem * 2, S.i32)
        b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_byte, zero, zero)
        b_pack = S.view(b_vec, S.Tensor((2, 4, 1), S.bf16))
        for ii in S.range(4):
            b_stage[0, b_step, b_row, b_n8 * 8 + ii] = b_pack[0, ii, 0]
            b_stage[0, b_step, b_row, b_n8 * 8 + 4 + ii] = b_pack[1, ii, 0]

    S.syncthreads()

    # Compute k_block 1 from buffer 1.
    for k_step in S.range(2):
        b_repack_lane = tid % 64
        b_repack_warp = (tid % 128) // 64
        b_col = b_repack_warp * 32 + (b_repack_lane % 32)
        b_quad = 4 * (b_repack_lane // 32)
        b0 = S.convert(S.bitcast(b_stage[1, k_step, b_quad + 0, b_col], S.u16), S.u32)
        b1 = S.convert(S.bitcast(b_stage[1, k_step, b_quad + 1, b_col], S.u16), S.u32)
        b2 = S.convert(S.bitcast(b_stage[1, k_step, b_quad + 2, b_col], S.u16), S.u32)
        b3 = S.convert(S.bitcast(b_stage[1, k_step, b_quad + 3, b_col], S.u16), S.u32)
        b_shared[1, k_step, b_repack_warp, b_repack_lane, 0] = (
            b0 | (b1 << S.convert(16, S.u32))
        )
        b_shared[1, k_step, b_repack_warp, b_repack_lane, 1] = (
            b2 | (b3 << S.convert(16, S.u32))
        )

        S.syncthreads()

        a_frag = S.view(a_shared[1, k_step, warp_row, lane], S.Tensor((1, 4, 1), S.bf16))
        b_frag = S.view(b_shared[1, k_step, warp_col, lane], S.Tensor((1, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)

    S.syncthreads()

    # While buffer 0 is consumed next, prefetch k_block 3 into buffer 1.
    if tid < 128:
        a_row = tid // 2
        a_k8 = tid % 2
        a_elem = (block_row + a_row) * K + 3 * BLOCK_K + a_k8 * 8
        a_byte = S.convert(a_elem * 2, S.i32)
        a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, a_byte, zero, zero)
        a_pack = S.view(a_vec, S.Tensor((2, 2), S.u32))
        a_warp = a_row // 32
        a_lane = a_row % 32
        a_shared[1, a_k8, a_warp, a_lane] = a_pack[0]
        a_shared[1, a_k8, a_warp, a_lane + 32] = a_pack[1]
    else:
        b_tid = tid - 128
        b_k = b_tid // 8
        b_n8 = b_tid % 8
        b_step = b_k // 8
        b_row = b_k % 8
        b_elem = (3 * BLOCK_K + b_k) * N + block_col + b_n8 * 8
        b_byte = S.convert(b_elem * 2, S.i32)
        b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, b_byte, zero, zero)
        b_pack = S.view(b_vec, S.Tensor((2, 4, 1), S.bf16))
        for ii in S.range(4):
            b_stage[1, b_step, b_row, b_n8 * 8 + ii] = b_pack[0, ii, 0]
            b_stage[1, b_step, b_row, b_n8 * 8 + 4 + ii] = b_pack[1, ii, 0]

    S.syncthreads()

    # Compute k_block 2 from buffer 0.
    for k_step in S.range(2):
        b_repack_lane = tid % 64
        b_repack_warp = (tid % 128) // 64
        b_col = b_repack_warp * 32 + (b_repack_lane % 32)
        b_quad = 4 * (b_repack_lane // 32)
        b0 = S.convert(S.bitcast(b_stage[0, k_step, b_quad + 0, b_col], S.u16), S.u32)
        b1 = S.convert(S.bitcast(b_stage[0, k_step, b_quad + 1, b_col], S.u16), S.u32)
        b2 = S.convert(S.bitcast(b_stage[0, k_step, b_quad + 2, b_col], S.u16), S.u32)
        b3 = S.convert(S.bitcast(b_stage[0, k_step, b_quad + 3, b_col], S.u16), S.u32)
        b_shared[0, k_step, b_repack_warp, b_repack_lane, 0] = (
            b0 | (b1 << S.convert(16, S.u32))
        )
        b_shared[0, k_step, b_repack_warp, b_repack_lane, 1] = (
            b2 | (b3 << S.convert(16, S.u32))
        )

        S.syncthreads()

        a_frag = S.view(a_shared[0, k_step, warp_row, lane], S.Tensor((1, 4, 1), S.bf16))
        b_frag = S.view(b_shared[0, k_step, warp_col, lane], S.Tensor((1, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)

    S.syncthreads()

    # Compute k_block 3 from buffer 1.
    for k_step in S.range(2):
        b_repack_lane = tid % 64
        b_repack_warp = (tid % 128) // 64
        b_col = b_repack_warp * 32 + (b_repack_lane % 32)
        b_quad = 4 * (b_repack_lane // 32)
        b0 = S.convert(S.bitcast(b_stage[1, k_step, b_quad + 0, b_col], S.u16), S.u32)
        b1 = S.convert(S.bitcast(b_stage[1, k_step, b_quad + 1, b_col], S.u16), S.u32)
        b2 = S.convert(S.bitcast(b_stage[1, k_step, b_quad + 2, b_col], S.u16), S.u32)
        b3 = S.convert(S.bitcast(b_stage[1, k_step, b_quad + 3, b_col], S.u16), S.u32)
        b_shared[1, k_step, b_repack_warp, b_repack_lane, 0] = (
            b0 | (b1 << S.convert(16, S.u32))
        )
        b_shared[1, k_step, b_repack_warp, b_repack_lane, 1] = (
            b2 | (b3 << S.convert(16, S.u32))
        )

        S.syncthreads()

        a_frag = S.view(a_shared[1, k_step, warp_row, lane], S.Tensor((1, 4, 1), S.bf16))
        b_frag = S.view(b_shared[1, k_step, warp_col, lane], S.Tensor((1, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)

    S.syncthreads()

    lane_col = lane % 32
    lane_row_quad = 4 * (lane // 32)

    for acc_idx in S.range(16):
        out_row = (
            block_row
            + warp_row * 32
            + 8 * (acc_idx // 4)
            + lane_row_quad
            + (acc_idx % 4)
        )
        out_col = block_col + warp_col * 32 + lane_col
        C[out_row, out_col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise ValueError("ModelNew expects A=(32768,64) and B=(64,32768)")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError("ModelNew expects bf16 inputs")
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)
        gemm_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS, 1, 1))](A, B, C)
        return C
