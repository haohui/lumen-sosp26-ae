import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 8205
K = 2949
N = 5921

BLOCK_M = 64
BLOCK_N = 64
WARP_M = 32
WARP_N = 32
WARP_SIZE = 64
WAVES_PER_BLOCK = 4
K_TILE = 16
NUM_K_TILES = 185
NUM_PIPE_PAIRS = 92

M_PAD = 8256
K_PAD = 2960
N_PAD = 5952

A_RANGE_BYTES = M * K * 2
B_RANGE_BYTES = K * N * 2
GRID_M = (M + BLOCK_M - 1) // BLOCK_M
GRID_N = (N + BLOCK_N - 1) // BLOCK_N


@substrate.jit
def gemm_mfma_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M_PAD, N_PAD), S.bf16),
    a_range_bytes: S.i32,
    b_range_bytes: S.i32,
):
    tid = S.thread_id(0)
    lane = tid % WARP_SIZE
    warp_id = tid // WARP_SIZE

    warp_row = warp_id // 2
    warp_col = warp_id % 2
    lane_group = lane // 32
    lane_rowcol = lane % 32

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N
    warp_row_base = block_row + warp_row * WARP_M
    warp_col_base = block_col + warp_col * WARP_N

    a_stage = S.make_shared((2, 2, 2, 2, 32, 4), S.bf16)
    b_stage = S.make_shared((2, 2, 2, 2, 32, 4), S.bf16)
    acc = S.full((16,), 0.0, S.f32)

    a_rsrc = S.amdgpu.make_rsrc(A, a_range_bytes)
    b_rsrc = S.amdgpu.make_rsrc(B, b_range_bytes)
    zero = S.convert(0, S.i32)

    if tid < 128:
        a_loader = tid
        a_warp_row = a_loader // 64
        a_row_local = a_loader % 64
        a_row = block_row + a_warp_row * 32 + (a_row_local % 32)
        a_kpair = a_row_local // 32

        a_offset = S.convert(((a_row * K) + a_kpair * 8) * 2, S.i32)
        a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset, 0)
        a_loaded = S.view(a_vec, S.Tensor((2, 4, 1), S.bf16))
        for elem_idx in S.range(4):
            a_stage[0, a_warp_row, a_kpair, 0, a_row_local % 32, elem_idx] = a_loaded[0, elem_idx, 0]
            a_stage[0, a_warp_row, a_kpair, 1, a_row_local % 32, elem_idx] = a_loaded[1, elem_idx, 0]

        a_offset = S.convert(((a_row * K) + 16 + a_kpair * 8) * 2, S.i32)
        a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset, 0)
        a_loaded = S.view(a_vec, S.Tensor((2, 4, 1), S.bf16))
        for elem_idx in S.range(4):
            a_stage[1, a_warp_row, a_kpair, 0, a_row_local % 32, elem_idx] = a_loaded[0, elem_idx, 0]
            a_stage[1, a_warp_row, a_kpair, 1, a_row_local % 32, elem_idx] = a_loaded[1, elem_idx, 0]
    else:
        b_loader = tid - 128
        b_k_local = b_loader % 16
        b_col_chunk = b_loader // 16
        b_warp_col = b_col_chunk // 4
        b_col_local = (b_col_chunk % 4) * 8
        b_stage_idx = b_k_local // 8
        b_k_group = (b_k_local % 8) // 4
        b_k_elem = b_k_local % 4

        b_offset = S.convert(((b_k_local * N) + block_col + b_col_chunk * 8) * 2, S.i32)
        b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset, 0)
        b_loaded = S.view(b_vec, S.Tensor((2, 4, 1), S.bf16))
        for elem_idx in S.range(4):
            b_stage[0, b_warp_col, b_stage_idx, b_k_group, b_col_local + elem_idx, b_k_elem] = b_loaded[0, elem_idx, 0]
            b_stage[0, b_warp_col, b_stage_idx, b_k_group, b_col_local + 4 + elem_idx, b_k_elem] = b_loaded[1, elem_idx, 0]

        b_offset = S.convert((((16 + b_k_local) * N) + block_col + b_col_chunk * 8) * 2, S.i32)
        b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset, 0)
        b_loaded = S.view(b_vec, S.Tensor((2, 4, 1), S.bf16))
        for elem_idx in S.range(4):
            b_stage[1, b_warp_col, b_stage_idx, b_k_group, b_col_local + elem_idx, b_k_elem] = b_loaded[0, elem_idx, 0]
            b_stage[1, b_warp_col, b_stage_idx, b_k_group, b_col_local + 4 + elem_idx, b_k_elem] = b_loaded[1, elem_idx, 0]

    S.syncthreads()

    for pair_idx in S.range(NUM_PIPE_PAIRS):
        a_frag = a_stage[0, warp_row, 0, lane_group, lane_rowcol]
        b_frag = b_stage[0, warp_col, 0, lane_group, lane_rowcol]
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

        a_frag = a_stage[0, warp_row, 1, lane_group, lane_rowcol]
        b_frag = b_stage[0, warp_col, 1, lane_group, lane_rowcol]
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

        if pair_idx < (NUM_PIPE_PAIRS - 1):
            k_even = (pair_idx + 1) * 32
            if tid < 128:
                a_loader = tid
                a_warp_row = a_loader // 64
                a_row_local = a_loader % 64
                a_row = block_row + a_warp_row * 32 + (a_row_local % 32)
                a_kpair = a_row_local // 32

                a_offset = S.convert(((a_row * K) + k_even + a_kpair * 8) * 2, S.i32)
                a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset, 0)
                a_loaded = S.view(a_vec, S.Tensor((2, 4, 1), S.bf16))
                for elem_idx in S.range(4):
                    a_stage[0, a_warp_row, a_kpair, 0, a_row_local % 32, elem_idx] = a_loaded[0, elem_idx, 0]
                    a_stage[0, a_warp_row, a_kpair, 1, a_row_local % 32, elem_idx] = a_loaded[1, elem_idx, 0]
            else:
                b_loader = tid - 128
                b_k_local = b_loader % 16
                b_col_chunk = b_loader // 16
                b_warp_col = b_col_chunk // 4
                b_col_local = (b_col_chunk % 4) * 8
                b_stage_idx = b_k_local // 8
                b_k_group = (b_k_local % 8) // 4
                b_k_elem = b_k_local % 4

                b_offset = S.convert((((k_even + b_k_local) * N) + block_col + b_col_chunk * 8) * 2, S.i32)
                b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset, 0)
                b_loaded = S.view(b_vec, S.Tensor((2, 4, 1), S.bf16))
                for elem_idx in S.range(4):
                    b_stage[0, b_warp_col, b_stage_idx, b_k_group, b_col_local + elem_idx, b_k_elem] = b_loaded[0, elem_idx, 0]
                    b_stage[0, b_warp_col, b_stage_idx, b_k_group, b_col_local + 4 + elem_idx, b_k_elem] = b_loaded[1, elem_idx, 0]

        a_frag = a_stage[1, warp_row, 0, lane_group, lane_rowcol]
        b_frag = b_stage[1, warp_col, 0, lane_group, lane_rowcol]
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

        a_frag = a_stage[1, warp_row, 1, lane_group, lane_rowcol]
        b_frag = b_stage[1, warp_col, 1, lane_group, lane_rowcol]
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

        if pair_idx < (NUM_PIPE_PAIRS - 1):
            k_odd = (pair_idx + 1) * 32 + 16
            if tid < 128:
                a_loader = tid
                a_warp_row = a_loader // 64
                a_row_local = a_loader % 64
                a_row = block_row + a_warp_row * 32 + (a_row_local % 32)
                a_kpair = a_row_local // 32

                a_offset = S.convert(((a_row * K) + k_odd + a_kpair * 8) * 2, S.i32)
                a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset, 0)
                a_loaded = S.view(a_vec, S.Tensor((2, 4, 1), S.bf16))
                for elem_idx in S.range(4):
                    a_stage[1, a_warp_row, a_kpair, 0, a_row_local % 32, elem_idx] = a_loaded[0, elem_idx, 0]
                    a_stage[1, a_warp_row, a_kpair, 1, a_row_local % 32, elem_idx] = a_loaded[1, elem_idx, 0]
            else:
                b_loader = tid - 128
                b_k_local = b_loader % 16
                b_col_chunk = b_loader // 16
                b_warp_col = b_col_chunk // 4
                b_col_local = (b_col_chunk % 4) * 8
                b_stage_idx = b_k_local // 8
                b_k_group = (b_k_local % 8) // 4
                b_k_elem = b_k_local % 4

                b_offset = S.convert((((k_odd + b_k_local) * N) + block_col + b_col_chunk * 8) * 2, S.i32)
                b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset, 0)
                b_loaded = S.view(b_vec, S.Tensor((2, 4, 1), S.bf16))
                for elem_idx in S.range(4):
                    b_stage[1, b_warp_col, b_stage_idx, b_k_group, b_col_local + elem_idx, b_k_elem] = b_loaded[0, elem_idx, 0]
                    b_stage[1, b_warp_col, b_stage_idx, b_k_group, b_col_local + 4 + elem_idx, b_k_elem] = b_loaded[1, elem_idx, 0]

        S.syncthreads()

    if tid < 128:
        a_loader = tid
        a_warp_row = a_loader // 64
        a_row_local = a_loader % 64
        a_row = block_row + a_warp_row * 32 + (a_row_local % 32)
        a_kpair = a_row_local // 32

        a_offset = S.convert(((a_row * K) + 2944 + a_kpair * 8) * 2, S.i32)
        a_vec = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset, 0)
        a_loaded = S.view(a_vec, S.Tensor((2, 4, 1), S.bf16))
        for elem_idx in S.range(4):
            a_stage[0, a_warp_row, a_kpair, 0, a_row_local % 32, elem_idx] = a_loaded[0, elem_idx, 0]
            a_stage[0, a_warp_row, a_kpair, 1, a_row_local % 32, elem_idx] = a_loaded[1, elem_idx, 0]
    else:
        b_loader = tid - 128
        b_k_local = b_loader % 16
        b_col_chunk = b_loader // 16
        b_warp_col = b_col_chunk // 4
        b_col_local = (b_col_chunk % 4) * 8
        b_stage_idx = b_k_local // 8
        b_k_group = (b_k_local % 8) // 4
        b_k_elem = b_k_local % 4

        b_offset = S.convert((((2944 + b_k_local) * N) + block_col + b_col_chunk * 8) * 2, S.i32)
        b_vec = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset, 0)
        b_loaded = S.view(b_vec, S.Tensor((2, 4, 1), S.bf16))
        for elem_idx in S.range(4):
            b_stage[0, b_warp_col, b_stage_idx, b_k_group, b_col_local + elem_idx, b_k_elem] = b_loaded[0, elem_idx, 0]
            b_stage[0, b_warp_col, b_stage_idx, b_k_group, b_col_local + 4 + elem_idx, b_k_elem] = b_loaded[1, elem_idx, 0]

    S.syncthreads()

    a_frag = a_stage[0, warp_row, 0, lane_group, lane_rowcol]
    b_frag = b_stage[0, warp_col, 0, lane_group, lane_rowcol]
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

    a_frag = a_stage[0, warp_row, 1, lane_group, lane_rowcol]
    b_frag = b_stage[0, warp_col, 1, lane_group, lane_rowcol]
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

    for acc_idx in S.range(16):
        row = warp_row_base + 8 * (acc_idx // 4) + 4 * lane_group + (acc_idx % 4)
        col = warp_col_base + lane_rowcol
        C[row, col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = {}

    def _get_buffers(self, A: torch.Tensor):
        key = (A.device.type, A.device.index)
        cached = self._cache.get(key)
        if cached is None:
            c_pad = torch.empty((M_PAD, N_PAD), device=A.device, dtype=torch.bfloat16)
            cached = {
                "c_pad": c_pad,
                "c_view": c_pad[:M, :N],
            }
            self._cache[key] = cached
        return cached

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise ValueError(f"Expected A {(M, K)} and B {(K, N)}, got {tuple(A.shape)} and {tuple(B.shape)}")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise TypeError("ModelNew expects bfloat16 inputs")

        A = A.contiguous()
        B = B.contiguous()
        bufs = self._get_buffers(A)
        c_pad = bufs["c_pad"]

        gemm_mfma_kernel[lambda: ((GRID_N, GRID_M, 1), (WAVES_PER_BLOCK * WARP_SIZE, 1, 1))](
            A,
            B,
            c_pad,
            A_RANGE_BYTES,
            B_RANGE_BYTES,
        )
        return bufs["c_view"]
