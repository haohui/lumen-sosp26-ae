import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 32768
K = 32
N = 32768

BLOCK_M = 64
BLOCK_N = 64
STAGE_K = 16
WAVE_SIZE = 64
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = WAVE_SIZE * WAVES_PER_BLOCK
BF16_BYTES = 2
A_LOAD_THREADS = 128
B_LOAD_THREADS = 128


@substrate.jit
def gemm_kernel_mfma(
    A: S.Tensor((32768, 32), S.bf16),
    B: S.Tensor((32, 32768), S.bf16),
    C: S.Tensor((32768, 32768), S.bf16),
    a_range_bytes: S.i32,
    b_range_bytes: S.i32,
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    warp = tid // WAVE_SIZE
    warp_row = warp // 2
    warp_col = warp % 2

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N

    shared_a = S.make_shared((2, BLOCK_M, STAGE_K), S.bf16)
    shared_b = S.make_shared((2, STAGE_K, BLOCK_N), S.bf16)
    scratch_a = S.make_shared((2, THREADS_PER_BLOCK, 8), S.bf16)
    scratch_b = S.make_shared((2, THREADS_PER_BLOCK, 8), S.bf16)
    shared_c = S.make_shared((BLOCK_M, BLOCK_N), S.bf16)

    a_rsrc = S.amdgpu.make_rsrc(A, a_range_bytes)
    b_rsrc = S.amdgpu.make_rsrc(B, b_range_bytes)
    zero = S.convert(0, S.i32)

    if tid < A_LOAD_THREADS:
        a_linear = tid * 8
        a_row = a_linear // STAGE_K
        a_col = a_linear % STAGE_K
        a_offset = ((block_row + a_row) * K + a_col) * BF16_BYTES
        a_pack = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, S.convert(a_offset, S.i32), 0)
        a_vals = S.view(a_pack, S.Tensor((8,), S.bf16))
        for i in S.range(8):
            shared_a[0, a_row, a_col + i] = a_vals[i]
    else:
        b_tid = tid - A_LOAD_THREADS
        b_linear = b_tid * 8
        b_row = b_linear // BLOCK_N
        b_col = b_linear % BLOCK_N
        b_offset = (b_row * N + block_col + b_col) * BF16_BYTES
        b_pack = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, S.convert(b_offset, S.i32), 0)
        b_vals = S.view(b_pack, S.Tensor((8,), S.bf16))
        for i in S.range(8):
            shared_b[0, b_row, b_col + i] = b_vals[i]

    if tid < A_LOAD_THREADS:
        a_linear = tid * 8
        a_row = a_linear // STAGE_K
        a_col = a_linear % STAGE_K
        a_offset = ((block_row + a_row) * K + STAGE_K + a_col) * BF16_BYTES
        a_pack = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, S.convert(a_offset, S.i32), 0)
        a_vals = S.view(a_pack, S.Tensor((8,), S.bf16))
        for i in S.range(8):
            shared_a[1, a_row, a_col + i] = a_vals[i]
    else:
        b_tid = tid - A_LOAD_THREADS
        b_linear = b_tid * 8
        b_row = b_linear // BLOCK_N
        b_col = b_linear % BLOCK_N
        b_offset = ((STAGE_K + b_row) * N + block_col + b_col) * BF16_BYTES
        b_pack = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, S.convert(b_offset, S.i32), 0)
        b_vals = S.view(b_pack, S.Tensor((8,), S.bf16))
        for i in S.range(8):
            shared_b[1, b_row, b_col + i] = b_vals[i]

    S.syncthreads()

    acc = S.full((16,), 0.0, S.f32)
    wave_row = warp_row * 32
    wave_col = warp_col * 32
    lane_row = lane % 32
    lane_col = lane % 32
    lane_k8 = (lane // 32) * 8

    for i in S.range(8):
        scratch_a[0, tid, i] = shared_a[0, wave_row + lane_row, lane_k8 + i]
        scratch_b[0, tid, i] = shared_b[0, lane_k8 + i, wave_col + lane_col]

    a_frag0 = S.view(scratch_a[0, tid], S.Tensor((2, 4, 1), S.bf16))
    b_frag0 = S.view(scratch_b[0, tid], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)

    for i in S.range(8):
        scratch_a[1, tid, i] = shared_a[1, wave_row + lane_row, lane_k8 + i]
        scratch_b[1, tid, i] = shared_b[1, lane_k8 + i, wave_col + lane_col]

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

    a_frag1 = S.view(scratch_a[1, tid], S.Tensor((2, 4, 1), S.bf16))
    b_frag1 = S.view(scratch_b[1, tid], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

    row_group = (lane // 32) * 4
    for i in S.range(16):
        out_row = wave_row + (i // 4) * 8 + row_group + (i % 4)
        shared_c[out_row, wave_col + lane_col] = S.convert(acc[i], S.bf16)

    S.syncthreads()

    store_row = tid // 4
    store_col = (tid % 4) * 16
    c_row_view = S.subview(C, (block_row + store_row, 0), (1, N), (1, 1))
    c_row_rsrc = S.amdgpu.make_rsrc(c_row_view, N * BF16_BYTES)
    c_base_offset = (block_col + store_col) * BF16_BYTES
    for i in S.range(8):
        scratch_a[0, tid, i] = shared_c[store_row, store_col + i]
        scratch_b[0, tid, i] = shared_c[store_row, store_col + 8 + i]
    c_pack0 = S.view(scratch_a[0, tid], S.Tensor((4,), S.u32))
    c_pack1 = S.view(scratch_b[0, tid], S.Tensor((4,), S.u32))
    S.amdgpu.raw_buffer_store_x4(c_pack0, c_row_rsrc, zero, S.convert(c_base_offset, S.i32), 0)
    S.amdgpu.raw_buffer_store_x4(
        c_pack1, c_row_rsrc, zero, S.convert(c_base_offset + 8 * BF16_BYTES, S.i32), 0
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._launch = None
        self._cached_ptrs = None

    def _get_launch(self, A, B):
        a_ptr = A.untyped_storage().data_ptr()
        b_ptr = B.untyped_storage().data_ptr()
        ptrs = (a_ptr, b_ptr)
        if self._launch is None or self._cached_ptrs != ptrs:
            self._cached_ptrs = ptrs
            self._launch = gemm_kernel_mfma[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))]
        return self._launch

    def forward(self, A, B):
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        launch = self._get_launch(A, B)
        launch(
            A,
            B,
            C,
            A.numel() * A.element_size(),
            B.numel() * B.element_size(),
        )
        return C
