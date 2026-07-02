import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 32768
K = 32
N = 32768

BLOCK_M = 64
BLOCK_N = 64
WAVE_SIZE = 64
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = WAVE_SIZE * WAVES_PER_BLOCK
BF16_BYTES = 2


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

    shared_a = S.make_shared((BLOCK_M, K), S.bf16)
    shared_b = S.make_shared((K, BLOCK_N), S.bf16)
    scratch_a = S.make_shared((THREADS_PER_BLOCK, 8), S.bf16)
    scratch_b = S.make_shared((THREADS_PER_BLOCK, 8), S.bf16)

    a_rsrc = S.amdgpu.make_rsrc(A, a_range_bytes)
    b_rsrc = S.amdgpu.make_rsrc(B, b_range_bytes)

    zero = S.convert(0, S.i32)

    a_linear = tid * 8
    a_row = a_linear // K
    a_col = a_linear % K
    a_offset = ((block_row + a_row) * K + a_col) * BF16_BYTES
    a_pack = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, S.convert(a_offset, S.i32), 0)
    a_vals = S.view(a_pack, S.Tensor((8,), S.bf16))
    for i in S.range(8):
        shared_a[a_row, a_col + i] = a_vals[i]

    b_linear = tid * 8
    b_row = b_linear // BLOCK_N
    b_col = b_linear % BLOCK_N
    b_offset = (b_row * N + block_col + b_col) * BF16_BYTES
    b_pack = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, S.convert(b_offset, S.i32), 0)
    b_vals = S.view(b_pack, S.Tensor((8,), S.bf16))
    for i in S.range(8):
        shared_b[b_row, b_col + i] = b_vals[i]

    S.syncthreads()

    acc = S.full((16,), 0.0, S.f32)
    wave_row = warp_row * 32
    wave_col = warp_col * 32
    lane_row = lane % 32
    lane_col = lane % 32
    lane_k8 = (lane // 32) * 8

    for phase in S.range(2):
        phase_k = phase * 16 + lane_k8
        for i in S.range(8):
            scratch_a[tid, i] = shared_a[wave_row + lane_row, phase_k + i]
            scratch_b[tid, i] = shared_b[phase_k + i, wave_col + lane_col]

        a_frag = S.view(scratch_a[tid], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(scratch_b[tid], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

    out_col = block_col + wave_col + lane_col
    row_group = (lane // 32) * 4
    for i in S.range(16):
        out_row = block_row + wave_row + (i // 4) * 8 + row_group + (i % 4)
        C[out_row, out_col] = S.convert(acc[i], S.bf16)


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
