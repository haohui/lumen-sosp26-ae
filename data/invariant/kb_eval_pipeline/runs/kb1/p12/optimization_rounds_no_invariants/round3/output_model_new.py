import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 4096
N = 4096
BLOCK_M = 64
BLOCK_N = 64
WAVE_SIZE = 64
THREADS = 256


@substrate.jit
def diag_left_kernel(
    A: S.Tensor((4096,), S.bf16),
    B: S.Tensor((4096, 4096), S.bf16),
    C: S.Tensor((4096, 4096), S.bf16),
    a_range_bytes: S.i32,
    b_range_bytes: S.i32,
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    wave = tid // WAVE_SIZE

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N

    warp_row = wave // 2
    warp_col = wave % 2

    wave_row = block_row + warp_row * 32
    wave_col = block_col + warp_col * 32

    row_in_wave = lane % 32
    col_group = lane // 32
    row = wave_row + row_in_wave
    col0 = wave_col + col_group * 16

    zero = S.convert(0, S.i32)
    two = S.convert(2, S.i32)
    eight = S.convert(8, S.i32)
    sixteen = S.convert(16, S.i32)
    mat_stride_elems = S.convert(4096, S.i32)

    a_rsrc = S.amdgpu.make_rsrc(A, a_range_bytes)
    b_rsrc = S.amdgpu.make_rsrc(B, b_range_bytes)
    a_words = S.make_shared((2, 16, 4), S.u32)
    b_words = S.make_shared((2, 512, 4), S.u32)

    a_slot = wave * 4 + row_in_wave // 8
    a_pack_row = wave_row + (a_slot - wave * 4) * 8
    a_offset = S.convert(a_pack_row, S.i32) * two

    b_slot0 = wave * 128 + lane * 2
    b_row_offset = S.convert(row, S.i32) * mat_stride_elems + S.convert(col0, S.i32)
    b_offset0 = b_row_offset * two
    b_offset1 = (b_row_offset + eight) * two

    if lane < 4:
        a_words[0, wave * 4 + lane] = S.amdgpu.raw_buffer_load_x4(
            a_rsrc, zero, a_offset + S.convert(lane, S.i32) * sixteen, 0
        )
    b_words[0, b_slot0 + 0] = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset0, 0)
    b_words[0, b_slot0 + 1] = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset1, 0)

    S.syncthreads()

    a_frag0 = S.view(a_words[0, wave * 4 + (lane % 4)], S.Tensor((2, 4, 1), S.bf16))
    b_frag0 = S.view(b_words[0, b_slot0], S.Tensor((2, 4, 1), S.bf16))
    mfma_acc = S.full((16,), 0.0, S.f32)
    mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], mfma_acc)

    if lane < 4:
        a_words[1, wave * 4 + lane] = S.amdgpu.raw_buffer_load_x4(
            a_rsrc, zero, a_offset + S.convert(lane, S.i32) * sixteen, 0
        )
    b_words[1, b_slot0 + 0] = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset0, 0)
    b_words[1, b_slot0 + 1] = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset1, 0)

    mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], mfma_acc)

    S.syncthreads()

    a_frag1 = S.view(a_words[1, wave * 4 + (lane % 4)], S.Tensor((2, 4, 1), S.bf16))
    b_frag1 = S.view(b_words[1, b_slot0], S.Tensor((2, 4, 1), S.bf16))
    mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], mfma_acc)
    mfma_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], mfma_acc)

    a_vals = S.view(a_words[0, a_slot], S.Tensor((8,), S.bf16))
    row_scale = a_vals[row_in_wave % 8]

    b_vals0 = S.view(b_words[0, b_slot0 + 0], S.Tensor((8,), S.bf16))
    b_vals1 = S.view(b_words[0, b_slot0 + 1], S.Tensor((8,), S.bf16))

    for j in S.range(8):
        C[row, col0 + j] = row_scale * b_vals0[j]
    for j in S.range(8):
        C[row, col0 + 8 + j] = row_scale * b_vals1[j]


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._a_range_bytes = M * 2
        self._b_range_bytes = M * N * 2

    def forward(self, A, B):
        if tuple(A.shape) != (4096,) or tuple(B.shape) != (4096, 4096):
            raise RuntimeError("ModelNew expects A=(4096,) and B=(4096, 4096)")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise RuntimeError("ModelNew expects bfloat16 inputs")
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((4096, 4096), device=B.device, dtype=B.dtype)
        diag_left_kernel[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS, 1, 1))](
            A,
            B,
            C,
            self._a_range_bytes,
            self._b_range_bytes,
        )
        return C
