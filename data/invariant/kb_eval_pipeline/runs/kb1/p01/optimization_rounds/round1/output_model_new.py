import torch
import torch.nn as nn

import substrate
import substrate.language as S


N = 4096
BLOCK_M = 64
BLOCK_N = 64
WAVE_SIZE = 64
WAVES_M = 2
WAVES_N = 2
BLOCK_THREADS = WAVE_SIZE * WAVES_M * WAVES_N
K_CHUNK = 16
K_TILES = N // K_CHUNK
RANGE_BYTES = N * N * 2


@substrate.jit
def gemm_kernel(
    A: S.Tensor((N, N), S.bf16),
    B: S.Tensor((N, N), S.bf16),
    C: S.Tensor((N, N), S.bf16),
):
    tid = S.thread_id(0)
    wave_id = tid >> 6
    lane = tid % WAVE_SIZE
    warp_row = wave_id >> 1
    warp_col = wave_id % 2

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N
    tile_row_base = block_row + warp_row * 32
    tile_col_base = block_col + warp_col * 32

    zero = S.convert(0, S.i32)
    a_rsrc = S.amdgpu.make_rsrc(A, S.convert(RANGE_BYTES, S.i32))
    b_rsrc = S.amdgpu.make_rsrc(B, S.convert(RANGE_BYTES, S.i32))

    # Per-wave swizzled LDS staging buffers. Each row is one 16-byte fragment.
    a_smem = S.make_shared((4 * 64 * 4,), S.u32)
    b_smem = S.make_shared((4 * 64 * 4,), S.u32)
    acc = S.make_local((16,), S.f32)
    frag_layout = S.make_layout((2, 4), (4, 1))
    b_lane_layout = S.make_layout((4 * 64, 8), (8, 1))
    b_smem_bf16 = S.view(b_smem, S.bf16, b_lane_layout)

    for i in S.range(16):
        acc[i] = S.convert(0.0, S.f32)

    for k_tile in S.range(K_TILES):
        k_base = k_tile * K_CHUNK

        a_row = lane >> 1
        a_half = lane & 1
        a_byte_off = ((tile_row_base + a_row) * N + k_base + a_half * 8) * 2
        a_words = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, S.convert(a_byte_off, S.i32), 0)

        a_base0 = (wave_id * 64 + a_row) * 4 + a_half * 2
        a_base1 = (wave_id * 64 + a_row + 32) * 4 + a_half * 2
        a_smem[a_base0] = a_words[0]
        a_smem[a_base0 + 1] = a_words[1]
        a_smem[a_base1] = a_words[2]
        a_smem[a_base1 + 1] = a_words[3]

        b_k = lane >> 2
        b_col_frag = lane & 3
        b_byte_off = ((k_base + b_k) * N + tile_col_base + b_col_frag * 8) * 2
        b_words = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, S.convert(b_byte_off, S.i32), 0)
        b_vals = S.view(b_words, S.Tensor((2, 4, 1), S.bf16))
        b_step = b_k >> 3
        b_quarter = (b_k >> 2) & 1
        b_elem = b_k & 3
        b_dst_idx = b_step * 4 + b_elem
        b_lane_base = wave_id * 64 + b_quarter * 32 + b_col_frag * 8
        for s in S.range(4):
            b_smem_bf16[b_lane_base + s, b_dst_idx] = b_vals[0, s, 0]
            b_smem_bf16[b_lane_base + 4 + s, b_dst_idx] = b_vals[1, s, 0]

        S.syncthreads()

        lane_base = (wave_id * 64 + lane) * 4
        a_lane_words = S.subview(a_smem, (lane_base,), (4,), (1,))
        b_lane_words = S.subview(b_smem, (lane_base,), (4,), (1,))
        a_frag = S.view(a_lane_words, S.bf16, frag_layout)
        b_frag = S.view(b_lane_words, S.bf16, frag_layout)

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        S.syncthreads()

    col = tile_col_base + (lane % 32)
    lane_row_quad = 4 * (lane >> 5)
    for acc_idx in S.range(16):
        row = tile_row_base + 8 * (acc_idx >> 2) + lane_row_quad + (acc_idx & 3)
        C[row, col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if tuple(A.shape) != (N, N) or tuple(B.shape) != (N, N):
            raise ValueError(f"expected {(N, N)} inputs, got {tuple(A.shape)} and {tuple(B.shape)}")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise TypeError(f"expected bfloat16 inputs, got {A.dtype} and {B.dtype}")
        if A.device != B.device:
            raise ValueError("A and B must be on the same device")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((N, N), device=A.device, dtype=torch.bfloat16)
        gemm_kernel[lambda: ((N // BLOCK_N, N // BLOCK_M, 1), (BLOCK_THREADS, 1, 1))](A, B, C)
        return C
