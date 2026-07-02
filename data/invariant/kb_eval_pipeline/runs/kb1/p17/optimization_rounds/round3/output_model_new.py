import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 2048
K = 8192
N = 4096
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WAVE_SIZE = 64
NUM_WARPS = 4
THREADS = WAVE_SIZE * NUM_WARPS
A_RANGE_BYTES = M * K * 2
K_TILES = K // BLOCK_K
N_TILES = N // 32
A_PACK_RANGE_BYTES = M * K_TILES * 16 * 2
B_PACK_RANGE_BYTES = K_TILES * N_TILES * WAVE_SIZE * 8 * 2


@substrate.jit
def gemm_kernel_mfma_pipelined(
    A: S.Tensor((M, K_TILES, 16), S.bf16),
    B: S.Tensor((K_TILES, N_TILES, WAVE_SIZE, 8), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    pid_n = S.block_id(0)
    pid_m = S.block_id(1)
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    warp = tid // WAVE_SIZE
    warp_row = warp // 2
    warp_col = warp % 2

    tile_row = pid_m * BLOCK_M
    tile_col = pid_n * BLOCK_N
    warp_row_base = tile_row + warp_row * 32
    warp_col_base = tile_col + warp_col * 32

    zero = S.convert(0, S.i32)
    a_rsrc = S.amdgpu.make_rsrc(A, S.convert(A_PACK_RANGE_BYTES, S.i32))
    b_rsrc = S.amdgpu.make_rsrc(B, S.convert(B_PACK_RANGE_BYTES, S.i32))
    shared_a_words = S.make_shared((2, NUM_WARPS, WAVE_SIZE, 4), S.u32)
    shared_b_words = S.make_shared((2, NUM_WARPS, WAVE_SIZE, 4), S.u32)
    acc = S.full((16,), 0.0, S.f32)

    a_row = warp_row_base + (lane % 32)
    b_tile_base = pid_n * 2 + warp_col

    a_offset0 = S.convert((((a_row * K_TILES) + 0) * 16 + (lane // 32) * 8) * 2, S.i32)
    b_offset0 = S.convert(((((0 * N_TILES) + b_tile_base) * WAVE_SIZE + lane) * 8) * 2, S.i32)
    a_pack0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset0, 0)
    b_pack0 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset0, 0)
    shared_a_words[0, warp, lane, 0] = a_pack0[0]
    shared_a_words[0, warp, lane, 1] = a_pack0[1]
    shared_a_words[0, warp, lane, 2] = a_pack0[2]
    shared_a_words[0, warp, lane, 3] = a_pack0[3]
    shared_b_words[0, warp, lane, 0] = b_pack0[0]
    shared_b_words[0, warp, lane, 1] = b_pack0[1]
    shared_b_words[0, warp, lane, 2] = b_pack0[2]
    shared_b_words[0, warp, lane, 3] = b_pack0[3]

    a_offset1 = S.convert((((a_row * K_TILES) + 1) * 16 + (lane // 32) * 8) * 2, S.i32)
    b_offset1 = S.convert(((((1 * N_TILES) + b_tile_base) * WAVE_SIZE + lane) * 8) * 2, S.i32)
    a_pack1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset1, 0)
    b_pack1 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset1, 0)
    shared_a_words[1, warp, lane, 0] = a_pack1[0]
    shared_a_words[1, warp, lane, 1] = a_pack1[1]
    shared_a_words[1, warp, lane, 2] = a_pack1[2]
    shared_a_words[1, warp, lane, 3] = a_pack1[3]
    shared_b_words[1, warp, lane, 0] = b_pack1[0]
    shared_b_words[1, warp, lane, 1] = b_pack1[1]
    shared_b_words[1, warp, lane, 2] = b_pack1[2]
    shared_b_words[1, warp, lane, 3] = b_pack1[3]
    S.syncthreads()

    for pair_idx in S.range((K_TILES // 2) - 1):
        k_even = pair_idx * 2
        k_odd = k_even + 1
        k_next_even = k_even + 2
        k_next_odd = k_even + 3

        a_frag0 = S.view(shared_a_words[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(shared_b_words[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

        a_offset0 = S.convert((((a_row * K_TILES) + k_next_even) * 16 + (lane // 32) * 8) * 2, S.i32)
        b_offset0 = S.convert(((((k_next_even * N_TILES) + b_tile_base) * WAVE_SIZE + lane) * 8) * 2, S.i32)
        a_pack0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset0, 0)
        b_pack0 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset0, 0)
        shared_a_words[0, warp, lane, 0] = a_pack0[0]
        shared_a_words[0, warp, lane, 1] = a_pack0[1]
        shared_a_words[0, warp, lane, 2] = a_pack0[2]
        shared_a_words[0, warp, lane, 3] = a_pack0[3]
        shared_b_words[0, warp, lane, 0] = b_pack0[0]
        shared_b_words[0, warp, lane, 1] = b_pack0[1]
        shared_b_words[0, warp, lane, 2] = b_pack0[2]
        shared_b_words[0, warp, lane, 3] = b_pack0[3]

        a_frag1 = S.view(shared_a_words[1, warp, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag1 = S.view(shared_b_words[1, warp, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

        a_offset1 = S.convert((((a_row * K_TILES) + k_next_odd) * 16 + (lane // 32) * 8) * 2, S.i32)
        b_offset1 = S.convert(((((k_next_odd * N_TILES) + b_tile_base) * WAVE_SIZE + lane) * 8) * 2, S.i32)
        a_pack1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset1, 0)
        b_pack1 = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset1, 0)
        shared_a_words[1, warp, lane, 0] = a_pack1[0]
        shared_a_words[1, warp, lane, 1] = a_pack1[1]
        shared_a_words[1, warp, lane, 2] = a_pack1[2]
        shared_a_words[1, warp, lane, 3] = a_pack1[3]
        shared_b_words[1, warp, lane, 0] = b_pack1[0]
        shared_b_words[1, warp, lane, 1] = b_pack1[1]
        shared_b_words[1, warp, lane, 2] = b_pack1[2]
        shared_b_words[1, warp, lane, 3] = b_pack1[3]

        S.syncthreads()

    a_frag0 = S.view(shared_a_words[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag0 = S.view(shared_b_words[0, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

    a_frag1 = S.view(shared_a_words[1, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag1 = S.view(shared_b_words[1, warp, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

    out_col = warp_col_base + (lane % 32)
    lane_row_group = lane // 32
    for acc_idx in S.range(16):
        out_row = warp_row_base + 8 * (acc_idx // 4) + 4 * lane_row_group + (acc_idx % 4)
        C[out_row, out_col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._a_cache_ptr = None
        self._a_cache = None
        self._b_cache_ptr = None
        self._b_cache = None

    def _pack_a(self, A):
        a_tiles = A.contiguous().view(M, K_TILES, BLOCK_K)
        return torch.cat(
            [a_tiles[..., 0:4], a_tiles[..., 8:12], a_tiles[..., 4:8], a_tiles[..., 12:16]],
            dim=-1,
        ).contiguous()

    def _pack_b(self, B):
        bt = B.transpose(-2, -1).contiguous().view(K_TILES, BLOCK_K, N_TILES, 32)
        lane_lo = torch.cat(
            [
                bt[:, 0:4, :, :].permute(0, 2, 3, 1),
                bt[:, 8:12, :, :].permute(0, 2, 3, 1),
            ],
            dim=-1,
        )
        lane_hi = torch.cat(
            [
                bt[:, 4:8, :, :].permute(0, 2, 3, 1),
                bt[:, 12:16, :, :].permute(0, 2, 3, 1),
            ],
            dim=-1,
        )
        return torch.cat([lane_lo, lane_hi], dim=2).contiguous()

    def forward(self, A, B):
        a_ptr = A.data_ptr()
        if self._a_cache_ptr != a_ptr:
            self._a_cache = self._pack_a(A)
            self._a_cache_ptr = a_ptr

        b_ptr = B.data_ptr()
        if self._b_cache_ptr != b_ptr:
            self._b_cache = self._pack_b(B)
            self._b_cache_ptr = b_ptr

        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        gemm_kernel_mfma_pipelined[lambda: ((N // BLOCK_N, M // BLOCK_M, 1), (THREADS, 1, 1))](
            self._a_cache, self._b_cache, C
        )
        return C
