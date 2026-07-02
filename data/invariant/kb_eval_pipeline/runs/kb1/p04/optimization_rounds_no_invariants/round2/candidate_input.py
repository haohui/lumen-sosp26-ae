import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 2048
K = 1048576
N = 1

WAVE_SIZE = 64
WARPS_M = 2
WARPS_N = 2
BLOCK_M = WARPS_M * 32
BLOCK_N = WARPS_N * 32
BLOCK_THREADS = WARPS_M * WARPS_N * WAVE_SIZE
K_TILE = 16
B_PADDED_N = BLOCK_N


@substrate.jit
def pack_b_kernel(
    B: S.Tensor((K, 1), S.bf16),
    B_tiled: S.Tensor((K, B_PADDED_N), S.bf16),
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    if idx < K:
        value = B[idx, 0]
        for col in S.range(B_PADDED_N):
            B_tiled[idx, col] = value


@substrate.jit
def gemv_mfma_kernel(
    A: S.Tensor((M, K), S.bf16),
    B_tiled: S.Tensor((K, B_PADDED_N), S.bf16),
    C: S.Tensor((M, 1), S.bf16),
):
    block_row = S.block_id(0) * BLOCK_M
    tid = S.thread_id(0)
    warp_id = tid // WAVE_SIZE
    lane = tid % WAVE_SIZE
    warp_row = warp_id // WARPS_N
    warp_col = warp_id % WARPS_N

    a_smem = S.make_shared((WARPS_M * WAVE_SIZE, 4), S.u32)
    b_smem = S.make_shared((WARPS_N * WAVE_SIZE, 4), S.u32)

    acc = S.full((16,), 0.0, S.f32)

    a_block = S.subview(A, (block_row, 0), (BLOCK_M, K), (1, 1))
    a_rsrc = S.amdgpu.make_rsrc(a_block, BLOCK_M * K * 2)
    b_rsrc = S.amdgpu.make_rsrc(B_tiled, K * B_PADDED_N * 2)
    zero = S.convert(0, S.i32)

    for k_tile in S.range(K // K_TILE):
        if tid < WARPS_M * WAVE_SIZE:
            a_frag = tid
            a_row = a_frag % 32
            a_k = k_tile * K_TILE + (a_frag // 32) * 8
            a_offset = S.convert((a_row * K + a_k) * 2, S.i32)
            packed_a = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, a_offset, 0)
            for i in S.range(4):
                a_smem[a_frag, i] = packed_a[i]

        if tid >= WARPS_M * WAVE_SIZE:
            b_frag = tid - WARPS_M * WAVE_SIZE
            b_row = k_tile * K_TILE + (b_frag // 8)
            b_col = (b_frag % 8) * 8
            b_offset = S.convert((b_row * B_PADDED_N + b_col) * 2, S.i32)
            packed_b = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, b_offset, 0)
            for i in S.range(4):
                b_smem[b_frag, i] = packed_b[i]

        S.syncthreads()

        a_idx = warp_row * WAVE_SIZE + lane
        b_idx = warp_col * WAVE_SIZE + lane
        a_frag = S.view(a_smem[a_idx], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_smem[b_idx], S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        S.syncthreads()

    if warp_col == 0:
        if lane == 0:
            for i in S.range(16):
                row = block_row + warp_row * 32 + (i % 4) + (i // 4) * 8
                if row < M:
                    C[row, 0] = S.convert(acc[i], S.bf16)
        if lane == 32:
            for i in S.range(16):
                row = block_row + warp_row * 32 + 4 + (i % 4) + (i // 4) * 8
                if row < M:
                    C[row, 0] = S.convert(acc[i], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._cached_b_ptr = None
        self._cached_b_tiled = None

    def _get_b_tiled(self, B: torch.Tensor) -> torch.Tensor:
        if (
            self._cached_b_tiled is None
            or self._cached_b_ptr != B.data_ptr()
            or self._cached_b_tiled.device != B.device
            or self._cached_b_tiled.dtype != B.dtype
        ):
            B_tiled = torch.empty((K, B_PADDED_N), device=B.device, dtype=B.dtype)
            grid = ((K + 255) // 256, 1, 1)
            pack_b_kernel[lambda: (grid, (256, 1, 1))](B, B_tiled)
            self._cached_b_ptr = B.data_ptr()
            self._cached_b_tiled = B_tiled
        return self._cached_b_tiled

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, 1):
            raise ValueError(f"Expected A {(M, K)} and B {(K, 1)}, got {tuple(A.shape)} and {tuple(B.shape)}")

        A = A.contiguous()
        B = B.contiguous()
        B_tiled = self._get_b_tiled(B)
        C = torch.empty((M, 1), device=A.device, dtype=A.dtype)

        grid = (M // BLOCK_M, 1, 1)
        gemv_mfma_kernel[lambda: (grid, (BLOCK_THREADS, 1, 1))](A, B_tiled, C)
        return C
