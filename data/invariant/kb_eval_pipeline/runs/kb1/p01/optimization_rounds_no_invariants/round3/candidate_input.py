import torch
import torch.nn as nn

import substrate
import substrate.language as S


N = 4096
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
THREADS = 256
WARP_SIZE = 64
K_TILES = N // BLOCK_K
K_TILE_PAIRS = K_TILES // 2


@substrate.jit
def gemm_kernel(
    A: S.Tensor((N, N), S.bf16),
    B: S.Tensor((N, N), S.bf16),
    C: S.Tensor((N, N), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WARP_SIZE
    warp = tid // WARP_SIZE
    warp_row = warp // 2
    warp_col = warp % 2

    block_m = S.block_id(1) * BLOCK_M
    block_n = S.block_id(0) * BLOCK_N

    range_bytes = S.convert(N * N * 2, S.i32)
    zero = S.convert(0, S.i32)
    a_rsrc = S.amdgpu.make_rsrc(A, range_bytes)
    b_rsrc = S.amdgpu.make_rsrc(B, range_bytes)

    a_shared = S.make_shared((2, 2, 64, 8), S.bf16)
    b_shared = S.make_shared((2, 2, 64, 8), S.bf16)

    acc = S.full((16,), 0.0, S.f32)

    if tid < 128:
        group = tid // 64
        frag = tid % 64
        row = block_m + group * 32 + (frag % 32)
        k0 = (frag // 32) * 8
        offset = S.convert((row * N + k0) * 2, S.i32)
        packed = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, offset, 0)
        frag_u32 = S.view(packed, S.Tensor((4,), S.u32))
        frag_halves = S.view(frag_u32, S.Tensor((2, 4), S.bf16))
        for i in S.range(4):
            a_shared[0, group, frag, i] = frag_halves[0, i]
            a_shared[0, group, frag, i + 4] = frag_halves[1, i]
    else:
        load_tid = tid - 128
        group = load_tid // 64
        frag = load_tid % 64
        k = frag // 4
        c0 = block_n + group * 32 + (frag % 4) * 8
        offset = S.convert((k * N + c0) * 2, S.i32)
        packed = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, offset, 0)
        frag_u32 = S.view(packed, S.Tensor((4,), S.u32))
        frag_halves = S.view(frag_u32, S.Tensor((2, 4), S.bf16))
        col_base = (frag % 4) * 8
        lane_base = 32 * (k // 8)
        k_elem = k % 8
        for i in S.range(4):
            b_shared[0, group, lane_base + col_base + i, k_elem] = frag_halves[0, i]
            b_shared[0, group, lane_base + col_base + i + 4, k_elem] = frag_halves[1, i]

    S.syncthreads()

    for ko_pair in S.range(K_TILE_PAIRS):
        k_base = ko_pair * (2 * BLOCK_K)
        next_k_base = k_base + BLOCK_K

        if tid < 128:
            group = tid // 64
            frag = tid % 64
            row = block_m + group * 32 + (frag % 32)
            k0 = next_k_base + (frag // 32) * 8
            offset = S.convert((row * N + k0) * 2, S.i32)
            packed = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, offset, 0)
            frag_u32 = S.view(packed, S.Tensor((4,), S.u32))
            frag_halves = S.view(frag_u32, S.Tensor((2, 4), S.bf16))
            for i in S.range(4):
                a_shared[1, group, frag, i] = frag_halves[0, i]
                a_shared[1, group, frag, i + 4] = frag_halves[1, i]
        else:
            load_tid = tid - 128
            group = load_tid // 64
            frag = load_tid % 64
            k = next_k_base + (frag // 4)
            c0 = block_n + group * 32 + (frag % 4) * 8
            offset = S.convert((k * N + c0) * 2, S.i32)
            packed = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, offset, 0)
            frag_u32 = S.view(packed, S.Tensor((4,), S.u32))
            frag_halves = S.view(frag_u32, S.Tensor((2, 4), S.bf16))
            col_base = (frag % 4) * 8
            lane_base = 32 * ((frag // 4) // 8)
            k_elem = (frag // 4) % 8
            for i in S.range(4):
                b_shared[1, group, lane_base + col_base + i, k_elem] = frag_halves[0, i]
                b_shared[1, group, lane_base + col_base + i + 4, k_elem] = frag_halves[1, i]

        a_frag0 = S.view(a_shared[0, warp_row, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(b_shared[0, warp_col, lane], S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

        S.syncthreads()

        a_frag1 = S.view(a_shared[1, warp_row, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag1 = S.view(b_shared[1, warp_col, lane], S.Tensor((2, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

        if ko_pair + 1 != K_TILE_PAIRS:
            next_pair_k_base = k_base + 2 * BLOCK_K

            if tid < 128:
                group = tid // 64
                frag = tid % 64
                row = block_m + group * 32 + (frag % 32)
                k0 = next_pair_k_base + (frag // 32) * 8
                offset = S.convert((row * N + k0) * 2, S.i32)
                packed = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, offset, 0)
                frag_u32 = S.view(packed, S.Tensor((4,), S.u32))
                frag_halves = S.view(frag_u32, S.Tensor((2, 4), S.bf16))
                for i in S.range(4):
                    a_shared[0, group, frag, i] = frag_halves[0, i]
                    a_shared[0, group, frag, i + 4] = frag_halves[1, i]
            else:
                load_tid = tid - 128
                group = load_tid // 64
                frag = load_tid % 64
                k = next_pair_k_base + (frag // 4)
                c0 = block_n + group * 32 + (frag % 4) * 8
                offset = S.convert((k * N + c0) * 2, S.i32)
                packed = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, offset, 0)
                frag_u32 = S.view(packed, S.Tensor((4,), S.u32))
                frag_halves = S.view(frag_u32, S.Tensor((2, 4), S.bf16))
                col_base = (frag % 4) * 8
                lane_base = 32 * ((frag // 4) // 8)
                k_elem = (frag // 4) % 8
                for i in S.range(4):
                    b_shared[0, group, lane_base + col_base + i, k_elem] = frag_halves[0, i]
                    b_shared[0, group, lane_base + col_base + i + 4, k_elem] = frag_halves[1, i]

            S.syncthreads()

    col = block_n + warp_col * 32 + (lane % 32)
    lane_row_group = lane // 32
    for i in S.range(16):
        row = (
            block_m
            + warp_row * 32
            + (i // 8) * 16
            + ((i % 8) // 4) * 8
            + lane_row_group * 4
            + (i % 4)
        )
        C[row, col] = S.convert(acc[i], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if (
            tuple(A.shape) != (N, N)
            or tuple(B.shape) != (N, N)
            or A.dtype != torch.bfloat16
            or B.dtype != torch.bfloat16
            or A.device != B.device
            or not A.is_cuda
        ):
            raise ValueError("ModelNew expects contiguous CUDA bfloat16 4096x4096 inputs")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((N, N), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: ((N // BLOCK_N, N // BLOCK_M, 1), (THREADS, 1, 1))](A, B, C)
        return C
