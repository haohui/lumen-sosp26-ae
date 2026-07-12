#!/usr/bin/env python3

import torch

import avelang
import avelang.language as al


WARP_SIZE = 64
NUM_WARPS = 4
BLOCK_SIZE = WARP_SIZE * NUM_WARPS

GROUP_M = 128
GROUP_N = 128
GROUP_K = 64

WARP_PER_ROW = 2
WARP_PER_COL = 2
WARP_MAT_M = GROUP_M // WARP_PER_ROW
WARP_MAT_N = GROUP_N // WARP_PER_COL
M_TILES_PER_WARP = WARP_MAT_M // 16
N_TILES_PER_WARP = WARP_MAT_N // 16

VEC_SIZE = 8
BF16_BYTES = 2


@avelang.jit
def _load_global(
    src_ptr: al.Pointer(al.bf16),
    reg: al.Tensor(((GROUP_M * GROUP_K) // BLOCK_SIZE,), al.bf16),
    row_offset: al.i32,
    k_offset: al.i32,
    rows: al.i32,
    k: al.i32,
):
    src = al.make_tensor(src_ptr, al.bf16, al.make_layout((rows * k,), (1,)))
    tid = al.thread_id(0)

    for load_idx in al.range((GROUP_M * GROUP_K) // BLOCK_SIZE):
        tile_offset = tid + load_idx * BLOCK_SIZE
        tile_row = tile_offset // GROUP_K
        tile_k = tile_offset % GROUP_K
        global_row = row_offset + tile_row
        global_k = k_offset + tile_k

        if global_row < rows and global_k < k:
            reg[load_idx] = src[global_row * k + global_k]
        else:
            reg[load_idx] = al.convert(0.0, al.bf16)


@avelang.jit
def _store_shm(
    shm: al.Tensor((GROUP_M, GROUP_K), al.bf16),
    reg: al.Tensor(((GROUP_M * GROUP_K) // BLOCK_SIZE,), al.bf16),
):
    tid = al.thread_id(0)

    for store_idx in al.range((GROUP_M * GROUP_K) // BLOCK_SIZE):
        tile_offset = tid + store_idx * BLOCK_SIZE
        tile_row = tile_offset // GROUP_K
        tile_k = tile_offset % GROUP_K
        shm[tile_row, tile_k] = reg[store_idx]


@avelang.jit
def _write_results(
    C_ptr: al.Pointer(al.bf16),
    acc: al.Tensor(((GROUP_M * GROUP_N) // BLOCK_SIZE,), al.f32),
    group_row: al.i32,
    group_col: al.i32,
    m: al.i32,
    n: al.i32,
):
    C = al.make_tensor(C_ptr, al.bf16, al.make_layout((m * n,), (1,)))
    tid = al.thread_id(0)

    for output_idx in al.range((GROUP_M * GROUP_N) // BLOCK_SIZE):
        tile_offset = tid + output_idx * BLOCK_SIZE
        row = group_row * GROUP_M + tile_offset // GROUP_N
        col = group_col * GROUP_N + tile_offset % GROUP_N

        if row < m and col < n:
            C[row * n + col] = al.convert(acc[output_idx], al.bf16)


@avelang.jit
def _gemm_pipeline_transposed_b_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    m: al.i32,
    n: al.i32,
    k: al.i32,
):
    tid = al.thread_id(0)
    n_groups = (n + GROUP_N - 1) // GROUP_N
    group_id = al.block_id(0)
    group_row = group_id // n_groups
    group_col = group_id % n_groups

    shm_a = al.make_shared((GROUP_M, GROUP_K), al.bf16)
    shm_b = al.make_shared((GROUP_N, GROUP_K), al.bf16)
    reg_a = al.make_local(((GROUP_M * GROUP_K) // BLOCK_SIZE,), al.bf16)
    reg_b = al.make_local(((GROUP_N * GROUP_K) // BLOCK_SIZE,), al.bf16)
    acc = al.make_local(((GROUP_M * GROUP_N) // BLOCK_SIZE,), al.f32)

    for output_idx in al.range((GROUP_M * GROUP_N) // BLOCK_SIZE):
        acc[output_idx] = al.convert(0.0, al.f32)

    for k_tile in al.range((k + GROUP_K - 1) // GROUP_K):
        k_offset = k_tile * GROUP_K
        _load_global(A_ptr, reg_a, group_row * GROUP_M, k_offset, m, k)
        _load_global(B_ptr, reg_b, group_col * GROUP_N, k_offset, n, k)
        _store_shm(shm_a, reg_a)
        _store_shm(shm_b, reg_b)
        al.syncthreads()

        for output_idx in al.range((GROUP_M * GROUP_N) // BLOCK_SIZE):
            tile_offset = tid + output_idx * BLOCK_SIZE
            tile_row = tile_offset // GROUP_N
            tile_col = tile_offset % GROUP_N
            value = acc[output_idx]

            for kk in al.range(GROUP_K):
                a = al.convert(shm_a[tile_row, kk], al.f32)
                b = al.convert(shm_b[tile_col, kk], al.f32)
                value += a * b

            acc[output_idx] = value

        al.syncthreads()

    _write_results(C_ptr, acc, group_row, group_col, m, n)


def gemm_pipeline_transposed_b(A, B, out=None):
    m = A.shape[0]
    k = A.shape[1]
    n = B.shape[0]

    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)

    m_groups = (m + GROUP_M - 1) // GROUP_M
    n_groups = (n + GROUP_N - 1) // GROUP_N
    grid_size = m_groups * n_groups
    block_size = BLOCK_SIZE

    _gemm_pipeline_transposed_b_kernel[
        lambda: ((grid_size, 1, 1), (block_size, 1, 1))
    ](A, B, out, m, n, k, num_warps=NUM_WARPS)
    return out
