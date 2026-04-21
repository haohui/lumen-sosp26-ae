import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 2048
K = 8192
N = 4096

WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
PIPELINE_STAGES = 2
GRID_TILES_N = N // BLOCK_N


@substrate.jit
def gemm_kernel(
    A: S.Tensor((2048, 8192), S.bf16),
    B: S.Tensor((8192, 4096), S.bf16),
    C: S.Tensor((2048, 4096), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WARP_SIZE
    warp = tid // WARP_SIZE
    warp_m = warp // 2
    warp_n = warp % 2

    block = S.block_id(0)
    tile_m = block // GRID_TILES_N
    tile_n = block % GRID_TILES_N
    block_row = tile_m * BLOCK_M
    block_col = tile_n * BLOCK_N

    a_shared = S.make_shared((PIPELINE_STAGES, 2, 64, 8), S.bf16)
    b_shared = S.make_shared((PIPELINE_STAGES, 2, 64, 8), S.bf16)
    a_shared_u32 = S.view(a_shared, S.Tensor((PIPELINE_STAGES, 2, 64, 4), S.u32))
    b_shared_u32 = S.view(b_shared, S.Tensor((PIPELINE_STAGES, 2, 64, 4), S.u32))
    a_rsrc = S.amdgpu.make_rsrc(A, M * K * 2)
    b_rsrc = S.amdgpu.make_rsrc(B, K * N * 2)

    acc = S.full((16,), 0.0, S.f32)

    if tid < 128:
        load_group = tid // 64
        load_id = tid % 64
        row = load_id % 32
        k_chunk = load_id // 32
        elem_base = k_chunk * 4
        global_row = block_row + load_group * 32 + row

        soffset0 = S.convert((global_row * K + elem_base * 2) * 2, S.i32)
        vec0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, soffset0, 0)
        frag0 = S.view(vec0, S.Tensor((2, 4, 1), S.bf16))
        for elem in S.range(4):
            a_shared[0, load_group, row, elem_base + elem] = frag0[0, elem, 0]
            a_shared[0, load_group, row + 32, elem_base + elem] = frag0[1, elem, 0]
    else:
        load_id = tid - 128
        load_group = load_id // 64
        local_id = load_id % 64
        k_local = local_id // 4
        col_chunk = local_id % 4
        col_base = block_col + load_group * 32 + col_chunk * 8
        lane_base = ((k_local % 8) // 4) * 32 + col_chunk * 8
        elem_idx = (k_local // 8) * 4 + (k_local % 4)

        soffset0 = S.convert(((k_local) * N + col_base) * 2, S.i32)
        vec0 = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, soffset0, 0)
        frag0 = S.view(vec0, S.Tensor((2, 4, 1), S.bf16))
        for elem in S.range(4):
            b_shared[0, load_group, lane_base + elem, elem_idx] = frag0[0, elem, 0]
            b_shared[0, load_group, lane_base + 4 + elem, elem_idx] = frag0[1, elem, 0]

    S.syncthreads()

    for kk0 in S.range(0, K, 2 * BLOCK_K):
        load_group = tid // 64

        if tid < 128:
            load_id = tid % 64
            row = load_id % 32
            k_chunk = load_id // 32
            elem_base = k_chunk * 4
            global_row = block_row + load_group * 32 + row
            soffset = S.convert((global_row * K + kk0 + BLOCK_K + elem_base * 2) * 2, S.i32)
            vec1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, soffset, 0)
            frag1 = S.view(vec1, S.Tensor((2, 4, 1), S.bf16))
            for elem in S.range(4):
                a_shared[1, load_group, row, elem_base + elem] = frag1[0, elem, 0]
                a_shared[1, load_group, row + 32, elem_base + elem] = frag1[1, elem, 0]
        else:
            load_id = tid - 128
            load_group_b = load_id // 64
            local_id = load_id % 64
            k_local = local_id // 4
            col_chunk = local_id % 4
            col_base = block_col + load_group_b * 32 + col_chunk * 8
            lane_base = ((k_local % 8) // 4) * 32 + col_chunk * 8
            elem_idx = (k_local // 8) * 4 + (k_local % 4)
            soffset = S.convert(((kk0 + BLOCK_K + k_local) * N + col_base) * 2, S.i32)
            vec1 = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, soffset, 0)
            frag1 = S.view(vec1, S.Tensor((2, 4, 1), S.bf16))
            for elem in S.range(4):
                b_shared[1, load_group_b, lane_base + elem, elem_idx] = frag1[0, elem, 0]
                b_shared[1, load_group_b, lane_base + 4 + elem, elem_idx] = frag1[1, elem, 0]

        a_frag0 = S.view(a_shared_u32[0, warp_m, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(b_shared_u32[0, warp_n, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

        S.syncthreads()

        if tid < 128:
            load_id = tid % 64
            row = load_id % 32
            k_chunk = load_id // 32
            elem_base = k_chunk * 4
            global_row = block_row + load_group * 32 + row
            soffset = S.convert((global_row * K + kk0 + 2 * BLOCK_K + elem_base * 2) * 2, S.i32)
            vec0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, 0, soffset, 0)
            frag0 = S.view(vec0, S.Tensor((2, 4, 1), S.bf16))
            for elem in S.range(4):
                a_shared[0, load_group, row, elem_base + elem] = frag0[0, elem, 0]
                a_shared[0, load_group, row + 32, elem_base + elem] = frag0[1, elem, 0]
        else:
            load_id = tid - 128
            load_group_b = load_id // 64
            local_id = load_id % 64
            k_local = local_id // 4
            col_chunk = local_id % 4
            col_base = block_col + load_group_b * 32 + col_chunk * 8
            lane_base = ((k_local % 8) // 4) * 32 + col_chunk * 8
            elem_idx = (k_local // 8) * 4 + (k_local % 4)
            soffset = S.convert(((kk0 + 2 * BLOCK_K + k_local) * N + col_base) * 2, S.i32)
            vec0 = S.amdgpu.raw_buffer_load_x4(b_rsrc, 0, soffset, 0)
            frag0 = S.view(vec0, S.Tensor((2, 4, 1), S.bf16))
            for elem in S.range(4):
                b_shared[0, load_group_b, lane_base + elem, elem_idx] = frag0[0, elem, 0]
                b_shared[0, load_group_b, lane_base + 4 + elem, elem_idx] = frag0[1, elem, 0]

        a_frag1 = S.view(a_shared_u32[1, warp_m, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag1 = S.view(b_shared_u32[1, warp_n, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

        S.syncthreads()

    tile_row_base = block_row + warp_m * 32
    tile_col_base = block_col + warp_n * 32
    col = tile_col_base + (lane % 32)
    row_quad = 4 * (lane // 32)
    for acc_idx in S.range(16):
        row = tile_row_base + 8 * (acc_idx // 4) + row_quad + (acc_idx % 4)
        C[row, col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if (
            tuple(A.shape) != (M, K)
            or tuple(B.shape) != (K, N)
            or A.dtype != torch.bfloat16
            or B.dtype != torch.bfloat16
            or not A.is_cuda
            or not B.is_cuda
        ):
            return torch.matmul(A, B)
        if torch.cuda.is_current_stream_capturing():
            return torch.matmul(A, B)

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        gemm_kernel[lambda: (((M // BLOCK_M) * (N // BLOCK_N), 1, 1), (THREADS, 1, 1))](
            A, B, C
        )
        return C
