import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 2048
K = 8192
N = 4096

# Tile sizes for 4-wave kernel
TILE_M = 64
TILE_N = 64
TILE_K = 8  # Single MFMA covers K=8


@substrate.jit
def gemm_mfma_kernel(
    A: S.Tensor((2048, 8192), S.bf16),
    B: S.Tensor((4096, 8192), S.bf16),  # B stored as (N, K) for contiguous K access
    C: S.Tensor((2048, 4096), S.bf16),
):
    bx = S.block_id(0)
    by = S.block_id(1)
    lane = S.thread_id(0)

    warp_id = lane // 64
    lane_in_warp = lane % 64

    warp_row = warp_id // 2
    warp_col = warp_id % 2

    tile_row_base = by * TILE_M + warp_row * 32
    tile_col_base = bx * TILE_N + warp_col * 32

    # Accumulator for MFMA results
    acc = S.full((16,), 0.0, S.f32)

    # Create resource descriptors for global memory access
    # Range is set to total buffer size in bytes for OOB handling
    rsrc_A = S.amdgpu.make_rsrc(A, M * K * 2)
    rsrc_B = S.amdgpu.make_rsrc(B, N * K * 2)

    # LDS for double buffering with K unrolled by 2
    # Each iteration processes 16 K values (2 MFMA operations)
    # Using separate dimension for k_half: A_lds[buffer, lane, k_half, elem]
    # k_half: 0 or 1 for the two K=8 tiles
    # elem: 0-3 for the 4 bf16 values
    A_lds = S.make_shared((2, 256, 2, 4), S.bf16)
    B_lds = S.make_shared((2, 256, 2, 4), S.bf16)

    # Determine lane's position in the MFMA swizzle pattern
    a_row = lane_in_warp % 32
    a_k_group = lane_in_warp // 32

    global_a_row = tile_row_base + a_row

    b_col = lane_in_warp % 32
    b_k_group = lane_in_warp // 32

    global_b_row = tile_col_base + b_col

    # Preload first K=16 tile into LDS buffer 0 (2 MFMA iterations worth)
    # First K tile (K=0:8) - k_half=0
    k_base_0 = 0
    a_offset_0 = global_a_row * K + k_base_0 + a_k_group * 4
    a_data_0 = S.amdgpu.raw_buffer_load_x2(rsrc_A, a_offset_0 * 2, 0, 0)
    a_frag_0 = S.view(a_data_0, S.Tensor((4,), S.bf16))

    A_lds[0, lane, 0, 0] = a_frag_0[0]
    A_lds[0, lane, 0, 1] = a_frag_0[1]
    A_lds[0, lane, 0, 2] = a_frag_0[2]
    A_lds[0, lane, 0, 3] = a_frag_0[3]

    b_offset_0 = global_b_row * K + k_base_0 + b_k_group * 4
    b_data_0 = S.amdgpu.raw_buffer_load_x2(rsrc_B, b_offset_0 * 2, 0, 0)
    b_frag_0 = S.view(b_data_0, S.Tensor((4,), S.bf16))

    B_lds[0, lane, 0, 0] = b_frag_0[0]
    B_lds[0, lane, 0, 1] = b_frag_0[1]
    B_lds[0, lane, 0, 2] = b_frag_0[2]
    B_lds[0, lane, 0, 3] = b_frag_0[3]

    # Second K tile (K=8:16) - k_half=1
    k_base_1 = 8
    a_offset_1 = global_a_row * K + k_base_1 + a_k_group * 4
    a_data_1 = S.amdgpu.raw_buffer_load_x2(rsrc_A, a_offset_1 * 2, 0, 0)
    a_frag_1 = S.view(a_data_1, S.Tensor((4,), S.bf16))

    A_lds[0, lane, 1, 0] = a_frag_1[0]
    A_lds[0, lane, 1, 1] = a_frag_1[1]
    A_lds[0, lane, 1, 2] = a_frag_1[2]
    A_lds[0, lane, 1, 3] = a_frag_1[3]

    b_offset_1 = global_b_row * K + k_base_1 + b_k_group * 4
    b_data_1 = S.amdgpu.raw_buffer_load_x2(rsrc_B, b_offset_1 * 2, 0, 0)
    b_frag_1 = S.view(b_data_1, S.Tensor((4,), S.bf16))

    B_lds[0, lane, 1, 0] = b_frag_1[0]
    B_lds[0, lane, 1, 1] = b_frag_1[1]
    B_lds[0, lane, 1, 2] = b_frag_1[2]
    B_lds[0, lane, 1, 3] = b_frag_1[3]

    S.syncthreads()

    # Main loop with K unrolled by 2 (process 16 K per iteration)
    num_k_iterations = K // 16  # 8192 / 16 = 512 iterations

    for k_iter in S.range(num_k_iterations):
        current_buf = k_iter % 2
        next_buf = 1 - current_buf

        # Prefetch next K=16 tile to next_buf (software pipelining)
        # No branch needed - OOB access handled by range in rsrc descriptor
        # OOB loads return 0, which doesn't affect computation
        next_k_base = (k_iter + 1) * 16

        # First K tile of next iteration - k_half=0
        a_offset_next_0 = global_a_row * K + next_k_base + a_k_group * 4
        a_data_next_0 = S.amdgpu.raw_buffer_load_x2(rsrc_A, a_offset_next_0 * 2, 0, 0)
        a_frag_next_0 = S.view(a_data_next_0, S.Tensor((4,), S.bf16))

        A_lds[next_buf, lane, 0, 0] = a_frag_next_0[0]
        A_lds[next_buf, lane, 0, 1] = a_frag_next_0[1]
        A_lds[next_buf, lane, 0, 2] = a_frag_next_0[2]
        A_lds[next_buf, lane, 0, 3] = a_frag_next_0[3]

        b_offset_next_0 = global_b_row * K + next_k_base + b_k_group * 4
        b_data_next_0 = S.amdgpu.raw_buffer_load_x2(rsrc_B, b_offset_next_0 * 2, 0, 0)
        b_frag_next_0 = S.view(b_data_next_0, S.Tensor((4,), S.bf16))

        B_lds[next_buf, lane, 0, 0] = b_frag_next_0[0]
        B_lds[next_buf, lane, 0, 1] = b_frag_next_0[1]
        B_lds[next_buf, lane, 0, 2] = b_frag_next_0[2]
        B_lds[next_buf, lane, 0, 3] = b_frag_next_0[3]

        # Second K tile of next iteration - k_half=1
        a_offset_next_1 = global_a_row * K + next_k_base + 8 + a_k_group * 4
        a_data_next_1 = S.amdgpu.raw_buffer_load_x2(rsrc_A, a_offset_next_1 * 2, 0, 0)
        a_frag_next_1 = S.view(a_data_next_1, S.Tensor((4,), S.bf16))

        A_lds[next_buf, lane, 1, 0] = a_frag_next_1[0]
        A_lds[next_buf, lane, 1, 1] = a_frag_next_1[1]
        A_lds[next_buf, lane, 1, 2] = a_frag_next_1[2]
        A_lds[next_buf, lane, 1, 3] = a_frag_next_1[3]

        b_offset_next_1 = global_b_row * K + next_k_base + 8 + b_k_group * 4
        b_data_next_1 = S.amdgpu.raw_buffer_load_x2(rsrc_B, b_offset_next_1 * 2, 0, 0)
        b_frag_next_1 = S.view(b_data_next_1, S.Tensor((4,), S.bf16))

        B_lds[next_buf, lane, 1, 0] = b_frag_next_1[0]
        B_lds[next_buf, lane, 1, 1] = b_frag_next_1[1]
        B_lds[next_buf, lane, 1, 2] = b_frag_next_1[2]
        B_lds[next_buf, lane, 1, 3] = b_frag_next_1[3]

        # Wait for LDS stores to complete
        S.amdgpu.s_waitcnt(0, 0, 15)

        # Issue two MFMA operations (K unrolled by 2)
        # First MFMA: K[0:8] - k_half=0
        a_frag_lds_0 = A_lds[current_buf, lane, 0]
        b_frag_lds_0 = B_lds[current_buf, lane, 0]
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_lds_0, b_frag_lds_0, acc)

        # Second MFMA: K[8:16] - k_half=1
        a_frag_lds_1 = A_lds[current_buf, lane, 1]
        b_frag_lds_1 = B_lds[current_buf, lane, 1]
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_lds_1, b_frag_lds_1, acc)

        # Synchronize before next iteration overwrites LDS
        S.syncthreads()

    # Output write using the accumulator swizzle
    for acc_idx in S.range(16):
        col = tile_col_base + (lane_in_warp % 32)
        row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane_in_warp // 32) + (acc_idx % 4)
        C[row, col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (2048, 8192) or tuple(B.shape) != (4096, 8192):
            raise ValueError(f"Shape mismatch: A={A.shape}, B={B.shape}")

        A2 = A.contiguous()
        B2 = B.contiguous()
        C = torch.empty((2048, 4096), device=A.device, dtype=A.dtype)

        grid = (N // TILE_N, M // TILE_M, 1)
        block = (256, 1, 1)

        gemm_mfma_kernel[lambda: (grid, block)](A2, B2, C)
        return C
