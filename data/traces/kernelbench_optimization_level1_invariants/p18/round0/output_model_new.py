import torch
import torch.nn as nn
import avelang
import avelang.language as al


M = 2048
K = 8192
N = 4096

BF16_BYTES = 2


@avelang.jit
def gemm_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M_val: al.i32,
    K_val: al.i32,
    N_val: al.i32,
    BLOCK_M: al.constexpr,
    BLOCK_N: al.constexpr,
    BLOCK_K: al.constexpr,
):
    # Thread, warp, and lane indexing (256 threads = 4 warps x 64 lanes)
    tid = al.thread_id(0)
    warp_id = tid >> 6
    lane = tid & 63
    warp_m = warp_id >> 1
    warp_n = warp_id & 1
    lane_col = lane & 31
    lane_group = lane >> 5

    block_m = al.block_id(0) * BLOCK_M
    block_n = al.block_id(1) * BLOCK_N

    # View global tensors as flat (for raw buffer loads)
    A_flat = al.make_tensor(A_ptr, al.bf16, al.make_layout((M_val * K_val,), (1,)))
    B_flat = al.make_tensor(B_ptr, al.bf16, al.make_layout((K_val * N_val,), (1,)))
    C_flat = al.make_tensor(C_ptr, al.bf16, al.make_layout((M_val * N_val,), (1,)))

    # Subview to block-owned region for resource descriptors
    A_block = al.subview(A_flat, (block_m * K_val,), (BLOCK_M * K_val,), (1,))
    B_block = al.subview(B_flat, (block_n * K_val,), (BLOCK_N * K_val,), (1,))

    A_rsrc = al.amdgpu.make_rsrc(A_block, BLOCK_M * K_val * BF16_BYTES)
    B_rsrc = al.amdgpu.make_rsrc(B_block, BLOCK_N * K_val * BF16_BYTES)

    # LDS: 4 warps, each needs BLOCK_M rows of 4 i32 per 16 K step = 64 LDS rows/warp
    # With 2 lane_groups per warp, 4 warps -> 256 total LDS rows
    a_smem = al.make_shared((256, 4), al.i32)
    b_smem = al.make_shared((256, 4), al.i32)

    C_out = al.make_tensor(C_ptr, al.bf16, al.make_layout((M_val, N_val), (N_val, 1)))
    zero = al.convert(0, al.i32)

    # Accumulator for this warp's 32x32 output tile
    acc = al.full((16,), 0.0, al.f32)

    k_tiles = K_val >> 4  # K / 16

    for kt in al.range(k_tiles):
        k_base = kt * 16 + lane_group * 8

        # --- Global-to-LDS: A tile ---
        a_row = warp_m * 32 + lane_col
        a_offset = al.convert((a_row * K_val + k_base) * BF16_BYTES, al.i32)
        a_smem[tid] = al.amdgpu.raw_buffer_load_x4(A_rsrc, zero, a_offset, 0)

        # --- Global-to-LDS: B tile ---
        b_col = warp_n * 32 + lane_col
        b_offset = al.convert((b_col * K_val + k_base) * BF16_BYTES, al.i32)
        b_smem[tid] = al.amdgpu.raw_buffer_load_x4(B_rsrc, zero, b_offset, 0)

        al.syncthreads()

        # --- LDS-to-register for MFMA ---
        a_words = a_smem[tid]
        b_words = b_smem[tid]

        a_frag = al.view(a_words, al.Tensor((2, 2, 1), al.u32))
        b_frag = al.view(b_words, al.Tensor((2, 2, 1), al.u32))

        # MFMA: (B, A, C) order (per tutorial: operands swapped for row-major C)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)

    # --- Writeback: accumulator to global C ---
    tile_row_base = block_m + warp_m * 32
    tile_col_base = block_n + warp_n * 32

    for acc_idx in al.range(16):
        row = tile_row_base + (lane & 31)
        col = tile_col_base + 8 * (acc_idx // 4) + 4 * (lane >> 5) + (acc_idx & 3)
        C_out[row, col] = al.convert(acc[acc_idx], al.bf16)
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        # A is (K, M) = (8192, 2048), B is (N, K) = (4096, 8192)
        # Compute C = A.T @ B.T = (M, K) @ (K, N) = (2048, 4096)
        AT = A.transpose(-2, -1).contiguous()  # (M, K) = (2048, 8192)
        B2 = B.contiguous()                     # (N, K) = (4096, 8192)

        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        grid_m = M // 64
        grid_n = N // 64
        gemm_kernel[lambda: ((grid_m, grid_n, 1), (256, 1, 1))](
            AT.data_ptr(), B2.data_ptr(), C.data_ptr(),
            M, K, N,
            64, 64, 16,
        )
        return C
