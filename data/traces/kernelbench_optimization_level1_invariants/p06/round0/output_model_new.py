import torch
import torch.nn as nn

import avelang
import avelang.language as al


M = 256
N = 256
K = 524288

# Tile constants
M_TILE = 64    # M tile per block (2x2 warp grid, each warp 32)
N_TILE = 64    # N tile per block
K_TILE = 64    # K tile per LDS staging iteration
K_GROUPS = K_TILE // 8  # number of 8-K groups per K_TILE


@avelang.jit
def gemm_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M_val: al.i32,
    N_val: al.i32,
    K_val: al.i32,
    stride_a_k: al.i32,
    stride_b_n: al.i32,
    stride_c_n: al.i32,
):
    # Build tensor views from pointers
    layout_a = al.make_layout((M_val, K_val), (stride_a_k, al.convert(1, al.i32)))
    A = al.make_tensor(A_ptr, al.bf16, layout_a)
    layout_b = al.make_layout((K_val, N_val), (stride_b_n, al.convert(1, al.i32)))
    B = al.make_tensor(B_ptr, al.bf16, layout_b)
    layout_c = al.make_layout((M_val, N_val), (stride_c_n, al.convert(1, al.i32)))
    C = al.make_tensor(C_ptr, al.bf16, layout_c)

    # Build buffer resources for raw_buffer_load
    A_rsrc = al.amdgpu.make_rsrc(A, al.convert(M_val * K_val * 2, al.i32))
    B_rsrc = al.amdgpu.make_rsrc(B, al.convert(K_val * N_val * 2, al.i32))

    # Block indices in 2D grid
    bm = al.block_id(0)
    bn = al.block_id(1)

    # Thread ID and warp/lane decomposition
    tid = al.thread_id(0)
    wid = tid // 64  # 0..3
    lid = tid % 64  # 0..63
    wi = wid // 2   # 0 or 1 (M direction in 2x2 warp grid)
    wj = wid % 2    # 0 or 1 (N direction in 2x2 warp grid)

    # Tile base for this warp
    tile_m = bm * M_TILE + wi * 32
    tile_n = bn * N_TILE + wj * 32

    # 16 FP32 accumulators per lane
    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = al.convert(0.0, al.f32)

    # LDS for A and B operands
    A_lds = al.make_shared((M_TILE, K_TILE), al.bf16)
    B_lds = al.make_shared((K_TILE, N_TILE), al.bf16)

    # u32 views of LDS: reinterpret as u32 and reshape for batch loading
    # A_lds: (64, K_TILE) bf16 -> (64, K_TILE//2) u32. Organize as (64, K_TILE//4, 2) u32.
    # Each chunk of 2 u32 = 4 bf16 = one MFMA A operand.
    A_U32_CHUNKS = K_TILE // 4  # number of 2-u32 chunks per row in A
    A_lds_u32 = al.view(
        A_lds, al.u32,
        al.make_layout((M_TILE, A_U32_CHUNKS, 2), (K_TILE // 2, 2, 1)),
    )
    # B_lds: (K_TILE, 64) bf16 -> (K_TILE, 32) u32. Organize as (K_TILE, N_TILE//4, 2) u32.
    B_U32_CHUNKS = N_TILE // 4  # 16
    B_lds_u32 = al.view(
        B_lds, al.u32,
        al.make_layout((K_TILE, B_U32_CHUNKS, 2), (N_TILE // 2, 2, 1)),
    )

    # Main K loop
    for k in al.range(0, K_val, K_TILE):
        # --- Cooperative global-to-LDS load for A ---
        for kg_pass in al.range(0, K_GROUPS, 4):
            a_row = tid // 4
            a_kg = kg_pass + (tid % 4)
            a_g_k = k + a_kg * 8
            a_g_row = bm * M_TILE + a_row
            a_byte_off = (a_g_row * stride_a_k + a_g_k) * 2
            a_raw = al.amdgpu.raw_buffer_load_x4(
                A_rsrc,
                al.convert(a_byte_off, al.i32),
                al.convert(0, al.i32),
                al.convert(0, al.i32),
            )
            a_v4_u32 = al.view(a_raw, al.Tensor((4,), al.u32))
            # Each load = 4 u32 = 8 bf16 = 2 chunks in the u32 view
            A_lds_u32[a_row, a_kg * 2, 0] = a_v4_u32[0]
            A_lds_u32[a_row, a_kg * 2, 1] = a_v4_u32[1]
            A_lds_u32[a_row, a_kg * 2 + 1, 0] = a_v4_u32[2]
            A_lds_u32[a_row, a_kg * 2 + 1, 1] = a_v4_u32[3]

        # --- Cooperative global-to-LDS load for B ---
        for kg_pass in al.range(0, K_TILE, 32):
            b_col_g = tid // 32        # 0..7 column groups
            b_kg = kg_pass + (tid % 32)  # K rows
            b_g_k = k + b_kg
            b_g_n = bn * N_TILE + b_col_g * 8
            b_byte_off = (b_g_k * stride_b_n + b_g_n) * 2
            b_raw = al.amdgpu.raw_buffer_load_x4(
                B_rsrc,
                al.convert(b_byte_off, al.i32),
                al.convert(0, al.i32),
                al.convert(0, al.i32),
            )
            b_v4_u32 = al.view(b_raw, al.Tensor((4,), al.u32))
            B_lds_u32[b_kg, b_col_g * 2, 0] = b_v4_u32[0]
            B_lds_u32[b_kg, b_col_g * 2, 1] = b_v4_u32[1]
            B_lds_u32[b_kg, b_col_g * 2 + 1, 0] = b_v4_u32[2]
            B_lds_u32[b_kg, b_col_g * 2 + 1, 1] = b_v4_u32[3]

        al.syncthreads()

        # --- Each warp performs MFMA from LDS ---
        for k_sub in al.range(0, K_TILE, 8):
            # Load A operand into registers first, then pass to MFMA
            a_lds_row = wi * 32 + (lid % 32)
            a_chunk_idx = k_sub // 4 + (lid // 32)
            a_reg = al.make_local((2,), al.u32)
            a_reg[0] = A_lds_u32[a_lds_row, a_chunk_idx, 0]
            a_reg[1] = A_lds_u32[a_lds_row, a_chunk_idx, 1]

            # Load B operand into registers first
            b_k0 = k_sub + (lid % 8)
            b_n_chunk = wj * 8 + (lid // 8)
            b_reg = al.make_local((2,), al.u32)
            b_reg[0] = B_lds_u32[b_k0, b_n_chunk, 0]
            b_reg[1] = B_lds_u32[b_k0, b_n_chunk, 1]

            # MFMA call
            acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_reg, b_reg, acc)

        al.syncthreads()

    # --- Writeback using accumulator invariant ---
    for a_idx in al.range(16):
        row = tile_m + 8 * (a_idx // 4) + 4 * (lid // 32) + (a_idx % 4)
        col = tile_n + (lid % 32)
        C[row, col] = al.convert(acc[a_idx], al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        M_val = A.shape[0]
        K_val = A.shape[1]
        N_val = B.shape[1]

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M_val, N_val), device=A.device, dtype=A.dtype)

        stride_a_k = A.stride(0)
        stride_b_n = B.stride(0)
        stride_c_n = C.stride(0)

        gemm_kernel[lambda: ((M_val // M_TILE, N_val // N_TILE, 1), (256, 1, 1))](
            A.data_ptr(),
            B.data_ptr(),
            C.data_ptr(),
            M_val,
            N_val,
            K_val,
            stride_a_k,
            stride_b_n,
            stride_c_n,
        )
        return C
