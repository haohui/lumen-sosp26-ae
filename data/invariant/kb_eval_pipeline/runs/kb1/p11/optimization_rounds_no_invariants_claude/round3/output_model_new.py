import torch
import torch.nn as nn

import substrate
import substrate.language as S


# Problem dimensions
BATCH = 8
I = 256
J = 512
L = 256
K_DIM = 768

# Tiling parameters
TILE_M = 64
TILE_N = 64
BLOCK_SIZE = 256
K_TILE_SIZE = 16

# Grid dimensions
M_TOTAL = BATCH * I * J
M_TILES = (M_TOTAL + TILE_M - 1) // TILE_M
N_TILES = (K_DIM + TILE_N - 1) // TILE_N
NUM_BLOCKS = M_TILES * N_TILES

# Byte sizes for range parameter (bf16 = 2 bytes)
A_TOTAL_BYTES = BATCH * I * J * L * 2  # 536870912
B_TOTAL_BYTES = L * K_DIM * 2  # 393216

outputs_per_thread = (TILE_M * TILE_N) // BLOCK_SIZE  # 16


@substrate.jit
def einsum_tiled_kernel(
    A: S.Tensor((8, 256, 512, 256), S.bf16),
    B: S.Tensor((256, 768), S.bf16),
    C: S.Tensor((8, 256, 512, 768), S.bf16),
):
    """
    Tiled einsum kernel with LDS staging and software pipelining.

    Computes: C[b,i,j,k] = sum_l A[b,i,j,l] * B[l,k]

    Features:
    - LDS for cooperative data staging
    - K-loop unrolled by 2 to minimize branching
    - f32 accumulation for numerical precision
    - 4 waves (256 threads) per block
    - Raw buffer loads with range to remove OOB branches

    Each block computes 64x64 output tile.
    Each thread computes 16 output elements.
    """
    bid = S.block_id(0)
    tid = S.thread_id(0)

    # Tile positions
    m_tile = bid // N_TILES
    n_tile = bid % N_TILES
    m_start = m_tile * TILE_M
    n_start = n_tile * TILE_N

    # Allocate LDS for A (64x16) and B (16x64)
    lds_A = S.make_shared((64, 16), S.bf16)
    lds_B = S.make_shared((16, 64), S.bf16)

    num_k_tiles = L // K_TILE_SIZE  # 16 K-tiles

    # Thread's load positions
    a_row = tid // 4
    a_col_offset = (tid % 4) * 4
    b_row = tid // 16
    b_col_offset = (tid % 16) * 4

    # Global memory positions for loading
    a_global_m = m_start + a_row
    b_global_n = n_start + b_col_offset

    # Create resource descriptors with range for OOB handling
    rsrc_A = S.amdgpu.make_rsrc(A, A_TOTAL_BYTES)
    rsrc_B = S.amdgpu.make_rsrc(B, B_TOTAL_BYTES)

    # Process each output element
    for out_idx in S.range(outputs_per_thread):
        flat_idx = tid * outputs_per_thread + out_idx
        local_m = flat_idx // TILE_N
        local_n = flat_idx % TILE_N

        m_global = m_start + local_m
        n_global = n_start + local_n

        # Compute output indices (unconditionally, tiles are exact)
        b_idx = m_global // (I * J)
        rem = m_global % (I * J)
        i_idx = rem // J
        j_idx = rem % J

        # Initialize f32 accumulator
        acc = S.convert(0.0, S.f32)

        # Process K tiles
        for k_tile in S.range(num_k_tiles):
            k_base = k_tile * K_TILE_SIZE

            # === LOAD: Cooperatively load this K tile into LDS using raw_buffer_load ===
            # Load 4 bf16 values from A (consecutive along L dimension)
            # A is (BATCH, I, J, L) with L as innermost dimension
            # Flat index = a_global_m * L + k_base + a_col_offset
            a_byte_offset = (a_global_m * L + k_base + a_col_offset) * 2
            a_data = S.amdgpu.raw_buffer_load_x2(rsrc_A, a_byte_offset, 0, 0)
            a_data_bf16 = S.view(a_data, S.Tensor((4,), S.bf16))
            for i in S.range(4):
                lds_A[a_row, a_col_offset + i] = a_data_bf16[i]

            # Load 4 bf16 values from B (consecutive along K_DIM dimension)
            # B is (L, K_DIM) with K_DIM as innermost dimension
            # Flat index = (k_base + b_row) * K_DIM + b_global_n
            b_byte_offset = ((k_base + b_row) * K_DIM + b_global_n) * 2
            b_data = S.amdgpu.raw_buffer_load_x2(rsrc_B, b_byte_offset, 0, 0)
            b_data_bf16 = S.view(b_data, S.Tensor((4,), S.bf16))
            for i in S.range(4):
                lds_B[b_row, b_col_offset + i] = b_data_bf16[i]

            S.syncthreads()

            # === COMPUTE: Accumulate over this K tile with K-loop unrolled by 2 ===
            for half in S.range(2):
                k_start = half * 8
                for kk in S.range(8):
                    a_k = k_start + kk
                    a_val = S.convert(lds_A[local_m, a_k], S.f32)
                    b_val = S.convert(lds_B[a_k, local_n], S.f32)
                    acc = acc + a_val * b_val

            S.syncthreads()

        # Write final result (unconditionally, tiles are exact)
        C[b_idx, i_idx, j_idx, n_global] = S.convert(acc, S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (8, 256, 512, 256) or tuple(B.shape) != (256, 768):
            return torch.einsum("bijl,lk->bijk", A, B)
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((8, 256, 512, 768), device=A.device, dtype=A.dtype)
        einsum_tiled_kernel[lambda: ((NUM_BLOCKS, 1, 1), (BLOCK_SIZE, 1, 1))](A, B, C)
        return C
