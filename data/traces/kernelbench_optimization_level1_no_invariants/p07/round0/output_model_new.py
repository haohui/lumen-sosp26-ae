import torch
import torch.nn as nn

import avelang
import avelang.language as al


BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16


@avelang.jit
def gemm_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
    stride_a_m: al.i32,
    stride_b_n: al.i32,
    stride_c_m: al.i32,
):
    A_bf16 = al.make_tensor(A_ptr, al.bf16, al.make_layout((M, K), (stride_a_m, 1)))
    B_bf16 = al.make_tensor(B_ptr, al.bf16, al.make_layout((N, K), (stride_b_n, 1)))
    C_bf16 = al.make_tensor(C_ptr, al.bf16, al.make_layout((M, N), (stride_c_m, 1)))

    k_vecs = K >> 3
    packed_stride = K >> 1

    A_vec = al.view(A_bf16, al.i32, al.make_layout((M, k_vecs, 4), (packed_stride, 4, 1)))
    B_vec = al.view(B_bf16, al.i32, al.make_layout((N, k_vecs, 4), (packed_stride, 4, 1)))

    tid = al.thread_id(0)
    warp_id = tid >> 6
    warp_m = warp_id >> 1
    warp_n = warp_id & 1
    lane = tid & 63
    lane_col = lane & 31
    lane_group = lane >> 5

    block_m = al.block_id(1) * BLOCK_M
    block_n = al.block_id(0) * BLOCK_N

    a_smem = al.make_shared((256, BLOCK_K >> 2), al.i32)
    b_smem = al.make_shared((256, BLOCK_K >> 2), al.i32)

    acc = al.full((16,), 0.0, al.f32)

    k_tiles = K // BLOCK_K
    for kt in al.range(k_tiles):
        k_vec = kt * 2 + lane_group
        a_row = warp_m * 32 + lane_col
        a_smem[tid] = A_vec[block_m + a_row, k_vec]
        b_row = warp_n * 32 + lane_col
        b_smem[tid] = B_vec[block_n + b_row, k_vec]

        al.syncthreads()

        a_words = a_smem[tid]
        b_words = b_smem[tid]
        a_frag = al.view(a_words, al.Tensor((2, 2, 1), al.u32))
        b_frag = al.view(b_words, al.Tensor((2, 2, 1), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)

        al.syncthreads()

    # Write: c_smem[col, row] — data lands at quadrant (dim0=warp_n*32, dim1=warp_m*32)
    c_smem = al.make_shared((BLOCK_M, BLOCK_N), al.f32)
    for r in al.range(16):
        col = warp_n * 32 + lane_col
        row = warp_m * 32 + ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        c_smem[col, row] = acc[r]

    al.syncthreads()

    # Read from same physical quadrant: local row = lane>>1, local col = (lane&1)*16+v
    # Read indices: (dim0 = warp_n*32 + local_row, dim1 = warp_m*32 + local_col)
    # Within the quadrant, the 1-wave data flow maps VALUE → output at (local_row, local_col).
    # Global output: (row = warp_m*32 + local_row, col = warp_n*32 + local_col)
    # Since read_row = warp_n*32 + local_row → local_row = read_row - warp_n*32
    local_row = lane >> 1
    local_col_base = (lane & 1) * (BLOCK_N >> 2)

    for v in al.range(BLOCK_N >> 2):
        local_col = local_col_base + v
        read_row = warp_n * 32 + local_row
        read_col = warp_m * 32 + local_col
        val_f32 = c_smem[read_row, read_col]

        c_row = block_m + warp_m * 32 + local_row
        c_col = block_n + warp_n * 32 + local_col
        if c_row < M and c_col < N:
            C_bf16[c_row, c_col] = al.convert(val_f32, al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        A = A.contiguous()
        B_t = B.T.contiguous()
        M_val, K_val = A.shape
        N_val = B.shape[1]

        C = torch.empty((M_val, N_val), device=A.device, dtype=A.dtype)

        grid_n = (N_val + BLOCK_N - 1) // BLOCK_N
        grid_m = (M_val + BLOCK_M - 1) // BLOCK_M

        gemm_kernel[lambda: ((grid_n, grid_m, 1), (256, 1, 1))](
            A.data_ptr(),
            B_t.data_ptr(),
            C.data_ptr(),
            M_val,
            N_val,
            K_val,
            A.stride(0),
            B_t.stride(0),
            C.stride(0),
        )
        return C
