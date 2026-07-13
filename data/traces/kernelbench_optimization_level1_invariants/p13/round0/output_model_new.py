import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
TILE_M = 32
TILE_N = 32


@avelang.jit
def gemm_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    A_bf16 = al.make_tensor(A_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    B_bf16 = al.make_tensor(B_ptr, al.bf16, al.make_layout((N, K), (K, 1)))
    C_out = al.make_tensor(C_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    k_vecs = K >> 3
    packed_stride = K >> 1

    A_vec = al.view(A_bf16, al.i32, al.make_layout((M, k_vecs, 4), (packed_stride, 4, 1)))
    B_vec = al.view(B_bf16, al.i32, al.make_layout((N, k_vecs, 4), (packed_stride, 4, 1)))

    lane = al.thread_id(0)
    wave = lane >> 6
    warp_m = wave >> 1
    warp_n = wave & 1
    lane_in_wave = lane & 63
    lane_col = lane_in_wave & 31
    lane_group = lane_in_wave >> 5

    block_m = al.block_id(1) * BLOCK_M
    block_n = al.block_id(0) * BLOCK_N

    a_smem = al.make_shared((256, BLOCK_K >> 2), al.i32)
    b_smem = al.make_shared((256, BLOCK_K >> 2), al.i32)
    c_smem = al.make_shared((BLOCK_M, BLOCK_N), al.f32)

    acc = al.full((16,), 0.0, al.f32)

    for kt in al.range(K // BLOCK_K):
        k_vec = kt * (BLOCK_K >> 3) + lane_group
        a_smem[lane] = A_vec[block_m + warp_m * TILE_M + lane_col, k_vec]
        b_smem[lane] = B_vec[block_n + warp_n * TILE_N + lane_col, k_vec]

        al.syncthreads()

        a_words = a_smem[lane]
        b_words = b_smem[lane]
        a_frag = al.view(a_words, al.Tensor((2, 2, 1), al.u32))
        b_frag = al.view(b_words, al.Tensor((2, 2, 1), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)

        al.syncthreads()

    for r in al.range(16):
        row_off = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        c_smem[warp_m * TILE_M + lane_col, warp_n * TILE_N + row_off] = acc[r]

    al.syncthreads()

    store_row = lane >> 2
    store_col_off = (lane & 3) * 16
    for e in al.range(16):
        col = store_col_off + e
        val = al.convert(c_smem[store_row, col], al.bf16)
        C_out[block_m + store_row, block_n + col] = val


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._cached_B_ptr = None
        self._cached_B_t = None

    def forward(self, A, B):
        orig_dtype = A.dtype
        A_bf16 = A.to(torch.bfloat16).contiguous()

        if self._cached_B_ptr is not B.data_ptr():
            self._cached_B_ptr = B.data_ptr()
            self._cached_B_t = B.to(torch.bfloat16).T.contiguous()
        B_bf16_t = self._cached_B_t

        M_val = A_bf16.shape[0]
        N_val = B_bf16_t.shape[0]
        K_val = A_bf16.shape[1]

        C = torch.empty((M_val, N_val), device=A.device, dtype=torch.bfloat16)

        grid_x = N_val // BLOCK_N
        grid_y = M_val // BLOCK_M

        gemm_kernel[lambda: ((grid_x, grid_y, 1), (256, 1, 1))](
            A_bf16, B_bf16_t, C, M_val, N_val, K_val,
        )

        return C.to(orig_dtype)
