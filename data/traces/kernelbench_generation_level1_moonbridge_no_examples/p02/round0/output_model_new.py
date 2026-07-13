import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def matmul_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.f32),
    M: al.u32,
    K: al.u32,
    N: al.u32,
):
    lane = al.thread_id(0)
    lane_col = lane & 31
    lane_group = lane >> 5

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 16

    block_m = al.block_id(1) * BLOCK_M
    block_n = al.block_id(0) * BLOCK_N

    k_vecs = K >> 3
    packed_row_stride = K >> 1

    A_bf16 = al.make_tensor(A_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    B_bf16 = al.make_tensor(B_ptr, al.bf16, al.make_layout((N, K), (K, 1)))
    C = al.make_tensor(C_ptr, al.f32, al.make_layout((M, N), (N, 1)))

    A_vec = al.view(
        A_bf16, al.i32,
        al.make_layout((M, k_vecs, 4), (packed_row_stride, 4, 1)),
    )
    B_vec = al.view(
        B_bf16, al.i32,
        al.make_layout((N, k_vecs, 4), (packed_row_stride, 4, 1)),
    )
    C_vec = al.view(
        C, al.i32,
        al.make_layout((M, N >> 2, 4), (N, 4, 1)),
    )

    a_smem = al.make_shared((BLOCK_M * (BLOCK_K >> 3), BLOCK_K >> 2), al.i32)
    b_smem = al.make_shared((BLOCK_N * (BLOCK_K >> 3), BLOCK_K >> 2), al.i32)
    c_smem = al.make_shared((BLOCK_M, BLOCK_N), al.f32)
    c_smem_vec = al.view(
        c_smem, al.i32,
        al.make_layout((BLOCK_M, BLOCK_N >> 2, 4), (BLOCK_N, 4, 1)),
    )

    acc = al.full((16,), 0.0, al.f32)

    num_k_tiles = K // BLOCK_K

    for kt in al.range(num_k_tiles):
        k_vec = kt * 2 + lane_group

        a_smem[lane] = A_vec[block_m + lane_col, k_vec]
        b_smem[lane] = B_vec[block_n + lane_col, k_vec]

        al.syncthreads()

        a_words = a_smem[lane]
        b_words = b_smem[lane]
        a_frag = al.view(a_words, al.Tensor((2, 2, 1), al.u32))
        b_frag = al.view(b_words, al.Tensor((2, 2, 1), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)

        al.syncthreads()

    for r in al.range(16):
        row_offset = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        c_smem[lane_col, row_offset] = acc[r]

    al.syncthreads()

    store_row = lane >> 1
    store_vec_base = (lane & 1) * (BLOCK_N >> 3)

    for v in al.range(BLOCK_N >> 3):
        C_vec[block_m + store_row, (block_n >> 2) + store_vec_base + v] = (
            c_smem_vec[store_row, store_vec_base + v]
        )


def _avelang_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA/HIP device."
    M, K = A.shape
    K2, N_mat = B.shape
    assert K == K2, "Inner dimensions must match."

    A_bf16 = A.to(torch.bfloat16).contiguous()
    B_T = B.T.contiguous().to(torch.bfloat16)

    C_f32 = torch.empty(M, N_mat, dtype=torch.float32, device=A.device)

    BLOCK_M = 32
    BLOCK_N = 32
    grid_x = N_mat // BLOCK_N
    grid_y = M // BLOCK_M

    matmul_kernel[lambda: ((grid_x, grid_y, 1), (64, 1, 1))](
        A_bf16, B_T, C_f32, M, K, N_mat
    )

    return C_f32.to(torch.bfloat16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _avelang_matmul(A, B)
