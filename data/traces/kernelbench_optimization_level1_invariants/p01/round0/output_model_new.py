import torch
import torch.nn as nn

import avelang
import avelang.language as al

N = 4096

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64


@avelang.jit
def gemm_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    N_val: al.i32,
):
    bm = al.block_id(0)
    bn = al.block_id(1)
    tid = al.thread_id(0)

    A_layout = al.make_layout((N_val, N_val), (N_val, 1))
    A = al.make_tensor(A_ptr, al.bf16, A_layout)
    B_layout = al.make_layout((N_val, N_val), (N_val, 1))
    B = al.make_tensor(B_ptr, al.bf16, B_layout)
    C_layout = al.make_layout((N_val, N_val), (N_val, 1))
    C = al.make_tensor(C_ptr, al.bf16, C_layout)

    lds_A = al.make_shared((BLOCK_M, BLOCK_K), al.bf16)
    lds_B = al.make_shared((BLOCK_K, BLOCK_N), al.bf16)

    tile_row = bm * BLOCK_M
    tile_col = bn * BLOCK_N

    thread_row = tid >> 4
    thread_col = tid & 15

    acc00 = al.convert(0.0, al.f32)
    acc01 = al.convert(0.0, al.f32)
    acc02 = al.convert(0.0, al.f32)
    acc03 = al.convert(0.0, al.f32)
    acc10 = al.convert(0.0, al.f32)
    acc11 = al.convert(0.0, al.f32)
    acc12 = al.convert(0.0, al.f32)
    acc13 = al.convert(0.0, al.f32)
    acc20 = al.convert(0.0, al.f32)
    acc21 = al.convert(0.0, al.f32)
    acc22 = al.convert(0.0, al.f32)
    acc23 = al.convert(0.0, al.f32)
    acc30 = al.convert(0.0, al.f32)
    acc31 = al.convert(0.0, al.f32)
    acc32 = al.convert(0.0, al.f32)
    acc33 = al.convert(0.0, al.f32)

    r0 = tile_row + thread_row * 4
    r1 = r0 + 1
    r2 = r0 + 2
    r3 = r0 + 3
    c0 = tile_col + thread_col * 4
    c1 = c0 + 1
    c2 = c0 + 2
    c3 = c0 + 3

    num_k_blocks = N_val // BLOCK_K
    for kb in al.range(num_k_blocks):
        k_start = kb * BLOCK_K

        # Each thread loads 16 elements of A tile into LDS
        # 256 threads x 16 = 4096 = 64x64
        base = tid * 16
        for i in al.range(16):
            idx = base + i
            a_ld_row = idx // 64
            a_ld_col = idx - a_ld_row * 64
            lds_A[a_ld_row, a_ld_col] = A[tile_row + a_ld_row, k_start + a_ld_col]

        # Each thread loads 16 elements of B tile into LDS
        for i in al.range(16):
            idx = base + i
            b_ld_row = idx // 64
            b_ld_col = idx - b_ld_row * 64
            lds_B[b_ld_row, b_ld_col] = B[k_start + b_ld_row, tile_col + b_ld_col]

        al.syncthreads()

        for ki in al.range(BLOCK_K):
            a0 = al.convert(lds_A[thread_row * 4 + 0, ki], al.f32)
            a1 = al.convert(lds_A[thread_row * 4 + 1, ki], al.f32)
            a2 = al.convert(lds_A[thread_row * 4 + 2, ki], al.f32)
            a3 = al.convert(lds_A[thread_row * 4 + 3, ki], al.f32)
            b0 = al.convert(lds_B[ki, thread_col * 4 + 0], al.f32)
            b1 = al.convert(lds_B[ki, thread_col * 4 + 1], al.f32)
            b2 = al.convert(lds_B[ki, thread_col * 4 + 2], al.f32)
            b3 = al.convert(lds_B[ki, thread_col * 4 + 3], al.f32)
            acc00 = acc00 + a0 * b0
            acc01 = acc01 + a0 * b1
            acc02 = acc02 + a0 * b2
            acc03 = acc03 + a0 * b3
            acc10 = acc10 + a1 * b0
            acc11 = acc11 + a1 * b1
            acc12 = acc12 + a1 * b2
            acc13 = acc13 + a1 * b3
            acc20 = acc20 + a2 * b0
            acc21 = acc21 + a2 * b1
            acc22 = acc22 + a2 * b2
            acc23 = acc23 + a2 * b3
            acc30 = acc30 + a3 * b0
            acc31 = acc31 + a3 * b1
            acc32 = acc32 + a3 * b2
            acc33 = acc33 + a3 * b3

        al.syncthreads()

    C[tile_row + thread_row * 4 + 0, tile_col + thread_col * 4 + 0] = al.convert(acc00, al.bf16)
    C[tile_row + thread_row * 4 + 0, tile_col + thread_col * 4 + 1] = al.convert(acc01, al.bf16)
    C[tile_row + thread_row * 4 + 0, tile_col + thread_col * 4 + 2] = al.convert(acc02, al.bf16)
    C[tile_row + thread_row * 4 + 0, tile_col + thread_col * 4 + 3] = al.convert(acc03, al.bf16)
    C[tile_row + thread_row * 4 + 1, tile_col + thread_col * 4 + 0] = al.convert(acc10, al.bf16)
    C[tile_row + thread_row * 4 + 1, tile_col + thread_col * 4 + 1] = al.convert(acc11, al.bf16)
    C[tile_row + thread_row * 4 + 1, tile_col + thread_col * 4 + 2] = al.convert(acc12, al.bf16)
    C[tile_row + thread_row * 4 + 1, tile_col + thread_col * 4 + 3] = al.convert(acc13, al.bf16)
    C[tile_row + thread_row * 4 + 2, tile_col + thread_col * 4 + 0] = al.convert(acc20, al.bf16)
    C[tile_row + thread_row * 4 + 2, tile_col + thread_col * 4 + 1] = al.convert(acc21, al.bf16)
    C[tile_row + thread_row * 4 + 2, tile_col + thread_col * 4 + 2] = al.convert(acc22, al.bf16)
    C[tile_row + thread_row * 4 + 2, tile_col + thread_col * 4 + 3] = al.convert(acc23, al.bf16)
    C[tile_row + thread_row * 4 + 3, tile_col + thread_col * 4 + 0] = al.convert(acc30, al.bf16)
    C[tile_row + thread_row * 4 + 3, tile_col + thread_col * 4 + 1] = al.convert(acc31, al.bf16)
    C[tile_row + thread_row * 4 + 3, tile_col + thread_col * 4 + 2] = al.convert(acc32, al.bf16)
    C[tile_row + thread_row * 4 + 3, tile_col + thread_col * 4 + 3] = al.convert(acc33, al.bf16)


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
        ):
            raise RuntimeError('Shape/dtype/device mismatch for AveLang kernel.')

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((N, N), device=A.device, dtype=A.dtype)

        gemm_kernel[lambda: ((N // BLOCK_M, N // BLOCK_N, 1), (256, 1, 1))](
            A, B, C, N
        )
        return C
