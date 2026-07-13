import torch
import torch.nn as nn
import avelang
import avelang.language as al

M = 256 * 8  # 2048
K = 131072 * 8  # 1048576

WAVEFRONT = 64
BLOCK_M = 4  # rows per block
UNROLL = 16  # elements per thread per tile


@avelang.jit
def gemv_bf16_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M: al.i32,
    K: al.i32,
):
    A = al.make_tensor(A_ptr, al.bf16, al.make_layout((M * K,), (1,)))
    B = al.make_tensor(B_ptr, al.bf16, al.make_layout((K,), (1,)))
    C = al.make_tensor(C_ptr, al.bf16, al.make_layout((M, 1), (1, 1)))

    bid = al.block_id(0)
    lane = al.thread_id(0)
    row_in_block = al.thread_id(1)

    row = bid * BLOCK_M + row_in_block

    if row < M:
        tile_size = WAVEFRONT * UNROLL
        k_tiles = K // tile_size
        row_off = row * K

        acc = al.convert(0.0, al.f32)
        for kt in al.range(k_tiles):
            k_base = kt * tile_size + lane * UNROLL
            off = row_off + k_base

            a0 = al.convert(A[off + 0], al.f32)
            a1 = al.convert(A[off + 1], al.f32)
            a2 = al.convert(A[off + 2], al.f32)
            a3 = al.convert(A[off + 3], al.f32)
            a4 = al.convert(A[off + 4], al.f32)
            a5 = al.convert(A[off + 5], al.f32)
            a6 = al.convert(A[off + 6], al.f32)
            a7 = al.convert(A[off + 7], al.f32)
            a8 = al.convert(A[off + 8], al.f32)
            a9 = al.convert(A[off + 9], al.f32)
            a10 = al.convert(A[off + 10], al.f32)
            a11 = al.convert(A[off + 11], al.f32)
            a12 = al.convert(A[off + 12], al.f32)
            a13 = al.convert(A[off + 13], al.f32)
            a14 = al.convert(A[off + 14], al.f32)
            a15 = al.convert(A[off + 15], al.f32)

            b0 = al.convert(B[k_base + 0], al.f32)
            b1 = al.convert(B[k_base + 1], al.f32)
            b2 = al.convert(B[k_base + 2], al.f32)
            b3 = al.convert(B[k_base + 3], al.f32)
            b4 = al.convert(B[k_base + 4], al.f32)
            b5 = al.convert(B[k_base + 5], al.f32)
            b6 = al.convert(B[k_base + 6], al.f32)
            b7 = al.convert(B[k_base + 7], al.f32)
            b8 = al.convert(B[k_base + 8], al.f32)
            b9 = al.convert(B[k_base + 9], al.f32)
            b10 = al.convert(B[k_base + 10], al.f32)
            b11 = al.convert(B[k_base + 11], al.f32)
            b12 = al.convert(B[k_base + 12], al.f32)
            b13 = al.convert(B[k_base + 13], al.f32)
            b14 = al.convert(B[k_base + 14], al.f32)
            b15 = al.convert(B[k_base + 15], al.f32)

            acc = acc + a0*b0 + a1*b1 + a2*b2 + a3*b3 + a4*b4 + a5*b5 + a6*b6 + a7*b7 + a8*b8 + a9*b9 + a10*b10 + a11*b11 + a12*b12 + a13*b13 + a14*b14 + a15*b15

        val = acc
        val = val + al.shuffle_down(val, 32, WAVEFRONT)
        val = val + al.shuffle_down(val, 16, WAVEFRONT)
        val = val + al.shuffle_down(val, 8, WAVEFRONT)
        val = val + al.shuffle_down(val, 4, WAVEFRONT)
        val = val + al.shuffle_down(val, 2, WAVEFRONT)
        val = val + al.shuffle_down(val, 1, WAVEFRONT)
        if lane == 0:
            C[row, 0] = al.convert(val, al.bf16)


def avelang_gemv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    A = A.contiguous().cuda().to(torch.bfloat16)
    if B.dim() == 2:
        B = B.contiguous().cuda().to(torch.bfloat16)
    else:
        B = B.contiguous().cuda().to(torch.bfloat16)

    m, k = A.shape
    out = torch.empty(m, 1, device=A.device, dtype=torch.bfloat16)

    grid = ((m + BLOCK_M - 1) // BLOCK_M, 1, 1)
    gemv_bf16_kernel[lambda: (grid, (WAVEFRONT, BLOCK_M, 1))](A, B, out, m, k)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return avelang_gemv(A, B)


def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, 1)
    return [A, B]


def get_init_inputs():
    return []
