import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def matmul_bf16_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    k: al.u32,
    BLOCK_M: al.constexpr,
    BLOCK_N: al.constexpr,
    BLOCK_K: al.constexpr,
):
    lane = al.thread_id(0)
    lane_col = lane & 31
    lane_group = lane >> 5

    block_m = al.block_id(1) * BLOCK_M
    block_n = al.block_id(0) * BLOCK_N

    A_bf16 = al.make_tensor(A_ptr, al.bf16, al.make_layout((m, k), (k, 1)))
    B_bf16 = al.make_tensor(B_ptr, al.bf16, al.make_layout((n, k), (k, 1)))
    C = al.make_tensor(C_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    k_vecs = k >> 3
    packed_row_stride = k >> 1

    A_vec = al.view(
        A_bf16, al.i32, al.make_layout((m, k_vecs, 4), (packed_row_stride, 4, 1))
    )
    B_vec = al.view(
        B_bf16, al.i32, al.make_layout((n, k_vecs, 4), (packed_row_stride, 4, 1))
    )

    a_smem = al.make_shared((BLOCK_M * (BLOCK_K >> 3), BLOCK_K >> 2), al.i32)
    b_smem = al.make_shared((BLOCK_N * (BLOCK_K >> 3), BLOCK_K >> 2), al.i32)
    c_smem = al.make_shared((BLOCK_M, BLOCK_N), al.f32)

    acc = al.full((16,), 0.0, al.f32)

    for kt in al.range(k // BLOCK_K):
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
    col_base = (lane & 1) * 16

    for v in al.range(16):
        c_val = c_smem[store_row, col_base + v]
        C[block_m + store_row, block_n + col_base + v] = al.convert(c_val, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    a_bf16 = _prepare_bf16_cuda_contiguous(a)
    b_bf16 = _prepare_bf16_cuda_contiguous(b)

    m, k = a_bf16.shape
    k_b, n = b_bf16.shape
    if k != k_b:
        raise ValueError(f"Inner dimension mismatch: A has K={k}, B has K={k_b}")

    b_t = b_bf16.T.contiguous()

    out = torch.empty((m, n), device=a_bf16.device, dtype=torch.bfloat16)
    grid = (n // 32, m // 32, 1)
    matmul_bf16_kernel[lambda: (grid, (64, 1, 1))](
        a_bf16, b_t, out, m, n, k, 32, 32, 16,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return avelang_matmul(A, B)
