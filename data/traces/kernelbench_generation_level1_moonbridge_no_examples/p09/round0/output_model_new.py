import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def matmul_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
    M: al.i32,
    K: al.i32,
    N: al.i32,
):
    pid_m = al.block_id(0)
    pid_n = al.block_id(1)
    tid = al.thread_id(0)
    num_threads = al.block_dim(0)

    m_start = pid_m * 128
    n_start = pid_n * 128

    a_layout = al.make_layout((M, K), (K, 1))
    b_layout = al.make_layout((K, N), (N, 1))
    c_layout = al.make_layout((M, N), (N, 1))
    a = al.make_tensor(a_ptr, al.bf16, a_layout)
    b = al.make_tensor(b_ptr, al.bf16, b_layout)
    c = al.make_tensor(c_ptr, al.bf16, c_layout)

    a_shared = al.make_shared((128, 32), al.bf16)
    b_shared = al.make_shared((32, 128), al.bf16)

    for idx in al.range(tid, 4096, num_threads):
        mi = idx // 32
        ki = idx % 32
        src_m = m_start + mi
        if src_m < M:
            a_shared[mi, ki] = a[src_m, ki]
        else:
            a_shared[mi, ki] = al.convert(0.0, al.bf16)

    for idx in al.range(tid, 4096, num_threads):
        ki = idx // 128
        ni = idx % 128
        src_n = n_start + ni
        if src_n < N:
            b_shared[ki, ni] = b[ki, src_n]
        else:
            b_shared[ki, ni] = al.convert(0.0, al.bf16)

    al.syncthreads()

    # 16x16 thread grid over 128x128 tile, each thread handles 8x8 sub-tile
    thread_m = tid // 16
    thread_n = tid % 16
    base_m = thread_m * 8
    base_n = thread_n * 8

    # 64 scalar accumulators for the 8x8 sub-tile, explicitly named for compiler
    acc00 = al.convert(0.0, al.f32)
    acc01 = al.convert(0.0, al.f32)
    acc02 = al.convert(0.0, al.f32)
    acc03 = al.convert(0.0, al.f32)
    acc04 = al.convert(0.0, al.f32)
    acc05 = al.convert(0.0, al.f32)
    acc06 = al.convert(0.0, al.f32)
    acc07 = al.convert(0.0, al.f32)
    acc10 = al.convert(0.0, al.f32)
    acc11 = al.convert(0.0, al.f32)
    acc12 = al.convert(0.0, al.f32)
    acc13 = al.convert(0.0, al.f32)
    acc14 = al.convert(0.0, al.f32)
    acc15 = al.convert(0.0, al.f32)
    acc16 = al.convert(0.0, al.f32)
    acc17 = al.convert(0.0, al.f32)
    acc20 = al.convert(0.0, al.f32)
    acc21 = al.convert(0.0, al.f32)
    acc22 = al.convert(0.0, al.f32)
    acc23 = al.convert(0.0, al.f32)
    acc24 = al.convert(0.0, al.f32)
    acc25 = al.convert(0.0, al.f32)
    acc26 = al.convert(0.0, al.f32)
    acc27 = al.convert(0.0, al.f32)
    acc30 = al.convert(0.0, al.f32)
    acc31 = al.convert(0.0, al.f32)
    acc32 = al.convert(0.0, al.f32)
    acc33 = al.convert(0.0, al.f32)
    acc34 = al.convert(0.0, al.f32)
    acc35 = al.convert(0.0, al.f32)
    acc36 = al.convert(0.0, al.f32)
    acc37 = al.convert(0.0, al.f32)
    acc40 = al.convert(0.0, al.f32)
    acc41 = al.convert(0.0, al.f32)
    acc42 = al.convert(0.0, al.f32)
    acc43 = al.convert(0.0, al.f32)
    acc44 = al.convert(0.0, al.f32)
    acc45 = al.convert(0.0, al.f32)
    acc46 = al.convert(0.0, al.f32)
    acc47 = al.convert(0.0, al.f32)
    acc50 = al.convert(0.0, al.f32)
    acc51 = al.convert(0.0, al.f32)
    acc52 = al.convert(0.0, al.f32)
    acc53 = al.convert(0.0, al.f32)
    acc54 = al.convert(0.0, al.f32)
    acc55 = al.convert(0.0, al.f32)
    acc56 = al.convert(0.0, al.f32)
    acc57 = al.convert(0.0, al.f32)
    acc60 = al.convert(0.0, al.f32)
    acc61 = al.convert(0.0, al.f32)
    acc62 = al.convert(0.0, al.f32)
    acc63 = al.convert(0.0, al.f32)
    acc64 = al.convert(0.0, al.f32)
    acc65 = al.convert(0.0, al.f32)
    acc66 = al.convert(0.0, al.f32)
    acc67 = al.convert(0.0, al.f32)
    acc70 = al.convert(0.0, al.f32)
    acc71 = al.convert(0.0, al.f32)
    acc72 = al.convert(0.0, al.f32)
    acc73 = al.convert(0.0, al.f32)
    acc74 = al.convert(0.0, al.f32)
    acc75 = al.convert(0.0, al.f32)
    acc76 = al.convert(0.0, al.f32)
    acc77 = al.convert(0.0, al.f32)

    for ki in al.range(32):
        a0 = al.convert(a_shared[base_m + 0, ki], al.f32)
        a1 = al.convert(a_shared[base_m + 1, ki], al.f32)
        a2 = al.convert(a_shared[base_m + 2, ki], al.f32)
        a3 = al.convert(a_shared[base_m + 3, ki], al.f32)
        a4 = al.convert(a_shared[base_m + 4, ki], al.f32)
        a5 = al.convert(a_shared[base_m + 5, ki], al.f32)
        a6 = al.convert(a_shared[base_m + 6, ki], al.f32)
        a7 = al.convert(a_shared[base_m + 7, ki], al.f32)

        b0 = al.convert(b_shared[ki, base_n + 0], al.f32)
        b1 = al.convert(b_shared[ki, base_n + 1], al.f32)
        b2 = al.convert(b_shared[ki, base_n + 2], al.f32)
        b3 = al.convert(b_shared[ki, base_n + 3], al.f32)
        b4 = al.convert(b_shared[ki, base_n + 4], al.f32)
        b5 = al.convert(b_shared[ki, base_n + 5], al.f32)
        b6 = al.convert(b_shared[ki, base_n + 6], al.f32)
        b7 = al.convert(b_shared[ki, base_n + 7], al.f32)

        acc00 = acc00 + a0 * b0
        acc01 = acc01 + a0 * b1
        acc02 = acc02 + a0 * b2
        acc03 = acc03 + a0 * b3
        acc04 = acc04 + a0 * b4
        acc05 = acc05 + a0 * b5
        acc06 = acc06 + a0 * b6
        acc07 = acc07 + a0 * b7
        acc10 = acc10 + a1 * b0
        acc11 = acc11 + a1 * b1
        acc12 = acc12 + a1 * b2
        acc13 = acc13 + a1 * b3
        acc14 = acc14 + a1 * b4
        acc15 = acc15 + a1 * b5
        acc16 = acc16 + a1 * b6
        acc17 = acc17 + a1 * b7
        acc20 = acc20 + a2 * b0
        acc21 = acc21 + a2 * b1
        acc22 = acc22 + a2 * b2
        acc23 = acc23 + a2 * b3
        acc24 = acc24 + a2 * b4
        acc25 = acc25 + a2 * b5
        acc26 = acc26 + a2 * b6
        acc27 = acc27 + a2 * b7
        acc30 = acc30 + a3 * b0
        acc31 = acc31 + a3 * b1
        acc32 = acc32 + a3 * b2
        acc33 = acc33 + a3 * b3
        acc34 = acc34 + a3 * b4
        acc35 = acc35 + a3 * b5
        acc36 = acc36 + a3 * b6
        acc37 = acc37 + a3 * b7
        acc40 = acc40 + a4 * b0
        acc41 = acc41 + a4 * b1
        acc42 = acc42 + a4 * b2
        acc43 = acc43 + a4 * b3
        acc44 = acc44 + a4 * b4
        acc45 = acc45 + a4 * b5
        acc46 = acc46 + a4 * b6
        acc47 = acc47 + a4 * b7
        acc50 = acc50 + a5 * b0
        acc51 = acc51 + a5 * b1
        acc52 = acc52 + a5 * b2
        acc53 = acc53 + a5 * b3
        acc54 = acc54 + a5 * b4
        acc55 = acc55 + a5 * b5
        acc56 = acc56 + a5 * b6
        acc57 = acc57 + a5 * b7
        acc60 = acc60 + a6 * b0
        acc61 = acc61 + a6 * b1
        acc62 = acc62 + a6 * b2
        acc63 = acc63 + a6 * b3
        acc64 = acc64 + a6 * b4
        acc65 = acc65 + a6 * b5
        acc66 = acc66 + a6 * b6
        acc67 = acc67 + a6 * b7
        acc70 = acc70 + a7 * b0
        acc71 = acc71 + a7 * b1
        acc72 = acc72 + a7 * b2
        acc73 = acc73 + a7 * b3
        acc74 = acc74 + a7 * b4
        acc75 = acc75 + a7 * b5
        acc76 = acc76 + a7 * b6
        acc77 = acc77 + a7 * b7

    c[m_start + base_m + 0, n_start + base_n + 0] = al.convert(acc00, al.bf16)
    c[m_start + base_m + 0, n_start + base_n + 1] = al.convert(acc01, al.bf16)
    c[m_start + base_m + 0, n_start + base_n + 2] = al.convert(acc02, al.bf16)
    c[m_start + base_m + 0, n_start + base_n + 3] = al.convert(acc03, al.bf16)
    c[m_start + base_m + 0, n_start + base_n + 4] = al.convert(acc04, al.bf16)
    c[m_start + base_m + 0, n_start + base_n + 5] = al.convert(acc05, al.bf16)
    c[m_start + base_m + 0, n_start + base_n + 6] = al.convert(acc06, al.bf16)
    c[m_start + base_m + 0, n_start + base_n + 7] = al.convert(acc07, al.bf16)
    c[m_start + base_m + 1, n_start + base_n + 0] = al.convert(acc10, al.bf16)
    c[m_start + base_m + 1, n_start + base_n + 1] = al.convert(acc11, al.bf16)
    c[m_start + base_m + 1, n_start + base_n + 2] = al.convert(acc12, al.bf16)
    c[m_start + base_m + 1, n_start + base_n + 3] = al.convert(acc13, al.bf16)
    c[m_start + base_m + 1, n_start + base_n + 4] = al.convert(acc14, al.bf16)
    c[m_start + base_m + 1, n_start + base_n + 5] = al.convert(acc15, al.bf16)
    c[m_start + base_m + 1, n_start + base_n + 6] = al.convert(acc16, al.bf16)
    c[m_start + base_m + 1, n_start + base_n + 7] = al.convert(acc17, al.bf16)
    c[m_start + base_m + 2, n_start + base_n + 0] = al.convert(acc20, al.bf16)
    c[m_start + base_m + 2, n_start + base_n + 1] = al.convert(acc21, al.bf16)
    c[m_start + base_m + 2, n_start + base_n + 2] = al.convert(acc22, al.bf16)
    c[m_start + base_m + 2, n_start + base_n + 3] = al.convert(acc23, al.bf16)
    c[m_start + base_m + 2, n_start + base_n + 4] = al.convert(acc24, al.bf16)
    c[m_start + base_m + 2, n_start + base_n + 5] = al.convert(acc25, al.bf16)
    c[m_start + base_m + 2, n_start + base_n + 6] = al.convert(acc26, al.bf16)
    c[m_start + base_m + 2, n_start + base_n + 7] = al.convert(acc27, al.bf16)
    c[m_start + base_m + 3, n_start + base_n + 0] = al.convert(acc30, al.bf16)
    c[m_start + base_m + 3, n_start + base_n + 1] = al.convert(acc31, al.bf16)
    c[m_start + base_m + 3, n_start + base_n + 2] = al.convert(acc32, al.bf16)
    c[m_start + base_m + 3, n_start + base_n + 3] = al.convert(acc33, al.bf16)
    c[m_start + base_m + 3, n_start + base_n + 4] = al.convert(acc34, al.bf16)
    c[m_start + base_m + 3, n_start + base_n + 5] = al.convert(acc35, al.bf16)
    c[m_start + base_m + 3, n_start + base_n + 6] = al.convert(acc36, al.bf16)
    c[m_start + base_m + 3, n_start + base_n + 7] = al.convert(acc37, al.bf16)
    c[m_start + base_m + 4, n_start + base_n + 0] = al.convert(acc40, al.bf16)
    c[m_start + base_m + 4, n_start + base_n + 1] = al.convert(acc41, al.bf16)
    c[m_start + base_m + 4, n_start + base_n + 2] = al.convert(acc42, al.bf16)
    c[m_start + base_m + 4, n_start + base_n + 3] = al.convert(acc43, al.bf16)
    c[m_start + base_m + 4, n_start + base_n + 4] = al.convert(acc44, al.bf16)
    c[m_start + base_m + 4, n_start + base_n + 5] = al.convert(acc45, al.bf16)
    c[m_start + base_m + 4, n_start + base_n + 6] = al.convert(acc46, al.bf16)
    c[m_start + base_m + 4, n_start + base_n + 7] = al.convert(acc47, al.bf16)
    c[m_start + base_m + 5, n_start + base_n + 0] = al.convert(acc50, al.bf16)
    c[m_start + base_m + 5, n_start + base_n + 1] = al.convert(acc51, al.bf16)
    c[m_start + base_m + 5, n_start + base_n + 2] = al.convert(acc52, al.bf16)
    c[m_start + base_m + 5, n_start + base_n + 3] = al.convert(acc53, al.bf16)
    c[m_start + base_m + 5, n_start + base_n + 4] = al.convert(acc54, al.bf16)
    c[m_start + base_m + 5, n_start + base_n + 5] = al.convert(acc55, al.bf16)
    c[m_start + base_m + 5, n_start + base_n + 6] = al.convert(acc56, al.bf16)
    c[m_start + base_m + 5, n_start + base_n + 7] = al.convert(acc57, al.bf16)
    c[m_start + base_m + 6, n_start + base_n + 0] = al.convert(acc60, al.bf16)
    c[m_start + base_m + 6, n_start + base_n + 1] = al.convert(acc61, al.bf16)
    c[m_start + base_m + 6, n_start + base_n + 2] = al.convert(acc62, al.bf16)
    c[m_start + base_m + 6, n_start + base_n + 3] = al.convert(acc63, al.bf16)
    c[m_start + base_m + 6, n_start + base_n + 4] = al.convert(acc64, al.bf16)
    c[m_start + base_m + 6, n_start + base_n + 5] = al.convert(acc65, al.bf16)
    c[m_start + base_m + 6, n_start + base_n + 6] = al.convert(acc66, al.bf16)
    c[m_start + base_m + 6, n_start + base_n + 7] = al.convert(acc67, al.bf16)
    c[m_start + base_m + 7, n_start + base_n + 0] = al.convert(acc70, al.bf16)
    c[m_start + base_m + 7, n_start + base_n + 1] = al.convert(acc71, al.bf16)
    c[m_start + base_m + 7, n_start + base_n + 2] = al.convert(acc72, al.bf16)
    c[m_start + base_m + 7, n_start + base_n + 3] = al.convert(acc73, al.bf16)
    c[m_start + base_m + 7, n_start + base_n + 4] = al.convert(acc74, al.bf16)
    c[m_start + base_m + 7, n_start + base_n + 5] = al.convert(acc75, al.bf16)
    c[m_start + base_m + 7, n_start + base_n + 6] = al.convert(acc76, al.bf16)
    c[m_start + base_m + 7, n_start + base_n + 7] = al.convert(acc77, al.bf16)


def avelang_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA/HIP device."
    M, K_a = A.shape
    K_b, N = B.shape
    assert K_a == K_b, "Inner dimensions must match."

    original_dtype = A.dtype
    A_bf16 = A.contiguous().to(torch.bfloat16)
    B_bf16 = B.contiguous().to(torch.bfloat16)
    C_bf16 = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)

    grid_m = (M + 127) // 128
    grid_n = (N + 127) // 128
    block = 256

    matmul_kernel[lambda: ((grid_m, grid_n, 1), (block, 1, 1))](
        A_bf16, B_bf16, C_bf16, M, K_a, N,
    )
    return C_bf16.to(original_dtype)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return avelang_matmul(A, B)
