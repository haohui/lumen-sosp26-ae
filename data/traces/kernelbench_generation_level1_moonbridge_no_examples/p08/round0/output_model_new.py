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
    stride_ak: al.i32,
    stride_bk: al.i32,
):
    A = al.make_tensor(a_ptr, al.bf16, al.make_layout((M, K), (stride_ak, 1)))
    B = al.make_tensor(b_ptr, al.bf16, al.make_layout((K, N), (stride_bk, 1)))
    C = al.make_tensor(c_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    pid_m = al.block_id(0)
    pid_n = al.block_id(1)

    tid_m = al.thread_id(0)  # 0..15
    tid_n = al.thread_id(1)  # 0..15

    a_tile = al.make_shared((128, 16), al.bf16)
    b_tile = al.make_shared((16, 128), al.bf16)

    acc_00 = al.convert(0.0, al.f32)
    acc_01 = al.convert(0.0, al.f32)
    acc_02 = al.convert(0.0, al.f32)
    acc_03 = al.convert(0.0, al.f32)
    acc_04 = al.convert(0.0, al.f32)
    acc_05 = al.convert(0.0, al.f32)
    acc_06 = al.convert(0.0, al.f32)
    acc_07 = al.convert(0.0, al.f32)
    acc_10 = al.convert(0.0, al.f32)
    acc_11 = al.convert(0.0, al.f32)
    acc_12 = al.convert(0.0, al.f32)
    acc_13 = al.convert(0.0, al.f32)
    acc_14 = al.convert(0.0, al.f32)
    acc_15 = al.convert(0.0, al.f32)
    acc_16 = al.convert(0.0, al.f32)
    acc_17 = al.convert(0.0, al.f32)
    acc_20 = al.convert(0.0, al.f32)
    acc_21 = al.convert(0.0, al.f32)
    acc_22 = al.convert(0.0, al.f32)
    acc_23 = al.convert(0.0, al.f32)
    acc_24 = al.convert(0.0, al.f32)
    acc_25 = al.convert(0.0, al.f32)
    acc_26 = al.convert(0.0, al.f32)
    acc_27 = al.convert(0.0, al.f32)
    acc_30 = al.convert(0.0, al.f32)
    acc_31 = al.convert(0.0, al.f32)
    acc_32 = al.convert(0.0, al.f32)
    acc_33 = al.convert(0.0, al.f32)
    acc_34 = al.convert(0.0, al.f32)
    acc_35 = al.convert(0.0, al.f32)
    acc_36 = al.convert(0.0, al.f32)
    acc_37 = al.convert(0.0, al.f32)
    acc_40 = al.convert(0.0, al.f32)
    acc_41 = al.convert(0.0, al.f32)
    acc_42 = al.convert(0.0, al.f32)
    acc_43 = al.convert(0.0, al.f32)
    acc_44 = al.convert(0.0, al.f32)
    acc_45 = al.convert(0.0, al.f32)
    acc_46 = al.convert(0.0, al.f32)
    acc_47 = al.convert(0.0, al.f32)
    acc_50 = al.convert(0.0, al.f32)
    acc_51 = al.convert(0.0, al.f32)
    acc_52 = al.convert(0.0, al.f32)
    acc_53 = al.convert(0.0, al.f32)
    acc_54 = al.convert(0.0, al.f32)
    acc_55 = al.convert(0.0, al.f32)
    acc_56 = al.convert(0.0, al.f32)
    acc_57 = al.convert(0.0, al.f32)
    acc_60 = al.convert(0.0, al.f32)
    acc_61 = al.convert(0.0, al.f32)
    acc_62 = al.convert(0.0, al.f32)
    acc_63 = al.convert(0.0, al.f32)
    acc_64 = al.convert(0.0, al.f32)
    acc_65 = al.convert(0.0, al.f32)
    acc_66 = al.convert(0.0, al.f32)
    acc_67 = al.convert(0.0, al.f32)
    acc_70 = al.convert(0.0, al.f32)
    acc_71 = al.convert(0.0, al.f32)
    acc_72 = al.convert(0.0, al.f32)
    acc_73 = al.convert(0.0, al.f32)
    acc_74 = al.convert(0.0, al.f32)
    acc_75 = al.convert(0.0, al.f32)
    acc_76 = al.convert(0.0, al.f32)
    acc_77 = al.convert(0.0, al.f32)

    for k_block in al.range(0, K, 16):
        for off in al.range(0, 8):
            gl_row = pid_m * 128 + tid_m + off * 16
            gl_col = k_block + tid_n
            sh_row = tid_m + off * 16
            if gl_row < M and gl_col < K:
                a_tile[sh_row, tid_n] = A[gl_row, gl_col]
            else:
                a_tile[sh_row, tid_n] = al.convert(0.0, al.bf16)

        for off in al.range(0, 8):
            gl_row = k_block + tid_m
            gl_col = pid_n * 128 + tid_n + off * 16
            sh_col = tid_n + off * 16
            if gl_row < K and gl_col < N:
                b_tile[tid_m, sh_col] = B[gl_row, gl_col]
            else:
                b_tile[tid_m, sh_col] = al.convert(0.0, al.bf16)

        al.syncthreads()

        for ki in al.range(0, 16):
            a0 = al.convert(a_tile[tid_m + 0, ki], al.f32)
            a1 = al.convert(a_tile[tid_m + 16, ki], al.f32)
            a2 = al.convert(a_tile[tid_m + 32, ki], al.f32)
            a3 = al.convert(a_tile[tid_m + 48, ki], al.f32)
            a4 = al.convert(a_tile[tid_m + 64, ki], al.f32)
            a5 = al.convert(a_tile[tid_m + 80, ki], al.f32)
            a6 = al.convert(a_tile[tid_m + 96, ki], al.f32)
            a7 = al.convert(a_tile[tid_m + 112, ki], al.f32)

            b0 = al.convert(b_tile[ki, tid_n + 0], al.f32)
            b1 = al.convert(b_tile[ki, tid_n + 16], al.f32)
            b2 = al.convert(b_tile[ki, tid_n + 32], al.f32)
            b3 = al.convert(b_tile[ki, tid_n + 48], al.f32)
            b4 = al.convert(b_tile[ki, tid_n + 64], al.f32)
            b5 = al.convert(b_tile[ki, tid_n + 80], al.f32)
            b6 = al.convert(b_tile[ki, tid_n + 96], al.f32)
            b7 = al.convert(b_tile[ki, tid_n + 112], al.f32)

            acc_00 = acc_00 + a0 * b0
            acc_01 = acc_01 + a0 * b1
            acc_02 = acc_02 + a0 * b2
            acc_03 = acc_03 + a0 * b3
            acc_04 = acc_04 + a0 * b4
            acc_05 = acc_05 + a0 * b5
            acc_06 = acc_06 + a0 * b6
            acc_07 = acc_07 + a0 * b7
            acc_10 = acc_10 + a1 * b0
            acc_11 = acc_11 + a1 * b1
            acc_12 = acc_12 + a1 * b2
            acc_13 = acc_13 + a1 * b3
            acc_14 = acc_14 + a1 * b4
            acc_15 = acc_15 + a1 * b5
            acc_16 = acc_16 + a1 * b6
            acc_17 = acc_17 + a1 * b7
            acc_20 = acc_20 + a2 * b0
            acc_21 = acc_21 + a2 * b1
            acc_22 = acc_22 + a2 * b2
            acc_23 = acc_23 + a2 * b3
            acc_24 = acc_24 + a2 * b4
            acc_25 = acc_25 + a2 * b5
            acc_26 = acc_26 + a2 * b6
            acc_27 = acc_27 + a2 * b7
            acc_30 = acc_30 + a3 * b0
            acc_31 = acc_31 + a3 * b1
            acc_32 = acc_32 + a3 * b2
            acc_33 = acc_33 + a3 * b3
            acc_34 = acc_34 + a3 * b4
            acc_35 = acc_35 + a3 * b5
            acc_36 = acc_36 + a3 * b6
            acc_37 = acc_37 + a3 * b7
            acc_40 = acc_40 + a4 * b0
            acc_41 = acc_41 + a4 * b1
            acc_42 = acc_42 + a4 * b2
            acc_43 = acc_43 + a4 * b3
            acc_44 = acc_44 + a4 * b4
            acc_45 = acc_45 + a4 * b5
            acc_46 = acc_46 + a4 * b6
            acc_47 = acc_47 + a4 * b7
            acc_50 = acc_50 + a5 * b0
            acc_51 = acc_51 + a5 * b1
            acc_52 = acc_52 + a5 * b2
            acc_53 = acc_53 + a5 * b3
            acc_54 = acc_54 + a5 * b4
            acc_55 = acc_55 + a5 * b5
            acc_56 = acc_56 + a5 * b6
            acc_57 = acc_57 + a5 * b7
            acc_60 = acc_60 + a6 * b0
            acc_61 = acc_61 + a6 * b1
            acc_62 = acc_62 + a6 * b2
            acc_63 = acc_63 + a6 * b3
            acc_64 = acc_64 + a6 * b4
            acc_65 = acc_65 + a6 * b5
            acc_66 = acc_66 + a6 * b6
            acc_67 = acc_67 + a6 * b7
            acc_70 = acc_70 + a7 * b0
            acc_71 = acc_71 + a7 * b1
            acc_72 = acc_72 + a7 * b2
            acc_73 = acc_73 + a7 * b3
            acc_74 = acc_74 + a7 * b4
            acc_75 = acc_75 + a7 * b5
            acc_76 = acc_76 + a7 * b6
            acc_77 = acc_77 + a7 * b7

        al.syncthreads()

    gl_r0 = pid_m * 128 + tid_m + 0
    gl_r1 = pid_m * 128 + tid_m + 16
    gl_r2 = pid_m * 128 + tid_m + 32
    gl_r3 = pid_m * 128 + tid_m + 48
    gl_r4 = pid_m * 128 + tid_m + 64
    gl_r5 = pid_m * 128 + tid_m + 80
    gl_r6 = pid_m * 128 + tid_m + 96
    gl_r7 = pid_m * 128 + tid_m + 112

    c0 = pid_n * 128 + tid_n + 0
    c1 = pid_n * 128 + tid_n + 16
    c2 = pid_n * 128 + tid_n + 32
    c3 = pid_n * 128 + tid_n + 48
    c4 = pid_n * 128 + tid_n + 64
    c5 = pid_n * 128 + tid_n + 80
    c6 = pid_n * 128 + tid_n + 96
    c7 = pid_n * 128 + tid_n + 112

    if gl_r0 < M and c0 < N: C[gl_r0, c0] = al.convert(acc_00, al.bf16)
    if gl_r0 < M and c1 < N: C[gl_r0, c1] = al.convert(acc_01, al.bf16)
    if gl_r0 < M and c2 < N: C[gl_r0, c2] = al.convert(acc_02, al.bf16)
    if gl_r0 < M and c3 < N: C[gl_r0, c3] = al.convert(acc_03, al.bf16)
    if gl_r0 < M and c4 < N: C[gl_r0, c4] = al.convert(acc_04, al.bf16)
    if gl_r0 < M and c5 < N: C[gl_r0, c5] = al.convert(acc_05, al.bf16)
    if gl_r0 < M and c6 < N: C[gl_r0, c6] = al.convert(acc_06, al.bf16)
    if gl_r0 < M and c7 < N: C[gl_r0, c7] = al.convert(acc_07, al.bf16)

    if gl_r1 < M and c0 < N: C[gl_r1, c0] = al.convert(acc_10, al.bf16)
    if gl_r1 < M and c1 < N: C[gl_r1, c1] = al.convert(acc_11, al.bf16)
    if gl_r1 < M and c2 < N: C[gl_r1, c2] = al.convert(acc_12, al.bf16)
    if gl_r1 < M and c3 < N: C[gl_r1, c3] = al.convert(acc_13, al.bf16)
    if gl_r1 < M and c4 < N: C[gl_r1, c4] = al.convert(acc_14, al.bf16)
    if gl_r1 < M and c5 < N: C[gl_r1, c5] = al.convert(acc_15, al.bf16)
    if gl_r1 < M and c6 < N: C[gl_r1, c6] = al.convert(acc_16, al.bf16)
    if gl_r1 < M and c7 < N: C[gl_r1, c7] = al.convert(acc_17, al.bf16)

    if gl_r2 < M and c0 < N: C[gl_r2, c0] = al.convert(acc_20, al.bf16)
    if gl_r2 < M and c1 < N: C[gl_r2, c1] = al.convert(acc_21, al.bf16)
    if gl_r2 < M and c2 < N: C[gl_r2, c2] = al.convert(acc_22, al.bf16)
    if gl_r2 < M and c3 < N: C[gl_r2, c3] = al.convert(acc_23, al.bf16)
    if gl_r2 < M and c4 < N: C[gl_r2, c4] = al.convert(acc_24, al.bf16)
    if gl_r2 < M and c5 < N: C[gl_r2, c5] = al.convert(acc_25, al.bf16)
    if gl_r2 < M and c6 < N: C[gl_r2, c6] = al.convert(acc_26, al.bf16)
    if gl_r2 < M and c7 < N: C[gl_r2, c7] = al.convert(acc_27, al.bf16)

    if gl_r3 < M and c0 < N: C[gl_r3, c0] = al.convert(acc_30, al.bf16)
    if gl_r3 < M and c1 < N: C[gl_r3, c1] = al.convert(acc_31, al.bf16)
    if gl_r3 < M and c2 < N: C[gl_r3, c2] = al.convert(acc_32, al.bf16)
    if gl_r3 < M and c3 < N: C[gl_r3, c3] = al.convert(acc_33, al.bf16)
    if gl_r3 < M and c4 < N: C[gl_r3, c4] = al.convert(acc_34, al.bf16)
    if gl_r3 < M and c5 < N: C[gl_r3, c5] = al.convert(acc_35, al.bf16)
    if gl_r3 < M and c6 < N: C[gl_r3, c6] = al.convert(acc_36, al.bf16)
    if gl_r3 < M and c7 < N: C[gl_r3, c7] = al.convert(acc_37, al.bf16)

    if gl_r4 < M and c0 < N: C[gl_r4, c0] = al.convert(acc_40, al.bf16)
    if gl_r4 < M and c1 < N: C[gl_r4, c1] = al.convert(acc_41, al.bf16)
    if gl_r4 < M and c2 < N: C[gl_r4, c2] = al.convert(acc_42, al.bf16)
    if gl_r4 < M and c3 < N: C[gl_r4, c3] = al.convert(acc_43, al.bf16)
    if gl_r4 < M and c4 < N: C[gl_r4, c4] = al.convert(acc_44, al.bf16)
    if gl_r4 < M and c5 < N: C[gl_r4, c5] = al.convert(acc_45, al.bf16)
    if gl_r4 < M and c6 < N: C[gl_r4, c6] = al.convert(acc_46, al.bf16)
    if gl_r4 < M and c7 < N: C[gl_r4, c7] = al.convert(acc_47, al.bf16)

    if gl_r5 < M and c0 < N: C[gl_r5, c0] = al.convert(acc_50, al.bf16)
    if gl_r5 < M and c1 < N: C[gl_r5, c1] = al.convert(acc_51, al.bf16)
    if gl_r5 < M and c2 < N: C[gl_r5, c2] = al.convert(acc_52, al.bf16)
    if gl_r5 < M and c3 < N: C[gl_r5, c3] = al.convert(acc_53, al.bf16)
    if gl_r5 < M and c4 < N: C[gl_r5, c4] = al.convert(acc_54, al.bf16)
    if gl_r5 < M and c5 < N: C[gl_r5, c5] = al.convert(acc_55, al.bf16)
    if gl_r5 < M and c6 < N: C[gl_r5, c6] = al.convert(acc_56, al.bf16)
    if gl_r5 < M and c7 < N: C[gl_r5, c7] = al.convert(acc_57, al.bf16)

    if gl_r6 < M and c0 < N: C[gl_r6, c0] = al.convert(acc_60, al.bf16)
    if gl_r6 < M and c1 < N: C[gl_r6, c1] = al.convert(acc_61, al.bf16)
    if gl_r6 < M and c2 < N: C[gl_r6, c2] = al.convert(acc_62, al.bf16)
    if gl_r6 < M and c3 < N: C[gl_r6, c3] = al.convert(acc_63, al.bf16)
    if gl_r6 < M and c4 < N: C[gl_r6, c4] = al.convert(acc_64, al.bf16)
    if gl_r6 < M and c5 < N: C[gl_r6, c5] = al.convert(acc_65, al.bf16)
    if gl_r6 < M and c6 < N: C[gl_r6, c6] = al.convert(acc_66, al.bf16)
    if gl_r6 < M and c7 < N: C[gl_r6, c7] = al.convert(acc_67, al.bf16)

    if gl_r7 < M and c0 < N: C[gl_r7, c0] = al.convert(acc_70, al.bf16)
    if gl_r7 < M and c1 < N: C[gl_r7, c1] = al.convert(acc_71, al.bf16)
    if gl_r7 < M and c2 < N: C[gl_r7, c2] = al.convert(acc_72, al.bf16)
    if gl_r7 < M and c3 < N: C[gl_r7, c3] = al.convert(acc_73, al.bf16)
    if gl_r7 < M and c4 < N: C[gl_r7, c4] = al.convert(acc_74, al.bf16)
    if gl_r7 < M and c5 < N: C[gl_r7, c5] = al.convert(acc_75, al.bf16)
    if gl_r7 < M and c6 < N: C[gl_r7, c6] = al.convert(acc_76, al.bf16)
    if gl_r7 < M and c7 < N: C[gl_r7, c7] = al.convert(acc_77, al.bf16)


def avelang_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Dimension mismatch: A has {K} cols but B has {K2} rows"

    A_bf16 = A.to(torch.bfloat16)
    B_bf16 = B.to(torch.bfloat16)

    C = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)

    grid_m = (M + 128 - 1) // 128
    grid_n = (N + 128 - 1) // 128

    stride_ak = K
    stride_bk = N

    matmul_kernel[lambda: ((grid_m, grid_n, 1), (16, 16, 1))](
        A_bf16, B_bf16, C, M, K, N, stride_ak, stride_bk,
    )

    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return avelang_matmul(A, B)
