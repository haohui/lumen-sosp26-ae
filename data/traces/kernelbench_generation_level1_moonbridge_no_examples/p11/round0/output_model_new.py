import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def gemm_kernel_tiled(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
    stride_am: al.i32,
    stride_ak: al.i32,
    stride_bk: al.i32,
    stride_bn: al.i32,
    stride_cm: al.i32,
    stride_cn: al.i32,
    BM: al.constexpr,
    BN: al.constexpr,
    BK: al.constexpr,
):
    pid_m = al.block_id(0)
    pid_n = al.block_id(1)

    m_off = pid_m * BM
    n_off = pid_n * BN

    a = al.make_tensor(a_ptr, al.bf16, al.make_layout((M, K), (stride_am, stride_ak)))
    b = al.make_tensor(b_ptr, al.bf16, al.make_layout((K, N), (stride_bk, stride_bn)))
    c = al.make_tensor(c_ptr, al.bf16, al.make_layout((M, N), (stride_cm, stride_cn)))

    a_sh = al.make_shared((BM, BK), al.bf16)
    b_sh = al.make_shared((BK, BN), al.bf16)
    acc = al.make_local((BM, BN), al.f32)

    for mi in al.range(BM):
        for ni in al.range(BN):
            acc[mi, ni] = al.convert(0.0, al.f32)

    for k_block in al.range(0, K, BK):
        for mi in al.range(BM):
            m_idx = m_off + mi
            if m_idx < M:
                for ki in al.range(BK):
                    k_idx = k_block + ki
                    if k_idx < K:
                        a_sh[mi, ki] = a[m_idx, k_idx]
                    else:
                        a_sh[mi, ki] = al.convert(0.0, al.bf16)

        for ki in al.range(BK):
            for ni in al.range(BN):
                n_idx = n_off + ni
                k_idx = k_block + ki
                if k_idx < K and n_idx < N:
                    b_sh[ki, ni] = b[k_idx, n_idx]
                else:
                    b_sh[ki, ni] = al.convert(0.0, al.bf16)

        al.syncthreads()

        for mi in al.range(BM):
            for ni in al.range(BN):
                dot = al.convert(0.0, al.f32)
                for ki in al.range(BK):
                    a_val = al.convert(a_sh[mi, ki], al.f32)
                    b_val = al.convert(b_sh[ki, ni], al.f32)
                    dot = dot + a_val * b_val
                acc[mi, ni] = acc[mi, ni] + dot

        al.syncthreads()

    for mi in al.range(BM):
        m_idx = m_off + mi
        if m_idx < M:
            for ni in al.range(BN):
                n_idx = n_off + ni
                if n_idx < N:
                    c[m_idx, n_idx] = al.convert(acc[mi, ni], al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA/HIP device."

        b, i, j, l = A.shape
        lb, k = B.shape
        assert l == lb, f"Contracting dimension mismatch: {l} vs {lb}"

        M = b * i * j
        N = k
        K_dim = l

        A_flat = A.reshape(M, K_dim).contiguous()
        B_contig = B.contiguous()
        C_flat = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)

        BM = 16
        BN = 8
        BK = 256

        grid_m = (M + BM - 1) // BM
        grid_n = (N + BN - 1) // BN

        stride_am = K_dim
        stride_ak = 1
        stride_bk = N
        stride_bn = 1
        stride_cm = N
        stride_cn = 1

        gemm_kernel_tiled[lambda: ((grid_m, grid_n, 1), (1, 1, 1))](
            A_flat.data_ptr(),
            B_contig.data_ptr(),
            C_flat.data_ptr(),
            M, N, K_dim,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BM, BN, BK,
        )

        return C_flat.reshape(b, i, j, k)
