import torch
import torch.nn as nn

import avelang
import avelang.language as al


@avelang.jit
def transpose_b_kernel(
    B_ptr: al.Pointer(al.bf16),
    Bt_ptr: al.Pointer(al.bf16),
    K: al.i32, L: al.i32,
):
    tid = al.thread_id(0)
    gid = al.block_id(0) * al.block_dim(0) + tid
    row = gid // L
    col = gid % L
    if row < K:
        layout_b = al.make_layout((K, L), (L, 1))
        B = al.make_tensor(B_ptr, al.bf16, layout_b)
        layout_bt = al.make_layout((L, K), (K, 1))
        Bt = al.make_tensor(Bt_ptr, al.bf16, layout_bt)
        Bt[col, row] = B[row, col]


# Directly use the proven reference GEMM kernel from avelang_kernels
from avelang_kernels.amdgpu_gemm import _gemm_pipeline_transposed_b_kernel

GROUP_M = 128
GROUP_N = 128
GROUP_K = 64
NUM_WARPS = 4
WARP_SIZE = 64
THREADS = NUM_WARPS * WARP_SIZE
BF16_BYTES = 2


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._Bt = None
        self._Bt_shape = None

    def forward(self, A, B):
        N_batch, M_val, K_val = A.shape
        K_b, L_val = B.shape

        if K_val != K_b:
            raise RuntimeError(f'K dimension mismatch: {K_val} vs {K_b}')

        A = A.contiguous()
        B = B.contiguous()

        # Transpose B to (L, K) format expected by reference gemm
        if self._Bt is None or self._Bt_shape != (L_val, K_val):
            self._Bt = torch.empty((L_val, K_val), device=B.device, dtype=B.dtype)
            self._Bt_shape = (L_val, K_val)

        block_t = 256
        grid_t = (K_val * L_val + block_t - 1) // block_t
        transpose_b_kernel[lambda: ((grid_t, 1, 1), (block_t, 1, 1))](
            B, self._Bt, K_val, L_val)

        C = torch.empty((N_batch, M_val, L_val), device=A.device, dtype=A.dtype)

        # Call reference GEMM for each batch
        m_groups = M_val // GROUP_M
        n_groups = L_val // GROUP_N
        grid_size = m_groups * n_groups

        for b in range(N_batch):
            _gemm_pipeline_transposed_b_kernel[
                lambda: ((grid_size, 1, 1), (THREADS, 1, 1))
            ](A[b], self._Bt, C[b], M_val, L_val, K_val)

        return C
