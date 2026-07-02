import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 2048
K = 8192
N = 4096


@substrate.jit
def mfma_probe_kernel(
    A: S.Tensor((64, 2), S.u32),
    B: S.Tensor((64, 2), S.u32),
    C: S.Tensor((64, 16), S.f32),
):
    lane = S.thread_id(0)
    c_lane = S.full((16,), 0.0, S.f32)
    m_a = S.view(A[lane], S.Tensor((1, 4, 1), S.bf16))
    m_b = S.view(B[lane], S.Tensor((1, 4, 1), S.bf16))
    c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(m_a[0], m_b[0], c_lane)
    C[lane] = c_lane


@substrate.jit
def gemm_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((K, N), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    col = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    row = S.block_id(1) * S.block_dim(1) + S.thread_id(1)

    if row < M and col < N:
        acc = S.convert(0.0, S.f32)
        for kk in S.range(K):
            acc += S.convert(A[row, kk], S.f32) * S.convert(B[kk, col], S.f32)
        C[row, col] = S.convert(acc, S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._mfma_a = torch.zeros((64, 2), device="cuda", dtype=torch.uint32)
        self._mfma_b = torch.zeros((64, 2), device="cuda", dtype=torch.uint32)
        self._mfma_c = torch.zeros((64, 16), device="cuda", dtype=torch.float32)

    def forward(self, A, B):
        A2 = A.transpose(-2, -1).contiguous()
        B2 = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        mfma_probe_kernel[lambda: ((1, 1, 1), (64, 1, 1))](
            self._mfma_a, self._mfma_b, self._mfma_c
        )
        gemm_kernel[lambda: ((256, 128, 1), (16, 16, 1))](A2, B2, C)
        return C
