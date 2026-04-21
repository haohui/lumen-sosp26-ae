import torch
import torch.nn as nn
import triton
import triton.language as tl


GROUP_M = 8


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "PIPELINE_STAGES": 4},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "PIPELINE_STAGES": 4},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "PIPELINE_STAGES": 4},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "PIPELINE_STAGES": 5},
            num_warps=4,
            num_stages=5,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_transposed_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_ak,
    stride_am,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    GROUP_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_in_group = pid % width
    pid_m = group_id * GROUP_M + (pid_in_group % group_size)
    pid_n = pid_in_group // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    for k_start in tl.range(0, K, BLOCK_K * 2, num_stages=PIPELINE_STAGES):
        k0 = k_start + offs_k
        a0 = tl.load(a_ptrs, mask=mask_m & (k0[None, :] < K), other=0)
        b0 = tl.load(b_ptrs, mask=mask_n & (k0[:, None] < K), other=0)
        acc = tl.dot(a0, b0, acc)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

        k1 = k_start + BLOCK_K + offs_k
        a1 = tl.load(a_ptrs, mask=mask_m & (k1[None, :] < K), other=0)
        b1 = tl.load(b_ptrs, mask=mask_n & (k1[:, None] < K), other=0)
        acc = tl.dot(a1, b1, acc)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=mask_m & mask_n)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("Expected 2D inputs")
        if A.device != B.device:
            raise ValueError("Inputs must be on the same device")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError("Expected bfloat16 inputs")
        if not A.is_cuda or not B.is_cuda:
            raise ValueError("Expected CUDA/HIP inputs")
        if A.shape[0] != B.shape[1]:
            raise ValueError(f"Incompatible shapes: {tuple(A.shape)} and {tuple(B.shape)}")

        if not A.is_contiguous():
            A = A.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()

        K, M = A.shape
        N = B.shape[0]
        C = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
        _matmul_transposed_kernel[grid](
            A,
            B,
            C,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            GROUP_M=GROUP_M,
        )
        return C
