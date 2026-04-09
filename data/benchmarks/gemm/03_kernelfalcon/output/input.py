"""
Triton matmul kernel implementation for:
    y = a @ b
where a: [M, K], b: [K, N], y: [M, N].

Fusion note:
- The test graph contains a single operation (matmul), so the fully fused implementation
  is a single Triton kernel that performs load->dot->store in one pass.
- No extra epilogue ops exist to fuse.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": gm},
            num_stages=ns,
            num_warps=nw,
        )
        for bm, bn, bk, gm, ns, nw in [
            (64, 64, 32, 8, 3, 4),
            (128, 64, 32, 8, 3, 4),
            (64, 128, 32, 8, 3, 4),
            (128, 128, 32, 8, 4, 8),
            (64, 64, 64, 8, 4, 4),
            (128, 64, 64, 8, 4, 8),
        ]
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Grouped ordering for better L2 reuse
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_idx = k0 * BLOCK_K + offs_k

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak
        b_ptrs = b_ptr + k_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        b_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Match PyTorch matmul output dtype behavior for fp16/bf16 inputs in this test context.
    out = acc.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, out, mask=c_mask)


def kernel_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Wrapper that validates inputs, allocates output, configures launch grid, and runs Triton kernel.

    Runtime restrictions compliance:
    - No PyTorch compute ops are used here for numerical math.
    - All core computation happens inside _matmul_kernel.
    """
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("kernel_function expects two torch.Tensor inputs")

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got a.ndim={a.ndim}, b.ndim={b.ndim}")

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes: a={a.shape}, b={b.shape}")

    if a.device != b.device:
        raise ValueError(f"Input device mismatch: a.device={a.device}, b.device={b.device}")

    if a.dtype != b.dtype:
        raise ValueError(f"Input dtype mismatch: a.dtype={a.dtype}, b.dtype={b.dtype}")

    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported dtype for this kernel: {a.dtype}")

    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Inputs must be on GPU device")

    M, K = a.shape
    _, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    _matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c