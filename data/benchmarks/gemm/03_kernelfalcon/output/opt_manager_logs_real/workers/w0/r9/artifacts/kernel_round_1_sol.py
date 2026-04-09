import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 16, "GROUP_M": 4}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 16, "GROUP_M": 4}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 16, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 16, "GROUP_M": 8}, num_stages=2, num_warps=4),
    ],
    key=["M", "N", "K", "DTYPE_ID"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["M"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
        "EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0,
    }
)
@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    DTYPE_ID,  # autotune key partitioning by dtype family
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
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

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

    rm = offs_m < M
    rn = offs_n < N

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if EVEN_K:
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            if EVEN_M:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(a_ptrs, mask=rm[:, None], other=0.0)

            if EVEN_N:
                b = tl.load(b_ptrs)
            else:
                b = tl.load(b_ptrs, mask=rn[None, :], other=0.0)

            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    else:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = offs_k + k * BLOCK_K < K

            if EVEN_M and EVEN_N:
                a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
            elif EVEN_M and (not EVEN_N):
                a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=k_mask[:, None] & rn[None, :], other=0.0)
            elif (not EVEN_M) and EVEN_N:
                a = tl.load(a_ptrs, mask=rm[:, None] & k_mask[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
            else:
                a = tl.load(a_ptrs, mask=rm[:, None] & k_mask[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=k_mask[:, None] & rn[None, :], other=0.0)

            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out = acc.to(c_ptr.dtype.element_ty)

    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, out)
    else:
        tl.store(c_ptrs, out, mask=rm[:, None] & rn[None, :])


def kernel_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Validation only (no math in wrapper).
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("kernel_function expects torch.Tensor inputs")
    if a.ndim != 2:
        raise ValueError(f"`a` must be 2D, got a.ndim={a.ndim}")
    if b.ndim not in (1, 2):
        raise ValueError(f"`b` must be 1D or 2D, got b.ndim={b.ndim}")
    if a.device != b.device:
        raise ValueError(f"Input device mismatch: a.device={a.device}, b.device={b.device}")
    if a.dtype != b.dtype:
        raise ValueError(f"Input dtype mismatch: a.dtype={a.dtype}, b.dtype={b.dtype}")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported dtype: {a.dtype}")
    if a.device.type == "cpu":
        raise ValueError("Inputs must be on a GPU device supported by Triton")

    M, K = a.shape
    if b.ndim == 2:
        if K != b.shape[0]:
            raise ValueError(f"Incompatible shapes: a={a.shape}, b={b.shape}")
        N = b.shape[1]
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        stride_bk, stride_bn = b.stride(0), b.stride(1)
        stride_cm, stride_cn = c.stride(0), c.stride(1)
    else:
        if K != b.shape[0]:
            raise ValueError(f"Incompatible shapes: a={a.shape}, b={b.shape}")
        N = 1
        c = torch.empty((M,), device=a.device, dtype=a.dtype)
        stride_bk, stride_bn = b.stride(0), 0
        stride_cm, stride_cn = c.stride(0), 0

    if a.dtype == torch.float16:
        dtype_id = 0
    elif a.dtype == torch.bfloat16:
        dtype_id = 1
    else:
        dtype_id = 2

    # No additional stage to fuse for plain GEMM/matvec: single Triton kernel already performs all compute.
    if M == 0 or N == 0:
        return c

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    _matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        dtype_id,
        a.stride(0),
        a.stride(1),
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
    )
    return c