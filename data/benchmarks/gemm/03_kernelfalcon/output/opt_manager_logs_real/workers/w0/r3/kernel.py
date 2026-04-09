import torch
import triton
import triton.language as tl


# -------------------------
# 2D x 2D : GEMM
# -------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
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
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc)

    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# -------------------------
# 2D x 1D : MATVEC
# -------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_K": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}, num_stages=2, num_warps=8),
    ],
    key=["M", "K"],
)
@triton.jit
def _matvec_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    M,
    K,
    stride_am,
    stride_ak,
    stride_x,
    stride_y,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        x_ptrs = x_ptr + offs_k * stride_x

        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        x = tl.load(x_ptrs, mask=mask_k, other=0.0)

        acc += tl.sum(a * x[None, :], axis=1)

    y = acc.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs_m * stride_y, y, mask=mask_m)


# -------------------------
# 1D x 2D : VECMAT
# -------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2, num_warps=8),
    ],
    key=["N", "K"],
)
@triton.jit
def _vecmat_kernel(
    x_ptr,
    b_ptr,
    y_ptr,
    N,
    K,
    stride_x,
    stride_bk,
    stride_bn,
    stride_y,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        x = tl.load(x_ptr + offs_k * stride_x, mask=mask_k, other=0.0)

        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.sum(x[:, None] * b, axis=0)

    y = acc.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs_n * stride_y, y, mask=mask_n)


def kernel_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Wrapper does only validation / allocation / launch (no math).
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("kernel_function expects torch.Tensor inputs")
    if a.device != b.device:
        raise ValueError(f"Input device mismatch: {a.device} vs {b.device}")
    if a.device.type == "cpu":
        raise ValueError("Inputs must be on a GPU device")
    if a.dtype != b.dtype:
        raise ValueError(f"Input dtype mismatch: {a.dtype} vs {b.dtype}")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported dtype: {a.dtype}")

    # Fused scope note:
    # This workload is a single linear algebra op (matmul/matvec/vecmat), so there are
    # no additional dependent stages to fuse beyond this kernelized computation.

    # Case 1: matrix @ matrix
    if a.ndim == 2 and b.ndim == 2:
        M, K = a.shape
        Kb, N = b.shape
        if K != Kb:
            raise ValueError(f"Incompatible shapes for matmul: {a.shape} @ {b.shape}")
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        if M == 0 or N == 0:
            return c

        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
        _matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )
        return c

    # Case 2: matrix @ vector
    if a.ndim == 2 and b.ndim == 1:
        M, K = a.shape
        if K != b.shape[0]:
            raise ValueError(f"Incompatible shapes for matvec: {a.shape} @ {b.shape}")
        y = torch.empty((M,), device=a.device, dtype=a.dtype)
        if M == 0:
            return y

        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
        _matvec_kernel[grid](
            a, b, y,
            M, K,
            a.stride(0), a.stride(1),
            b.stride(0), y.stride(0),
        )
        return y

    # Case 3: vector @ matrix
    if a.ndim == 1 and b.ndim == 2:
        K = a.shape[0]
        Kb, N = b.shape
        if K != Kb:
            raise ValueError(f"Incompatible shapes for vecmat: {a.shape} @ {b.shape}")
        y = torch.empty((N,), device=a.device, dtype=a.dtype)
        if N == 0:
            return y

        grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
        _vecmat_kernel[grid](
            a, b, y,
            N, K,
            a.stride(0),
            b.stride(0), b.stride(1),
            y.stride(0),
        )
        return y

    raise ValueError(
        f"Unsupported input ranks: a.ndim={a.ndim}, b.ndim={b.ndim}. "
        "Supported: (2,2), (2,1), (1,2)."
    )