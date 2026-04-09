import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 16, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 4}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4}, num_stages=2, num_warps=8),
    ],
    key=["M", "N", "K"],
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

    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in tl.range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            if EVEN_M:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)

            if EVEN_N:
                b = tl.load(b_ptrs)
            else:
                b = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)
        else:
            k_mask = offs_k < (K - k_tile * BLOCK_K)

            if EVEN_M:
                a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            else:
                a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            if EVEN_N:
                b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
            else:
                b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, c)
    elif EVEN_M:
        tl.store(c_ptrs, c, mask=n_mask[None, :])
    elif EVEN_N:
        tl.store(c_ptrs, c, mask=m_mask[:, None])
    else:
        tl.store(c_ptrs, c, mask=m_mask[:, None] & n_mask[None, :])


def kernel_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Wrapper does only validation/allocation/launch.
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("kernel_function expects tensor inputs")

    if a.ndim != 2:
        raise ValueError(f"`a` must be 2D, got shape={tuple(a.shape)}")

    b_was_vector = False
    if b.ndim == 1:
        b_was_vector = True
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Incompatible shapes: a={tuple(a.shape)}, b={tuple(b.shape)}")
        b_mat = b.reshape(b.shape[0], 1)
    elif b.ndim == 2:
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Incompatible shapes: a={tuple(a.shape)}, b={tuple(b.shape)}")
        b_mat = b
    else:
        raise ValueError(f"`b` must be 1D or 2D, got shape={tuple(b.shape)}")

    if a.device != b.device:
        raise ValueError(f"Input device mismatch: a.device={a.device}, b.device={b.device}")
    if a.device.type == "cpu":
        raise ValueError("Inputs must be on a GPU device")
    if a.dtype != b.dtype:
        raise ValueError(f"Input dtype mismatch: a.dtype={a.dtype}, b.dtype={b.dtype}")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported dtype: {a.dtype}")

    M, K = a.shape
    _, N = b_mat.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    # Single fused stage for this problem: GEMM (matrix-vector is GEMM with N=1).
    _matmul_kernel[grid](
        a,
        b_mat,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_mat.stride(0),
        b_mat.stride(1),
        c.stride(0),
        c.stride(1),
    )

    return c.squeeze(1) if b_was_vector else c