import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16, "GROUP_M": 8, "matrix_instr_nonkdim": 16, "waves_per_eu": 2},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16, "GROUP_M": 8, "matrix_instr_nonkdim": 16, "waves_per_eu": 1},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "matrix_instr_nonkdim": 16, "waves_per_eu": 2},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "matrix_instr_nonkdim": 16, "waves_per_eu": 1},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 8, "GROUP_M": 8, "matrix_instr_nonkdim": 16, "waves_per_eu": 2},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 8, "GROUP_M": 8, "matrix_instr_nonkdim": 16, "waves_per_eu": 1},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 16, "GROUP_M": 4, "matrix_instr_nonkdim": 16, "waves_per_eu": 2},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 16, "GROUP_M": 4, "matrix_instr_nonkdim": 16, "waves_per_eu": 2},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16, "GROUP_M": 2, "matrix_instr_nonkdim": 16, "waves_per_eu": 1},
            num_stages=2,
            num_warps=8,
        ),
    ],
    key=["M", "N", "K", "DTYPE_ID"],
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0,
        "EVEN_M": lambda args: args["M"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
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
    DTYPE_ID,  # used for autotune key specialization
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
    EVEN_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
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

    offs_am = offs_m if EVEN_M else (offs_m % M)
    offs_bn = offs_n if EVEN_N else (offs_n % N)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_mask = offs_k < (K - k_tile * BLOCK_K)
            a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, c)
    else:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


def kernel_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("kernel_function expects torch.Tensor inputs")

    if a.ndim != 2:
        raise ValueError(f"Expected a to be 2D, got a.ndim={a.ndim}")

    # Optional convenience: allow matrix-vector by treating vector as (K, 1)
    squeeze_output = False
    if b.ndim == 1:
        b = b[:, None]
        squeeze_output = True
    elif b.ndim != 2:
        raise ValueError(f"Expected b to be 1D or 2D, got b.ndim={b.ndim}")

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes: a={a.shape}, b={b.shape}")
    if a.device != b.device:
        raise ValueError(f"Input device mismatch: a.device={a.device}, b.device={b.device}")
    if a.dtype != b.dtype:
        raise ValueError(f"Input dtype mismatch: a.dtype={a.dtype}, b.dtype={b.dtype}")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported dtype for this kernel: {a.dtype}")

    # Fix: `.is_hip` is not a standard Tensor attribute.
    # Accept any non-CPU accelerator device (CUDA/ROCm/private backends).
    if a.device.type == "cpu" or b.device.type == "cpu":
        raise ValueError("Inputs must be on a GPU/accelerator device")

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    if a.dtype == torch.float16:
        dtype_id = 0
    elif a.dtype == torch.bfloat16:
        dtype_id = 1
    else:
        dtype_id = 2

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
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )

    if squeeze_output:
        return c[:, 0]
    return c