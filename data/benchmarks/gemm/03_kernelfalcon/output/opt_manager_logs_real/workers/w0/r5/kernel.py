import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=2, num_warps=2),
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
    # Grouped launch ordering for better cache behavior
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Full fused matmul pipeline: load -> dot-accumulate -> store
    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def kernel_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("kernel_function expects torch.Tensor inputs")

    if a.ndim != 2:
        raise ValueError(f"`a` must be 2D, got shape={tuple(a.shape)}")

    # Support matrix-vector by treating vector as (K, 1)
    return_vector = False
    if b.ndim == 1:
        b_mat = b.unsqueeze(1)
        return_vector = True
    elif b.ndim == 2:
        b_mat = b
    else:
        raise ValueError(f"`b` must be 1D or 2D, got shape={tuple(b.shape)}")

    if a.shape[1] != b_mat.shape[0]:
        raise ValueError(f"Incompatible shapes: a={tuple(a.shape)}, b={tuple(b_mat.shape)}")
    if a.device != b_mat.device:
        raise ValueError(f"Device mismatch: a.device={a.device}, b.device={b_mat.device}")
    if a.dtype != b_mat.dtype:
        raise ValueError(f"Dtype mismatch: a.dtype={a.dtype}, b.dtype={b_mat.dtype}")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported dtype: {a.dtype}")

    # ROCm tensors report is_cuda=True; avoid non-portable .is_hip attribute
    if not (a.is_cuda and b_mat.is_cuda):
        raise ValueError("Inputs must be on CUDA/HIP GPU device")

    M, K = a.shape
    _, N = b_mat.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

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

    return c[:, 0] if return_vector else c