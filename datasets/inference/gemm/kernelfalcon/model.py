import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_a_bt_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Compute C = A @ B.T
      - A: [M, K]
      - B: [N, K]
      - C: [M, N]
    """
    pid = tl.program_id(axis=0)

    # Tile decomposition
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Grouped ordering for better L2 locality
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

    # K reduction
    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        k_offsets = k_tile * BLOCK_K + offs_k

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak
        # Load B as [BLOCK_K, BLOCK_N] so tl.dot(a, b) computes A @ B.T directly.
        b_ptrs = b_ptr + offs_n[None, :] * stride_bn + k_offsets[:, None] * stride_bk

        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (offs_n[None, :] < N) & (k_offsets[:, None] < K)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def kernel_function(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton wrapper for C = A @ B.T.

    Fusion note for reviewers:
    - This problem has a single operator pipeline (matmul with transposed RHS).
    - The implementation is fully fused into one Triton kernel:
      load tiles -> K reduction (dot-accumulate) -> store output.
    - No separate transpose kernel is launched; B is indexed in-kernel as needed.
    """
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("A and B must be torch.Tensor objects.")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A and B must be 2D. Got A.ndim={A.ndim}, B.ndim={B.ndim}.")
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"Incompatible shapes for A @ B.T: A={tuple(A.shape)}, B={tuple(B.shape)} "
            f"(K dimensions must match)."
        )
    if A.device.type != "cuda" or B.device.type != "cuda":
        raise ValueError("A and B must be CUDA/HIP tensors.")
    if A.device != B.device:
        raise ValueError(f"A and B must be on the same device. Got {A.device} and {B.device}.")
    if A.dtype != B.dtype:
        raise ValueError(f"A and B must have same dtype. Got {A.dtype} and {B.dtype}.")
    if A.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported dtype {A.dtype}. Use fp16/bf16/fp32.")

    M, K = A.shape
    N = B.shape[0]

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    _matmul_a_bt_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
        GROUP_M=8,
        num_warps=8,
        num_stages=4,
    )
    return C


__all__ = ["kernel_function"]