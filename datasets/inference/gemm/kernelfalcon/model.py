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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute C = A @ B^T
      A: [M, K]
      B: [N, K]
      C: [M, N]
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)

        # A tile: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B^T tile via B access pattern:
        # B is [N, K], we need [K, N] for dot -> shape [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc)

    c = acc.to(c_ptr.dtype.element_ty)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _launch_matmul_a_bt(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """
    Launch C = A @ B^T into a caller-provided output tensor.
    """
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("kernel_function expects two torch.Tensor inputs")

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got A.ndim={A.ndim}, B.ndim={B.ndim}")

    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"Incompatible shapes for A @ B^T: A.shape={tuple(A.shape)}, B.shape={tuple(B.shape)}"
        )

    if A.device != B.device:
        raise ValueError(f"A and B must be on same device, got {A.device} and {B.device}")

    if A.dtype != B.dtype:
        raise ValueError(f"A and B must have same dtype, got {A.dtype} and {B.dtype}")

    if A.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported dtype for this kernel: {A.dtype}")

    M, K = A.shape
    N = B.shape[0]

    if C.shape != (M, N):
        raise ValueError(f"Expected C shape {(M, N)}, got {tuple(C.shape)}")
    if C.device != A.device or C.dtype != A.dtype:
        raise ValueError("C must share device and dtype with A")

    # Nothing to launch for empty outputs.
    if M == 0 or N == 0:
        return C

    # Fixed tile sizes chosen for BF16/FP16/FP32 GEMM-style workloads.
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )

    _matmul_a_bt_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=8,
        num_stages=3,
    )

    return C


def kernel_function(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton wrapper for C = A @ B^T.

    Fusion note:
    - The model pipeline is a single matmul only.
    - We "fuse" transpose handling into the kernel memory access pattern:
      B is never explicitly transposed; kernel loads B with [k, n] indexing.
    - No extra PyTorch compute ops are used in the wrapper.
    """
    C = torch.empty((A.shape[0], B.shape[0]), device=A.device, dtype=A.dtype)
    return _launch_matmul_a_bt(A, B, C)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return kernel_function(A, B)

    def build_call(self, *, a_mk: torch.Tensor, b_nk: torch.Tensor):
        a = a_mk.contiguous()
        b = b_nk.contiguous()
        out = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=a.dtype)

        def call():
            return _launch_matmul_a_bt(a, b, out)

        return call
