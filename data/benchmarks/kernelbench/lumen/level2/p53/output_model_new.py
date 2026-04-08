"""
Fused GEMM + epilogue kernel for:
  y = GELU(clamp(x @ W.T * scale + bias, min, max))

Simple one-element-per-thread GEMM with element-wise epilogue.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import substrate
import substrate.language as S
import math
import struct

# Tiling configuration - each thread computes one output element
BLOCK_M = 16
BLOCK_N = 16
THREADS = BLOCK_M * BLOCK_N  # 256 threads per block


def float_to_u32_bits(f: float) -> int:
    """Convert a Python float to its IEEE 754 binary representation as u32."""
    return struct.unpack('<I', struct.pack('<f', f))[0]


@substrate.jit
def _gemm_epilogue_kernel(
    A_ptr: S.Pointer(S.bf16),
    B_ptr: S.Pointer(S.bf16),
    C_ptr: S.Pointer(S.bf16),
    bias_ptr: S.Pointer(S.bf16),
    M: S.i32,
    N: S.i32,
    K: S.i32,
    scale_bits: S.u32,
    clamp_min_bits: S.u32,
    clamp_max_bits: S.u32,
):
    # Convert bit patterns to f32
    scale = S.bitcast(scale_bits, S.f32)
    clamp_min = S.bitcast(clamp_min_bits, S.f32)
    clamp_max = S.bitcast(clamp_max_bits, S.f32)

    # Thread and block indices
    tid = S.thread_id(0)
    bid = S.block_id(0)

    # Compute global row and column
    blocks_n = (N + BLOCK_N - 1) // BLOCK_N
    block_row = bid // blocks_n
    block_col = bid % blocks_n

    row = block_row * BLOCK_M + tid // BLOCK_N
    col = block_col * BLOCK_N + tid % BLOCK_N

    # Create tensor views
    a_layout = S.make_layout((M, K), (K, 1))
    b_layout = S.make_layout((N, K), (K, 1))
    c_layout = S.make_layout((M, N), (N, 1))
    bias_layout = S.make_layout((N,), (1,))

    A = S.make_tensor(A_ptr, S.bf16, a_layout)
    B = S.make_tensor(B_ptr, S.bf16, b_layout)
    C = S.make_tensor(C_ptr, S.bf16, c_layout)
    bias_tensor = S.make_tensor(bias_ptr, S.bf16, bias_layout)

    # Only compute if within bounds
    if row < M and col < N:
        # Simple scalar accumulation
        zero_f32 = S.bitcast(0x00000000, S.f32)
        acc = zero_f32

        # Compute dot product: sum over K
        for k in S.range(K):
            a_val = S.convert(A[row, k], S.f32)
            b_val = S.convert(B[col, k], S.f32)
            acc = acc + a_val * b_val

        # Add bias
        bias_val = S.convert(bias_tensor[col], S.f32)
        acc = acc + bias_val

        # Scale
        acc = acc * scale

        # Clamp (hardtanh)
        if acc < clamp_min:
            acc = clamp_min
        if acc > clamp_max:
            acc = clamp_max

        # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        half = S.bitcast(0x3F000000, S.f32)
        one = S.bitcast(0x3F800000, S.f32)
        sqrt_2 = S.bitcast(0x3FB8AA3B, S.f32)  # sqrt(2) ≈ 1.4142135
        x_over_sqrt2 = acc / sqrt_2
        erf_val = S.erf(x_over_sqrt2)
        acc = acc * half * (one + erf_val)

        # Store result
        C[row, col] = S.convert(acc, S.bf16)


def fused_gemm_epilogue(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
    clamp_min: float,
    clamp_max: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute: out = GELU(clamp(x @ weight_t * scale + bias, clamp_min, clamp_max))"""
    if x.device.type != "cuda" or weight_t.device.type != "cuda":
        raise ValueError("Tensors must be on CUDA device")

    m, k = x.shape
    n, k2 = weight_t.shape
    if k != k2:
        raise ValueError(f"K dimension mismatch: x has K={k}, weight_t has K={k2}")

    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)

    grid_m = (m + BLOCK_M - 1) // BLOCK_M
    grid_n = (n + BLOCK_N - 1) // BLOCK_N
    grid = (grid_m * grid_n, 1, 1)
    block = (THREADS, 1, 1)

    scale_bits = float_to_u32_bits(scale)
    clamp_min_bits = float_to_u32_bits(clamp_min)
    clamp_max_bits = float_to_u32_bits(clamp_max)

    _gemm_epilogue_kernel[lambda: (grid, block)](
        x, weight_t, out, bias, m, n, k, scale_bits, clamp_min_bits, clamp_max_bits
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, scaling, hardtanh, and GELU activation.
    Uses fused Substrate kernel for computation.

    Matches the structure of the reference Model for proper weight loading.
    """
    def __init__(self, in_features: int, out_features: int, scaling_factor: float,
                 hardtanh_min: float, hardtanh_max: float):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

        # Create gemm submodule to match reference model structure
        self.gemm = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous and BF16
        x = x.contiguous()
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        # Convert gemm weights to BF16
        weight_bf16 = self.gemm.weight.to(torch.bfloat16)
        bias_bf16 = self.gemm.bias.to(torch.bfloat16)

        return fused_gemm_epilogue(
            x, weight_bf16, bias_bf16,
            self.scaling_factor, self.hardtanh_min, self.hardtanh_max
        )
