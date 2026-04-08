import torch
import torch.nn as nn
import substrate
import substrate.language as S
import math


# Problem sizes
BATCH_SIZE = 1024
INPUT_SIZE = 8192
OUTPUT_SIZE = 8192
DIVISOR = 10.0

# GELU constants
SQRT_2 = 1.4142135623730951

# Tiling parameters for GEMM
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 16
THREADS = 256

LOADS_A = (BLOCK_M * BLOCK_K) // THREADS
LOADS_B = (BLOCK_K * BLOCK_N) // THREADS


@substrate.jit
def linear_div_gelu_bf16_kernel(
    X: S.Pointer(S.bf16),
    W: S.Pointer(S.bf16),
    bias: S.Pointer(S.bf16),
    Y: S.Pointer(S.bf16),
    M: S.u32,
    N: S.u32,
    K: S.u32,
):
    """Fused kernel: Y = GELU((X @ W.T + bias) / divisor)"""
    tid = S.thread_id(0)
    bid_n = S.block_id(0)
    bid_m = S.block_id(1)

    layout_x = S.make_layout((M, K), (K, 1))
    layout_w = S.make_layout((N, K), (K, 1))
    layout_y = S.make_layout((M, N), (N, 1))

    gX = S.make_tensor(X, S.bf16, layout_x)
    gW = S.make_tensor(W, S.bf16, layout_w)
    gY = S.make_tensor(Y, S.bf16, layout_y)

    sA = S.make_shared((BLOCK_M, BLOCK_K), S.bf16)
    sB = S.make_shared((BLOCK_N, BLOCK_K), S.bf16)

    # Each thread handles multiple output elements for better register usage
    local_row = tid // 16
    local_col = tid % 16

    # Compute output rows and columns
    row0 = bid_m * BLOCK_M + local_row * 2
    row1 = row0 + 1
    col0 = bid_n * BLOCK_N + local_col * 2
    col1 = col0 + 1

    # Accumulators for each output element
    acc00 = S.convert(0.0, S.f32)
    acc01 = S.convert(0.0, S.f32)
    acc10 = S.convert(0.0, S.f32)
    acc11 = S.convert(0.0, S.f32)

    k_tiles = (K + BLOCK_K - 1) // BLOCK_K

    for kt in S.range(k_tiles):
        k_base = kt * BLOCK_K

        # Load A tile (X): [BLOCK_M, BLOCK_K]
        for i in S.range(LOADS_A):
            idx = tid + i * THREADS
            a_r = idx // BLOCK_K
            a_c = idx % BLOCK_K
            g_r = bid_m * BLOCK_M + a_r
            g_c = k_base + a_c
            if g_r < M and g_c < K:
                sA[a_r, a_c] = gX[g_r, g_c]
            else:
                sA[a_r, a_c] = S.convert(0.0, S.bf16)

        # Load B tile (W transposed, so we load W rows): [BLOCK_N, BLOCK_K]
        for i in S.range(LOADS_B):
            idx = tid + i * THREADS
            b_r = idx // BLOCK_K
            b_c = idx % BLOCK_K
            g_r = bid_n * BLOCK_N + b_r
            g_c = k_base + b_c
            if g_r < N and g_c < K:
                sB[b_r, b_c] = gW[g_r, g_c]
            else:
                sB[b_r, b_c] = S.convert(0.0, S.bf16)

        S.syncthreads()

        # Compute partial matmul
        for kk in S.range(BLOCK_K):
            a0 = S.convert(sA[local_row * 2, kk], S.f32)
            a1 = S.convert(sA[local_row * 2 + 1, kk], S.f32)
            b0 = S.convert(sB[local_col * 2, kk], S.f32)
            b1 = S.convert(sB[local_col * 2 + 1, kk], S.f32)
            acc00 = acc00 + a0 * b0
            acc01 = acc01 + a0 * b1
            acc10 = acc10 + a1 * b0
            acc11 = acc11 + a1 * b1

        S.syncthreads()

    # Build layout for bias access
    layout_bias = S.make_layout((N,), (1,))
    gBias = S.make_tensor(bias, S.bf16, layout_bias)

    # Constants for division and GELU
    divisor = S.convert(DIVISOR, S.f32)
    half = S.convert(0.5, S.f32)
    one = S.convert(1.0, S.f32)
    sqrt2 = S.convert(SQRT_2, S.f32)

    # Write results with bias, division, and GELU
    if row0 < M and col0 < N:
        bias_val = S.convert(gBias[col0], S.f32)
        val = acc00 + bias_val
        val = val / divisor
        # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        x_scaled = val / sqrt2
        erf_x = S.erf(x_scaled)
        cdf = half * (one + erf_x)
        result = val * cdf
        gY[row0, col0] = S.convert(result, S.bf16)
    if row0 < M and col1 < N:
        bias_val = S.convert(gBias[col1], S.f32)
        val = acc01 + bias_val
        val = val / divisor
        x_scaled = val / sqrt2
        erf_x = S.erf(x_scaled)
        cdf = half * (one + erf_x)
        result = val * cdf
        gY[row0, col1] = S.convert(result, S.bf16)
    if row1 < M and col0 < N:
        bias_val = S.convert(gBias[col0], S.f32)
        val = acc10 + bias_val
        val = val / divisor
        x_scaled = val / sqrt2
        erf_x = S.erf(x_scaled)
        cdf = half * (one + erf_x)
        result = val * cdf
        gY[row1, col0] = S.convert(result, S.bf16)
    if row1 < M and col1 < N:
        bias_val = S.convert(gBias[col1], S.f32)
        val = acc11 + bias_val
        val = val / divisor
        x_scaled = val / sqrt2
        erf_x = S.erf(x_scaled)
        cdf = half * (one + erf_x)
        result = val * cdf
        gY[row1, col1] = S.convert(result, S.bf16)


class ModelNew(nn.Module):
    """
    Optimized model that performs Linear, division by scalar, and GELU activation
    using fused Substrate GPU kernels.
    """
    def __init__(self, input_size: int, output_size: int, divisor: float):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.divisor = divisor

        # Create weight and bias as parameters (same as nn.Linear)
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

        # Ensure inputs are contiguous and on GPU
        x_dev = x.contiguous()
        if not x_dev.is_cuda:
            x_dev = x_dev.cuda()

        # Convert to BF16 for computation
        x_bf16 = x_dev.to(torch.bfloat16)
        w_bf16 = self.weight.to(torch.bfloat16).contiguous()
        b_bf16 = self.bias.to(torch.bfloat16).contiguous()

        M = x_bf16.shape[0]
        N = self.output_size
        K = self.input_size

        # Allocate output
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x_bf16.device)

        # Compute grid dimensions
        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        grid_n = (N + BLOCK_N - 1) // BLOCK_N

        linear_div_gelu_bf16_kernel[lambda: ((grid_n, grid_m, 1), (THREADS, 1, 1))](
            x_bf16, w_bf16, b_bf16, out, M, N, K
        )

        return out


batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, output_size, divisor]
