import torch
import torch.nn as nn
import substrate
import substrate.language as S

# Tile configuration
# Each block computes a TILE_M x TILE_N output tile
TILE_M = 32
TILE_N = 64
TILE_K = 32
BLOCK_THREADS = 256


@substrate.jit
def gemm_scale_bias_kernel(
    A: S.Pointer(S.bf16),
    B: S.Pointer(S.bf16),
    bias: S.Pointer(S.bf16),
    C: S.Pointer(S.bf16),
    M: S.i32,
    N: S.i32,
    K: S.i32,
    scale_num: S.i32,
    scale_denom: S.i32,
):
    """
    GEMM with fused bias and scaling: C = (A @ B + bias) * (scale_num / scale_denom)

    Each block computes a TILE_M x TILE_N output tile.
    Each thread computes multiple output elements.
    """
    tid = S.thread_id(0)

    # Block coordinates
    block_row = S.block_id(0) * TILE_M
    block_col = S.block_id(1) * TILE_N

    # Create tensor views
    layout_a = S.make_layout((M, K), (K, 1))
    layout_b = S.make_layout((K, N), (N, 1))
    layout_c = S.make_layout((M, N), (N, 1))
    layout_bias = S.make_layout((N,), (1,))

    g_a = S.make_tensor(A, S.bf16, layout_a)
    g_b = S.make_tensor(B, S.bf16, layout_b)
    g_c = S.make_tensor(C, S.bf16, layout_c)
    g_bias = S.make_tensor(bias, S.bf16, layout_bias)

    # Convert scale to f32
    scale = S.convert(scale_num, S.f32) / S.convert(scale_denom, S.f32)

    # Each thread computes TILE_M * TILE_N / BLOCK_THREADS output elements
    # For TILE_M=32, TILE_N=64, BLOCK_THREADS=256: each thread handles 8 elements
    elements_per_thread = TILE_M * TILE_N // BLOCK_THREADS

    # Local accumulator
    acc = S.make_local((elements_per_thread,), S.f32)
    for i in S.range(elements_per_thread):
        acc[i] = S.convert(0.0, S.f32)

    # Loop over K tiles
    k_tiles = K // TILE_K
    for k_tile in S.range(k_tiles):
        k_base = k_tile * TILE_K

        # Each thread loads and computes partial results
        for elem in S.range(elements_per_thread):
            elem_idx = tid * elements_per_thread + elem
            local_row = elem_idx // TILE_N
            local_col = elem_idx % TILE_N

            global_row = block_row + local_row
            global_col = block_col + local_col

            # Compute partial dot product for this K tile
            for k in S.range(TILE_K):
                a_val = S.convert(g_a[global_row, k_base + k], S.f32)
                b_val = S.convert(g_b[k_base + k, global_col], S.f32)
                acc[elem] = acc[elem] + a_val * b_val

    # Add bias, scale, and store results
    for elem in S.range(elements_per_thread):
        elem_idx = tid * elements_per_thread + elem
        local_row = elem_idx // TILE_N
        local_col = elem_idx % TILE_N

        global_row = block_row + local_row
        global_col = block_col + local_col

        if global_row < M:
            if global_col < N:
                bias_val = S.convert(g_bias[global_col], S.f32)
                acc[elem] = acc[elem] + bias_val
                acc[elem] = acc[elem] * scale
                g_c[global_row, global_col] = S.convert(acc[elem], S.bf16)


def gemm_scale_bias(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Launch GEMM kernel with bias and scaling."""
    assert A.is_cuda and B.is_cuda and bias.is_cuda
    assert A.dtype == torch.bfloat16
    assert B.dtype == torch.bfloat16
    assert bias.dtype == torch.bfloat16

    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)

    # Grid dimensions
    grid_m = (M + TILE_M - 1) // TILE_M
    grid_n = (N + TILE_N - 1) // TILE_N

    # Convert scale to numerator/denominator
    if abs(scale - 1.5) < 0.001:
        scale_num, scale_denom = 3, 2
    elif abs(scale - 1.0) < 0.001:
        scale_num, scale_denom = 1, 1
    elif abs(scale - 2.0) < 0.001:
        scale_num, scale_denom = 2, 1
    else:
        scale_num = int(scale * 1000)
        scale_denom = 1000

    gemm_scale_bias_kernel[lambda: ((grid_m, grid_n, 1), (BLOCK_THREADS, 1, 1))](
        A, B, bias, C, M, N, K, scale_num, scale_denom
    )

    return C


class ModelNew(nn.Module):
    """Optimized model using Substrate DSL kernels."""

    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor

        # Create weight and bias tensors (matching nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # Pre-transpose weight for efficient GEMM
        self.weight_t = None

    def forward(self, x):
        # Ensure input is bf16
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        # Transpose weight for GEMM: we need (K, N) layout
        if self.weight_t is None or self.weight_t.shape != (self.in_features, self.out_features):
            self.weight_t = self.weight.t().contiguous().to(torch.bfloat16)

        # Ensure weight_t is on the same device
        if self.weight_t.device != x.device:
            self.weight_t = self.weight_t.to(x.device)

        bias_bf16 = self.bias.to(torch.bfloat16)
        if bias_bf16.device != x.device:
            bias_bf16 = bias_bf16.to(x.device)

        # Compute scale factor: output = (matmul + bias) * (1 + scaling_factor)
        scale = 1.0 + self.scaling_factor

        # Launch optimized kernel
        return gemm_scale_bias(x, self.weight_t, bias_bf16, scale)


# Required functions for evaluation
batch_size = 16384
in_features = 4096
out_features = 4096
scaling_factor = 0.5


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, scaling_factor]
