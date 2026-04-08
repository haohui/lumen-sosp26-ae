import torch
import torch.nn as nn
import substrate
import substrate.language as S

# Problem sizes
BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192

# GEMM tile configuration
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 32
THREADS = BLOCK_M * BLOCK_N
K_TILES = IN_FEATURES // BLOCK_K

# GELU constants
SQRT_2 = 1.4142135623730951
INV_SQRT_2 = 0.7071067811865476

# Block size for other kernels
BLOCK_SIZE = 256


# ============================================================================
# GEMM Kernel (with bias addition)
# Computes: C = A @ W^T + bias
# A: (M, K), W: (N, K), bias: (N,), C: (M, N)
# ============================================================================

@substrate.jit
def gemm_bias_bf16_kernel(
    A: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((OUT_FEATURES, IN_FEATURES), S.bf16),
    bias: S.Tensor((OUT_FEATURES,), S.bf16),
    C: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    tid = S.thread_id(0)
    bx = S.block_id(0)  # N tile
    by = S.block_id(1)  # M tile

    local_m = tid // BLOCK_N
    local_n = tid % BLOCK_N

    row = by * BLOCK_M + local_m
    col = bx * BLOCK_N + local_n

    sA = S.make_shared((BLOCK_M, BLOCK_K), S.bf16)
    sW = S.make_shared((BLOCK_N, BLOCK_K), S.bf16)

    # FP32 accumulation
    acc = S.convert(0.0, S.f32)

    for kt in S.range(K_TILES):
        k_base = kt * BLOCK_K

        # Cooperative load A tile: [BLOCK_M, BLOCK_K]
        for i in S.range((BLOCK_M * BLOCK_K) // THREADS):
            idx = tid + i * THREADS
            a_r = idx // BLOCK_K
            a_c = idx % BLOCK_K
            sA[a_r, a_c] = A[by * BLOCK_M + a_r, k_base + a_c]

        # Cooperative load W tile: [BLOCK_N, BLOCK_K] (W is stored row-major)
        # We need W^T, so we load W columns as W rows
        for i in S.range((BLOCK_N * BLOCK_K) // THREADS):
            idx = tid + i * THREADS
            w_r = idx // BLOCK_K
            w_c = idx % BLOCK_K
            sW[w_r, w_c] = W[bx * BLOCK_N + w_r, k_base + w_c]

        S.syncthreads()

        for kk in S.range(BLOCK_K):
            av = S.convert(sA[local_m, kk], S.f32)
            wv = S.convert(sW[local_n, kk], S.f32)
            acc = acc + av * wv

        S.syncthreads()

    # Add bias
    bias_val = S.convert(bias[col], S.f32)
    acc = acc + bias_val

    C[row, col] = S.convert(acc, S.bf16)


# ============================================================================
# Max Reduction Kernel
# Computes max over dim 1 (columns) with keepdim=True
# Input: (BATCH_SIZE, OUT_FEATURES), Output: (BATCH_SIZE, 1)
# ============================================================================

@substrate.jit
def max_dim1_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
    out: S.Tensor((BATCH_SIZE, 1), S.bf16),
):
    row = S.block_id(0)
    tid = S.thread_id(0)

    # Initialize with first element (or very negative value)
    # Use FP32 for accumulation
    neg_inf = S.convert(-3.4e38, S.f32)
    max_val = neg_inf

    # Each thread processes multiple elements
    for col_base in S.range((OUT_FEATURES + BLOCK_SIZE - 1) // BLOCK_SIZE):
        col = col_base * BLOCK_SIZE + tid
        if col < OUT_FEATURES:
            val = S.convert(x[row, col], S.f32)
            if val > max_val:
                max_val = val

    # Parallel reduction within block
    for stride in S.range(128):
        other = S.shuffle_down(max_val, 128 - stride, BLOCK_SIZE)
        if tid < BLOCK_SIZE // 2:
            if other > max_val:
                max_val = other

    for stride in S.range(64):
        other = S.shuffle_down(max_val, 64 - stride, BLOCK_SIZE)
        if tid < BLOCK_SIZE // 4:
            if other > max_val:
                max_val = other

    for stride in S.range(32):
        other = S.shuffle_down(max_val, 32 - stride, BLOCK_SIZE)
        if tid < BLOCK_SIZE // 8:
            if other > max_val:
                max_val = other

    for stride in S.range(16):
        other = S.shuffle_down(max_val, 16 - stride, BLOCK_SIZE)
        if tid < BLOCK_SIZE // 16:
            if other > max_val:
                max_val = other

    for stride in S.range(8):
        other = S.shuffle_down(max_val, 8 - stride, BLOCK_SIZE)
        if tid < BLOCK_SIZE // 32:
            if other > max_val:
                max_val = other

    for stride in S.range(4):
        other = S.shuffle_down(max_val, 4 - stride, BLOCK_SIZE)
        if tid < BLOCK_SIZE // 64:
            if other > max_val:
                max_val = other

    for stride in S.range(2):
        other = S.shuffle_down(max_val, 2 - stride, BLOCK_SIZE)
        if tid < BLOCK_SIZE // 128:
            if other > max_val:
                max_val = other

    # Thread 0 writes the result
    if tid == 0:
        out[row, 0] = S.convert(max_val, S.bf16)


# ============================================================================
# Mean Subtraction Kernel
# For a (BATCH_SIZE, 1) tensor, mean over dim 1 is just the value itself
# So x - mean = 0
# ============================================================================

@substrate.jit
def mean_subtract_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, 1), S.bf16),
    out: S.Tensor((BATCH_SIZE, 1), S.bf16),
):
    row = S.block_id(0)
    # For a single-element row, mean is the element itself
    # So x - x.mean() = 0
    out[row, 0] = S.convert(0.0, S.bf16)


# ============================================================================
# GELU Kernel
# GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
# ============================================================================

@substrate.jit
def gelu_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, 1), S.bf16),
    out: S.Tensor((BATCH_SIZE, 1), S.bf16),
):
    row = S.block_id(0)

    xv = S.convert(x[row, 0], S.f32)
    inv_sqrt2 = S.convert(INV_SQRT_2, S.f32)
    half = S.convert(0.5, S.f32)
    one = S.convert(1.0, S.f32)

    # GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    erf_arg = xv * inv_sqrt2
    erf_val = S.erf(erf_arg)
    gelu_val = xv * half * (one + erf_val)

    out[row, 0] = S.convert(gelu_val, S.bf16)


# ============================================================================
# Host wrapper functions
# ============================================================================

def substrate_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Compute x @ weight.T + bias using Substrate kernel."""
    if x.device.type != "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device required for Substrate kernels.")
        x = x.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch_size, in_features = x.shape
    out_features = weight.shape[0]

    assert batch_size == BATCH_SIZE
    assert in_features == IN_FEATURES
    assert out_features == OUT_FEATURES

    out = torch.empty((batch_size, out_features), dtype=torch.bfloat16, device=x.device)

    grid = (OUT_FEATURES // BLOCK_N, BATCH_SIZE // BLOCK_M, 1)
    block = (THREADS, 1, 1)

    gemm_bias_bf16_kernel[lambda: (grid, block)](x, weight, bias, out)

    return out


def substrate_max_dim1(x: torch.Tensor) -> torch.Tensor:
    """Compute max over dim 1 with keepdim=True."""
    if x.device.type != "cuda":
        x = x.cuda()

    x = x.contiguous()
    batch_size, features = x.shape

    assert batch_size == BATCH_SIZE
    assert features == OUT_FEATURES

    out = torch.empty((batch_size, 1), dtype=torch.bfloat16, device=x.device)

    max_dim1_bf16_kernel[lambda: ((BATCH_SIZE, 1, 1), (BLOCK_SIZE, 1, 1))](x, out)

    return out


def substrate_mean_subtract(x: torch.Tensor) -> torch.Tensor:
    """Compute x - x.mean(dim=1, keepdim=True)."""
    if x.device.type != "cuda":
        x = x.cuda()

    x = x.contiguous()
    batch_size, cols = x.shape

    assert batch_size == BATCH_SIZE

    out = torch.empty((batch_size, 1), dtype=torch.bfloat16, device=x.device)

    mean_subtract_bf16_kernel[lambda: ((BATCH_SIZE, 1, 1), (1, 1, 1))](x, out)

    return out


def substrate_gelu(x: torch.Tensor) -> torch.Tensor:
    """Compute GELU activation."""
    if x.device.type != "cuda":
        x = x.cuda()

    x = x.contiguous()
    batch_size, cols = x.shape

    assert batch_size == BATCH_SIZE

    out = torch.empty((batch_size, 1), dtype=torch.bfloat16, device=x.device)

    gelu_bf16_kernel[lambda: ((BATCH_SIZE, 1, 1), (1, 1, 1))](x, out)

    return out


# ============================================================================
# Model
# ============================================================================

class ModelNew(nn.Module):
    """
    Optimized Substrate implementation of:
    GEMM -> max(dim=1) -> subtract mean -> GELU
    """
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_dim = max_dim

        # Store weights as nn.Parameter for compatibility
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.bfloat16))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16))

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to BF16 if needed
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        # 1. GEMM (Linear): x = x @ W^T + bias
        x = substrate_linear(x, self.weight, self.bias)

        # 2. Max over dim 1 (columns) with keepdim=True
        x = substrate_max_dim1(x)

        # 3. Subtract mean over dim 1
        x = substrate_mean_subtract(x)

        # 4. GELU activation
        x = substrate_gelu(x)

        return x


batch_size = BATCH_SIZE
in_features = IN_FEATURES
out_features = OUT_FEATURES
max_dim = 1


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, max_dim]
