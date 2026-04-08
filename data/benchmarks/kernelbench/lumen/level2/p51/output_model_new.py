import torch
import torch.nn as nn
import substrate
import substrate.language as S

# Problem sizes (compile-time constants)
BATCH_SIZE = 2048
IN_FEATURES = 8192
OUT_FEATURES = 8192

# GEMM tiling parameters
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 16
THREADS = BLOCK_M * BLOCK_N
K_TILES = IN_FEATURES // BLOCK_K

# Number of loads per thread
A_LOADS = (BLOCK_M * BLOCK_K) // THREADS
B_LOADS = (BLOCK_N * BLOCK_K) // THREADS

# Block size for elementwise kernels
BLOCK_SIZE = 256


@substrate.jit
def linear_f32_kernel(
    x: S.Tensor((BATCH_SIZE, IN_FEATURES), S.f32),
    weight: S.Tensor((OUT_FEATURES, IN_FEATURES), S.f32),
    bias: S.Tensor((OUT_FEATURES,), S.f32),
    out: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.f32),
):
    """GEMM kernel for Linear layer: out = x @ weight.T + bias"""
    tid = S.thread_id(0)
    bx = S.block_id(0)  # N tile (out_features)
    by = S.block_id(1)  # M tile (batch)

    local_m = tid // BLOCK_N
    local_n = tid % BLOCK_N

    row = by * BLOCK_M + local_m
    col = bx * BLOCK_N + local_n

    sX = S.make_shared((BLOCK_M, BLOCK_K), S.f32)
    sW = S.make_shared((BLOCK_N, BLOCK_K), S.f32)

    acc = S.convert(0.0, S.f32)

    for kt in S.range(K_TILES):
        k_base = kt * BLOCK_K

        # Load X tile: [BLOCK_M, BLOCK_K]
        for i in S.range(A_LOADS):
            idx = tid + i * THREADS
            x_r = idx // BLOCK_K
            x_c = idx % BLOCK_K
            sX[x_r, x_c] = x[by * BLOCK_M + x_r, k_base + x_c]

        # Load W tile: weight is (OUT, IN), we want [BLOCK_N, BLOCK_K]
        for i in S.range(B_LOADS):
            idx = tid + i * THREADS
            w_r = idx // BLOCK_K
            w_c = idx % BLOCK_K
            sW[w_r, w_c] = weight[bx * BLOCK_N + w_r, k_base + w_c]

        S.syncthreads()

        for kk in S.range(BLOCK_K):
            acc = acc + sX[local_m, kk] * sW[local_n, kk]

        S.syncthreads()

    # Add bias
    out[row, col] = acc + bias[col]


@substrate.jit
def linear_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    weight: S.Tensor((OUT_FEATURES, IN_FEATURES), S.bf16),
    bias: S.Tensor((OUT_FEATURES,), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    """GEMM kernel for Linear layer in BF16 with FP32 accumulation"""
    tid = S.thread_id(0)
    bx = S.block_id(0)  # N tile (out_features)
    by = S.block_id(1)  # M tile (batch)

    local_m = tid // BLOCK_N
    local_n = tid % BLOCK_N

    row = by * BLOCK_M + local_m
    col = bx * BLOCK_N + local_n

    sX = S.make_shared((BLOCK_M, BLOCK_K), S.bf16)
    sW = S.make_shared((BLOCK_N, BLOCK_K), S.bf16)

    acc = S.convert(0.0, S.f32)

    for kt in S.range(K_TILES):
        k_base = kt * BLOCK_K

        # Load X tile: [BLOCK_M, BLOCK_K]
        for i in S.range(A_LOADS):
            idx = tid + i * THREADS
            x_r = idx // BLOCK_K
            x_c = idx % BLOCK_K
            sX[x_r, x_c] = x[by * BLOCK_M + x_r, k_base + x_c]

        # Load W tile: weight is (OUT, IN), we want [BLOCK_N, BLOCK_K]
        for i in S.range(B_LOADS):
            idx = tid + i * THREADS
            w_r = idx // BLOCK_K
            w_c = idx % BLOCK_K
            sW[w_r, w_c] = weight[bx * BLOCK_N + w_r, k_base + w_c]

        S.syncthreads()

        for kk in S.range(BLOCK_K):
            xv = S.convert(sX[local_m, kk], S.f32)
            wv = S.convert(sW[local_n, kk], S.f32)
            acc = acc + xv * wv

        S.syncthreads()

    # Add bias
    bias_val = S.convert(bias[col], S.f32)
    out[row, col] = S.convert(acc + bias_val, S.bf16)


@substrate.jit
def subtract_f32_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.f32),
    sub: S.Tensor((OUT_FEATURES,), S.f32),
    out: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.f32),
):
    """Elementwise subtract with row-wise broadcast: out = x - sub"""
    row = S.block_id(1)
    col = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if row < BATCH_SIZE and col < OUT_FEATURES:
        out[row, col] = x[row, col] - sub[col]


@substrate.jit
def subtract_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
    sub: S.Tensor((OUT_FEATURES,), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    """Elementwise subtract with row-wise broadcast: out = x - sub"""
    row = S.block_id(1)
    col = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if row < BATCH_SIZE and col < OUT_FEATURES:
        xv = S.convert(x[row, col], S.f32)
        sv = S.convert(sub[col], S.f32)
        out[row, col] = S.convert(xv - sv, S.bf16)


@substrate.jit
def mean_dim1_f32_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.f32),
    out: S.Tensor((BATCH_SIZE, 1), S.f32),
):
    """Mean reduction over dim=1: out[b, 0] = mean(x[b, :])"""
    b = S.block_id(0)

    acc = S.convert(0.0, S.f32)
    inv = S.convert(1.0 / OUT_FEATURES, S.f32)

    for c in S.range(OUT_FEATURES):
        acc = acc + x[b, c]

    out[b, 0] = acc * inv


@substrate.jit
def mean_dim1_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
    out: S.Tensor((BATCH_SIZE, 1), S.bf16),
):
    """Mean reduction over dim=1: out[b, 0] = mean(x[b, :])"""
    b = S.block_id(0)

    acc = S.convert(0.0, S.f32)
    inv = S.convert(1.0 / OUT_FEATURES, S.f32)

    for c in S.range(OUT_FEATURES):
        acc = acc + S.convert(x[b, c], S.f32)

    out[b, 0] = S.convert(acc * inv, S.bf16)


@substrate.jit
def logsumexp_dim1_f32_kernel(
    x: S.Tensor((BATCH_SIZE, 1), S.f32),
    out: S.Tensor((BATCH_SIZE, 1), S.f32),
):
    """LogSumExp over dim=1. Since dim=1 has size 1, logsumexp(x) = x."""
    b = S.block_id(0)
    out[b, 0] = x[b, 0]


@substrate.jit
def logsumexp_dim1_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, 1), S.bf16),
    out: S.Tensor((BATCH_SIZE, 1), S.bf16),
):
    """LogSumExp over dim=1. Since dim=1 has size 1, logsumexp(x) = x."""
    b = S.block_id(0)
    out[b, 0] = x[b, 0]


@substrate.jit
def gelu_f32_kernel(
    x: S.Pointer(S.f32),
    y: S.Pointer(S.f32),
    n: S.u32,
):
    """GELU activation: y = x * 0.5 * (1 + erf(x / sqrt(2)))"""
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx < n:
        layout = S.make_layout((n,), (1,))
        gx = S.make_tensor(x, S.f32, layout)
        gy = S.make_tensor(y, S.f32, layout)

        xv = gx[idx]
        half = S.convert(0.5, S.f32)
        one = S.convert(1.0, S.f32)
        sqrt2 = S.convert(1.41421356237, S.f32)

        # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        arg = xv / sqrt2
        erf_val = S.erf(arg)
        gelu_out = xv * half * (one + erf_val)

        gy[idx] = gelu_out


@substrate.jit
def gelu_bf16_kernel(
    x: S.Pointer(S.bf16),
    y: S.Pointer(S.bf16),
    n: S.u32,
):
    """GELU activation: y = x * 0.5 * (1 + erf(x / sqrt(2)))"""
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx < n:
        layout = S.make_layout((n,), (1,))
        gx = S.make_tensor(x, S.bf16, layout)
        gy = S.make_tensor(y, S.bf16, layout)

        xv = S.convert(gx[idx], S.f32)
        half = S.convert(0.5, S.f32)
        one = S.convert(1.0, S.f32)
        sqrt2 = S.convert(1.41421356237, S.f32)

        # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        arg = xv / sqrt2
        erf_val = S.erf(arg)
        gelu_out = xv * half * (one + erf_val)

        gy[idx] = S.convert(gelu_out, S.bf16)


@substrate.jit
def broadcast_add_f32_kernel(
    x: S.Tensor((BATCH_SIZE, 1), S.f32),
    original: S.Tensor((BATCH_SIZE, IN_FEATURES), S.f32),
    out: S.Tensor((BATCH_SIZE, IN_FEATURES), S.f32),
):
    """Broadcast add: out = x + original where x is (B, 1) and original is (B, IN)"""
    row = S.block_id(1)
    col = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if row < BATCH_SIZE and col < IN_FEATURES:
        out[row, col] = x[row, 0] + original[row, col]


@substrate.jit
def broadcast_add_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, 1), S.bf16),
    original: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    out: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
):
    """Broadcast add: out = x + original where x is (B, 1) and original is (B, IN)"""
    row = S.block_id(1)
    col = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if row < BATCH_SIZE and col < IN_FEATURES:
        xv = S.convert(x[row, 0], S.f32)
        ov = S.convert(original[row, col], S.f32)
        out[row, col] = S.convert(xv + ov, S.bf16)


class ModelNew(nn.Module):
    """
    Optimized Substrate DSL implementation of the model:
    Gemm -> Subtract -> GlobalAvgPool -> LogSumExp -> GELU -> ResidualAdd
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(ModelNew, self).__init__()

        # Match reference model's structure exactly
        # Reference: self.gemm = nn.Linear(in_features, out_features, bias=bias)
        #            self.subtract = nn.Parameter(torch.randn(out_features))
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move to GPU if needed
        orig_device = x.device
        need_copy_back = not x.is_cuda

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device required for Substrate kernels.")

        if need_copy_back:
            x_dev = x.contiguous().cuda()
        else:
            x_dev = x.contiguous()

        # Save original for residual connection
        original_x = x_dev.clone().detach()

        # Determine compute dtype - preserve input dtype behavior
        compute_dtype = x.dtype

        # Prepare weight tensors (nn.Linear stores weight as (out_features, in_features))
        weight = self.gemm.weight.data.to(compute_dtype).contiguous()
        bias = self.gemm.bias.data.to(compute_dtype).contiguous() if self.gemm.bias is not None else torch.zeros(OUT_FEATURES, device=x_dev.device, dtype=compute_dtype)
        subtract = self.subtract.data.to(compute_dtype).contiguous()

        # Grid configurations
        grid_gemm = (OUT_FEATURES // BLOCK_N, BATCH_SIZE // BLOCK_M, 1)
        block_gemm = (THREADS, 1, 1)
        grid_sub = ((OUT_FEATURES + BLOCK_SIZE - 1) // BLOCK_SIZE, BATCH_SIZE, 1)
        block_elem = (BLOCK_SIZE, 1, 1)
        grid_add = ((IN_FEATURES + BLOCK_SIZE - 1) // BLOCK_SIZE, BATCH_SIZE, 1)

        if compute_dtype == torch.float32:
            # Float32 compute path
            # 1. Linear (GEMM + bias)
            gemm_out = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x_dev.device, dtype=torch.float32)
            linear_f32_kernel[lambda: (grid_gemm, block_gemm)](x_dev, weight, bias, gemm_out)

            # 2. Subtract
            sub_out = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x_dev.device, dtype=torch.float32)
            subtract_f32_kernel[lambda: (grid_sub, block_elem)](gemm_out, subtract, sub_out)

            # 3. GlobalAvgPool (mean over dim=1)
            mean_out = torch.empty((BATCH_SIZE, 1), device=x_dev.device, dtype=torch.float32)
            mean_dim1_f32_kernel[lambda: ((BATCH_SIZE, 1, 1), (1, 1, 1))](sub_out, mean_out)

            # 4. LogSumExp (effectively identity since dim=1 is 1)
            lse_out = torch.empty((BATCH_SIZE, 1), device=x_dev.device, dtype=torch.float32)
            logsumexp_dim1_f32_kernel[lambda: ((BATCH_SIZE, 1, 1), (1, 1, 1))](mean_out, lse_out)

            # 5. GELU
            gelu_out = torch.empty((BATCH_SIZE, 1), device=x_dev.device, dtype=torch.float32)
            gelu_f32_kernel[lambda: ((BATCH_SIZE, 1, 1), (1, 1, 1))](lse_out.view(-1), gelu_out.view(-1), BATCH_SIZE)

            # 6. ResidualAdd (broadcast add with original input)
            result = torch.empty((BATCH_SIZE, IN_FEATURES), device=x_dev.device, dtype=torch.float32)
            broadcast_add_f32_kernel[lambda: (grid_add, block_elem)](gelu_out, original_x, result)

        elif compute_dtype == torch.bfloat16:
            # BF16 compute path with FP32 accumulation
            # 1. Linear (GEMM + bias)
            gemm_out = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x_dev.device, dtype=torch.bfloat16)
            linear_bf16_kernel[lambda: (grid_gemm, block_gemm)](x_dev, weight, bias, gemm_out)

            # 2. Subtract
            sub_out = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x_dev.device, dtype=torch.bfloat16)
            subtract_bf16_kernel[lambda: (grid_sub, block_elem)](gemm_out, subtract, sub_out)

            # 3. GlobalAvgPool (mean over dim=1)
            mean_out = torch.empty((BATCH_SIZE, 1), device=x_dev.device, dtype=torch.bfloat16)
            mean_dim1_bf16_kernel[lambda: ((BATCH_SIZE, 1, 1), (1, 1, 1))](sub_out, mean_out)

            # 4. LogSumExp (effectively identity since dim=1 is 1)
            lse_out = torch.empty((BATCH_SIZE, 1), device=x_dev.device, dtype=torch.bfloat16)
            logsumexp_dim1_bf16_kernel[lambda: ((BATCH_SIZE, 1, 1), (1, 1, 1))](mean_out, lse_out)

            # 5. GELU
            gelu_out = torch.empty((BATCH_SIZE, 1), device=x_dev.device, dtype=torch.bfloat16)
            gelu_bf16_kernel[lambda: ((BATCH_SIZE, 1, 1), (1, 1, 1))](lse_out.view(-1), gelu_out.view(-1), BATCH_SIZE)

            # 6. ResidualAdd (broadcast add with original input)
            result = torch.empty((BATCH_SIZE, IN_FEATURES), device=x_dev.device, dtype=torch.bfloat16)
            broadcast_add_bf16_kernel[lambda: (grid_add, block_elem)](gelu_out, original_x, result)

        else:
            raise TypeError(f"Unsupported dtype: {compute_dtype}. Supported: float32, bfloat16.")

        if need_copy_back:
            return result.to(orig_device)
        return result


batch_size = BATCH_SIZE
in_features = IN_FEATURES
out_features = OUT_FEATURES


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
