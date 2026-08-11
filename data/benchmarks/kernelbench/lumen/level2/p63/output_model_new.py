import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Problem sizes from target model
M = 1024
K = 8192
N = 8192

# Tile configuration for BF16 GEMM
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 16

THREADS_X = 16
THREADS_Y = 16
THREADS = THREADS_X * THREADS_Y

LOADS_A = (BLOCK_M * BLOCK_K) // THREADS  # 2
LOADS_B = (BLOCK_K * BLOCK_N) // THREADS  # 2

# Reciprocal of divisor for multiplication (divisor = 2.0)
RECIPROCAL = 0.5


@substrate.jit
def linear_relu_div_bf16_kernel(
    A: S.Pointer(S.bf16),
    B: S.Pointer(S.bf16),
    Bias: S.Pointer(S.bf16),
    C: S.Pointer(S.bf16),
    m: S.u32,
    n: S.u32,
    k: S.u32,
):
    """Fused GEMM + bias + ReLU + divide kernel for BF16."""
    bid_n = S.block_id(0)
    bid_m = S.block_id(1)
    tid = S.thread_id(0)

    layout_a = S.make_layout((m, k), (k, 1))
    layout_b = S.make_layout((n, k), (k, 1))  # B is transposed: (N, K)
    layout_c = S.make_layout((m, n), (n, 1))

    gA = S.make_tensor(A, S.bf16, layout_a)
    gB = S.make_tensor(B, S.bf16, layout_b)
    gC = S.make_tensor(C, S.bf16, layout_c)
    gBias = S.make_tensor(Bias, S.bf16, S.make_layout((n,), (1,)))

    sA = S.make_shared((BLOCK_M, BLOCK_K), S.bf16)
    sB = S.make_shared((BLOCK_N, BLOCK_K), S.bf16)

    tx = tid % THREADS_X
    ty = tid // THREADS_X

    # Each thread computes 2x2 output block
    local_r0 = ty * 2
    local_r1 = local_r0 + 1
    local_c0 = tx * 2
    local_c1 = local_c0 + 1

    row0 = bid_m * BLOCK_M + local_r0
    row1 = row0 + 1
    col0 = bid_n * BLOCK_N + local_c0
    col1 = col0 + 1

    # FP32 accumulation for BF16 compute
    acc00 = S.convert(0.0, S.f32)
    acc01 = S.convert(0.0, S.f32)
    acc10 = S.convert(0.0, S.f32)
    acc11 = S.convert(0.0, S.f32)

    k_tiles = (k + BLOCK_K - 1) // BLOCK_K

    for kt in S.range(k_tiles):
        k_base = kt * BLOCK_K

        # Load A tile: A is (M, K) row-major
        for i in S.range(LOADS_A):
            idx = tid + i * THREADS
            a_r = idx // BLOCK_K
            a_c = idx % BLOCK_K
            g_r = bid_m * BLOCK_M + a_r
            g_c = k_base + a_c
            if g_r < m and g_c < k:
                sA[a_r, a_c] = gA[g_r, g_c]
            else:
                sA[a_r, a_c] = S.convert(0.0, S.bf16)

        # Load B tile: B is (N, K) row-major (weight transposed)
        for i in S.range(LOADS_B):
            idx = tid + i * THREADS
            b_r = idx // BLOCK_K
            b_c = idx % BLOCK_K
            g_r = bid_n * BLOCK_N + b_r
            g_c = k_base + b_c
            if g_r < n and g_c < k:
                sB[b_r, b_c] = gB[g_r, g_c]
            else:
                sB[b_r, b_c] = S.convert(0.0, S.bf16)

        S.syncthreads()

        # Compute partial dot products
        for kk in S.range(BLOCK_K):
            a0 = S.convert(sA[local_r0, kk], S.f32)
            a1 = S.convert(sA[local_r1, kk], S.f32)
            b0 = S.convert(sB[local_c0, kk], S.f32)
            b1 = S.convert(sB[local_c1, kk], S.f32)
            acc00 = acc00 + a0 * b0
            acc01 = acc01 + a0 * b1
            acc10 = acc10 + a1 * b0
            acc11 = acc11 + a1 * b1

        S.syncthreads()

    # Load bias values for this block's columns
    zero_f = S.convert(0.0, S.f32)
    bias_c0 = S.convert(gBias[col0], S.f32)
    bias_c1 = S.convert(gBias[col1], S.f32)

    # Add bias, apply ReLU, and multiply by reciprocal (equivalent to divide by 2)
    if row0 < m and col0 < n:
        val00 = acc00 + bias_c0
        val00 = val00 * S.convert(RECIPROCAL, S.f32)
        if val00 < zero_f:
            val00 = zero_f
        gC[row0, col0] = S.convert(val00, S.bf16)

    if row0 < m and col1 < n:
        val01 = acc01 + bias_c1
        val01 = val01 * S.convert(RECIPROCAL, S.f32)
        if val01 < zero_f:
            val01 = zero_f
        gC[row0, col1] = S.convert(val01, S.bf16)

    if row1 < m and col0 < n:
        val10 = acc10 + bias_c0
        val10 = val10 * S.convert(RECIPROCAL, S.f32)
        if val10 < zero_f:
            val10 = zero_f
        gC[row1, col0] = S.convert(val10, S.bf16)

    if row1 < m and col1 < n:
        val11 = acc11 + bias_c1
        val11 = val11 * S.convert(RECIPROCAL, S.f32)
        if val11 < zero_f:
            val11 = zero_f
        gC[row1, col1] = S.convert(val11, S.bf16)


@substrate.jit
def linear_relu_div_f32_kernel(
    A: S.Pointer(S.f32),
    B: S.Pointer(S.f32),
    Bias: S.Pointer(S.f32),
    C: S.Pointer(S.f32),
    m: S.u32,
    n: S.u32,
    k: S.u32,
):
    """Fused GEMM + bias + ReLU + divide kernel for FP32."""
    bid_n = S.block_id(0)
    bid_m = S.block_id(1)
    tid = S.thread_id(0)

    layout_a = S.make_layout((m, k), (k, 1))
    layout_b = S.make_layout((n, k), (k, 1))
    layout_c = S.make_layout((m, n), (n, 1))

    gA = S.make_tensor(A, S.f32, layout_a)
    gB = S.make_tensor(B, S.f32, layout_b)
    gC = S.make_tensor(C, S.f32, layout_c)
    gBias = S.make_tensor(Bias, S.f32, S.make_layout((n,), (1,)))

    sA = S.make_shared((BLOCK_M, BLOCK_K), S.f32)
    sB = S.make_shared((BLOCK_N, BLOCK_K), S.f32)

    tx = tid % THREADS_X
    ty = tid // THREADS_X

    local_r0 = ty * 2
    local_r1 = local_r0 + 1
    local_c0 = tx * 2
    local_c1 = local_c0 + 1

    row0 = bid_m * BLOCK_M + local_r0
    row1 = row0 + 1
    col0 = bid_n * BLOCK_N + local_c0
    col1 = col0 + 1

    acc00 = S.convert(0.0, S.f32)
    acc01 = S.convert(0.0, S.f32)
    acc10 = S.convert(0.0, S.f32)
    acc11 = S.convert(0.0, S.f32)

    k_tiles = (k + BLOCK_K - 1) // BLOCK_K

    for kt in S.range(k_tiles):
        k_base = kt * BLOCK_K

        for i in S.range(LOADS_A):
            idx = tid + i * THREADS
            a_r = idx // BLOCK_K
            a_c = idx % BLOCK_K
            g_r = bid_m * BLOCK_M + a_r
            g_c = k_base + a_c
            if g_r < m and g_c < k:
                sA[a_r, a_c] = gA[g_r, g_c]
            else:
                sA[a_r, a_c] = S.convert(0.0, S.f32)

        for i in S.range(LOADS_B):
            idx = tid + i * THREADS
            b_r = idx // BLOCK_K
            b_c = idx % BLOCK_K
            g_r = bid_n * BLOCK_N + b_r
            g_c = k_base + b_c
            if g_r < n and g_c < k:
                sB[b_r, b_c] = gB[g_r, g_c]
            else:
                sB[b_r, b_c] = S.convert(0.0, S.f32)

        S.syncthreads()

        for kk in S.range(BLOCK_K):
            a0 = sA[local_r0, kk]
            a1 = sA[local_r1, kk]
            b0 = sB[local_c0, kk]
            b1 = sB[local_c1, kk]
            acc00 = acc00 + a0 * b0
            acc01 = acc01 + a0 * b1
            acc10 = acc10 + a1 * b0
            acc11 = acc11 + a1 * b1

        S.syncthreads()

    # Load bias values for this block's columns
    zero_f = S.convert(0.0, S.f32)
    bias_c0 = gBias[col0]
    bias_c1 = gBias[col1]

    if row0 < m and col0 < n:
        val00 = acc00 + bias_c0
        val00 = val00 * S.convert(RECIPROCAL, S.f32)
        if val00 < zero_f:
            val00 = zero_f
        gC[row0, col0] = val00

    if row0 < m and col1 < n:
        val01 = acc01 + bias_c1
        val01 = val01 * S.convert(RECIPROCAL, S.f32)
        if val01 < zero_f:
            val01 = zero_f
        gC[row0, col1] = val01

    if row1 < m and col0 < n:
        val10 = acc10 + bias_c0
        val10 = val10 * S.convert(RECIPROCAL, S.f32)
        if val10 < zero_f:
            val10 = zero_f
        gC[row1, col0] = val10

    if row1 < m and col1 < n:
        val11 = acc11 + bias_c1
        val11 = val11 * S.convert(RECIPROCAL, S.f32)
        if val11 < zero_f:
            val11 = zero_f
        gC[row1, col1] = val11


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _launch_bf16(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    n = weight.shape[0]  # weight is (out_features, in_features)
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    grid_x = _ceil_div(n, BLOCK_N)
    grid_y = _ceil_div(m, BLOCK_M)
    linear_relu_div_bf16_kernel[lambda: ((grid_x, grid_y, 1), (THREADS, 1, 1))](
        x, weight, bias, out, m, n, k
    )
    return out


def _launch_f32(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    grid_x = _ceil_div(n, BLOCK_N)
    grid_y = _ceil_div(m, BLOCK_M)
    linear_relu_div_f32_kernel[lambda: ((grid_x, grid_y, 1), (THREADS, 1, 1))](
        x, weight, bias, out, m, n, k
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication, applies ReLU, and divides by a constant.
    Uses fused Substrate GPU kernels for improved performance.
    """
    def __init__(self, in_features: int, out_features: int, divisor: float):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.divisor = divisor

        # Store weight as (out_features, in_features) in row-major (transposed for GEMM)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got shape {tuple(x.shape)}")

        orig_device = x.device
        need_host_copy_back = not x.is_cuda

        if need_host_copy_back:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")
            x_dev = x.contiguous().cuda()
            weight_dev = self.weight.cuda()
            bias_dev = self.bias.cuda()
        else:
            x_dev = x.contiguous()
            weight_dev = self.weight
            bias_dev = self.bias

        # Determine compute dtype (prefer BF16 for AMD MI300X)
        compute_dtype = torch.bfloat16
        if x_dev.dtype != compute_dtype:
            x_dev = x_dev.to(compute_dtype).contiguous()
        if weight_dev.dtype != compute_dtype:
            weight_dev = weight_dev.to(compute_dtype).contiguous()
        if bias_dev.dtype != compute_dtype:
            bias_dev = bias_dev.to(compute_dtype).contiguous()

        if compute_dtype == torch.bfloat16:
            out_dev = _launch_bf16(x_dev, weight_dev, bias_dev)
        else:
            out_dev = _launch_f32(x_dev, weight_dev, bias_dev)

        if need_host_copy_back:
            return out_dev.to(orig_device)
        return out_dev


batch_size = 1024
in_features = 8192
out_features = 8192
divisor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, divisor]
