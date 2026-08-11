import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Problem sizes
BATCH_SIZE = 128
IN_FEATURES = 32768
OUT_FEATURES = 32768
SCALING_FACTOR = 2.0

# GEMM tiling parameters
BLOCK_M = 16
BLOCK_N = 32
BLOCK_K = 32
THREADS = BLOCK_M * BLOCK_N // 2  # 256 threads for cooperative loading

# Derived constants for GEMM
M = BATCH_SIZE
N = OUT_FEATURES
K = IN_FEATURES

K_TILES = K // BLOCK_K
A_LOADS_PER_THREAD = (BLOCK_M * BLOCK_K) // THREADS
B_LOADS_PER_THREAD = (BLOCK_K * BLOCK_N) // THREADS

# Elementwise kernel parameters
ELEM_THREADS = 256

# Constants for sigmoid computation
LOG2E = 1.4426950408889634  # log2(e)


@substrate.jit
def gemm_bias_bf16_kernel(
    A: S.Pointer(S.bf16),
    B: S.Pointer(S.bf16),
    bias: S.Pointer(S.bf16),
    C: S.Pointer(S.bf16),
    m: S.u32,
    n: S.u32,
    k: S.u32,
):
    """Compute C = A @ B^T + bias for BF16 inputs with FP32 accumulation."""
    bid_n = S.block_id(0)
    bid_m = S.block_id(1)
    tid = S.thread_id(0)

    # Create tensor views with layouts
    layout_a = S.make_layout((m, k), (k, 1))
    layout_b = S.make_layout((n, k), (k, 1))  # B is stored as (n, k), we read B^T
    layout_c = S.make_layout((m, n), (n, 1))

    gA = S.make_tensor(A, S.bf16, layout_a)
    gB = S.make_tensor(B, S.bf16, layout_b)
    gC = S.make_tensor(C, S.bf16, layout_c)
    gBias = S.make_tensor(bias, S.bf16, S.make_layout((n,), (1,)))

    # Shared memory tiles
    sA = S.make_shared((BLOCK_M, BLOCK_K), S.bf16)
    sB = S.make_shared((BLOCK_N, BLOCK_K), S.bf16)

    # Thread mapping: each thread computes multiple output elements
    tx = tid % (BLOCK_N // 2)
    ty = tid // (BLOCK_N // 2)

    # Each thread computes 2x2 block of outputs
    local_m0 = ty * 2
    local_m1 = local_m0 + 1
    local_n0 = tx * 2
    local_n1 = local_n0 + 1

    # Global output coordinates
    row0 = bid_m * BLOCK_M + local_m0
    row1 = bid_m * BLOCK_M + local_m1
    col0 = bid_n * BLOCK_N + local_n0
    col1 = bid_n * BLOCK_N + local_n1

    # FP32 accumulators for 2x2 output block
    acc00 = S.convert(0.0, S.f32)
    acc01 = S.convert(0.0, S.f32)
    acc10 = S.convert(0.0, S.f32)
    acc11 = S.convert(0.0, S.f32)

    k_tiles = k // BLOCK_K

    for kt in S.range(k_tiles):
        k_base = kt * BLOCK_K

        # Cooperative load A tile: [BLOCK_M, BLOCK_K]
        for i in S.range(A_LOADS_PER_THREAD):
            idx = tid + i * THREADS
            a_r = idx // BLOCK_K
            a_c = idx % BLOCK_K
            sA[a_r, a_c] = gA[bid_m * BLOCK_M + a_r, k_base + a_c]

        # Cooperative load B tile: [BLOCK_N, BLOCK_K] (B is already transposed in memory)
        for i in S.range(B_LOADS_PER_THREAD):
            idx = tid + i * THREADS
            b_r = idx // BLOCK_K
            b_c = idx % BLOCK_K
            sB[b_r, b_c] = gB[bid_n * BLOCK_N + b_r, k_base + b_c]

        S.syncthreads()

        # Compute 2x2 output block
        for kk in S.range(BLOCK_K):
            a0 = S.convert(sA[local_m0, kk], S.f32)
            a1 = S.convert(sA[local_m1, kk], S.f32)
            b0 = S.convert(sB[local_n0, kk], S.f32)
            b1 = S.convert(sB[local_n1, kk], S.f32)
            acc00 = acc00 + a0 * b0
            acc01 = acc01 + a0 * b1
            acc10 = acc10 + a1 * b0
            acc11 = acc11 + a1 * b1

        S.syncthreads()

    # Add bias and store results
    bias0 = S.convert(gBias[col0], S.f32)
    bias1 = S.convert(gBias[col1], S.f32)

    if row0 < m and col0 < n:
        gC[row0, col0] = S.convert(acc00 + bias0, S.bf16)
    if row0 < m and col1 < n:
        gC[row0, col1] = S.convert(acc01 + bias1, S.bf16)
    if row1 < m and col0 < n:
        gC[row1, col0] = S.convert(acc10 + bias0, S.bf16)
    if row1 < m and col1 < n:
        gC[row1, col1] = S.convert(acc11 + bias1, S.bf16)


@substrate.jit
def swish_scale_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    y_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    """Compute y = x * sigmoid(x) * SCALING_FACTOR elementwise for BF16."""
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx < n:
        layout = S.make_layout((n,), (1,))
        x = S.make_tensor(x_ptr, S.bf16, layout)
        y = S.make_tensor(y_ptr, S.bf16, layout)

        # Load BF16 value and convert to FP32 for computation
        xv = S.convert(x[idx], S.f32)

        # Compute sigmoid(x) = 1 / (1 + exp(-x))
        # exp(-x) = exp2(-x * log2(e))
        log2e = S.convert(LOG2E, S.f32)
        neg_x = S.convert(0.0, S.f32) - xv
        exp_neg_x = S.exp2(neg_x * log2e)
        one = S.convert(1.0, S.f32)
        sigmoid_x = one / (one + exp_neg_x)

        # Swish: x * sigmoid(x), then scale by compile-time constant
        scale = S.convert(SCALING_FACTOR, S.f32)
        swish = xv * sigmoid_x
        result = swish * scale

        # Store as BF16
        y[idx] = S.convert(result, S.bf16)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _launch_gemm_bias(
    A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Launch GEMM with bias kernel for BF16."""
    m, k = A.shape
    n = B.shape[0]  # B is (n, k) stored as transposed
    assert B.shape[1] == k

    out = torch.empty((m, n), device=A.device, dtype=torch.bfloat16)

    grid_x = _ceil_div(n, BLOCK_N)
    grid_y = _ceil_div(m, BLOCK_M)

    gemm_bias_bf16_kernel[lambda: ((grid_x, grid_y, 1), (THREADS, 1, 1))](
        A, B, bias, out, m, n, k
    )
    return out


def _launch_swish_scale(x: torch.Tensor) -> torch.Tensor:
    """Launch Swish + scale kernel for BF16."""
    n = x.numel()
    out = torch.empty_like(x)

    grid = _ceil_div(n, ELEM_THREADS)
    swish_scale_bf16_kernel[lambda: ((grid, 1, 1), (ELEM_THREADS, 1, 1))](
        x, out, n
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication with Swish activation and scaling
    using Substrate GPU kernels optimized for BF16 on AMD MI300X.
    """

    def __init__(self, in_features: int, out_features: int, scaling_factor: float):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor

        # Store weight as transposed for efficient kernel access: (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous and on GPU
        orig_device = x.device
        moved = False

        if not x.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA/HIP device is required for Substrate kernels")
            x = x.cuda()
            moved = True

        x = x.contiguous()

        # Convert to BF16 for computation
        x_bf16 = x.to(torch.bfloat16)
        weight_bf16 = self.weight.to(torch.bfloat16).contiguous()
        bias_bf16 = self.bias.to(torch.bfloat16).contiguous()

        # GEMM: x @ W^T + bias (W is stored transposed, so we do x @ W directly)
        out = _launch_gemm_bias(x_bf16, weight_bf16, bias_bf16)

        # Swish activation + scaling
        out = _launch_swish_scale(out)

        if moved:
            out = out.to(orig_device)

        return out


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_FEATURES)]


def get_init_inputs():
    return [IN_FEATURES, OUT_FEATURES, SCALING_FACTOR]
