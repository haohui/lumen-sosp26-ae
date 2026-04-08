import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Problem sizes
BATCH_SIZE = 128
INPUT_SIZE = 32768
HIDDEN_SIZE = 32768

# GEMM tiling parameters
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 16
THREADS_X = 16
THREADS_Y = 16
THREADS = THREADS_X * THREADS_Y

LOADS_A = (BLOCK_M * BLOCK_K) // THREADS
LOADS_B = (BLOCK_K * BLOCK_N) // THREADS

# Sigmoid constant: log2(e)
LOG2E = 1.4426950408889634

# Reduction block size
BLOCK_SIZE = 256


@substrate.jit
def linear_bf16_kernel(
    A: S.Pointer(S.bf16),
    B: S.Pointer(S.bf16),
    bias: S.Pointer(S.bf16),
    C: S.Pointer(S.bf16),
    m: S.u32,
    n: S.u32,
    k: S.u32,
):
    """GEMM kernel for linear layer: C = A @ B.T + bias"""
    bid_n = S.block_id(0)
    bid_m = S.block_id(1)
    tid = S.thread_id(0)

    layout_a = S.make_layout((m, k), (k, 1))
    layout_b = S.make_layout((n, k), (k, 1))  # B is (hidden_size, input_size), we access B[row, :] for each output column
    layout_c = S.make_layout((m, n), (n, 1))

    gA = S.make_tensor(A, S.bf16, layout_a)
    gB = S.make_tensor(B, S.bf16, layout_b)
    gC = S.make_tensor(C, S.bf16, layout_c)
    g_bias = S.make_tensor(bias, S.bf16, S.make_layout((n,), (1,)))

    sA = S.make_shared((BLOCK_M, BLOCK_K), S.bf16)
    sB = S.make_shared((BLOCK_N, BLOCK_K), S.bf16)

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

        # Load A tile: A is (M, K), load [BLOCK_M, BLOCK_K]
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

        # Load B tile: B is (N, K), load [BLOCK_N, BLOCK_K]
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

    # Add bias and store
    # Load bias values for each column
    b0 = S.convert(g_bias[col0], S.f32)
    b1 = S.convert(g_bias[col1], S.f32)

    if row0 < m and col0 < n:
        gC[row0, col0] = S.convert(acc00 + b0, S.bf16)
    if row0 < m and col1 < n:
        gC[row0, col1] = S.convert(acc01 + b1, S.bf16)
    if row1 < m and col0 < n:
        gC[row1, col0] = S.convert(acc10 + b0, S.bf16)
    if row1 < m and col1 < n:
        gC[row1, col1] = S.convert(acc11 + b1, S.bf16)


@substrate.jit
def sigmoid_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    y_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    """Element-wise sigmoid: y = 1 / (1 + exp(-x))"""
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx < n:
        layout = S.make_layout((n,), (1,))
        x = S.make_tensor(x_ptr, S.bf16, layout)
        y = S.make_tensor(y_ptr, S.bf16, layout)

        xv = S.convert(x[idx], S.f32)
        one = S.convert(1.0, S.f32)
        log2e = S.convert(LOG2E, S.f32)

        # sigmoid(x) = 1 / (1 + exp(-x)) = 1 / (1 + exp2(-x * log2(e)))
        neg_x_log2e = -xv * log2e
        exp_term = S.exp2(neg_x_log2e)
        sig = one / (one + exp_term)

        y[idx] = S.convert(sig, S.bf16)


@substrate.jit
def sum_dim1_bf16_kernel(
    x: S.Pointer(S.bf16),
    out: S.Pointer(S.bf16),
    batch_size: S.u32,
    hidden_size: S.u32,
):
    """Sum reduction along dim=1 with keepdim: out[b, 0] = sum_c x[b, c]"""
    tid = S.thread_id(0)
    b = S.block_id(0)

    if b < batch_size:
        layout_x = S.make_layout((batch_size, hidden_size), (hidden_size, 1))
        layout_out = S.make_layout((batch_size, 1), (1, 1))

        gx = S.make_tensor(x, S.bf16, layout_x)
        gout = S.make_tensor(out, S.bf16, layout_out)

        acc = S.convert(0.0, S.f32)
        for c in S.range(hidden_size):
            acc = acc + S.convert(gx[b, c], S.f32)

        gout[b, 0] = S.convert(acc, S.bf16)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def substrate_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Compute y = x @ weight.T + bias using Substrate GEMM kernel."""
    assert x.ndim == 2 and weight.ndim == 2
    m, k = x.shape
    n, k2 = weight.shape
    assert k == k2

    orig_device = x.device
    moved = False
    if not x.is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device required for Substrate kernels.")
        x = x.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        moved = True

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Convert to BF16 for optimized computation
    x_bf16 = x.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)

    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)

    grid_x = _ceil_div(n, BLOCK_N)
    grid_y = _ceil_div(m, BLOCK_M)

    linear_bf16_kernel[lambda: ((grid_x, grid_y, 1), (THREADS, 1, 1))](
        x_bf16, weight_bf16, bias_bf16, out, m, n, k
    )

    if moved:
        return out.to(orig_device)
    return out


def substrate_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Element-wise sigmoid using Substrate kernel."""
    orig_device = x.device
    moved = False
    if not x.is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device required for Substrate kernels.")
        x = x.cuda()
        moved = True

    x = x.contiguous()
    x_bf16 = x.to(torch.bfloat16)

    y = torch.empty_like(x_bf16)
    n = x_bf16.numel()

    if n > 0:
        grid = _ceil_div(n, BLOCK_SIZE)
        sigmoid_bf16_kernel[lambda: ((grid, 1, 1), (BLOCK_SIZE, 1, 1))](x_bf16, y, n)

    if moved:
        return y.to(orig_device)
    return y


def substrate_sum_dim1_keepdim(x: torch.Tensor) -> torch.Tensor:
    """Sum along dim=1 with keepdim=True using Substrate kernel."""
    assert x.ndim == 2
    batch_size, hidden_size = x.shape

    orig_device = x.device
    moved = False
    if not x.is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device required for Substrate kernels.")
        x = x.cuda()
        moved = True

    x = x.contiguous()
    x_bf16 = x.to(torch.bfloat16)

    out = torch.empty((batch_size, 1), device=x.device, dtype=torch.bfloat16)

    sum_dim1_bf16_kernel[lambda: ((batch_size, 1, 1), (1, 1, 1))](
        x_bf16, out, batch_size, hidden_size
    )

    if moved:
        return out.to(orig_device)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Linear, Sigmoid, and Sum using Substrate GPU kernels.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear: x @ weight.T + bias
        x = substrate_linear(x, self.weight, self.bias)
        # Sigmoid
        x = substrate_sigmoid(x)
        # Sum along dim=1 with keepdim=True
        x = substrate_sum_dim1_keepdim(x)
        return x


batch_size = BATCH_SIZE
input_size = INPUT_SIZE
hidden_size = HIDDEN_SIZE


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size]
