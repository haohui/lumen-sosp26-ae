import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Problem sizes from target model.py
BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192

# GEMM tiling parameters
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 16
THREADS = BLOCK_M * BLOCK_N

# Elementwise kernel block size
ELEMENTWISE_BLOCK = 256

# Constants for Mish activation
LOG2E = 1.4426950408889634  # 1/log(2) for exp to exp2 conversion


@substrate.jit
def linear_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    w_ptr: S.Pointer(S.bf16),
    bias_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    m: S.u32,
    n: S.u32,
    k: S.u32,
):
    """GEMM kernel for linear layer: out = x @ w.T + bias"""
    bid_n = S.block_id(0)
    bid_m = S.block_id(1)
    tid = S.thread_id(0)

    # Create layouts for input tensors
    layout_x = S.make_layout((m, k), (k, 1))
    layout_w = S.make_layout((n, k), (k, 1))  # W is (n, k), we access W[j, :] for row j
    layout_bias = S.make_layout((n,), (1,))
    layout_out = S.make_layout((m, n), (n, 1))

    gx = S.make_tensor(x_ptr, S.bf16, layout_x)
    gw = S.make_tensor(w_ptr, S.bf16, layout_w)
    gbias = S.make_tensor(bias_ptr, S.bf16, layout_bias)
    gout = S.make_tensor(out_ptr, S.bf16, layout_out)

    # Shared memory tiles
    sx = S.make_shared((BLOCK_M, BLOCK_K), S.bf16)
    sw = S.make_shared((BLOCK_N, BLOCK_K), S.bf16)

    # Thread-to-output mapping
    local_m = tid // BLOCK_N
    local_n = tid % BLOCK_N

    row = bid_m * BLOCK_M + local_m
    col = bid_n * BLOCK_N + local_n

    # FP32 accumulation
    acc = S.convert(0.0, S.f32)

    k_tiles = (k + BLOCK_K - 1) // BLOCK_K
    loads_per_thread_x = (BLOCK_M * BLOCK_K) // THREADS
    loads_per_thread_w = (BLOCK_N * BLOCK_K) // THREADS

    for kt in S.range(k_tiles):
        k_base = kt * BLOCK_K

        # Load X tile: [BLOCK_M, BLOCK_K]
        for i in S.range(loads_per_thread_x):
            idx = tid + i * THREADS
            lx_r = idx // BLOCK_K
            lx_c = idx % BLOCK_K
            g_r = bid_m * BLOCK_M + lx_r
            g_c = k_base + lx_c
            if g_r < m and g_c < k:
                sx[lx_r, lx_c] = gx[g_r, g_c]
            else:
                sx[lx_r, lx_c] = S.convert(0.0, S.bf16)

        # Load W tile: [BLOCK_N, BLOCK_K] (W is stored as (n, k))
        for i in S.range(loads_per_thread_w):
            idx = tid + i * THREADS
            lw_r = idx // BLOCK_K
            lw_c = idx % BLOCK_K
            g_r = bid_n * BLOCK_N + lw_r
            g_c = k_base + lw_c
            if g_r < n and g_c < k:
                sw[lw_r, lw_c] = gw[g_r, g_c]
            else:
                sw[lw_r, lw_c] = S.convert(0.0, S.bf16)

        S.syncthreads()

        # Compute partial product
        for kk in S.range(BLOCK_K):
            xv = S.convert(sx[local_m, kk], S.f32)
            wv = S.convert(sw[local_n, kk], S.f32)
            acc = acc + xv * wv

        S.syncthreads()

    # Add bias and store result
    if row < m and col < n:
        bias_val = S.convert(gbias[col], S.f32)
        result = acc + bias_val
        gout[row, col] = S.convert(result, S.bf16)


@substrate.jit
def mish_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    """Mish activation: mish(x) = x * tanh(softplus(x)) where softplus(x) = ln(1 + exp(x))"""
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx < n:
        layout = S.make_layout((n,), (1,))
        gx = S.make_tensor(x_ptr, S.bf16, layout)
        gout = S.make_tensor(out_ptr, S.bf16, layout)

        # Load and convert to f32 for compute
        xv = S.convert(gx[idx], S.f32)

        # Compute softplus(x) = ln(1 + exp(x))
        # Use exp2 for better performance: exp(x) = exp2(x * log2(e))
        log2e = S.convert(LOG2E, S.f32)
        one = S.convert(1.0, S.f32)

        exp_x = S.exp2(xv * log2e)
        softplus_x = S.log(one + exp_x)

        # Compute tanh(softplus(x))
        tanh_sp = S.tanh(softplus_x)

        # mish(x) = x * tanh(softplus(x))
        mish_val = xv * tanh_sp

        # Store result
        gout[idx] = S.convert(mish_val, S.bf16)


def _launch_linear_bf16(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Launch the linear kernel for BF16 inputs."""
    m, k = x.shape
    n, k2 = w.shape
    assert k == k2, f"Incompatible shapes: x={x.shape}, w={w.shape}"

    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)

    grid_x = (n + BLOCK_N - 1) // BLOCK_N
    grid_y = (m + BLOCK_M - 1) // BLOCK_M

    linear_bf16_kernel[lambda: ((grid_x, grid_y, 1), (THREADS, 1, 1))](
        x, w, bias, out, m, n, k
    )
    return out


def _launch_mish_bf16(x: torch.Tensor) -> torch.Tensor:
    """Launch the Mish kernel for BF16 inputs."""
    x_flat = x.view(-1)
    n = x_flat.numel()
    out = torch.empty_like(x_flat)

    if n > 0:
        grid = (n + ELEMENTWISE_BLOCK - 1) // ELEMENTWISE_BLOCK
        mish_bf16_kernel[lambda: ((grid, 1, 1), (ELEMENTWISE_BLOCK, 1, 1))](x_flat, out, n)

    return out.view_as(x)


def substrate_linear_mish_mish(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Compute linear layer followed by double Mish activation using Substrate kernels."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

    # Move to GPU if needed
    orig_device = x.device
    moved = not x.is_cuda
    if moved:
        x = x.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    # Ensure contiguous and correct dtype
    x = x.contiguous().to(torch.bfloat16)
    weight = weight.contiguous().to(torch.bfloat16)
    bias = bias.contiguous().to(torch.bfloat16)

    # Linear: x @ W.T + bias
    x = _launch_linear_bf16(x, weight, bias)

    # First Mish
    x = _launch_mish_bf16(x)

    # Second Mish
    x = _launch_mish_bf16(x)

    if moved:
        x = x.to(orig_device)

    return x


class ModelNew(nn.Module):
    """
    Optimized model using Substrate GPU kernels for:
    - Linear layer (GEMM + bias)
    - Mish activation applied twice
    """
    def __init__(self, in_features: int, out_features: int):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Store weight and bias as parameters (matching nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return substrate_linear_mish_mish(x, self.weight, self.bias)


batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
