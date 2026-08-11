import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Problem sizes from target model.py
BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
SUBTRACT_VALUE = 2.0
MULTIPLY_VALUE = 1.5

# Tiling parameters
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 32
THREADS = BLOCK_M * BLOCK_N
K_TILES = IN_FEATURES // BLOCK_K


@substrate.jit
def linear_fused_relu_bf16_kernel(
    X: S.Pointer(S.bf16),
    W: S.Pointer(S.bf16),
    Bias: S.Pointer(S.bf16),
    Out: S.Pointer(S.bf16),
    m: S.u32,
    n: S.u32,
    k: S.u32,
):
    """GEMM + bias + subtract + multiply + ReLU fused kernel for BF16."""
    tid = S.thread_id(0)
    bx = S.block_id(0)  # N tile
    by = S.block_id(1)  # M tile

    # Create layout views
    layout_x = S.make_layout((m, k), (k, 1))
    layout_w = S.make_layout((n, k), (k, 1))  # W is stored as (n, k) = (out_features, in_features)
    layout_bias = S.make_layout((n,), (1,))
    layout_out = S.make_layout((m, n), (n, 1))

    gX = S.make_tensor(X, S.bf16, layout_x)
    gW = S.make_tensor(W, S.bf16, layout_w)
    gBias = S.make_tensor(Bias, S.bf16, layout_bias)
    gOut = S.make_tensor(Out, S.bf16, layout_out)

    # Local thread coordinates within the block
    local_m = tid // BLOCK_N
    local_n = tid % BLOCK_N

    # Global coordinates
    row = by * BLOCK_M + local_m
    col = bx * BLOCK_N + local_n

    # Shared memory tiles
    sA = S.make_shared((BLOCK_M, BLOCK_K), S.bf16)
    sB = S.make_shared((BLOCK_N, BLOCK_K), S.bf16)

    # FP32 accumulation
    acc = S.convert(0.0, S.f32)

    # Hardcoded constants
    subtract_val = S.convert(SUBTRACT_VALUE, S.f32)
    multiply_val = S.convert(MULTIPLY_VALUE, S.f32)
    zero = S.convert(0.0, S.f32)

    # Compute GEMM: C[row, col] = sum_k X[row, k] * W[col, k]
    for kt in S.range(K_TILES):
        k_base = kt * BLOCK_K

        # Cooperative load X tile: [BLOCK_M, BLOCK_K]
        # Each thread loads (BLOCK_M * BLOCK_K) / THREADS elements
        for load_idx in S.range((BLOCK_M * BLOCK_K) // THREADS):
            idx = tid + load_idx * THREADS
            a_r = idx // BLOCK_K
            a_c = idx % BLOCK_K
            sA[a_r, a_c] = gX[by * BLOCK_M + a_r, k_base + a_c]

        # Cooperative load W tile: [BLOCK_N, BLOCK_K]
        # W is stored as (n, k), so we access W[col, k_idx]
        for load_idx in S.range((BLOCK_N * BLOCK_K) // THREADS):
            idx = tid + load_idx * THREADS
            b_r = idx // BLOCK_K
            b_c = idx % BLOCK_K
            sB[b_r, b_c] = gW[bx * BLOCK_N + b_r, k_base + b_c]

        S.syncthreads()

        # Compute partial product
        for kk in S.range(BLOCK_K):
            av = S.convert(sA[local_m, kk], S.f32)
            bv = S.convert(sB[local_n, kk], S.f32)
            acc = acc + av * bv

        S.syncthreads()

    # Fuse: result = relu((acc + bias - subtract_val) * multiply_val)
    if row < m and col < n:
        bias = S.convert(gBias[col], S.f32)
        result = acc + bias
        result = result - subtract_val
        result = result * multiply_val

        # ReLU
        if result < zero:
            result = zero

        gOut[row, col] = S.convert(result, S.bf16)


def substrate_linear_fused_relu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Host wrapper for the fused linear + activation kernel."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

    # Ensure inputs are contiguous and on GPU
    x_work = x.contiguous()
    if not x_work.is_cuda:
        x_work = x_work.cuda()

    # Weight is stored as (out_features, in_features), we need it as (out_features, in_features)
    # for the kernel to work as designed
    weight_work = weight.contiguous()
    if not weight_work.is_cuda:
        weight_work = weight_work.cuda()

    bias_work = bias.contiguous()
    if not bias_work.is_cuda:
        bias_work = bias_work.cuda()

    # Convert to BF16 if needed
    x_bf16 = x_work.to(torch.bfloat16)
    weight_bf16 = weight_work.to(torch.bfloat16)
    bias_bf16 = bias_work.to(torch.bfloat16)

    m = x_bf16.shape[0]
    k = x_bf16.shape[1]
    n = weight_bf16.shape[0]

    out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)

    grid_m = m // BLOCK_M
    grid_n = n // BLOCK_N

    linear_fused_relu_bf16_kernel[lambda: ((grid_n, grid_m, 1), (THREADS, 1, 1))](
        x_bf16, weight_bf16, bias_bf16, out, m, n, k
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model using Substrate GPU kernels.
    Performs: linear(x) - subtract_value) * multiply_value, then ReLU.
    """
    def __init__(self, in_features: int, out_features: int, subtract_value: float, multiply_value: float):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

        # Create weight and bias as parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        # Initialize weights (matching PyTorch default initialization)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return substrate_linear_fused_relu(x, self.weight, self.bias)


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_FEATURES)]


def get_init_inputs():
    return [IN_FEATURES, OUT_FEATURES, SUBTRACT_VALUE, MULTIPLY_VALUE]
