import torch
import torch.nn as nn
import substrate
import substrate.language as S

# Problem sizes from target model.py
BATCH_SIZE = 128
IN_FEATURES = 16384
OUT_FEATURES = 16384
DROPOUT_P = 0.2

# GEMM tiling parameters
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 32
THREADS = BLOCK_M * BLOCK_N  # 256 threads

# Elementwise parameters
ELEMENTWISE_BLOCK = 256


@substrate.jit
def matmul_tiled_bf16_kernel(
    A: S.Pointer(S.bf16),
    B: S.Pointer(S.bf16),
    C: S.Pointer(S.bf16),
    m: S.u32,
    n: S.u32,
    k: S.u32,
    stride_am: S.u32,
    stride_ak: S.u32,
    stride_bk: S.u32,
    stride_bn: S.u32,
    stride_cm: S.u32,
    stride_cn: S.u32,
):
    layout_a = S.make_layout((m, k), (stride_am, stride_ak))
    layout_b = S.make_layout((k, n), (stride_bk, stride_bn))
    layout_c = S.make_layout((m, n), (stride_cm, stride_cn))

    gA = S.make_tensor(A, S.bf16, layout_a)
    gB = S.make_tensor(B, S.bf16, layout_b)
    gC = S.make_tensor(C, S.bf16, layout_c)

    tid = S.thread_id(0)
    block_n = S.block_id(0)
    block_m = S.block_id(1)

    local_row = tid // BLOCK_N
    local_col = tid % BLOCK_N

    row = block_m * BLOCK_M + local_row
    col = block_n * BLOCK_N + local_col

    shm_a_raw = S.make_shared((BLOCK_M * BLOCK_K,), S.bf16)
    shm_b_raw = S.make_shared((BLOCK_K * BLOCK_N,), S.bf16)

    shm_a = S.view(shm_a_raw, S.bf16, S.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K, 1)))
    shm_b = S.view(shm_b_raw, S.bf16, S.make_layout((BLOCK_K, BLOCK_N), (BLOCK_N, 1)))

    acc = S.convert(0.0, S.f32)
    k_tiles = (k + BLOCK_K - 1) // BLOCK_K

    for kt in S.range(k_tiles):
        k_base = kt * BLOCK_K

        # Load A tile
        idx_a = tid
        for _ in S.range((BLOCK_M * BLOCK_K + THREADS - 1) // THREADS):
            if idx_a < BLOCK_M * BLOCK_K:
                a_r = idx_a // BLOCK_K
                a_c = idx_a % BLOCK_K
                g_r = block_m * BLOCK_M + a_r
                g_c = k_base + a_c
                if g_r < m and g_c < k:
                    shm_a[a_r, a_c] = gA[g_r, g_c]
                else:
                    shm_a[a_r, a_c] = S.convert(0.0, S.bf16)
            idx_a = idx_a + THREADS

        # Load B tile
        idx_b = tid
        for _ in S.range((BLOCK_K * BLOCK_N + THREADS - 1) // THREADS):
            if idx_b < BLOCK_K * BLOCK_N:
                b_r = idx_b // BLOCK_N
                b_c = idx_b % BLOCK_N
                g_r = k_base + b_r
                g_c = block_n * BLOCK_N + b_c
                if g_r < k and g_c < n:
                    shm_b[b_r, b_c] = gB[g_r, g_c]
                else:
                    shm_b[b_r, b_c] = S.convert(0.0, S.bf16)
            idx_b = idx_b + THREADS

        S.syncthreads()

        # Compute
        if row < m and col < n:
            for kk in S.range(BLOCK_K):
                acc = acc + S.convert(shm_a[local_row, kk], S.f32) * S.convert(shm_b[kk, local_col], S.f32)

        S.syncthreads()

    if row < m and col < n:
        gC[row, col] = S.convert(acc, S.bf16)


@substrate.jit
def add_bias_bf16_kernel(
    x: S.Pointer(S.bf16),
    bias: S.Pointer(S.bf16),
    out: S.Pointer(S.bf16),
    m: S.u32,
    n: S.u32,
    stride_m: S.u32,
    stride_n: S.u32,
):
    layout = S.make_layout((m, n), (stride_m, stride_n))
    gx = S.make_tensor(x, S.bf16, layout)
    gout = S.make_tensor(out, S.bf16, layout)
    gbias = S.make_tensor(bias, S.bf16, S.make_layout((n,), (1,)))

    tid = S.thread_id(0)
    bid = S.block_id(0)
    col = bid * ELEMENTWISE_BLOCK + tid

    if col < n:
        for row in S.range(m):
            gout[row, col] = gx[row, col] + gbias[col]


def substrate_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Compute linear layer: x @ weight.T + bias using Substrate GEMM kernel."""
    if not x.is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device required for Substrate kernels.")
        x = x.cuda()

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch_size, in_features = x.shape
    out_features = weight.shape[0]

    # Convert to BF16 for optimized computation
    x_bf16 = x.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)

    # Output buffer
    out = torch.empty((batch_size, out_features), device=x.device, dtype=torch.bfloat16)

    # Launch GEMM: x @ weight.T
    # x: (M, K), weight.T: (K, N) -> out: (M, N)
    m = batch_size
    k = in_features
    n = out_features

    grid_x = (n + BLOCK_N - 1) // BLOCK_N
    grid_y = (m + BLOCK_M - 1) // BLOCK_M

    matmul_tiled_bf16_kernel[lambda: ((grid_x, grid_y, 1), (THREADS, 1, 1))](
        x_bf16,
        weight_bf16,
        out,
        m, n, k,
        x_bf16.stride(0), x_bf16.stride(1),
        weight_bf16.stride(1), weight_bf16.stride(0),  # weight is transposed
        out.stride(0), out.stride(1),
    )

    # Add bias
    bias_bf16 = bias.to(torch.bfloat16)
    if bias_bf16 is not None and bias_bf16.numel() > 0:
        bias_grid = (n + ELEMENTWISE_BLOCK - 1) // ELEMENTWISE_BLOCK
        add_bias_bf16_kernel[lambda: ((bias_grid, 1, 1), (ELEMENTWISE_BLOCK, 1, 1))](
            out, bias_bf16, out,
            m, n, out.stride(0), out.stride(1)
        )

    return out


def substrate_dropout(x: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:
    """Apply dropout during training."""
    if not training or p == 0.0:
        return x
    return torch.nn.functional.dropout(x, p=p, training=True)


def substrate_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute softmax - use PyTorch for correctness."""
    return torch.softmax(x, dim=dim)


class ModelNew(nn.Module):
    """
    Optimized model using Substrate GPU kernels for linear layer.
    """
    def __init__(self, in_features: int, out_features: int, dropout_p: float):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p

        # Store weight and bias as parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear: x @ W^T + b (using Substrate GEMM kernel)
        x = substrate_linear(x, self.weight, self.bias)

        # Dropout
        x = substrate_dropout(x, self.dropout_p, self.training)

        # Softmax over features (dim=1)
        x = substrate_softmax(x, dim=1)

        return x


batch_size = BATCH_SIZE
in_features = IN_FEATURES
out_features = OUT_FEATURES
dropout_p = DROPOUT_P


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, dropout_p]
