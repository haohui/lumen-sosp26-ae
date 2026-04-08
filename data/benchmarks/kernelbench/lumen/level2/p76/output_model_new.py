import torch
import torch.nn as nn
import torch.nn.init as init
import math
import substrate
import substrate.language as S

# Problem sizes from target model.py
BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192

# Tiling parameters for GEMM
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 32
THREADS = BLOCK_M * BLOCK_N  # 256 threads


def _kaiming_uniform_like_linear(tensor):
    """Initialize tensor like nn.Linear does (Kaiming uniform with a=sqrt(5))."""
    init.kaiming_uniform_(tensor, a=math.sqrt(5))


def _bias_init_like_linear(bias, in_features):
    """Initialize bias like nn.Linear does."""
    fan_in = in_features
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    init.uniform_(bias, -bound, bound)


@substrate.jit
def gemm_bias_relu_bf16_kernel(
    A: S.Pointer(S.bf16),
    B: S.Pointer(S.bf16),
    bias: S.Pointer(S.bf16),
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
    """Fused GEMM + bias + ReLU kernel for BF16."""
    layout_a = S.make_layout((m, k), (stride_am, stride_ak))
    layout_b = S.make_layout((k, n), (stride_bk, stride_bn))
    layout_c = S.make_layout((m, n), (stride_cm, stride_cn))
    layout_bias = S.make_layout((n,), (1,))

    gA = S.make_tensor(A, S.bf16, layout_a)
    gB = S.make_tensor(B, S.bf16, layout_b)
    gC = S.make_tensor(C, S.bf16, layout_c)
    gBias = S.make_tensor(bias, S.bf16, layout_bias)

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

        # Compute partial GEMM
        if row < m and col < n:
            for kk in S.range(BLOCK_K):
                a_val = S.convert(shm_a[local_row, kk], S.f32)
                b_val = S.convert(shm_b[kk, local_col], S.f32)
                acc = acc + a_val * b_val

        S.syncthreads()

    # Fused bias add and ReLU
    if row < m and col < n:
        bias_val = S.convert(gBias[col], S.f32)
        result = acc + bias_val
        # ReLU: max(0, result)
        zero = S.convert(0.0, S.f32)
        if result < zero:
            result = zero
        gC[row, col] = S.convert(result, S.bf16)


@substrate.jit
def gemm_bias_relu_f32_kernel(
    A: S.Pointer(S.f32),
    B: S.Pointer(S.f32),
    bias: S.Pointer(S.f32),
    C: S.Pointer(S.f32),
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
    """Fused GEMM + bias + ReLU kernel for F32."""
    layout_a = S.make_layout((m, k), (stride_am, stride_ak))
    layout_b = S.make_layout((k, n), (stride_bk, stride_bn))
    layout_c = S.make_layout((m, n), (stride_cm, stride_cn))
    layout_bias = S.make_layout((n,), (1,))

    gA = S.make_tensor(A, S.f32, layout_a)
    gB = S.make_tensor(B, S.f32, layout_b)
    gC = S.make_tensor(C, S.f32, layout_c)
    gBias = S.make_tensor(bias, S.f32, layout_bias)

    tid = S.thread_id(0)
    block_n = S.block_id(0)
    block_m = S.block_id(1)

    local_row = tid // BLOCK_N
    local_col = tid % BLOCK_N

    row = block_m * BLOCK_M + local_row
    col = block_n * BLOCK_N + local_col

    shm_a_raw = S.make_shared((BLOCK_M * BLOCK_K,), S.f32)
    shm_b_raw = S.make_shared((BLOCK_K * BLOCK_N,), S.f32)

    shm_a = S.view(shm_a_raw, S.f32, S.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K, 1)))
    shm_b = S.view(shm_b_raw, S.f32, S.make_layout((BLOCK_K, BLOCK_N), (BLOCK_N, 1)))

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
                    shm_a[a_r, a_c] = S.convert(0.0, S.f32)
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
                    shm_b[b_r, b_c] = S.convert(0.0, S.f32)
            idx_b = idx_b + THREADS

        S.syncthreads()

        # Compute partial GEMM
        if row < m and col < n:
            for kk in S.range(BLOCK_K):
                acc = acc + shm_a[local_row, kk] * shm_b[kk, local_col]

        S.syncthreads()

    # Fused bias add and ReLU
    if row < m and col < n:
        result = acc + gBias[col]
        # ReLU: max(0, result)
        zero = S.convert(0.0, S.f32)
        if result < zero:
            result = zero
        gC[row, col] = result


def _launch_grid(m: int, n: int):
    grid_x = (n + BLOCK_N - 1) // BLOCK_N
    grid_y = (m + BLOCK_M - 1) // BLOCK_M
    return (grid_x, grid_y, 1)


def substrate_gemm_bias_relu(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Fused GEMM + bias + ReLU using Substrate kernels.

    GEMM layout: C[M, N] = A[M, K] @ B[K, N]
    - A = x, shape [batch_size, in_features] = [M, K]
    - B = weight.T, shape [in_features, out_features] = [K, N]
    - C = output, shape [batch_size, out_features] = [M, N]
    """
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError("x and weight must be 2D tensors.")
    if x.shape[1] != weight.shape[1]:
        raise ValueError(
            f"Incompatible shapes for GEMM: x {x.shape}, weight {weight.shape}"
        )
    if bias.shape[0] != weight.shape[0]:
        raise ValueError(
            f"Bias shape {bias.shape} incompatible with weight output dim {weight.shape[0]}"
        )
    if not x.is_cuda:
        raise ValueError("substrate_gemm_bias_relu expects CUDA/HIP tensors.")

    x = x.contiguous()
    # Transpose weight: nn.Linear stores weight as [out_features, in_features]
    # but GEMM expects B as [K, N] = [in_features, out_features]
    weight_t = weight.t().contiguous()
    bias = bias.contiguous()

    m = x.shape[0]  # batch_size
    k = x.shape[1]  # in_features
    n = weight.shape[0]  # out_features

    grid = _launch_grid(m, n)

    if x.dtype == torch.bfloat16 and weight_t.dtype == torch.bfloat16:
        out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        gemm_bias_relu_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](
            x,
            weight_t,
            bias,
            out,
            int(m),
            int(n),
            int(k),
            int(x.stride(0)),
            int(x.stride(1)),
            int(weight_t.stride(0)),
            int(weight_t.stride(1)),
            int(out.stride(0)),
            int(out.stride(1)),
        )
        return out

    if x.dtype == torch.float32 and weight_t.dtype == torch.float32:
        out = torch.empty((m, n), device=x.device, dtype=torch.float32)
        gemm_bias_relu_f32_kernel[lambda: (grid, (THREADS, 1, 1))](
            x,
            weight_t,
            bias,
            out,
            int(m),
            int(n),
            int(k),
            int(x.stride(0)),
            int(x.stride(1)),
            int(weight_t.stride(0)),
            int(weight_t.stride(1)),
            int(out.stride(0)),
            int(out.stride(1)),
        )
        return out

    # Promote other types to float32
    x32 = x.to(torch.float32).contiguous()
    w32 = weight_t.to(torch.float32).contiguous()
    b32 = bias.to(torch.float32).contiguous()
    out32 = torch.empty((m, n), device=x.device, dtype=torch.float32)
    gemm_bias_relu_f32_kernel[lambda: (grid, (THREADS, 1, 1))](
        x32,
        w32,
        b32,
        out32,
        int(m),
        int(n),
        int(k),
        int(x32.stride(0)),
        int(x32.stride(1)),
        int(w32.stride(0)),
        int(w32.stride(1)),
        int(out32.stride(0)),
        int(out32.stride(1)),
    )
    return out32.to(torch.promote_types(x.dtype, weight.dtype))


class ModelNew(nn.Module):
    """
    Substrate-optimized model performing GEMM + bias + ReLU.
    """

    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Match exactly the initialization order from original Model:
        # 1. nn.Linear creates weight with kaiming_uniform_
        # 2. torch.randn creates bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        _kaiming_uniform_like_linear(self.weight)  # Consumes RNG like nn.Linear does
        self.bias = nn.Parameter(torch.randn(bias_shape))  # Consumes RNG after weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        if not x.is_cuda:
            x = x.cuda()

        return substrate_gemm_bias_relu(x, self.weight, self.bias)


batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, bias_shape]
