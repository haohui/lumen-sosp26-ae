import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Problem sizes
BATCH_SIZE = 1024
INPUT_SIZE = 8192
HIDDEN_SIZE = 8192
SCALING_FACTOR = 2.0

# GEMM tiling parameters
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 16
THREADS_X = 16
THREADS_Y = 16
THREADS = THREADS_X * THREADS_Y

LOADS_A = (BLOCK_M * BLOCK_K) // THREADS
LOADS_B = (BLOCK_K * BLOCK_N) // THREADS

# Elementwise kernel parameters
EW_THREADS = 256


@substrate.jit
def gemm_bf16_kernel(
    A: S.Pointer(S.bf16),
    B: S.Pointer(S.bf16),
    C: S.Pointer(S.bf16),
    m: S.u32,
    n: S.u32,
    k: S.u32,
):
    bid_n = S.block_id(0)
    bid_m = S.block_id(1)
    tid = S.thread_id(0)

    layout_a = S.make_layout((m, k), (k, 1))
    layout_b = S.make_layout((k, n), (n, 1))
    layout_c = S.make_layout((m, n), (n, 1))

    gA = S.make_tensor(A, S.bf16, layout_a)
    gB = S.make_tensor(B, S.bf16, layout_b)
    gC = S.make_tensor(C, S.bf16, layout_c)

    sA = S.make_shared((BLOCK_M, BLOCK_K), S.bf16)
    sB = S.make_shared((BLOCK_K, BLOCK_N), S.bf16)

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
                sA[a_r, a_c] = S.convert(0.0, S.bf16)

        for i in S.range(LOADS_B):
            idx = tid + i * THREADS
            b_r = idx // BLOCK_N
            b_c = idx % BLOCK_N
            g_r = k_base + b_r
            g_c = bid_n * BLOCK_N + b_c
            if g_r < k and g_c < n:
                sB[b_r, b_c] = gB[g_r, g_c]
            else:
                sB[b_r, b_c] = S.convert(0.0, S.bf16)

        S.syncthreads()

        for kk in S.range(BLOCK_K):
            a0 = S.convert(sA[local_r0, kk], S.f32)
            a1 = S.convert(sA[local_r1, kk], S.f32)
            b0 = S.convert(sB[kk, local_c0], S.f32)
            b1 = S.convert(sB[kk, local_c1], S.f32)
            acc00 = acc00 + a0 * b0
            acc01 = acc01 + a0 * b1
            acc10 = acc10 + a1 * b0
            acc11 = acc11 + a1 * b1

        S.syncthreads()

    if row0 < m and col0 < n:
        gC[row0, col0] = S.convert(acc00, S.bf16)
    if row0 < m and col1 < n:
        gC[row0, col1] = S.convert(acc01, S.bf16)
    if row1 < m and col0 < n:
        gC[row1, col0] = S.convert(acc10, S.bf16)
    if row1 < m and col1 < n:
        gC[row1, col1] = S.convert(acc11, S.bf16)


@substrate.jit
def sigmoid_scale_residual_bf16_kernel(
    gemm_out: S.Pointer(S.bf16),
    final_out: S.Pointer(S.bf16),
    n: S.u32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    layout = S.make_layout((n,), (1,))
    g_in = S.make_tensor(gemm_out, S.bf16, layout)
    g_out = S.make_tensor(final_out, S.bf16, layout)

    if idx < n:
        x_bf16 = g_in[idx]
        x = S.convert(x_bf16, S.f32)

        # sigmoid(x) = 0.5 * (1 + tanh(0.5 * x))
        half = S.convert(0.5, S.f32)
        one = S.convert(1.0, S.f32)
        half_x = half * x
        tanh_half_x = S.tanh(half_x)
        sigmoid_x = half * (one + tanh_half_x)

        # scale * sigmoid(x) + x
        scale = S.convert(SCALING_FACTOR, S.f32)
        result = scale * sigmoid_x + x

        g_out[idx] = S.convert(result, S.bf16)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def substrate_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    m, k = A.shape
    k2, n = B.shape
    assert k == k2

    out = torch.empty((m, n), device=A.device, dtype=torch.bfloat16)
    grid_x = _ceil_div(n, BLOCK_N)
    grid_y = _ceil_div(m, BLOCK_M)
    gemm_bf16_kernel[lambda: ((grid_x, grid_y, 1), (THREADS, 1, 1))](A, B, out, m, n, k)
    return out


def substrate_sigmoid_scale_residual(gemm_out: torch.Tensor) -> torch.Tensor:
    n = gemm_out.numel()
    out = torch.empty_like(gemm_out)
    grid = _ceil_div(n, EW_THREADS)
    sigmoid_scale_residual_bf16_kernel[lambda: ((grid, 1, 1), (EW_THREADS, 1, 1))](
        gemm_out, out, n
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model implementing Gemm_Sigmoid_Scaling_ResidualAdd pattern
    using Substrate GPU kernels.
    """

    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_device = x.device
        need_device_move = not x.is_cuda

        if need_device_move:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")
            x_dev = x.cuda()
        else:
            x_dev = x

        # Convert to BF16 for computation
        x_bf16 = x_dev.to(torch.bfloat16).contiguous()

        # GEMM: x @ weight.T + bias
        # weight is (hidden_size, input_size), weight.T is (input_size, hidden_size)
        weight_t = self.gemm.weight.t().contiguous()
        gemm_out = substrate_gemm(x_bf16, weight_t)

        # Add bias
        if self.gemm.bias is not None:
            gemm_out = gemm_out + self.gemm.bias.to(torch.bfloat16)

        # Fused: sigmoid, scale, residual add
        out = substrate_sigmoid_scale_residual(gemm_out)

        if need_device_move:
            out = out.to(original_device)

        return out


batch_size = BATCH_SIZE
input_size = INPUT_SIZE
hidden_size = HIDDEN_SIZE
scaling_factor = SCALING_FACTOR


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
