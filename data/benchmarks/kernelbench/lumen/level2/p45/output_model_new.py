import torch
import torch.nn as nn
import substrate
import substrate.language as S

# Problem sizes
BATCH_SIZE = 16384
INPUT_SIZE = 2048
HIDDEN_SIZE = 4096
OUTPUT_SIZE = 1024

# Tiling parameters for GEMM
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 32
THREADS = BLOCK_M * BLOCK_N

# Activation constants
LOG2E = 1.4426950408889634  # log2(e) for exp(x) = exp2(x * log2(e))

# Block size for elementwise and reduction kernels
BLOCK_SIZE = 256


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

        if row < m and col < n:
            for kk in S.range(BLOCK_K):
                acc = acc + S.convert(shm_a[local_row, kk], S.f32) * S.convert(shm_b[kk, local_col], S.f32)

        S.syncthreads()

    if row < m and col < n:
        gC[row, col] = S.convert(acc, S.bf16)


@substrate.jit
def add_bias_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    bias_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    m: S.u32,
    n: S.u32,
):
    layout = S.make_layout((m, n), (n, 1))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    bias = S.make_tensor(bias_ptr, S.bf16, S.make_layout((n,), (1,)))
    out = S.make_tensor(out_ptr, S.bf16, layout)

    row = S.block_id(1)
    col = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if row < m and col < n:
        out[row, col] = x[row, col] + bias[col]


@substrate.jit
def sigmoid_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    layout = S.make_layout((n,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    out = S.make_tensor(out_ptr, S.bf16, layout)

    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx < n:
        xv = S.convert(x[idx], S.f32)
        zero = S.convert(0.0, S.f32)
        one = S.convert(1.0, S.f32)
        log2e = S.convert(LOG2E, S.f32)

        # sigmoid(x) = 1 / (1 + exp(-x)) = 1 / (1 + 2^(-x * log2(e)))
        neg_x = zero - xv
        exp_arg = neg_x * log2e
        exp_val = S.exp2(exp_arg)
        denom = one + exp_val
        result = one / denom

        out[idx] = S.convert(result, S.bf16)


@substrate.jit
def logsumexp_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    m: S.u32,
    n: S.u32,
):
    layout = S.make_layout((m, n), (n, 1))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    out = S.make_tensor(out_ptr, S.bf16, S.make_layout((m,), (1,)))

    row = S.block_id(0)

    if row < m:
        # Find max value in the row
        max_val = S.convert(x[row, 0], S.f32)
        for col in S.range(1, n):
            val = S.convert(x[row, col], S.f32)
            if val > max_val:
                max_val = val

        # Compute sum of exp(x - max)
        log2e = S.convert(LOG2E, S.f32)
        sum_exp = S.convert(0.0, S.f32)

        for col in S.range(n):
            val = S.convert(x[row, col], S.f32)
            shifted = val - max_val
            exp_arg = shifted * log2e
            exp_val = S.exp2(exp_arg)
            sum_exp = sum_exp + exp_val

        # logsumexp = max + log(sum_exp)
        log_sum = S.log(sum_exp)
        result = max_val + log_sum

        out[row] = S.convert(result, S.bf16)


def substrate_matmul_bias(A: torch.Tensor, W: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Compute A @ W.T + bias for nn.Linear

    A: (m, k) - input matrix
    W: (n, k) - weight matrix (transposed in the computation)
    bias: (n,) - bias vector

    Computes: C = A @ W.T + bias
    """
    m, k = A.shape
    n = W.shape[0]  # W is (out_features, in_features) = (n, k)

    # Ensure BF16 for kernel
    A_bf16 = A.to(torch.bfloat16).contiguous()
    W_bf16 = W.to(torch.bfloat16).contiguous()
    bias_bf16 = bias.to(torch.bfloat16).contiguous()

    # Allocate output
    C = torch.empty((m, n), device=A.device, dtype=torch.bfloat16)

    # Launch GEMM
    # For A @ W.T where W is (n, k):
    # We compute C[i,j] = sum_k A[i,k] * W[j,k]
    # The kernel expects B[k,j] but we want W[j,k]
    # So we need to transpose the strides:
    # - stride for k dimension = W.stride(1) = 1
    # - stride for n dimension = W.stride(0) = k
    grid_x = (n + BLOCK_N - 1) // BLOCK_N
    grid_y = (m + BLOCK_M - 1) // BLOCK_M

    matmul_tiled_bf16_kernel[lambda: ((grid_x, grid_y, 1), (THREADS, 1, 1))](
        A_bf16, W_bf16, C,
        m, n, k,
        int(A_bf16.stride(0)), int(A_bf16.stride(1)),  # A strides: (k, 1) for row-major
        int(W_bf16.stride(1)), int(W_bf16.stride(0)),  # W strides SWAPPED for transpose
        int(C.stride(0)), int(C.stride(1)),
    )

    # Add bias in-place
    bias_grid_x = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    add_bias_bf16_kernel[lambda: ((bias_grid_x, m, 1), (BLOCK_SIZE, 1, 1))](
        C, bias_bf16, C, m, n
    )

    return C


def substrate_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Compute sigmoid elementwise"""
    x_bf16 = x.to(torch.bfloat16).contiguous()
    out = torch.empty_like(x_bf16)
    n = x_bf16.numel()

    if n > 0:
        grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
        sigmoid_bf16_kernel[lambda: ((grid, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_bf16, out, n
        )

    return out


def substrate_logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute logsumexp along dim=1"""
    if dim != 1:
        raise NotImplementedError("Only dim=1 supported")

    m, n = x.shape
    x_bf16 = x.to(torch.bfloat16).contiguous()
    out = torch.empty((m,), device=x.device, dtype=torch.bfloat16)

    logsumexp_bf16_kernel[lambda: ((m, 1, 1), (1, 1, 1))](x_bf16, out, m, n)

    return out


class ModelNew(nn.Module):
    """
    Optimized model using Substrate GPU kernels:
    Linear1 -> Sigmoid -> Linear2 -> LogSumExp
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Use nn.Linear to match reference parameter naming and initialization
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move input to GPU if needed
        src_device = x.device
        if not x.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA/HIP device required")
            x = x.cuda()

        # Linear1: x @ W1.T + b1
        x = substrate_matmul_bias(x, self.linear1.weight, self.linear1.bias)

        # Sigmoid activation
        x = substrate_sigmoid(x)

        # Linear2: x @ W2.T + b2
        x = substrate_matmul_bias(x, self.linear2.weight, self.linear2.bias)

        # LogSumExp over dim=1
        x = substrate_logsumexp(x, dim=1)

        return x


batch_size = BATCH_SIZE
input_size = INPUT_SIZE
hidden_size = HIDDEN_SIZE
output_size = OUTPUT_SIZE


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, output_size]
