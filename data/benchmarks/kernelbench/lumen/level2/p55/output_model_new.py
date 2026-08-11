import torch
import torch.nn as nn
import substrate
import substrate.language as S

# Problem sizes from input_model.py
BATCH_SIZE = 128
IN_FEATURES = 32768
OUT_FEATURES = 32768
KERNEL_SIZE = 2
SCALE_FACTOR = 0.5

# After maxpool with kernel_size=2, stride=2
POOL_OUT_LEN = OUT_FEATURES // 2  # 16384

# GEMM tiling parameters
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 32
THREADS = BLOCK_M * BLOCK_N  # 256 threads


# ============= GEMM Kernel for Linear Layer =============

@substrate.jit
def matmul_linear_bf16_kernel(
    A: S.Pointer(S.bf16),
    W: S.Pointer(S.bf16),
    C: S.Pointer(S.bf16),
    m: S.u32,
    n: S.u32,
    k: S.u32,
    stride_am: S.u32,
    stride_ak: S.u32,
    stride_wk: S.u32,
    stride_wn: S.u32,
    stride_cm: S.u32,
    stride_cn: S.u32,
):
    layout_a = S.make_layout((m, k), (stride_am, stride_ak))
    layout_w = S.make_layout((k, n), (stride_wk, stride_wn))
    layout_c = S.make_layout((m, n), (stride_cm, stride_cn))

    gA = S.make_tensor(A, S.bf16, layout_a)
    gW = S.make_tensor(W, S.bf16, layout_w)
    gC = S.make_tensor(C, S.bf16, layout_c)

    tid = S.thread_id(0)
    block_n = S.block_id(0)
    block_m = S.block_id(1)

    local_row = tid // BLOCK_N
    local_col = tid % BLOCK_N

    row = block_m * BLOCK_M + local_row
    col = block_n * BLOCK_N + local_col

    shm_a_raw = S.make_shared((BLOCK_M * BLOCK_K,), S.bf16)
    shm_w_raw = S.make_shared((BLOCK_K * BLOCK_N,), S.bf16)

    shm_a = S.view(shm_a_raw, S.bf16, S.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K, 1)))
    shm_w = S.view(shm_w_raw, S.bf16, S.make_layout((BLOCK_K, BLOCK_N), (BLOCK_N, 1)))

    acc = S.convert(0.0, S.f32)
    k_tiles = (k + BLOCK_K - 1) // BLOCK_K

    for kt in S.range(k_tiles):
        k_base = kt * BLOCK_K

        # Load A tile (column-major in shared memory)
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

        # Load W tile
        idx_w = tid
        for _ in S.range((BLOCK_K * BLOCK_N + THREADS - 1) // THREADS):
            if idx_w < BLOCK_K * BLOCK_N:
                w_r = idx_w // BLOCK_N
                w_c = idx_w % BLOCK_N
                g_r = k_base + w_r
                g_c = block_n * BLOCK_N + w_c
                if g_r < k and g_c < n:
                    shm_w[w_r, w_c] = gW[g_r, g_c]
                else:
                    shm_w[w_r, w_c] = S.convert(0.0, S.bf16)
            idx_w = idx_w + THREADS

        S.syncthreads()

        if row < m and col < n:
            for kk in S.range(BLOCK_K):
                acc = acc + S.convert(shm_a[local_row, kk], S.f32) * S.convert(shm_w[kk, local_col], S.f32)

        S.syncthreads()

    if row < m and col < n:
        gC[row, col] = S.convert(acc, S.bf16)


# ============= Add Bias Kernel =============

@substrate.jit
def add_bias_bf16_kernel(
    x: S.Pointer(S.bf16),
    bias: S.Pointer(S.bf16),
    out: S.Pointer(S.bf16),
    total: S.u32,
    n: S.u32,
):
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if tid < total:
        layout_x = S.make_layout((total // n, n), (n, 1))
        layout_bias = S.make_layout((n,), (1,))
        g_x = S.make_tensor(x, S.bf16, layout_x)
        g_bias = S.make_tensor(bias, S.bf16, layout_bias)
        g_out = S.make_tensor(out, S.bf16, layout_x)

        row = tid // n
        col = tid - row * n

        v = g_x[row, col]
        b = g_bias[col]
        g_out[row, col] = v + b


# ============= MaxPool1d Kernel =============

NEG_INF_F32 = -3.4028234663852886e38


@substrate.jit
def maxpool1d_bf16_kernel(
    x: S.Pointer(S.bf16),
    y: S.Pointer(S.bf16),
    batch_size: S.u32,
    channels: S.u32,
    in_len: S.u32,
    out_len: S.u32,
    kernel_size: S.u32,
    stride: S.u32,
):
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = batch_size * channels * out_len

    if tid < total:
        layout_x = S.make_layout((batch_size * channels, in_len), (in_len, 1))
        layout_y = S.make_layout((batch_size * channels, out_len), (out_len, 1))
        g_x = S.make_tensor(x, S.bf16, layout_x)
        g_y = S.make_tensor(y, S.bf16, layout_y)

        ow = tid % out_len
        remainder = tid // out_len
        bc = remainder

        iw_start = S.convert(ow, S.i32) * S.convert(stride, S.i32)
        in_len_i32 = S.convert(in_len, S.i32)

        maxv = S.convert(NEG_INF_F32, S.f32)

        for kk in S.range(KERNEL_SIZE):
            iw = iw_start + S.convert(kk, S.i32)
            if iw < in_len_i32:
                v = S.convert(g_x[bc, S.convert(iw, S.u32)], S.f32)
                maxv = v if v > maxv else maxv

        g_y[bc, ow] = S.convert(maxv, S.bf16)


# ============= Sum + Scale Kernel =============
# Scale factor is fixed at 0.5, so we use constexpr

SCALE_INV = 2.0  # 1 / 0.5 = 2.0 for division instead of multiplication


@substrate.jit
def sum_scale_bf16_kernel(
    x: S.Pointer(S.bf16),
    out: S.Pointer(S.bf16),
    batch_size: S.u32,
    dim1: S.u32,
):
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if tid < batch_size:
        layout_x = S.make_layout((batch_size, dim1), (dim1, 1))
        layout_out = S.make_layout((batch_size,), (1,))
        g_x = S.make_tensor(x, S.bf16, layout_x)
        g_out = S.make_tensor(out, S.bf16, layout_out)

        acc = S.convert(0.0, S.f32)
        for c in S.range(dim1):
            acc = acc + S.convert(g_x[tid, c], S.f32)

        # scale = 0.5, so acc * 0.5 = acc / 2.0
        g_out[tid] = S.convert(acc / S.convert(SCALE_INV, S.f32), S.bf16)


# ============= Host Wrappers =============

def _launch_gemm_linear(A: torch.Tensor, W: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Perform A @ W.T + bias for Linear layer."""
    m, k = A.shape  # (128, 32768)
    n = W.shape[0]  # (32768,) - W is (out_features, in_features)

    # W is stored as (out_features, in_features) = (n, k)
    # We need A @ W.T which is (m, k) @ (k, n)
    # W.T would be (in_features, out_features) = (k, n)
    # But we can pass W as (k, n) by treating W.T implicitly

    grid_x = (n + BLOCK_N - 1) // BLOCK_N
    grid_y = (m + BLOCK_M - 1) // BLOCK_M

    C = torch.empty((m, n), device=A.device, dtype=torch.bfloat16)

    # A is (m, k), W.T is (k, n) - we need to pass W.T
    # PyTorch Linear: output = input @ weight.T + bias
    # So W is (out_features, in_features) = (n, k), W.T is (k, n)
    W_T = W.T.contiguous()

    matmul_linear_bf16_kernel[lambda: ((grid_x, grid_y, 1), (THREADS, 1, 1))](
        A,
        W_T,
        C,
        int(m),
        int(n),
        int(k),
        int(A.stride(0)),
        int(A.stride(1)),
        int(W_T.stride(0)),
        int(W_T.stride(1)),
        int(C.stride(0)),
        int(C.stride(1)),
    )

    # Add bias
    total = m * n
    threads = 256
    grid_bias = (total + threads - 1) // threads
    add_bias_bf16_kernel[lambda: ((grid_bias, 1, 1), (threads, 1, 1))](
        C, bias, C, total, n
    )

    return C


def _launch_maxpool1d(x: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """MaxPool1d on (batch, channels, length) tensor."""
    batch, channels, in_len = x.shape
    out_len = (in_len - kernel_size) // stride + 1

    out = torch.empty((batch, channels, out_len), device=x.device, dtype=x.dtype)

    total = batch * channels * out_len
    threads = 256
    grid = (total + threads - 1) // threads

    maxpool1d_bf16_kernel[lambda: ((grid, 1, 1), (threads, 1, 1))](
        x, out, batch, channels, in_len, out_len, kernel_size, stride
    )

    return out


def _launch_sum_scale(x: torch.Tensor) -> torch.Tensor:
    """Sum over dim=1 and scale by 0.5."""
    batch_size, dim1 = x.shape
    out = torch.empty((batch_size,), device=x.device, dtype=x.dtype)

    threads = 256
    grid = (batch_size + threads - 1) // threads

    sum_scale_bf16_kernel[lambda: ((grid, 1, 1), (threads, 1, 1))](
        x, out, batch_size, dim1
    )

    return out


# ============= ModelNew =============

class ModelNew(nn.Module):
    """
    Optimized Substrate implementation of matmul -> maxpool -> sum -> scale.
    """
    def __init__(self, in_features: int, out_features: int, kernel_size: int, scale_factor: float):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

        # Initialize weights and bias like nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        src_device = x.device

        if not x.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA/HIP device required for Substrate kernels.")
            x = x.cuda()

        x = x.contiguous()

        # Convert to bf16 if needed
        input_dtype = x.dtype
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        # Step 1: Linear (matmul + bias)
        x = _launch_gemm_linear(x, self.weight, self.bias)

        # Step 2: MaxPool1d
        # x has shape (batch, out_features) = (128, 32768)
        # unsqueeze to (batch, 1, out_features) for MaxPool1d
        x = x.unsqueeze(1)  # (128, 1, 32768)
        x = _launch_maxpool1d(x, self.kernel_size, self.kernel_size)  # stride = kernel_size
        x = x.squeeze(1)  # (128, 16384)

        # Step 3: Sum + Scale
        x = _launch_sum_scale(x)

        # Convert back to original dtype if needed
        if input_dtype != torch.bfloat16:
            x = x.to(input_dtype)

        if src_device.type != "cuda":
            return x.to(src_device)

        return x


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_FEATURES)]


def get_init_inputs():
    return [IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, SCALE_FACTOR]
