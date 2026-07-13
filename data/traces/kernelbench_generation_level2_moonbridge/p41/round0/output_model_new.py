import torch
import torch.nn as nn
import avelang
import avelang.language as al

batch_size = 16384
in_features = 4096
out_features = 4096

# =============================================================================
# GEMM: shared-memory tiled, no tensor cores (MFMA unavailable on this backend)
# =============================================================================
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
GEMM_THREADS = 256
THREAD_M = 16
THREAD_N = 16
M_PER_THREAD = BLOCK_M // THREAD_M
N_PER_THREAD = BLOCK_N // THREAD_N

# =============================================================================
# BN + GELU + ReLU kernel constants
# =============================================================================
BN_BLOCK_SIZE = 256


@avelang.jit
def gemm_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    m: al.i32,
    n: al.i32,
    k: al.i32,
):
    tid = al.thread_id(0)
    block_n_idx = al.block_id(0)
    block_m_idx = al.block_id(1)

    thread_m = tid // THREAD_N
    thread_n = tid % THREAD_N

    # Views into global memory
    x_memref = al.make_tensor(x_ptr, al.bf16, al.make_layout((m, k), (k, 1)))
    w_memref = al.make_tensor(w_ptr, al.bf16, al.make_layout((n, k), (k, 1)))
    g_bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((n,), (1,)))
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    # Shared memory tiles
    shm_a = al.make_shared((BLOCK_M, BLOCK_K), al.bf16)
    shm_b = al.make_shared((BLOCK_K, BLOCK_N), al.bf16)

    # Register accumulator: each thread owns M_PER_THREAD x N_PER_THREAD outputs
    acc = al.make_local((M_PER_THREAD, N_PER_THREAD), al.f32)
    for mi in al.range(M_PER_THREAD):
        for ni in al.range(N_PER_THREAD):
            acc[mi, ni] = al.convert(0.0, al.f32)

    k_tiles = k // BLOCK_K
    for kt in al.range(k_tiles):
        k_off = kt * BLOCK_K

        # Load A tile [block_m*BM : (block_m+1)*BM, k_off : k_off+BK] into shm_a
        a_elems = BLOCK_M * BLOCK_K
        a_idx = tid
        for _ in al.range(a_elems // GEMM_THREADS):
            a_row = a_idx // BLOCK_K
            a_col = a_idx % BLOCK_K
            global_row = block_m_idx * BLOCK_M + a_row
            global_col = k_off + a_col
            shm_a[a_row, a_col] = x_memref[global_row, global_col]
            a_idx += GEMM_THREADS

        # Load B tile [block_n*BN : (block_n+1)*BN, k_off : k_off+BK] into shm_b
        # w is (n, k); we want shm_b[k_local, n_local] = w[block_n*BN + n_local, k_off + k_local]
        b_elems = BLOCK_N * BLOCK_K
        b_idx = tid
        for _ in al.range(b_elems // GEMM_THREADS):
            b_row = b_idx // BLOCK_N
            b_col = b_idx % BLOCK_N
            global_w_row = block_n_idx * BLOCK_N + b_col
            global_w_col = k_off + b_row
            shm_b[b_row, b_col] = w_memref[global_w_row, global_w_col]
            b_idx += GEMM_THREADS

        al.syncthreads()

        # Accumulate: for each kk in BLOCK_K, dot thread's A rows with B cols
        for kk in al.range(BLOCK_K):
            for mi in al.range(M_PER_THREAD):
                a_val = al.convert(shm_a[thread_m * M_PER_THREAD + mi, kk], al.f32)
                for ni in al.range(N_PER_THREAD):
                    b_val = al.convert(shm_b[kk, thread_n * N_PER_THREAD + ni], al.f32)
                    acc[mi, ni] = acc[mi, ni] + a_val * b_val

        al.syncthreads()

    # Write back with bias
    for mi in al.range(M_PER_THREAD):
        global_row = block_m_idx * BLOCK_M + thread_m * M_PER_THREAD + mi
        for ni in al.range(N_PER_THREAD):
            global_col = block_n_idx * BLOCK_N + thread_n * N_PER_THREAD + ni
            bias_val = al.convert(g_bias[global_col], al.f32)
            result = acc[mi, ni] + bias_val
            g_out[global_row, global_col] = al.convert(result, al.bf16)


# =============================================================================
# BatchNorm (eval mode, uses provided running stats) + GELU + ReLU
# Each CTA handles one feature column across the batch dimension.
# =============================================================================
@avelang.jit
def bn_gelu_relu_kernel(
    in_out_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    running_mean_ptr: al.Pointer(al.bf16),
    running_var_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    n_features: al.i32,
    eps: al.f32,
):
    tid = al.thread_id(0)
    col = al.block_id(0)

    if col < n_features:
        io_layout = al.make_layout((batch_size * n_features,), (1,))
        io = al.make_tensor(in_out_ptr, al.bf16, io_layout)

        param_layout = al.make_layout((n_features,), (1,))
        g = al.make_tensor(gamma_ptr, al.bf16, param_layout)
        b = al.make_tensor(beta_ptr, al.bf16, param_layout)

        gamma_val = al.convert(g[col], al.f32)
        beta_val = al.convert(b[col], al.f32)
        rm = al.make_tensor(running_mean_ptr, al.bf16, param_layout)
        rv = al.make_tensor(running_var_ptr, al.bf16, param_layout)
        running_mean = al.convert(rm[col], al.f32)
        running_var = al.convert(rv[col], al.f32)
        rstd_val = al.convert(1.0, al.f32) / al.sqrt(running_var + eps)

        # ---- Apply BN (eval mode) + GELU (tanh approx) + ReLU ----
        sqrt_2_over_pi = al.convert(0.7978845608028654, al.f32)
        coeff = al.convert(0.044715, al.f32)
        half = al.convert(0.5, al.f32)
        one = al.convert(1.0, al.f32)
        zero = al.convert(0.0, al.f32)

        for row in al.range(tid, batch_size, BN_BLOCK_SIZE):
            idx = row * n_features + col
            val = al.convert(io[idx], al.f32)

            normalized = (val - running_mean) * rstd_val
            y = normalized * gamma_val + beta_val

            y_cube = y * y * y
            inner = sqrt_2_over_pi * (y + coeff * y_cube)
            gelu_val = half * y * (one + al.tanh(inner))

            if gelu_val < zero:
                gelu_val = zero

            io[idx] = al.convert(gelu_val, al.bf16)


# =============================================================================
# Host helpers
# =============================================================================
def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight)
    bias_bf16 = _prepare_bf16_cuda_contiguous(bias)

    m, k = x_bf16.shape
    n_val, weight_k = weight_bf16.shape

    out = torch.empty((m, n_val), device=x_bf16.device, dtype=torch.bfloat16)
    grid = (n_val // BLOCK_N, m // BLOCK_M, 1)
    gemm_bf16_kernel[lambda: (grid, (GEMM_THREADS, 1, 1))](
        x_bf16, weight_bf16, bias_bf16, out, m, n_val, k
    )
    return out


def avelang_bn_gelu_relu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    x_bf16 = x if (x.is_contiguous() and x.dtype == torch.bfloat16) else x.contiguous().to(dtype=torch.bfloat16)
    gamma_bf16 = _prepare_bf16_cuda_contiguous(gamma)
    beta_bf16 = _prepare_bf16_cuda_contiguous(beta)
    rm_bf16 = _prepare_bf16_cuda_contiguous(running_mean)
    rv_bf16 = _prepare_bf16_cuda_contiguous(running_var)

    batch_size_val, n_features_val = x_bf16.shape

    grid = (n_features_val, 1, 1)
    bn_gelu_relu_kernel[lambda: (grid, (BN_BLOCK_SIZE, 1, 1))](
        x_bf16, gamma_bf16, beta_bf16, rm_bf16, rv_bf16,
        batch_size_val, n_features_val, eps
    )
    return x_bf16


# =============================================================================
# ModelNew
# =============================================================================
class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.bn_weight = nn.Parameter(torch.empty(out_features))
        self.bn_bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer("running_mean", torch.zeros(out_features))
        self.register_buffer("running_var", torch.ones(out_features))

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.ones_(self.bn_weight)
        nn.init.zeros_(self.bn_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        gemm_out = avelang_gemm(x, self.weight, self.bias)
        result = avelang_bn_gelu_relu(gemm_out, self.bn_weight, self.bn_bias,
                                      self.running_mean, self.running_var)
        if orig_dtype != result.dtype:
            result = result.to(orig_dtype)
        return result


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
