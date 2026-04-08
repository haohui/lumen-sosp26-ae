import torch
import torch.nn as nn
import substrate
import substrate.language as S

# Problem sizes
BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192

# Tiling for GEMM
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 32
THREADS = BLOCK_M * BLOCK_N
K_TILES = IN_FEATURES // BLOCK_K

# Constants for activation functions
LOG2E = 1.4426950408889634  # 1 / ln(2)
SQRT2 = 1.4142135623730951
INV_SQRT2 = 0.7071067811865475  # 1 / sqrt(2)


@substrate.jit
def linear_activation_chain_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    w_ptr: S.Pointer(S.bf16),
    bias_ptr: S.Pointer(S.bf16),
    add_val_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    m: S.u32,
    n: S.u32,
    k: S.u32,
):
    """
    Fused kernel for: Linear(x) + add_val -> Swish -> Tanh -> GELU -> Hardtanh

    Input x: [m, k]
    Weight W: [n, k] (stored transposed for row-major access)
    Bias: [n]
    Add value: [n]
    Output: [m, n]
    """
    tid = S.thread_id(0)
    bid_n = S.block_id(0)
    bid_m = S.block_id(1)

    # Local coordinates in tile
    local_m = tid // BLOCK_N
    local_n = tid % BLOCK_N

    # Global coordinates
    row = bid_m * BLOCK_M + local_m
    col = bid_n * BLOCK_N + local_n

    # Create tensor views
    layout_x = S.make_layout((m, k), (k, 1))
    layout_w = S.make_layout((n, k), (k, 1))
    layout_out = S.make_layout((m, n), (n, 1))
    layout_vec = S.make_layout((n,), (1,))

    x = S.make_tensor(x_ptr, S.bf16, layout_x)
    w = S.make_tensor(w_ptr, S.bf16, layout_w)
    bias = S.make_tensor(bias_ptr, S.bf16, layout_vec)
    add_val = S.make_tensor(add_val_ptr, S.bf16, layout_vec)
    out = S.make_tensor(out_ptr, S.bf16, layout_out)

    # Shared memory for tiles
    sA = S.make_shared((BLOCK_M, BLOCK_K), S.bf16)
    sB = S.make_shared((BLOCK_N, BLOCK_K), S.bf16)

    # FP32 accumulation for matmul
    acc = S.convert(0.0, S.f32)

    # Load counts per thread
    A_LOADS = (BLOCK_M * BLOCK_K) // THREADS
    B_LOADS = (BLOCK_N * BLOCK_K) // THREADS

    # Matmul: output = x @ W.T
    for kt in S.range(K_TILES):
        k_base = kt * BLOCK_K

        # Load A tile (x)
        for i in S.range(A_LOADS):
            idx = tid + i * THREADS
            a_r = idx // BLOCK_K
            a_c = idx % BLOCK_K
            sA[a_r, a_c] = x[bid_m * BLOCK_M + a_r, k_base + a_c]

        # Load B tile (W, already transposed)
        for i in S.range(B_LOADS):
            idx = tid + i * THREADS
            b_r = idx // BLOCK_K
            b_c = idx % BLOCK_K
            sB[b_r, b_c] = w[bid_n * BLOCK_N + b_r, k_base + b_c]

        S.syncthreads()

        # Compute partial dot product
        for kk in S.range(BLOCK_K):
            av = S.convert(sA[local_m, kk], S.f32)
            bv = S.convert(sB[local_n, kk], S.f32)
            acc = acc + av * bv

        S.syncthreads()

    # Add bias and add_value (both broadcast)
    if row < m and col < n:
        bias_val = S.convert(bias[col], S.f32)
        add_v = S.convert(add_val[col], S.f32)
        acc = acc + bias_val + add_v

        # Swish: sigmoid(x) * x
        # sigmoid(x) = 1 / (1 + exp(-x)) = exp2(x * log2(e)) / (exp2(x * log2(e)) + 1)
        # Numerically stable: if x >= 0: sigmoid = 1 / (1 + exp(-x))
        #                     if x < 0: sigmoid = exp(x) / (1 + exp(x))
        log2e = S.convert(LOG2E, S.f32)
        one = S.convert(1.0, S.f32)
        zero = S.convert(0.0, S.f32)

        # Compute sigmoid(x)
        neg_x = zero - acc
        exp_neg_x = S.exp2(neg_x * log2e)
        sig = one / (one + exp_neg_x)

        # Swish = sigmoid(x) * x
        swish_out = sig * acc

        # Tanh
        tanh_out = S.tanh(swish_out)

        # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        inv_sqrt2 = S.convert(INV_SQRT2, S.f32)
        half = S.convert(0.5, S.f32)
        erf_arg = tanh_out * inv_sqrt2
        erf_val = S.erf(erf_arg)
        gelu_out = half * tanh_out * (one + erf_val)

        # Hardtanh: clamp to [-1, 1]
        neg_one = S.convert(-1.0, S.f32)
        result = gelu_out
        if result < neg_one:
            result = neg_one
        if result > one:
            result = one

        out[row, col] = S.convert(result, S.bf16)


@substrate.jit
def linear_activation_chain_f32_kernel(
    x_ptr: S.Pointer(S.f32),
    w_ptr: S.Pointer(S.f32),
    bias_ptr: S.Pointer(S.f32),
    add_val_ptr: S.Pointer(S.f32),
    out_ptr: S.Pointer(S.f32),
    m: S.u32,
    n: S.u32,
    k: S.u32,
):
    """
    Fused kernel for: Linear(x) + add_val -> Swish -> Tanh -> GELU -> Hardtanh
    FP32 version.
    """
    tid = S.thread_id(0)
    bid_n = S.block_id(0)
    bid_m = S.block_id(1)

    local_m = tid // BLOCK_N
    local_n = tid % BLOCK_N

    row = bid_m * BLOCK_M + local_m
    col = bid_n * BLOCK_N + local_n

    layout_x = S.make_layout((m, k), (k, 1))
    layout_w = S.make_layout((n, k), (k, 1))
    layout_out = S.make_layout((m, n), (n, 1))
    layout_vec = S.make_layout((n,), (1,))

    x = S.make_tensor(x_ptr, S.f32, layout_x)
    w = S.make_tensor(w_ptr, S.f32, layout_w)
    bias = S.make_tensor(bias_ptr, S.f32, layout_vec)
    add_val = S.make_tensor(add_val_ptr, S.f32, layout_vec)
    out = S.make_tensor(out_ptr, S.f32, layout_out)

    sA = S.make_shared((BLOCK_M, BLOCK_K), S.f32)
    sB = S.make_shared((BLOCK_N, BLOCK_K), S.f32)

    acc = S.convert(0.0, S.f32)

    A_LOADS = (BLOCK_M * BLOCK_K) // THREADS
    B_LOADS = (BLOCK_N * BLOCK_K) // THREADS

    for kt in S.range(K_TILES):
        k_base = kt * BLOCK_K

        for i in S.range(A_LOADS):
            idx = tid + i * THREADS
            a_r = idx // BLOCK_K
            a_c = idx % BLOCK_K
            sA[a_r, a_c] = x[bid_m * BLOCK_M + a_r, k_base + a_c]

        for i in S.range(B_LOADS):
            idx = tid + i * THREADS
            b_r = idx // BLOCK_K
            b_c = idx % BLOCK_K
            sB[b_r, b_c] = w[bid_n * BLOCK_N + b_r, k_base + b_c]

        S.syncthreads()

        for kk in S.range(BLOCK_K):
            acc = acc + sA[local_m, kk] * sB[local_n, kk]

        S.syncthreads()

    if row < m and col < n:
        acc = acc + bias[col] + add_val[col]

        # Swish
        log2e = S.convert(LOG2E, S.f32)
        one = S.convert(1.0, S.f32)
        zero = S.convert(0.0, S.f32)

        neg_x = zero - acc
        exp_neg_x = S.exp2(neg_x * log2e)
        sig = one / (one + exp_neg_x)
        swish_out = sig * acc

        # Tanh
        tanh_out = S.tanh(swish_out)

        # GELU
        inv_sqrt2 = S.convert(INV_SQRT2, S.f32)
        half = S.convert(0.5, S.f32)
        erf_arg = tanh_out * inv_sqrt2
        erf_val = S.erf(erf_arg)
        gelu_out = half * tanh_out * (one + erf_val)

        # Hardtanh
        neg_one = S.convert(-1.0, S.f32)
        result = gelu_out
        if result < neg_one:
            result = neg_one
        if result > one:
            result = one

        out[row, col] = result


class ModelNew(nn.Module):
    """
    Optimized model using fused Substrate GPU kernel.
    Performs: Linear -> Add -> Swish -> Tanh -> GELU -> Hardtanh
    """

    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Use nn.Linear to match reference model's state_dict structure
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

        # Move to GPU if needed
        orig_device = x.device
        moved = not x.is_cuda

        if moved:
            x = x.cuda()

        x_contig = x.contiguous()
        # nn.Linear weight is shape (out_features, in_features), use directly
        weight_contig = self.matmul.weight.contiguous()
        bias_contig = self.matmul.bias.contiguous()
        add_val_contig = self.add_value.contiguous()

        # Determine dtype
        if x_contig.dtype == torch.bfloat16:
            compute_dtype = torch.bfloat16
        elif x_contig.dtype == torch.float32:
            compute_dtype = torch.float32
        else:
            # Default to bfloat16 for optimization
            x_contig = x_contig.to(torch.bfloat16)
            compute_dtype = torch.bfloat16

        # Cast weights to compute dtype if needed
        weight_typed = weight_contig.to(compute_dtype)
        bias_typed = bias_contig.to(compute_dtype)
        add_val_typed = add_val_contig.to(compute_dtype)

        # Allocate output
        out = torch.empty((batch_size, self.out_features), device=x_contig.device, dtype=compute_dtype)

        # Launch configuration
        grid_n = (self.out_features + BLOCK_N - 1) // BLOCK_N
        grid_m = (batch_size + BLOCK_M - 1) // BLOCK_M

        if compute_dtype == torch.bfloat16:
            linear_activation_chain_bf16_kernel[lambda: ((grid_n, grid_m, 1), (THREADS, 1, 1))](
                x_contig, weight_typed, bias_typed, add_val_typed, out,
                batch_size, self.out_features, self.in_features
            )
        else:
            linear_activation_chain_f32_kernel[lambda: ((grid_n, grid_m, 1), (THREADS, 1, 1))](
                x_contig, weight_typed, bias_typed, add_val_typed, out,
                batch_size, self.out_features, self.in_features
            )

        if moved:
            out = out.to(orig_device)

        return out


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_FEATURES)]


def get_init_inputs():
    return [IN_FEATURES, OUT_FEATURES, (OUT_FEATURES,)]
