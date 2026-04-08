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
EW_THREADS = 256


@substrate.jit
def linear_bf16_kernel(
    X: S.Pointer(S.bf16),
    W: S.Pointer(S.bf16),
    Bias: S.Pointer(S.bf16),
    Y: S.Pointer(S.bf16),
    m: S.u32,
    n: S.u32,
    k: S.u32,
):
    """GEMM kernel: Y = X @ W.T + Bias for row-major X (m, k), W (n, k)"""
    tid = S.thread_id(0)
    bx = S.block_id(0)  # N tile (output column)
    by = S.block_id(1)  # M tile (output row / batch)

    # Layout for row-major tensors
    layout_x = S.make_layout((m, k), (k, 1))
    layout_w = S.make_layout((n, k), (k, 1))
    layout_y = S.make_layout((m, n), (n, 1))
    layout_bias = S.make_layout((n,), (1,))

    gX = S.make_tensor(X, S.bf16, layout_x)
    gW = S.make_tensor(W, S.bf16, layout_w)
    gY = S.make_tensor(Y, S.bf16, layout_y)
    gBias = S.make_tensor(Bias, S.bf16, layout_bias)

    sX = S.make_shared((BLOCK_M, BLOCK_K), S.bf16)
    sW = S.make_shared((BLOCK_N, BLOCK_K), S.bf16)

    local_m = tid // BLOCK_N
    local_n = tid % BLOCK_N

    row = by * BLOCK_M + local_m
    col = bx * BLOCK_N + local_n

    # FP32 accumulation
    acc = S.convert(0.0, S.f32)

    # Compute K tiles dynamically
    k_tiles = k // BLOCK_K

    for kt in S.range(k_tiles):
        k_base = kt * BLOCK_K

        # Load X tile: [BLOCK_M, BLOCK_K]
        # Each thread loads one element
        x_r = tid // BLOCK_K
        x_c = tid % BLOCK_K
        sX[x_r, x_c] = gX[by * BLOCK_M + x_r, k_base + x_c]

        # Load W tile: [BLOCK_N, BLOCK_K]
        w_r = tid // BLOCK_K
        w_c = tid % BLOCK_K
        sW[w_r, w_c] = gW[bx * BLOCK_N + w_r, k_base + w_c]

        S.syncthreads()

        # Compute: each thread computes one output element
        for kk in S.range(BLOCK_K):
            xv = S.convert(sX[local_m, kk], S.f32)
            wv = S.convert(sW[local_n, kk], S.f32)
            acc = acc + xv * wv

        S.syncthreads()

    # Add bias
    bias_val = S.convert(gBias[col], S.f32)
    acc = acc + bias_val

    if row < m and col < n:
        gY[row, col] = S.convert(acc, S.bf16)


@substrate.jit
def fused_activation_bf16_kernel(
    X: S.Pointer(S.bf16),
    Y: S.Pointer(S.bf16),
    n: S.u32,
):
    """
    Fused activation kernel:
    1. Swish: x * sigmoid(x)
    2. Divide by 2
    3. Clamp between -1 and 1
    4. Tanh
    5. Clamp between -1 and 1
    """
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx < n:
        layout = S.make_layout((n,), (1,))
        gx = S.make_tensor(X, S.bf16, layout)
        gy = S.make_tensor(Y, S.bf16, layout)

        v = S.convert(gx[idx], S.f32)

        # Step 1: Swish activation: x * sigmoid(x)
        # sigmoid(x) = 1 / (1 + exp(-x))
        one = S.convert(1.0, S.f32)
        neg_v = S.convert(0.0, S.f32) - v
        # exp(-x) using exp2: exp(x) = exp2(x * log2(e))
        log2e = S.convert(1.4426950408889634, S.f32)
        exp_neg_v = S.exp2(neg_v * log2e)
        sigmoid_v = one / (one + exp_neg_v)
        v = v * sigmoid_v

        # Step 2: Divide by 2
        half = S.convert(0.5, S.f32)
        v = v * half

        # Step 3: Clamp between -1 and 1
        neg_one = S.convert(-1.0, S.f32)
        pos_one = S.convert(1.0, S.f32)
        if v < neg_one:
            v = neg_one
        if v > pos_one:
            v = pos_one

        # Step 4: Tanh activation
        v = S.tanh(v)

        # Step 5: Clamp between -1 and 1 (redundant for tanh but part of spec)
        if v < neg_one:
            v = neg_one
        if v > pos_one:
            v = pos_one

        gy[idx] = S.convert(v, S.bf16)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class ModelNew(nn.Module):
    """
    Optimized model with Substrate GPU kernels for:
    Linear -> Swish -> /2 -> Clamp(-1,1) -> Tanh -> Clamp(-1,1)
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Create weight tensor (out_features, in_features) - let the eval framework handle dtype
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Move to CUDA if needed
        orig_device = x.device
        need_copy_back = not x.is_cuda

        if not x.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")
            x = x.cuda()

        # Convert to BF16 for kernel computation
        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = self.weight.to(torch.bfloat16).contiguous()
        bias_bf16 = self.bias.to(torch.bfloat16).contiguous() if self.bias is not None else torch.zeros(self.out_features, dtype=torch.bfloat16, device=x.device)

        # Allocate output tensor for GEMM
        gemm_out = torch.empty(batch_size, self.out_features, dtype=torch.bfloat16, device=x.device)

        # Launch GEMM kernel
        m = batch_size
        n = self.out_features
        k = self.in_features

        grid_x = _ceil_div(n, BLOCK_N)
        grid_y = _ceil_div(m, BLOCK_M)

        linear_bf16_kernel[lambda: ((grid_x, grid_y, 1), (THREADS, 1, 1))](
            x_bf16, w_bf16, bias_bf16, gemm_out, m, n, k
        )

        # Allocate output for fused activation
        out = torch.empty_like(gemm_out)
        numel = gemm_out.numel()

        # Launch fused activation kernel
        ew_grid = _ceil_div(numel, EW_THREADS)
        fused_activation_bf16_kernel[lambda: ((ew_grid, 1, 1), (EW_THREADS, 1, 1))](
            gemm_out, out, numel
        )

        if need_copy_back:
            return out.to(orig_device)
        return out


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_FEATURES)]


def get_init_inputs():
    return [IN_FEATURES, OUT_FEATURES]
