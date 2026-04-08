import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Problem sizes from target model.py
M = 1024
K = 8192
N = 8192

# Tiling parameters optimized for this problem size
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 32
THREADS = BLOCK_M * BLOCK_N
K_TILES = K // BLOCK_K

# Constants
MULTIPLIER = 2.0
NEGATIVE_SLOPE = 0.1


@substrate.jit
def fused_gemm_mul_leaky_relu_f32_kernel(
    A: S.Tensor((M, K), S.f32),
    B: S.Tensor((N, K), S.f32),
    Bias: S.Tensor((N,), S.f32),
    C: S.Tensor((M, N), S.f32),
):
    """
    Fused kernel: C = LeakyReLU((A @ B.T + Bias) * multiplier, negative_slope)
    B is stored in transposed form (N, K) for direct access.
    """
    tid = S.thread_id(0)
    bx = S.block_id(0)  # N tile
    by = S.block_id(1)  # M tile

    local_m = tid // BLOCK_N
    local_n = tid % BLOCK_N

    row = by * BLOCK_M + local_m
    col = bx * BLOCK_N + local_n

    sA = S.make_shared((BLOCK_M, BLOCK_K), S.f32)
    sB = S.make_shared((BLOCK_N, BLOCK_K), S.f32)

    # FP32 accumulation
    acc = S.convert(0.0, S.f32)

    for kt in S.range(K_TILES):
        k_base = kt * BLOCK_K

        # Cooperative load A tile: [BLOCK_M, BLOCK_K]
        # A is (M, K), row-major
        for i in S.range(BLOCK_M * BLOCK_K // THREADS):
            idx = tid + i * THREADS
            a_r = idx // BLOCK_K
            a_c = idx % BLOCK_K
            sA[a_r, a_c] = A[by * BLOCK_M + a_r, k_base + a_c]

        # Cooperative load B tile: [BLOCK_N, BLOCK_K]
        # B is (N, K), row-major (B.T in column-major for matmul)
        for i in S.range(BLOCK_N * BLOCK_K // THREADS):
            idx = tid + i * THREADS
            b_r = idx // BLOCK_K
            b_c = idx % BLOCK_K
            sB[b_r, b_c] = B[bx * BLOCK_N + b_r, k_base + b_c]

        S.syncthreads()

        # Compute dot product
        for kk in S.range(BLOCK_K):
            acc = acc + sA[local_m, kk] * sB[local_n, kk]

        S.syncthreads()

    # Add bias
    if col < N:
        acc = acc + Bias[col]

    # Multiply by multiplier
    acc = acc * S.convert(MULTIPLIER, S.f32)

    # Apply LeakyReLU: max(0, x) + negative_slope * min(0, x)
    # Equivalently: x if x > 0 else x * negative_slope
    zero = S.convert(0.0, S.f32)
    slope = S.convert(NEGATIVE_SLOPE, S.f32)
    if acc > zero:
        C[row, col] = acc
    else:
        C[row, col] = acc * slope


@substrate.jit
def fused_gemm_mul_leaky_relu_bf16_kernel(
    A: S.Tensor((M, K), S.bf16),
    B: S.Tensor((N, K), S.bf16),
    Bias: S.Tensor((N,), S.bf16),
    C: S.Tensor((M, N), S.bf16),
):
    """
    Fused kernel: C = LeakyReLU((A @ B.T + Bias) * multiplier, negative_slope)
    BF16 inputs/outputs with FP32 accumulation.
    """
    tid = S.thread_id(0)
    bx = S.block_id(0)  # N tile
    by = S.block_id(1)  # M tile

    local_m = tid // BLOCK_N
    local_n = tid % BLOCK_N

    row = by * BLOCK_M + local_m
    col = bx * BLOCK_N + local_n

    sA = S.make_shared((BLOCK_M, BLOCK_K), S.bf16)
    sB = S.make_shared((BLOCK_N, BLOCK_K), S.bf16)

    # FP32 accumulation for BF16 compute
    acc = S.convert(0.0, S.f32)

    for kt in S.range(K_TILES):
        k_base = kt * BLOCK_K

        # Cooperative load A tile
        for i in S.range(BLOCK_M * BLOCK_K // THREADS):
            idx = tid + i * THREADS
            a_r = idx // BLOCK_K
            a_c = idx % BLOCK_K
            sA[a_r, a_c] = A[by * BLOCK_M + a_r, k_base + a_c]

        # Cooperative load B tile
        for i in S.range(BLOCK_N * BLOCK_K // THREADS):
            idx = tid + i * THREADS
            b_r = idx // BLOCK_K
            b_c = idx % BLOCK_K
            sB[b_r, b_c] = B[bx * BLOCK_N + b_r, k_base + b_c]

        S.syncthreads()

        # Compute dot product with FP32 accumulation
        for kk in S.range(BLOCK_K):
            av = S.convert(sA[local_m, kk], S.f32)
            bv = S.convert(sB[local_n, kk], S.f32)
            acc = acc + av * bv

        S.syncthreads()

    # Add bias
    if col < N:
        acc = acc + S.convert(Bias[col], S.f32)

    # Multiply by multiplier
    acc = acc * S.convert(MULTIPLIER, S.f32)

    # Apply LeakyReLU
    zero = S.convert(0.0, S.f32)
    slope = S.convert(NEGATIVE_SLOPE, S.f32)
    if acc > zero:
        C[row, col] = S.convert(acc, S.bf16)
    else:
        C[row, col] = S.convert(acc * slope, S.bf16)


class ModelNew(nn.Module):
    """
    Optimized model that performs GEMM + multiply + LeakyReLU using fused Substrate GPU kernels.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.multiplier = multiplier
        self.negative_slope = negative_slope

        # Initialize weight and bias like nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle device placement
        orig_device = x.device
        need_copy_back = not x.is_cuda

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

        if not x.is_cuda:
            x_dev = x.cuda()
        else:
            x_dev = x

        x_contig = x_dev.contiguous()
        weight_contig = self.weight.contiguous()
        bias_contig = self.bias.contiguous()

        # Output tensor
        out = torch.empty((x_contig.shape[0], self.out_features),
                          device=x_contig.device, dtype=x_contig.dtype)

        grid = (N // BLOCK_N, M // BLOCK_M, 1)
        block = (THREADS, 1, 1)

        if x_contig.dtype == torch.float32:
            fused_gemm_mul_leaky_relu_f32_kernel[lambda: (grid, block)](
                x_contig, weight_contig, bias_contig, out
            )
        elif x_contig.dtype == torch.bfloat16:
            fused_gemm_mul_leaky_relu_bf16_kernel[lambda: (grid, block)](
                x_contig, weight_contig, bias_contig, out
            )
        else:
            raise TypeError(f"Unsupported dtype: {x_contig.dtype}")

        if need_copy_back:
            return out.to(orig_device)
        return out


batch_size = 1024
in_features = 8192
out_features = 8192
multiplier = 2.0
negative_slope = 0.1


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]
