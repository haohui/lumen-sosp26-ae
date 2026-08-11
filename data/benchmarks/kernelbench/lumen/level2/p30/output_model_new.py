import torch
import torch.nn as nn
import substrate
import substrate.language as S

# Problem sizes
BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
NUM_GROUPS = 16
GROUP_SIZE = OUT_FEATURES // NUM_GROUPS  # 512

EPS = 1e-5
HARDTANH_MIN = -2.0
HARDTANH_MAX = 2.0

# GEMM tiling parameters
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 32
THREADS = BLOCK_M * BLOCK_N


@substrate.jit
def matmul_bf16_kernel(
    A: S.Pointer(S.bf16),
    B: S.Pointer(S.bf16),
    C: S.Pointer(S.bf16),
    m: S.u32,
    n: S.u32,
    k: S.u32,
):
    layout_a = S.make_layout((m, k), (k, 1))
    layout_b = S.make_layout((k, n), (n, 1))
    layout_c = S.make_layout((m, n), (n, 1))

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

    shm_a = S.make_shared((BLOCK_M, BLOCK_K), S.bf16)
    shm_b = S.make_shared((BLOCK_K, BLOCK_N), S.bf16)

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
def group_norm_kernel(
    x: S.Pointer(S.bf16),
    gamma: S.Pointer(S.bf16),
    beta: S.Pointer(S.bf16),
    y: S.Pointer(S.bf16),
    batch_size: S.u32,
    num_features: S.u32,
    num_groups: S.u32,
    group_size: S.u32,
):
    layout_x = S.make_layout((batch_size, num_features), (num_features, 1))
    layout_gamma = S.make_layout((num_features,), (1,))
    layout_beta = S.make_layout((num_features,), (1,))
    layout_y = S.make_layout((batch_size, num_features), (num_features, 1))

    gx = S.make_tensor(x, S.bf16, layout_x)
    ggamma = S.make_tensor(gamma, S.bf16, layout_gamma)
    gbeta = S.make_tensor(beta, S.bf16, layout_beta)
    gy = S.make_tensor(y, S.bf16, layout_y)

    b = S.block_id(0)
    g = S.block_id(1)

    inv_gs = S.convert(1.0 / GROUP_SIZE, S.f32)
    eps = S.convert(EPS, S.f32)

    mean = S.convert(0.0, S.f32)
    var = S.convert(0.0, S.f32)

    for f in S.range(GROUP_SIZE):
        feat_idx = g * GROUP_SIZE + f
        val = S.convert(gx[b, feat_idx], S.f32)
        mean = mean + val

    mean = mean * inv_gs

    for f in S.range(GROUP_SIZE):
        feat_idx = g * GROUP_SIZE + f
        val = S.convert(gx[b, feat_idx], S.f32)
        diff = val - mean
        var = var + diff * diff

    var = var * inv_gs
    rstd = S.convert(1.0, S.f32) / S.sqrt(var + eps)

    for f in S.range(GROUP_SIZE):
        feat_idx = g * GROUP_SIZE + f
        val = S.convert(gx[b, feat_idx], S.f32)
        normalized = (val - mean) * rstd
        w = S.convert(ggamma[feat_idx], S.f32)
        b_val = S.convert(gbeta[feat_idx], S.f32)
        out_val = normalized * w + b_val
        gy[b, feat_idx] = S.convert(out_val, S.bf16)


@substrate.jit
def hardtanh_bf16_kernel(
    x: S.Pointer(S.bf16),
    y: S.Pointer(S.bf16),
    n: S.u32,
):
    min_val = S.convert(HARDTANH_MIN, S.f32)
    max_val = S.convert(HARDTANH_MAX, S.f32)

    layout = S.make_layout((n,), (1,))
    gx = S.make_tensor(x, S.bf16, layout)
    gy = S.make_tensor(y, S.bf16, layout)

    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx < n:
        v = S.convert(gx[idx], S.f32)
        if v < min_val:
            v = min_val
        if v > max_val:
            v = max_val
        gy[idx] = S.convert(v, S.bf16)


def substrate_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Computes A @ B where A is (m, k) and B is (k, n)."""
    if not A.is_cuda or not B.is_cuda:
        raise ValueError("Substrate kernels require CUDA/HIP tensors.")

    A = A.contiguous()
    B = B.contiguous()

    m = A.shape[0]
    k = A.shape[1]
    n = B.shape[1]

    grid_x = (n + BLOCK_N - 1) // BLOCK_N
    grid_y = (m + BLOCK_M - 1) // BLOCK_M

    C = torch.empty((m, n), device=A.device, dtype=torch.bfloat16)
    matmul_bf16_kernel[lambda: ((grid_x, grid_y, 1), (THREADS, 1, 1))](
        A, B, C, m, n, k
    )
    return C


def substrate_group_norm(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("Substrate kernels require CUDA/HIP tensors.")

    x = x.contiguous()
    batch_size = x.shape[0]
    num_features = x.shape[1]

    y = torch.empty_like(x)
    grid = (batch_size, NUM_GROUPS, 1)
    block = (1, 1, 1)

    group_norm_kernel[lambda: (grid, block)](
        x, gamma, beta, y, batch_size, num_features, NUM_GROUPS, GROUP_SIZE
    )
    return y


def substrate_hardtanh(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("Substrate kernels require CUDA/HIP tensors.")

    x = x.contiguous()
    n = x.numel()
    y = torch.empty_like(x)

    block = 256
    grid = ((n + block - 1) // block, 1, 1)

    hardtanh_bf16_kernel[lambda: (grid, (block, 1, 1))](x, y, n)
    return y


class ModelNew(nn.Module):
    """
    Optimized model that performs GEMM + GroupNorm + HardTanh using Substrate GPU kernels.
    Matches the reference Model structure for correct weight initialization.
    """

    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        # Match reference structure exactly for weight initialization compatibility
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to bfloat16 for kernel execution
        x = x.to(torch.bfloat16)

        if not x.is_cuda:
            x = x.cuda()

        x_contig = x.contiguous()

        # Get weights in bfloat16
        W_t = self.gemm.weight.t().to(torch.bfloat16).contiguous()
        bias = self.gemm.bias.to(torch.bfloat16).contiguous() if self.gemm.bias is not None else None

        # GEMM: x @ weight.T + bias
        gemm_out = substrate_matmul(x_contig, W_t)
        if bias is not None:
            gemm_out = gemm_out + bias

        # Group Normalization
        gamma = self.group_norm.weight.to(torch.bfloat16).contiguous()
        beta = self.group_norm.bias.to(torch.bfloat16).contiguous()
        norm_out = substrate_group_norm(gemm_out, gamma, beta)

        # HardTanh
        out = substrate_hardtanh(norm_out)

        return out


batch_size = BATCH_SIZE
in_features = IN_FEATURES
out_features = OUT_FEATURES
num_groups = NUM_GROUPS
hardtanh_min = HARDTANH_MIN
hardtanh_max = HARDTANH_MAX


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]
