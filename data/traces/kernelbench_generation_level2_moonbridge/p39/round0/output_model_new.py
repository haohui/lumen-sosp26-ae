import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── GEMM tile constants ──
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS
GROUP_M = 128
GROUP_N = 128
GROUP_K = 16
MMA_M = 32
MMA_N = 32
VEC_ELEMS = 8
BF16_BYTES = 2
ACC_SIZE = 16
WARPS_M = 2
WARPS_N = 2
M_TILES_PER_WARP = GROUP_M // (WARPS_M * MMA_M)
N_TILES_PER_WARP = GROUP_N // (WARPS_N * MMA_N)
A_VECS_PER_ROW = GROUP_K // VEC_ELEMS
B_VECS_PER_ROW = GROUP_K // VEC_ELEMS
SHM_A_VECS = GROUP_M * A_VECS_PER_ROW
SHM_B_VECS = GROUP_N * B_VECS_PER_ROW
GLOBAL_LOADS_A = SHM_A_VECS // THREADS
GLOBAL_LOADS_B = SHM_B_VECS // THREADS
ROW_U32 = A_VECS_PER_ROW * 4

# ── BatchNorm constants ──
BN_BLOCK_SIZE = 256


# ═══════════════════════════════════════════════════════════════
#  GEMM helper kernels
# ═══════════════════════════════════════════════════════════════

@avelang.jit
def _load_global_a_to_shm(
    shm_a: al.Tensor((SHM_A_VECS, 4), al.u32),
    x_rsrc: al.Tensor((4,), al.u32),
    block_m: al.u32,
    k_base: al.u32,
    k: al.u32,
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    idx = tid
    for _ in al.range(GLOBAL_LOADS_A):
        row = idx // A_VECS_PER_ROW
        col_vec = idx % A_VECS_PER_ROW
        off = ((block_m * GROUP_M + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
        shm_a[idx] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, off, 0)
        idx += THREADS


@avelang.jit
def _load_global_b_to_shm(
    shm_b: al.Tensor((SHM_B_VECS, 4), al.u32),
    w_rsrc: al.Tensor((4,), al.u32),
    block_n: al.u32,
    k_base: al.u32,
    k: al.u32,
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    idx = tid
    for _ in al.range(GLOBAL_LOADS_B):
        row = idx // B_VECS_PER_ROW
        col_vec = idx % B_VECS_PER_ROW
        off = ((block_n * GROUP_N + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
        shm_b[idx] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, off, 0)
        idx += THREADS


@avelang.jit
def _fetch_mfma_operand_32x32x16(
    ret: al.Tensor((2, 4), al.bf16),
    shm: al.Tensor((SHM_A_VECS, 4), al.u32),
    tile_idx: al.u32,
    lane: al.u32,
):
    ret_u32 = al.view(ret, al.Tensor((4,), al.u32))
    shm_u32 = al.view(shm, al.Tensor((SHM_A_VECS * 4,), al.u32))
    row = tile_idx * MMA_M + (lane % MMA_M)
    k_group_u32 = (lane // MMA_M) * 2
    row_base = row * ROW_U32

    ret_u32[0] = shm_u32[row_base + k_group_u32]
    ret_u32[1] = shm_u32[row_base + k_group_u32 + 1]
    ret_u32[2] = shm_u32[row_base + 4 + k_group_u32]
    ret_u32[3] = shm_u32[row_base + 5 + k_group_u32]


# ═══════════════════════════════════════════════════════════════
#  Kernel 1: Fused GEMM + bias + scale
# ═══════════════════════════════════════════════════════════════

@avelang.jit
def gemm_scale_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    scale_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    k: al.u32,
):
    tid = al.thread_id(0)
    block_n = al.block_id(0)
    block_m = al.block_id(1)
    wid = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    warp_row = wid // WARPS_N
    warp_col = wid % WARPS_N

    x_memref = al.make_tensor(x_ptr, al.bf16, al.make_layout((m * k,), (1,)))
    w_memref = al.make_tensor(w_ptr, al.bf16, al.make_layout((n * k,), (1,)))
    g_bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((n,), (1,)))
    g_scale = al.make_tensor(scale_ptr, al.bf16, al.make_layout((n,), (1,)))
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    x_rsrc = al.amdgpu.make_rsrc(x_memref, m * k * BF16_BYTES)
    w_rsrc = al.amdgpu.make_rsrc(w_memref, n * k * BF16_BYTES)

    shm_a = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_B_VECS, 4), al.u32)
    a_reg = al.make_local((M_TILES_PER_WARP, 2, 4), al.bf16)
    b_reg = al.make_local((N_TILES_PER_WARP, 2, 4), al.bf16)
    acc = al.make_local((M_TILES_PER_WARP * N_TILES_PER_WARP, ACC_SIZE), al.f32)

    for i in al.range(M_TILES_PER_WARP * N_TILES_PER_WARP):
        for j in al.range(ACC_SIZE):
            acc[i, j] = 0

    k_tiles = k // GROUP_K
    for kt in al.range(k_tiles):
        k_base = kt * GROUP_K
        _load_global_a_to_shm(shm_a, x_rsrc, block_m, k_base, k, tid)
        _load_global_b_to_shm(shm_b, w_rsrc, block_n, k_base, k, tid)
        al.syncthreads()

        for i in al.range(M_TILES_PER_WARP):
            _fetch_mfma_operand_32x32x16(a_reg[i], shm_a, warp_row * M_TILES_PER_WARP + i, lane)
        for j in al.range(N_TILES_PER_WARP):
            _fetch_mfma_operand_32x32x16(b_reg[j], shm_b, warp_col * N_TILES_PER_WARP + j, lane)

        for i in al.range(M_TILES_PER_WARP):
            for j in al.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                a0 = al.view(a_reg[i, 0], al.Tensor((2,), al.u32))
                b0 = al.view(b_reg[j, 0], al.Tensor((2,), al.u32))
                a1 = al.view(a_reg[i, 1], al.Tensor((2,), al.u32))
                b1 = al.view(b_reg[j, 1], al.Tensor((2,), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a0, b0, acc[acc_idx])
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a1, b1, acc[acc_idx])

        al.syncthreads()

    # Epilogue: bias + scale then store as BF16
    lane_group = lane // MMA_N
    lane_col = lane % MMA_N
    block_row_base = block_m * GROUP_M
    block_col_base = block_n * GROUP_N

    for j in al.range(N_TILES_PER_WARP):
        col = block_col_base + (warp_col * N_TILES_PER_WARP + j) * MMA_N + lane_col
        bias_val = al.convert(g_bias[col], al.f32)
        scale_val = al.convert(g_scale[col], al.f32)
        for i in al.range(M_TILES_PER_WARP):
            acc_idx = i * N_TILES_PER_WARP + j
            row_base = block_row_base + (warp_row * M_TILES_PER_WARP + i) * MMA_M
            for t in al.range(ACC_SIZE):
                row = row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
                result = acc[acc_idx, t] + bias_val
                result = result * scale_val
                g_out[row, col] = al.convert(result, al.bf16)


# ═══════════════════════════════════════════════════════════════
#  Kernel 2: BatchNorm reduce — per-feature sum and sum_sq
# ═══════════════════════════════════════════════════════════════

@avelang.jit
def batchnorm_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    B: al.i32,
    N: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    feature = bid * BN_BLOCK_SIZE + tid

    if feature < N:
        layout_in = al.make_layout((B, N), (N, 1))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        s = al.convert(0.0, al.f32)
        sq = al.convert(0.0, al.f32)

        for b in al.range(B):
            val = al.convert(x[b, feature], al.f32)
            s = s + val
            sq = sq + val * val

        layout_ps = al.make_layout((N,), (1,))
        ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
        psq = al.make_tensor(partial_sq_ptr, al.f32, layout_ps)
        ps[feature] = s
        psq[feature] = sq


# ═══════════════════════════════════════════════════════════════
#  Kernel 3: BatchNorm finalize — compute mean, rstd from partials
# ═══════════════════════════════════════════════════════════════

@avelang.jit
def batchnorm_finalize_kernel(
    partial_sum_ptr: al.Pointer(al.f32),
    partial_sq_ptr: al.Pointer(al.f32),
    mean_ptr: al.Pointer(al.f32),
    rstd_ptr: al.Pointer(al.f32),
    B: al.i32,
    N: al.i32,
    eps: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    feature = bid * BN_BLOCK_SIZE + tid

    if feature < N:
        layout_ps = al.make_layout((N,), (1,))
        ps = al.make_tensor(partial_sum_ptr, al.f32, layout_ps)
        psq = al.make_tensor(partial_sq_ptr, al.f32, layout_ps)

        s_val = ps[feature]
        sq_val = psq[feature]
        B_f32 = al.convert(B, al.f32)

        mean = s_val / B_f32
        var = sq_val / B_f32 - mean * mean
        rstd = al.convert(1.0, al.f32) / al.sqrt(var + eps)

        layout_out = al.make_layout((N,), (1,))
        m_out = al.make_tensor(mean_ptr, al.f32, layout_out)
        r_out = al.make_tensor(rstd_ptr, al.f32, layout_out)
        m_out[feature] = mean
        r_out[feature] = rstd


# ═══════════════════════════════════════════════════════════════
#  Kernel 4: BatchNorm apply — normalize + affine
# ═══════════════════════════════════════════════════════════════

@avelang.jit
def batchnorm_apply_kernel(
    x_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    rstd_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    N: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    idx = bid * BN_BLOCK_SIZE + tid
    total = B * N

    if idx < total:
        batch_idx = idx // N
        feat = idx - batch_idx * N

        layout_in = al.make_layout((B, N), (N, 1))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        layout_1d = al.make_layout((N,), (1,))
        gamma = al.make_tensor(gamma_ptr, al.bf16, layout_1d)
        beta = al.make_tensor(beta_ptr, al.bf16, layout_1d)
        mean_t = al.make_tensor(mean_ptr, al.f32, layout_1d)
        rstd_t = al.make_tensor(rstd_ptr, al.f32, layout_1d)

        layout_out = al.make_layout((B, N), (N, 1))
        out = al.make_tensor(out_ptr, al.bf16, layout_out)

        x_val = al.convert(x[batch_idx, feat], al.f32)
        g_val = al.convert(gamma[feat], al.f32)
        b_val = al.convert(beta[feat], al.f32)
        m = mean_t[feat]
        r = rstd_t[feat]

        normalized = (x_val - m) * r
        result = normalized * g_val + b_val
        out[batch_idx, feat] = al.convert(result, al.bf16)


# ═══════════════════════════════════════════════════════════════
#  Host wrappers
# ═══════════════════════════════════════════════════════════════

def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_gemm_scale_batchnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
    momentum: float,
    *,
    training: bool = True,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = _prepare_bf16_cuda_contiguous(weight)
    bias_bf16 = _prepare_bf16_cuda_contiguous(bias)
    scale_bf16 = _prepare_bf16_cuda_contiguous(scale)
    gamma_bf16 = _prepare_bf16_cuda_contiguous(gamma)
    beta_bf16 = _prepare_bf16_cuda_contiguous(beta)

    m, k_in = x_bf16.shape
    n, w_k = w_bf16.shape
    if w_k != k_in:
        raise ValueError(f"Weight/input K mismatch: x has K={k_in}, weight has K={w_k}")
    if m % GROUP_M != 0 or n % GROUP_N != 0 or k_in % GROUP_K != 0:
        raise ValueError(
            f"Expected m % {GROUP_M} == 0, n % {GROUP_N} == 0, k % {GROUP_K} == 0 "
            f"(got m={m}, n={n}, k={k_in})"
        )

    # Stage 1: GEMM + bias + scale → intermediate buffer
    intermediate = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)
    grid_gemm = (n // GROUP_N, m // GROUP_M, 1)
    gemm_scale_kernel[lambda: (grid_gemm, (THREADS, 1, 1))](
        x_bf16, w_bf16, bias_bf16, scale_bf16, intermediate, m, n, k_in
    )

    bn_grid = ((n + BN_BLOCK_SIZE - 1) // BN_BLOCK_SIZE, 1, 1)
    bn_block = (BN_BLOCK_SIZE, 1, 1)

    if training:
        # Stage 2: BatchNorm reduce
        partial_sum = torch.empty((n,), device=x_bf16.device, dtype=torch.float32)
        partial_sq = torch.empty((n,), device=x_bf16.device, dtype=torch.float32)

        batchnorm_reduce_kernel[lambda: (bn_grid, bn_block)](
            intermediate, partial_sum, partial_sq, m, n
        )

        # Stage 3: BatchNorm finalize
        mean_out = torch.empty((n,), device=x_bf16.device, dtype=torch.float32)
        rstd_out = torch.empty((n,), device=x_bf16.device, dtype=torch.float32)

        batchnorm_finalize_kernel[lambda: (bn_grid, bn_block)](
            partial_sum, partial_sq, mean_out, rstd_out, m, n, eps
        )

        # Update running stats
        with torch.no_grad():
            var = partial_sq / m - mean_out * mean_out
            running_mean.copy_((1.0 - momentum) * running_mean + momentum * mean_out)
            running_var.copy_((1.0 - momentum) * running_var + momentum * var)
    else:
        # Eval mode: use stored running stats
        rmean = running_mean.to(dtype=torch.float32, device=x_bf16.device)
        rvar = running_var.to(dtype=torch.float32, device=x_bf16.device)
        mean_out = rmean.clone()
        rstd_out = torch.rsqrt(rvar + eps)

    # Stage 4: BatchNorm apply — normalize + affine → final output
    out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)
    total_elems = m * n
    apply_grid = ((total_elems + BN_BLOCK_SIZE - 1) // BN_BLOCK_SIZE, 1, 1)

    # Make sure mean_out and rstd_out are float32 on GPU
    mean_f32 = mean_out.to(device=x_bf16.device, dtype=torch.float32).contiguous()
    rstd_f32 = rstd_out.to(device=x_bf16.device, dtype=torch.float32).contiguous()

    batchnorm_apply_kernel[lambda: (apply_grid, bn_block)](
        intermediate, gamma_bf16, beta_bf16, mean_f32, rstd_f32, out, m, n
    )

    return out


# ═══════════════════════════════════════════════════════════════
#  ModelNew — matching the reference parameter structure
# ═══════════════════════════════════════════════════════════════

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication, scaling, and
    batch normalization using AveLang DSL kernels.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        return avelang_gemm_scale_batchnorm(
            x=x,
            weight=self.gemm.weight,
            bias=self.gemm.bias,
            scale=self.scale,
            gamma=self.bn.weight,
            beta=self.bn.bias,
            running_mean=self.bn.running_mean,
            running_var=self.bn.running_var,
            eps=self.bn.eps,
            momentum=self.bn.momentum,
            training=self.training,
        )


batch_size = 16384
in_features = 4096
out_features = 4096
scale_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scale_shape]
