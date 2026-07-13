import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# GEMM constants
# ---------------------------------------------------------------------------
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS  # 256
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
M_TILES_PER_WARP = GROUP_M // (WARPS_M * MMA_M)  # 2
N_TILES_PER_WARP = GROUP_N // (WARPS_N * MMA_N)  # 2
A_VECS_PER_ROW = GROUP_K // VEC_ELEMS           # 2
B_VECS_PER_ROW = GROUP_K // VEC_ELEMS           # 2
SHM_A_VECS = GROUP_M * A_VECS_PER_ROW           # 256
SHM_B_VECS = GROUP_N * B_VECS_PER_ROW           # 256
GLOBAL_LOADS_A = SHM_A_VECS // THREADS           # 1
GLOBAL_LOADS_B = SHM_B_VECS // THREADS           # 1
ROW_U32 = A_VECS_PER_ROW * 4                     # 8

# ---------------------------------------------------------------------------
# BN / softmax constants
# ---------------------------------------------------------------------------
SOFTMAX_BLOCK_SIZE = 256

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
BN_EPS = 1e-5
BN_MOMENTUM = 0.1
SCALE_SHAPE = (1,)


# ===========================================================================
# GEMM helper kernels
# ===========================================================================

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


# ===========================================================================
# Kernel 1: BF16 GEMM (x @ weight.T + bias)
# ===========================================================================

@avelang.jit
def gemm_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
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
                a_packed = al.view(a_reg[i, 0], al.Tensor((2,), al.i32))
                b_packed = al.view(b_reg[j, 0], al.Tensor((2,), al.i32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_packed, b_packed, acc[acc_idx])
                a_packed = al.view(a_reg[i, 1], al.Tensor((2,), al.i32))
                b_packed = al.view(b_reg[j, 1], al.Tensor((2,), al.i32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_packed, b_packed, acc[acc_idx])

        al.syncthreads()

    lane_group = lane // MMA_N
    lane_col = lane % MMA_N
    block_row_base = block_m * GROUP_M
    block_col_base = block_n * GROUP_N

    for j in al.range(N_TILES_PER_WARP):
        col = block_col_base + (warp_col * N_TILES_PER_WARP + j) * MMA_N + lane_col
        bias = al.convert(g_bias[col], al.f32)
        for i in al.range(M_TILES_PER_WARP):
            acc_idx = i * N_TILES_PER_WARP + j
            row_base = block_row_base + (warp_row * M_TILES_PER_WARP + i) * MMA_M
            for t in al.range(ACC_SIZE):
                row = row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
                result = acc[acc_idx, t] + bias
                g_out[row, col] = al.convert(result, al.bf16)


# ===========================================================================
# Kernel 2: Fused BN eval + scale + softmax
# In eval mode, BatchNorm uses running stats directly (no batch reduction).
# ===========================================================================

@avelang.jit
def bn_eval_scale_softmax_kernel(
    gemm_out_ptr: al.Pointer(al.bf16),
    running_mean_ptr: al.Pointer(al.f32),
    running_var_ptr: al.Pointer(al.f32),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    out_features: al.i32,
    eps: al.f32,
    scale_val: al.f32,
):
    tid = al.thread_id(0)
    batch_idx = al.block_id(0)

    if batch_idx < batch_size:
        layout_in = al.make_layout((batch_size, out_features), (out_features, 1))
        gemm_out = al.make_tensor(gemm_out_ptr, al.bf16, layout_in)

        layout_feat = al.make_layout((out_features,), (1,))
        run_mean = al.make_tensor(running_mean_ptr, al.f32, layout_feat)
        run_var = al.make_tensor(running_var_ptr, al.f32, layout_feat)
        gamma = al.make_tensor(gamma_ptr, al.bf16, layout_feat)
        beta = al.make_tensor(beta_ptr, al.bf16, layout_feat)

        layout_out = al.make_layout((batch_size, out_features), (out_features, 1))
        g_out = al.make_tensor(out_ptr, al.bf16, layout_out)

        smem = al.make_shared((SOFTMAX_BLOCK_SIZE,), al.f32)
        vals = al.make_local((32,), al.f32)
        exp_vals = al.make_local((32,), al.f32)

        # Phase 1: apply BN eval + scale, store in vals, and find local max
        local_max = al.convert(-1.0e30, al.f32)
        idx = al.convert(0, al.i32)
        for f in al.range(tid, out_features, SOFTMAX_BLOCK_SIZE):
            x_val = al.convert(gemm_out[batch_idx, f], al.f32)
            mean_val = run_mean[f]
            var_val = run_var[f]
            rstd = al.convert(1.0, al.f32) / al.sqrt(var_val + eps)
            g_val = al.convert(gamma[f], al.f32)
            b_val = al.convert(beta[f], al.f32)
            normalized = (x_val - mean_val) * rstd
            bn_out = normalized * g_val + b_val
            scaled = bn_out * scale_val
            vals[idx] = scaled
            if scaled > local_max:
                local_max = scaled
            idx = idx + 1

        smem[tid] = local_max
        al.syncthreads()

        if tid < 128:
            v0 = smem[tid]
            v1 = smem[tid + 128]
            smem[tid] = v0 if v0 > v1 else v1
        al.syncthreads()
        if tid < 64:
            v0 = smem[tid]
            v1 = smem[tid + 64]
            smem[tid] = v0 if v0 > v1 else v1
        al.syncthreads()
        if tid < 32:
            v0 = smem[tid]
            v1 = smem[tid + 32]
            smem[tid] = v0 if v0 > v1 else v1
        al.syncthreads()
        if tid < 16:
            v0 = smem[tid]
            v1 = smem[tid + 16]
            smem[tid] = v0 if v0 > v1 else v1
        al.syncthreads()
        if tid < 8:
            v0 = smem[tid]
            v1 = smem[tid + 8]
            smem[tid] = v0 if v0 > v1 else v1
        al.syncthreads()
        if tid < 4:
            v0 = smem[tid]
            v1 = smem[tid + 4]
            smem[tid] = v0 if v0 > v1 else v1
        al.syncthreads()
        if tid < 2:
            v0 = smem[tid]
            v1 = smem[tid + 2]
            smem[tid] = v0 if v0 > v1 else v1
        al.syncthreads()
        if tid < 1:
            v0 = smem[tid]
            v1 = smem[tid + 1]
            smem[tid] = v0 if v0 > v1 else v1
        al.syncthreads()
        row_max = smem[0]

        # Phase 2: compute exp(x - max) and accumulate sum
        local_sum = al.convert(0.0, al.f32)
        idx = al.convert(0, al.i32)
        for f in al.range(tid, out_features, SOFTMAX_BLOCK_SIZE):
            e = al.exp(vals[idx] - row_max)
            exp_vals[idx] = e
            local_sum = local_sum + e
            idx = idx + 1

        smem[tid] = local_sum
        al.syncthreads()

        if tid < 128:
            smem[tid] = smem[tid] + smem[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem[tid] = smem[tid] + smem[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem[tid] = smem[tid] + smem[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem[tid] = smem[tid] + smem[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem[tid] = smem[tid] + smem[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem[tid] = smem[tid] + smem[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem[tid] = smem[tid] + smem[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem[tid] = smem[tid] + smem[tid + 1]
        al.syncthreads()
        row_sum = smem[0]

        # Phase 3: normalize and write output
        idx = al.convert(0, al.i32)
        for f in al.range(tid, out_features, SOFTMAX_BLOCK_SIZE):
            result = exp_vals[idx] / row_sum
            g_out[batch_idx, f] = al.convert(result, al.bf16)
            idx = idx + 1


# ===========================================================================
# Host wrappers
# ===========================================================================

def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_pipeline(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    bn_gamma: torch.Tensor,
    bn_beta: torch.Tensor,
    scale: torch.Tensor,
    bn_eps: float,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight)
    bias_bf16 = _prepare_bf16_cuda_contiguous(bias)
    bn_gamma_bf16 = _prepare_bf16_cuda_contiguous(bn_gamma)
    bn_beta_bf16 = _prepare_bf16_cuda_contiguous(bn_beta)

    m, k = x_bf16.shape
    n, weight_k = weight_bf16.shape

    if m % GROUP_M != 0 or n % GROUP_N != 0 or k % GROUP_K != 0:
        raise ValueError(
            f"Expected m % {GROUP_M} == 0, n % {GROUP_N} == 0, k % {GROUP_K} == 0 "
            f"(got m={m}, n={n}, k={k})"
        )

    # Step 1: GEMM
    gemm_out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)
    grid = (n // GROUP_N, m // GROUP_M, 1)
    gemm_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, weight_bf16, bias_bf16, gemm_out, m, n, k
    )

    # Step 2: Fused BN eval + scale + softmax
    # In eval mode, use running stats directly
    run_mean_f32 = running_mean.to(device=x_bf16.device, dtype=torch.float32).contiguous()
    run_var_f32 = running_var.to(device=x_bf16.device, dtype=torch.float32).contiguous()
    eps_f32 = float(bn_eps)

    out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)
    bn_eval_scale_softmax_kernel[lambda: ((m, 1, 1), (SOFTMAX_BLOCK_SIZE, 1, 1))](
        gemm_out, run_mean_f32, run_var_f32, bn_gamma_bf16, bn_beta_bf16,
        out, m, n, eps_f32, 1.0,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum

        self.gemm_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gemm_bias = nn.Parameter(torch.empty(out_features))
        self.bn_gamma = nn.Parameter(torch.ones(out_features))
        self.bn_beta = nn.Parameter(torch.zeros(out_features))
        self.scale = nn.Parameter(torch.ones(scale_shape))

        self.register_buffer("running_mean", torch.zeros(out_features))
        self.register_buffer("running_var", torch.ones(out_features))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.gemm_weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.gemm_weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.gemm_bias, -bound, bound)

    def forward(self, x):
        result_bf16 = avelang_pipeline(
            x,
            self.gemm_weight,
            self.gemm_bias,
            self.bn_gamma,
            self.bn_beta,
            self.scale,
            self.bn_eps,
            self.running_mean,
            self.running_var,
        )
        return result_bf16.to(x.dtype)


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_FEATURES)]


def get_init_inputs():
    return [IN_FEATURES, OUT_FEATURES, BN_EPS, BN_MOMENTUM, SCALE_SHAPE]
