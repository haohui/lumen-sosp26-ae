import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── Problem constants ──────────────────────────────────────────────
batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0

# ── GEMM tile constants ────────────────────────────────────────────
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

# ── Reduction constants ────────────────────────────────────────────
REDUCE_THREADS = 256
POOLED_ELEMS = out_features // pool_kernel_size  # 512


# ═══════════════════════════════════════════════════════════════════
#  Shared helpers for GEMM operand loading
# ═══════════════════════════════════════════════════════════════════

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
def _fetch_mfma_operand_32x32x8_u32(
    dst: al.Tensor((4,), al.u32),
    shm: al.Tensor((SHM_A_VECS, 4), al.u32),
    tile_idx: al.u32,
    lane: al.u32,
):
    shm_u32 = al.view(shm, al.Tensor((SHM_A_VECS * 4,), al.u32))
    row = tile_idx * MMA_M + (lane % MMA_M)
    k_group_u32 = (lane // MMA_M) * 2
    row_base = row * ROW_U32

    dst[0] = shm_u32[row_base + k_group_u32]
    dst[1] = shm_u32[row_base + k_group_u32 + 1]
    dst[2] = shm_u32[row_base + 4 + k_group_u32]
    dst[3] = shm_u32[row_base + 5 + k_group_u32]


# ═══════════════════════════════════════════════════════════════════
#  Kernel 1: BF16 GEMM with bias
# ═══════════════════════════════════════════════════════════════════

@avelang.jit
def gemm_bias_bf16_kernel(
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
    a_reg = al.make_local((M_TILES_PER_WARP, 4), al.u32)
    b_reg = al.make_local((N_TILES_PER_WARP, 4), al.u32)
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
            _fetch_mfma_operand_32x32x8_u32(
                a_reg[i], shm_a, warp_row * M_TILES_PER_WARP + i, lane
            )
        for j in al.range(N_TILES_PER_WARP):
            _fetch_mfma_operand_32x32x8_u32(
                b_reg[j], shm_b, warp_col * N_TILES_PER_WARP + j, lane
            )

        for i in al.range(M_TILES_PER_WARP):
            for j in al.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                a_view = al.view(a_reg[i], al.Tensor((2, 2), al.u32))
                b_view = al.view(b_reg[j], al.Tensor((2, 2), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(
                    a_view[0], b_view[0], acc[acc_idx]
                )
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(
                    a_view[1], b_view[1], acc[acc_idx]
                )

        al.syncthreads()

    lane_group = lane // MMA_N
    lane_col = lane % MMA_N
    block_row_base = block_m * GROUP_M
    block_col_base = block_n * GROUP_N

    for j in al.range(N_TILES_PER_WARP):
        col = block_col_base + (warp_col * N_TILES_PER_WARP + j) * MMA_N + lane_col
        bias_val = al.convert(g_bias[col], al.f32)
        for i in al.range(M_TILES_PER_WARP):
            acc_idx = i * N_TILES_PER_WARP + j
            row_base = block_row_base + (warp_row * M_TILES_PER_WARP + i) * MMA_M
            for t in al.range(ACC_SIZE):
                row = row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
                result = acc[acc_idx, t] + bias_val
                g_out[row, col] = al.convert(result, al.bf16)


# ═══════════════════════════════════════════════════════════════════
#  Kernel 2: AvgPool(k=16) → GELU → Scale → Max reduction over dim=1
# ═══════════════════════════════════════════════════════════════════

# GELU tanh-approximation constants
GELU_SQRT_2_OVER_PI = 0.7978845608028654
GELU_COEFF = 0.044715


@avelang.jit
def avgpool_gelu_scale_max_kernel(
    in_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    n_cols: al.i32,
    pool_size: al.i32,
    pooled_elems: al.i32,
    scale_val: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < batch_size:
        smem = al.make_shared((REDUCE_THREADS,), al.f32)

        layout_in = al.make_layout((batch_size, n_cols), (n_cols, 1))
        in_t = al.make_tensor(in_ptr, al.bf16, layout_in)

        local_max = al.convert(-3.402823e38, al.f32)

        sqrt_2_pi = al.convert(GELU_SQRT_2_OVER_PI, al.f32)
        gelu_coeff = al.convert(GELU_COEFF, al.f32)
        half = al.convert(0.5, al.f32)
        one = al.convert(1.0, al.f32)
        pool_size_f = al.convert(pool_size, al.f32)

        tid_i32 = al.convert(tid, al.i32)
        thread_step = al.convert(REDUCE_THREADS, al.i32)

        p_idx = tid_i32
        while p_idx < pooled_elems:
            base_col = p_idx * pool_size
            acc_sum = al.convert(0.0, al.f32)
            for off in al.range(pool_size):
                col = base_col + off
                val = al.convert(in_t[bid, col], al.f32)
                acc_sum = acc_sum + val
            avg = acc_sum / pool_size_f

            # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            x_cube = avg * avg * avg
            inner = sqrt_2_pi * (avg + gelu_coeff * x_cube)
            tanh_val = al.tanh(inner)
            gelu_val = half * avg * (one + tanh_val)

            scaled = gelu_val * scale_val

            if scaled > local_max:
                local_max = scaled
            p_idx = p_idx + thread_step

        smem[tid] = local_max
        al.syncthreads()

        # Shared-memory max reduction tree
        if tid < 128:
            a = smem[tid]
            b = smem[tid + 128]
            smem[tid] = a if a > b else b
        al.syncthreads()
        if tid < 64:
            a = smem[tid]
            b = smem[tid + 64]
            smem[tid] = a if a > b else b
        al.syncthreads()
        if tid < 32:
            a = smem[tid]
            b = smem[tid + 32]
            smem[tid] = a if a > b else b
        al.syncthreads()
        if tid < 16:
            a = smem[tid]
            b = smem[tid + 16]
            smem[tid] = a if a > b else b
        al.syncthreads()
        if tid < 8:
            a = smem[tid]
            b = smem[tid + 8]
            smem[tid] = a if a > b else b
        al.syncthreads()
        if tid < 4:
            a = smem[tid]
            b = smem[tid + 4]
            smem[tid] = a if a > b else b
        al.syncthreads()
        if tid < 2:
            a = smem[tid]
            b = smem[tid + 2]
            smem[tid] = a if a > b else b
        al.syncthreads()
        if tid < 1:
            a = smem[tid]
            b = smem[tid + 1]
            smem[tid] = a if a > b else b
        al.syncthreads()

        if tid == 0:
            layout_out = al.make_layout((batch_size,), (1,))
            out_t = al.make_tensor(out_ptr, al.bf16, layout_out)
            out_t[bid] = al.convert(smem[0], al.bf16)


# ═══════════════════════════════════════════════════════════════════
#  Host wrappers
# ═══════════════════════════════════════════════════════════════════

def _prepare_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_matmul_bias(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_contiguous(x)
    weight_bf16 = _prepare_bf16_contiguous(weight)
    bias_bf16 = _prepare_bf16_contiguous(bias)

    m_val, k_val = x_bf16.shape
    n_val, w_k = weight_bf16.shape
    if w_k != k_val:
        raise ValueError(
            f"Weight/input K mismatch: x has K={k_val}, weight has K={w_k}"
        )

    out = torch.empty((m_val, n_val), device=x_bf16.device, dtype=torch.bfloat16)
    grid = (n_val // GROUP_N, m_val // GROUP_M, 1)
    gemm_bias_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, weight_bf16, bias_bf16, out, m_val, n_val, k_val
    )
    return out


def avelang_avgpool_gelu_scale_max(
    x: torch.Tensor,
    pool_size: int,
    scale_val: float,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_contiguous(x)
    bsize, ncols = x_bf16.shape
    pooled = ncols // pool_size

    out = torch.empty((bsize,), device=x_bf16.device, dtype=torch.bfloat16)

    grid = (bsize, 1, 1)
    avgpool_gelu_scale_max_kernel[lambda: (grid, (REDUCE_THREADS, 1, 1))](
        x_bf16, out, bsize, ncols, pool_size, pooled, scale_val
    )
    return out


# ═══════════════════════════════════════════════════════════════════
#  ModelNew
# ═══════════════════════════════════════════════════════════════════

class ModelNew(nn.Module):
    """
    Matmul_AvgPool_GELU_Scale_Max using AveLang DSL kernels.
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor

    def forward(self, x):
        weight = self.matmul.weight.data
        bias = self.matmul.bias.data

        x = avelang_matmul_bias(x, weight, bias)
        x = avelang_avgpool_gelu_scale_max(x, self.pool_kernel_size, self.scale_factor)
        return x


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]
