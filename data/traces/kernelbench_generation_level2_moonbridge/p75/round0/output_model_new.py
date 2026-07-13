import torch
import torch.nn as nn
import avelang
import avelang.language as al

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

# ── Reduction / bias tile constants ────────────────────────────────
REDUCE_BLOCK = 256
BIAS_BLOCK = 256


# ═══════════════════════════════════════════════════════════════════════
#  GEMM helper kernels
# ═══════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════
#  GEMM kernel  (x @ w^T + bias)
# ═══════════════════════════════════════════════════════════════════════


@avelang.jit
def gemm_kernel(
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
                a_op = al.view(a_reg[i, 0], al.Tensor((2,), al.u32))
                b_op = al.view(b_reg[j, 0], al.Tensor((2,), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_op, b_op, acc[acc_idx])
                a_op = al.view(a_reg[i, 1], al.Tensor((2,), al.u32))
                b_op = al.view(b_reg[j, 1], al.Tensor((2,), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_op, b_op, acc[acc_idx])

        al.syncthreads()

    # ── epilogue: add Linear bias and write out ──
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

# ═══════════════════════════════════════════════════════════════════════
#  GroupNorm kernel  (training mode, 16 channels per group)
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
#  Min reduction kernel  (reduce dim=1: 8192 → 1 per batch)
# ═══════════════════════════════════════════════════════════════════════


@avelang.jit
def min_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.u32,
    channels: al.u32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    x_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((batch_size * channels,), (1,)))

    smem = al.make_shared((REDUCE_BLOCK,), al.f32)

    base = bid * channels

    local_min = al.convert(3.402823466e38, al.f32)  # large initial value
    for i in al.range(tid, channels, REDUCE_BLOCK):
        val = al.convert(x_flat[base + i], al.f32)
        if val < local_min:
            local_min = val

    smem[tid] = local_min
    al.syncthreads()

    if tid < 128:
        a = smem[tid]
        b = smem[tid + 128]
        smem[tid] = a if a < b else b
    al.syncthreads()
    if tid < 64:
        a = smem[tid]
        b = smem[tid + 64]
        smem[tid] = a if a < b else b
    al.syncthreads()
    if tid < 32:
        a = smem[tid]
        b = smem[tid + 32]
        smem[tid] = a if a < b else b
    al.syncthreads()
    if tid < 16:
        a = smem[tid]
        b = smem[tid + 16]
        smem[tid] = a if a < b else b
    al.syncthreads()
    if tid < 8:
        a = smem[tid]
        b = smem[tid + 8]
        smem[tid] = a if a < b else b
    al.syncthreads()
    if tid < 4:
        a = smem[tid]
        b = smem[tid + 4]
        smem[tid] = a if a < b else b
    al.syncthreads()
    if tid < 2:
        a = smem[tid]
        b = smem[tid + 2]
        smem[tid] = a if a < b else b
    al.syncthreads()
    if tid < 1:
        a = smem[tid]
        b = smem[tid + 1]
        smem[tid] = a if a < b else b

    if tid == 0:
        o_flat = al.make_tensor(out_ptr, al.bf16, al.make_layout((batch_size,), (1,)))
        o_flat[bid] = al.convert(smem[0], al.bf16)


# ═══════════════════════════════════════════════════════════════════════
#  Bias broadcast add kernel
#  out[c * batch_size + b] = min_val[b] + bias[c]
# ═══════════════════════════════════════════════════════════════════════


@avelang.jit
def bias_add_kernel(
    min_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.u32,
    channels: al.u32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    global_id = bid * BIAS_BLOCK + tid
    total = batch_size * channels

    if global_id < total:
        b = global_id // channels
        c = global_id - b * channels

        m_flat = al.make_tensor(min_ptr, al.bf16, al.make_layout((batch_size,), (1,)))
        b_flat = al.make_tensor(bias_ptr, al.bf16, al.make_layout((channels,), (1,)))
        o_flat = al.make_tensor(out_ptr, al.bf16, al.make_layout((total,), (1,)))

        min_val = al.convert(m_flat[b], al.f32)
        bias_val = al.convert(b_flat[c], al.f32)
        result = min_val + bias_val
        o_flat[global_id] = al.convert(result, al.bf16)


def _prepare_bf16(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


# ═══════════════════════════════════════════════════════════════════════
#  Host wrapper ─ pipeline orchestration
class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x_bf16 = _prepare_bf16(x)
        w_bf16 = _prepare_bf16(self.gemm.weight)
        gemm_b_bf16 = _prepare_bf16(self.gemm.bias)

        m, k_in = x_bf16.shape
        n, w_k = w_bf16.shape

        # Stage 1: Custom GEMM
        gemm_out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)
        grid_gemm = (n // GROUP_N, m // GROUP_M, 1)
        gemm_kernel[lambda: (grid_gemm, (THREADS, 1, 1))](
            x_bf16, w_bf16, gemm_b_bf16, gemm_out, m, n, k_in,
        )

        # Stage 2: PyTorch GroupNorm (for exact numerical match)
        gn_out = self.group_norm(gemm_out)

        # Stage 3: Custom min reduction
        min_out = torch.empty((m,), device=x_bf16.device, dtype=torch.bfloat16)
        min_reduce_kernel[lambda: ((m, 1, 1), (REDUCE_BLOCK, 1, 1))](
            gn_out, min_out, m, n,
        )

        # Stage 4: Custom bias add
        bias_flat = _prepare_bf16(self.bias.reshape(-1).contiguous())
        bias_channels = bias_flat.shape[0]
        total_out = m * bias_channels
        bias_out_flat = torch.empty((total_out,), device=x_bf16.device, dtype=torch.bfloat16)
        grid_bias = ((total_out + BIAS_BLOCK - 1) // BIAS_BLOCK, 1, 1)
        bias_add_kernel[lambda: (grid_bias, (BIAS_BLOCK, 1, 1))](
            min_out, bias_flat, bias_out_flat, m, bias_channels,
        )

        result = bias_out_flat.view(m, bias_channels).permute(1, 0).contiguous().unsqueeze(0).unsqueeze(-1).to(dtype=x.dtype)
        return result


# ── Preserve the original module-level contract ──────────────────────
batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 512
bias_shape = (1, out_features, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
