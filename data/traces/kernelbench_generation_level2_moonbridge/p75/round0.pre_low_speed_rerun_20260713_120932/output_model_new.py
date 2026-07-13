import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── GEMM constants ────────────────────────────────────────────────────────────
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

# ── Min + Bias constants ──────────────────────────────────────────────────────
MIN_BLOCK_SIZE = 256


# ═══════════════════════════════════════════════════════════════════════════════
# GEMM helper kernels
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 1: BF16 GEMM with MFMA (BF16 in/out, FP32 accumulation)
# ═══════════════════════════════════════════════════════════════════════════════

@avelang.jit
def gemm_linear_kernel(
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
                a_op0 = al.view(a_reg[i, 0], al.Tensor((2,), al.u32))
                a_op1 = al.view(a_reg[i, 1], al.Tensor((2,), al.u32))
                b_op0 = al.view(b_reg[j, 0], al.Tensor((2,), al.u32))
                b_op1 = al.view(b_reg[j, 1], al.Tensor((2,), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_op0, b_op0, acc[acc_idx])
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_op1, b_op1, acc[acc_idx])

        al.syncthreads()

    # Epilogue: linear bias, write BF16
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


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 2: Global min across features + bias broadcast
# ═══════════════════════════════════════════════════════════════════════════════

@avelang.jit
def min_bias_kernel(
    gn_out_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    out_features: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < batch_size:
        smem = al.make_shared((MIN_BLOCK_SIZE,), al.f32)

        layout_2d = al.make_layout((batch_size, out_features), (out_features, 1))
        gn_out = al.make_tensor(gn_out_ptr, al.bf16, layout_2d)

        # Load first value
        local_min = al.convert(gn_out[bid, tid], al.f32)

        # Strided loop over remaining values
        for i in al.range(tid + MIN_BLOCK_SIZE, out_features, MIN_BLOCK_SIZE):
            val = al.convert(gn_out[bid, i], al.f32)
            if val < local_min:
                local_min = val

        smem[tid] = local_min
        al.syncthreads()

        # Tree reduction: 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
        if tid < 128:
            other = smem[tid + 128]
            smem[tid] = other if other < smem[tid] else smem[tid]
        al.syncthreads()
        if tid < 64:
            other = smem[tid + 64]
            smem[tid] = other if other < smem[tid] else smem[tid]
        al.syncthreads()
        if tid < 32:
            other = smem[tid + 32]
            smem[tid] = other if other < smem[tid] else smem[tid]
        al.syncthreads()
        if tid < 16:
            other = smem[tid + 16]
            smem[tid] = other if other < smem[tid] else smem[tid]
        al.syncthreads()
        if tid < 8:
            other = smem[tid + 8]
            smem[tid] = other if other < smem[tid] else smem[tid]
        al.syncthreads()
        if tid < 4:
            other = smem[tid + 4]
            smem[tid] = other if other < smem[tid] else smem[tid]
        al.syncthreads()
        if tid < 2:
            other = smem[tid + 2]
            smem[tid] = other if other < smem[tid] else smem[tid]
        al.syncthreads()
        if tid < 1:
            other = smem[tid + 1]
            smem[tid] = other if other < smem[tid] else smem[tid]
        al.syncthreads()

        global_min = smem[0]

        # Write output: out[0, c, bid, 0] = global_min + bias[c]
        bias_1d = al.make_tensor(bias_ptr, al.f32, al.make_layout((out_features,), (1,)))
        out_total = out_features * batch_size
        out_flat = al.make_tensor(out_ptr, al.bf16, al.make_layout((out_total,), (1,)))

        for c in al.range(tid, out_features, MIN_BLOCK_SIZE):
            bias_val = bias_1d[c]
            result = global_min + bias_val
            out_idx = c * batch_size + bid
            out_flat[out_idx] = al.convert(result, al.bf16)


# ═══════════════════════════════════════════════════════════════════════════════
# Host wrapper helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def _prepare_fp32_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.float32 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.float32)
    return t.contiguous().cuda().to(dtype=torch.float32)


def avelang_gemm_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = _prepare_bf16_cuda_contiguous(weight)
    b_bf16 = _prepare_bf16_cuda_contiguous(bias)

    m, k_in = x_bf16.shape
    n, k_w = w_bf16.shape
    if k_in != k_w:
        raise ValueError(f"K mismatch: x has K={k_in}, weight has K={k_w}")
    if m % GROUP_M != 0 or n % GROUP_N != 0 or k_in % GROUP_K != 0:
        raise ValueError(
            f"Expected m%{GROUP_M}==0, n%{GROUP_N}==0, k%{GROUP_K}==0 "
            f"(got m={m}, n={n}, k={k_in})"
        )

    out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)
    grid = (n // GROUP_N, m // GROUP_M, 1)
    gemm_linear_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, w_bf16, b_bf16, out, m, n, k_in
    )
    return out


def avelang_min_bias(
    x: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    b_fp32 = _prepare_fp32_cuda_contiguous(bias)

    batch_size, out_features = x_bf16.shape
    out = torch.empty(
        (1, out_features, batch_size, 1),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    grid = (batch_size, 1, 1)
    min_bias_kernel[lambda: (grid, (MIN_BLOCK_SIZE, 1, 1))](
        x_bf16, b_fp32, out, batch_size, out_features
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# ModelNew entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Step 1: GEMM via AveLang MFMA kernel (BF16 in/out)
        ge_out = avelang_gemm_linear(x, self.gemm.weight, self.gemm.bias)

        # Step 2: GroupNorm via PyTorch eager (matches reference exactly)
        gn_out = self.group_norm(ge_out)

        # Step 3: Min + Bias broadcast via AveLang kernel
        final_bias = self.bias.view(-1)
        result = avelang_min_bias(gn_out, final_bias)

        return result


# ── Preserve get_inputs / get_init_inputs from the reference problem ──────────
batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 512
bias_shape = (1, out_features, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
