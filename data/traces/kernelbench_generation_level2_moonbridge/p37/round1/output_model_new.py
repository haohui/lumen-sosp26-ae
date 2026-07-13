import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── Problem constants ───────────────────────────────────────────────
BATCH_SIZE = 32768
IN_FEATURES = 1024
OUT_FEATURES = 4096
NUM_GROUPS = 64
GROUP_SIZE = OUT_FEATURES // NUM_GROUPS  # 64
EPS = 1e-5

# ── GEMM tile constants (32x32 MFMA) ─────────────────────────────────
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS          # 256
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
M_TILES_PER_WARP = GROUP_M // (WARPS_M * MMA_M)   # 2
N_TILES_PER_WARP = GROUP_N // (WARPS_N * MMA_N)   # 2
A_VECS_PER_ROW = GROUP_K // VEC_ELEMS              # 2
B_VECS_PER_ROW = GROUP_K // VEC_ELEMS              # 2
SHM_A_VECS = GROUP_M * A_VECS_PER_ROW             # 256
SHM_B_VECS = GROUP_N * B_VECS_PER_ROW             # 256
GLOBAL_LOADS_A = SHM_A_VECS // THREADS             # 1
GLOBAL_LOADS_B = SHM_B_VECS // THREADS             # 1
ROW_U32 = A_VECS_PER_ROW * 4                       # 8


# ═════════════════════════════════════════════════════════════════════
# GEMM helpers
# ═════════════════════════════════════════════════════════════════════

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


# Fetch MFMA operand for 32x32 from shared memory into u32 register tile.
# This stores 4 u32 (= 8 bf16, K=16 slice) per tile.

@avelang.jit
def _fetch_mfma_operand_32x32x16(
    ret: al.Tensor((4,), al.u32),
    shm: al.Tensor((SHM_A_VECS, 4), al.u32),
    tile_idx: al.u32,
    lane: al.u32,
):
    shm_u32 = al.view(shm, al.Tensor((SHM_A_VECS * 4,), al.u32))
    row = tile_idx * MMA_M + (lane % MMA_M)
    k_group_u32 = (lane // MMA_M) * 2
    row_base = row * ROW_U32

    ret[0] = shm_u32[row_base + k_group_u32]
    ret[1] = shm_u32[row_base + k_group_u32 + 1]
    ret[2] = shm_u32[row_base + 4 + k_group_u32]
    ret[3] = shm_u32[row_base + 5 + k_group_u32]


# ═════════════════════════════════════════════════════════════════════
# Kernel 1: fused GEMM + Swish + bias addition
# ═════════════════════════════════════════════════════════════════════

@avelang.jit
def linear_fused_swish_bias_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    lin_bias_ptr: al.Pointer(al.bf16),
    add_bias_ptr: al.Pointer(al.bf16),
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
    g_lin_bias = al.make_tensor(lin_bias_ptr, al.bf16, al.make_layout((n,), (1,)))
    g_add_bias = al.make_tensor(add_bias_ptr, al.bf16, al.make_layout((n,), (1,)))
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    x_rsrc = al.amdgpu.make_rsrc(x_memref, m * k * BF16_BYTES)
    w_rsrc = al.amdgpu.make_rsrc(w_memref, n * k * BF16_BYTES)

    shm_a = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_B_VECS, 4), al.u32)

    # Store A/B operands as u32 (matches library pattern)
    a_data = al.make_local((M_TILES_PER_WARP, 4), al.u32)
    b_data = al.make_local((N_TILES_PER_WARP, 4), al.u32)
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
            _fetch_mfma_operand_32x32x16(a_data[i], shm_a, warp_row * M_TILES_PER_WARP + i, lane)
        for j in al.range(N_TILES_PER_WARP):
            _fetch_mfma_operand_32x32x16(b_data[j], shm_b, warp_col * N_TILES_PER_WARP + j, lane)

        # MFMA with u32 view: 32x32 tile, K=16 total, two K=8 halves
        for i in al.range(M_TILES_PER_WARP):
            for j in al.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                # View a_data[i] as (2, 2, 1) u32 -> frag[0] is K-low (2 u32 = 4 bf16)
                # frag[1] is K-high
                frag_a = al.view(a_data[i], al.Tensor((2, 2, 1), al.u32))
                frag_b = al.view(b_data[j], al.Tensor((2, 2, 1), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a[0], frag_b[0], acc[acc_idx])
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a[1], frag_b[1], acc[acc_idx])

        al.syncthreads()

    # Epilogue: Swish + bias + writeback
    one = al.convert(1.0, al.f32)
    lane_group = lane // MMA_N    # 0 or 1
    lane_col = lane % MMA_N       # 0..31
    block_row_base = block_m * GROUP_M
    block_col_base = block_n * GROUP_N

    for j in al.range(N_TILES_PER_WARP):
        col = block_col_base + (warp_col * N_TILES_PER_WARP + j) * MMA_N + lane_col
        lin_b = al.convert(g_lin_bias[col], al.f32)
        add_b = al.convert(g_add_bias[col], al.f32)
        for i in al.range(M_TILES_PER_WARP):
            acc_idx = i * N_TILES_PER_WARP + j
            row_base = block_row_base + (warp_row * M_TILES_PER_WARP + i) * MMA_M
            for t in al.range(ACC_SIZE):
                # Standard 32x32 MFMA accumulator mapping:
                # row = row_base + (t//4)*8 + lane_group*4 + (t%4)
                # col = col_base + lane_col
                row = row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
                val = acc[acc_idx, t] + lin_b
                # Swish: val * sigmoid(val)
                neg_val = al.convert(0.0, al.f32) - val
                sigmoid_val = one / (one + al.exp(neg_val))
                swish_val = sigmoid_val * val
                result = swish_val + add_b
                g_out[row, col] = al.convert(result, al.bf16)


# ═════════════════════════════════════════════════════════════════════
# Kernel 2: GroupNorm — one CTA per (batch, group), batch stats
# ═════════════════════════════════════════════════════════════════════

@avelang.jit
def groupnorm_kernel(
    x_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    out_features: al.i32,
    num_groups: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    group_size = 64
    batch_idx = bid // num_groups
    group_idx = bid - batch_idx * num_groups

    if batch_idx < batch_size and tid < group_size:
        layout_flat = al.make_layout((batch_size * out_features,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_flat)
        out = al.make_tensor(out_ptr, al.bf16, layout_flat)
        layout_wb = al.make_layout((out_features,), (1,))
        w = al.make_tensor(weight_ptr, al.bf16, layout_wb)
        b = al.make_tensor(bias_ptr, al.bf16, layout_wb)

        group_start = batch_idx * out_features + group_idx * group_size
        gidx = group_start + tid
        x_val = al.convert(x[gidx], al.f32)

        smem = al.make_shared((64,), al.f32)

        # -- Pass 1: compute mean --
        smem[tid] = x_val
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

        if tid == 0:
            gs = al.convert(group_size, al.f32)
            mean = smem[0] / gs
            smem[0] = mean

        al.syncthreads()

        # -- Pass 2: compute var from squared deviations --
        mean = smem[0]
        dev = x_val - mean
        smem[tid] = dev * dev
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

        if tid == 0:
            gs = al.convert(group_size, al.f32)
            var = smem[0] / gs
            eps_f32 = al.convert(1e-5, al.f32)
            rstd = al.convert(1.0, al.f32) / al.sqrt(var + eps_f32)
            smem[0] = rstd
            smem[1] = mean

        al.syncthreads()

        rstd = smem[0]
        mean = smem[1]
        normalized = (x_val - mean) * rstd


        w_idx = group_idx * group_size + tid
        w_val = al.convert(w[w_idx], al.f32)
        b_val = al.convert(b[w_idx], al.f32)
        result = normalized * w_val + b_val
        out[gidx] = al.convert(result, al.bf16)


# ═════════════════════════════════════════════════════════════════════
# Host wrappers
# ═════════════════════════════════════════════════════════════════════

def _to_bf16(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    return t.contiguous().to(dtype=torch.bfloat16)


def avelang_linear_swish_bias(
    x: torch.Tensor,
    weight: torch.Tensor,
    lin_bias: torch.Tensor,
    add_bias: torch.Tensor,
) -> torch.Tensor:
    x_bf16 = _to_bf16(x)
    w_bf16 = _to_bf16(weight)
    lb_bf16 = _to_bf16(lin_bias)
    ab_bf16 = _to_bf16(add_bias)

    m, k = x_bf16.shape
    n, wk = w_bf16.shape
    if wk != k:
        raise ValueError(f"Weight/input K mismatch: x has K={k}, weight has K={wk}")
    if m % GROUP_M != 0 or n % GROUP_N != 0 or k % GROUP_K != 0:
        raise ValueError(
            f"Expected m % {GROUP_M} == 0, n % {GROUP_N} == 0, k % {GROUP_K} == 0 "
            f"(got m={m}, n={n}, k={k})"
        )

    out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)
    grid_m = m // GROUP_M
    grid_n = n // GROUP_N
    grid = (grid_n, grid_m, 1)
    linear_fused_swish_bias_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, w_bf16, lb_bf16, ab_bf16, out, m, n, k
    )
    return out


def avelang_groupnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    x_bf16 = _to_bf16(x)
    w_bf16 = _to_bf16(weight)
    b_bf16 = _to_bf16(bias)

    batch_size, out_features = x_bf16.shape
    num_groups = NUM_GROUPS
    if out_features % num_groups != 0:
        raise ValueError(f"out_features {out_features} not divisible by num_groups {num_groups}")

    out = torch.empty_like(x_bf16)
    total_groups = batch_size * num_groups

    groupnorm_kernel[lambda: ((total_groups, 1, 1), (GROUP_SIZE, 1, 1))](
        x_bf16, w_bf16, b_bf16, out, batch_size, out_features, num_groups
    )
    return out


# ═════════════════════════════════════════════════════════════════════
# ModelNew
# ═════════════════════════════════════════════════════════════════════

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        w = self.matmul.weight.data
        lb = self.matmul.bias.data
        ab = self.bias.data
        gn_w = self.group_norm.weight.data
        gn_b = self.group_norm.bias.data

        x = avelang_linear_swish_bias(x, w, lb, ab)
        x = avelang_groupnorm(x, gn_w, gn_b)
        return x
