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

# ── Elementwise kernel constants ──
EW_THREADS = 256
LOG2_E = 1.4426950408889634  # log2(e), for computing e^x = 2^(x * log2(e))


# ══════════════════════════════════════════════════════════════════════
# GEMM kernel
# ══════════════════════════════════════════════════════════════════════

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
                a_low = al.view(a_reg[i, 0], al.Tensor((2,), al.i32))
                b_low = al.view(b_reg[j, 0], al.Tensor((2,), al.i32))
                a_high = al.view(a_reg[i, 1], al.Tensor((2,), al.i32))
                b_high = al.view(b_reg[j, 1], al.Tensor((2,), al.i32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_low, b_low, acc[acc_idx])
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_high, b_high, acc[acc_idx])

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


# ══════════════════════════════════════════════════════════════════════
# Elementwise epilogue: Swish → Multiply → Swish
# ══════════════════════════════════════════════════════════════════════

@avelang.jit
def epilogue_kernel(
    in_ptr: al.Pointer(al.bf16),
    multiply_weight_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    rows: al.u32,
    cols: al.u32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    gid = bid * EW_THREADS + tid
    total_elems = rows * cols

    if gid < total_elems:
        r = gid // cols
        c = gid - r * cols

        layout_2d = al.make_layout((rows, cols), (cols, 1))
        in_tensor = al.make_tensor(in_ptr, al.bf16, layout_2d)
        out_tensor = al.make_tensor(out_ptr, al.bf16, layout_2d)
        layout_1d = al.make_layout((cols,), (1,))
        mw = al.make_tensor(multiply_weight_ptr, al.bf16, layout_1d)

        x_val = al.convert(in_tensor[r, c], al.f32)

        # Swish 1
        log2e = al.convert(LOG2_E, al.f32)
        neg_x_log2e = (al.convert(0.0, al.f32) - x_val) * log2e
        sig1_f32 = al.convert(1.0, al.f32) / (al.convert(1.0, al.f32) + al.exp2(neg_x_log2e))
        sig1_bf16 = al.convert(sig1_f32, al.bf16)
        sig1 = al.convert(sig1_bf16, al.f32)
        result_f32 = x_val * sig1
        result = al.convert(al.convert(result_f32, al.bf16), al.f32)

        # Multiply
        mw_val = al.convert(mw[c], al.f32)
        result_f32 = result * mw_val
        result = al.convert(al.convert(result_f32, al.bf16), al.f32)

        # Swish 2
        neg_r_log2e = (al.convert(0.0, al.f32) - result) * log2e
        sig2_f32 = al.convert(1.0, al.f32) / (al.convert(1.0, al.f32) + al.exp2(neg_r_log2e))
        sig2_bf16 = al.convert(sig2_f32, al.bf16)
        sig2 = al.convert(sig2_bf16, al.f32)
        result_f32 = result * sig2
        result = al.convert(al.convert(result_f32, al.bf16), al.f32)

        out_tensor[r, c] = al.convert(result, al.bf16)


# ══════════════════════════════════════════════════════════════════════
# Host wrappers
# ══════════════════════════════════════════════════════════════════════

def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_gemm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight)
    bias_bf16 = _prepare_bf16_cuda_contiguous(bias)

    m, k = x_bf16.shape
    n, weight_k = weight_bf16.shape
    if weight_k != k:
        raise ValueError(f"Weight/input K mismatch: x has K={k}, weight has K={weight_k}")
    if m % GROUP_M != 0 or n % GROUP_N != 0 or k % GROUP_K != 0:
        raise ValueError(
            f"Expected m % {GROUP_M} == 0, n % {GROUP_N} == 0, k % {GROUP_K} == 0 "
            f"(got m={m}, n={n}, k={k})"
        )

    out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)
    grid = (n // GROUP_N, m // GROUP_M, 1)
    gemm_kernel[lambda: (grid, (THREADS, 1, 1))](x_bf16, weight_bf16, bias_bf16, out, m, n, k)
    return out


def avelang_epilogue(
    x: torch.Tensor, multiply_weight: torch.Tensor
) -> torch.Tensor:
    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    mw_bf16 = _prepare_bf16_cuda_contiguous(multiply_weight)

    total = x_bf16.numel()
    out = torch.empty_like(x_bf16)
    grid = ((total + EW_THREADS - 1) // EW_THREADS, 1, 1)
    rows, cols = x_bf16.shape
    epilogue_kernel[lambda: (grid, (EW_THREADS, 1, 1))](x_bf16, mw_bf16, out, rows, cols)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

    def forward(self, x):
        # GEMM via PyTorch (exact match to reference)
        x = self.gemm(x)
        # GroupNorm via PyTorch (original module, exact behavior)
        x = self.group_norm(x)
        # Swish + Multiply + Swish via AveLang elementwise kernel
        x = avelang_epilogue(x, self.multiply_weight.data)
        return x


batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 256
multiply_weight_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]
