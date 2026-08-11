import torch
import torch.nn as nn
import substrate
import substrate.language as S
import math


# ============================================================================
# GEMM Configuration (from amdgpu_gemm.py)
# ============================================================================
WARP_SIZE = 64
NUM_WARPS = 4
GROUP_M = 128
GROUP_N = 128
GROUP_K = 64
VEC_SIZE = 8
THREADS = WARP_SIZE * NUM_WARPS

MMA_M = 16
MMA_N = 16
MMA_K = 16

TILE_M = 16
TILE_N = 16

WARPS_M = 2
WARPS_N = 2

WARP_PER_ROW = 2
WARP_PER_COL = 2
WARP_MAT_M = GROUP_M // WARP_PER_ROW
WARP_MAT_N = GROUP_N // WARP_PER_COL
M_TILES_PER_WARP = WARP_MAT_M // MMA_M
N_TILES_PER_WARP = WARP_MAT_N // MMA_N
BATCH_K = GROUP_K // MMA_K
MATMUL_K_VEC = GROUP_K // VEC_SIZE
MATMUL_K_TILES = GROUP_K // (VEC_SIZE * BATCH_K)
ROW_STRIDE = MATMUL_K_VEC * 4
COL_STRIDE = 4

SHM_PAD_ROWS = 4
SHM_PAD_BYTES = 32
SHM_PAD_U32 = SHM_PAD_BYTES // 4
SHM_A_U32 = GROUP_M * ROW_STRIDE + (GROUP_M // SHM_PAD_ROWS) * SHM_PAD_U32
SHM_B_U32 = GROUP_N * ROW_STRIDE + (GROUP_N // SHM_PAD_ROWS) * SHM_PAD_U32


# ============================================================================
# GEMM Kernels
# ============================================================================
@substrate.jit
def load_global(
    A: S.Pointer(S.u32),
    B: S.Pointer(S.u32),
    m: S.u32,
    n: S.u32,
    k: S.u32,
    tile_a_idx_row: S.u32,
    tile_a_idx_col: S.u32,
    tile_b_idx_row: S.u32,
    tile_b_idx_col: S.u32,
    reg_a: S.Tensor((GROUP_M * GROUP_K // VEC_SIZE // THREADS, 4,), S.u32,),
    reg_b: S.Tensor((GROUP_N * GROUP_K // VEC_SIZE // THREADS, 4,), S.u32,),
):
    K_VEC = GROUP_K // VEC_SIZE
    k_vec = k // VEC_SIZE
    layout_a = S.make_layout(
        (m, k_vec, 4),
        (k_vec * 4, 4, 1),
    )
    layout_b = S.make_layout(
        (n, k_vec, 4),
        (k_vec * 4, 4, 1),
    )
    g_a = S.make_tensor(A, S.u32, layout_a)
    g_b = S.make_tensor(B, S.u32, layout_b)

    tid = S.thread_id(0)
    idx = tid
    for i in S.range(GROUP_M * GROUP_K // VEC_SIZE // THREADS):
        row = idx // K_VEC
        col = idx % K_VEC
        reg_a[i] = g_a[tile_a_idx_row * GROUP_M + row, tile_a_idx_col * K_VEC + col]
        idx = idx + THREADS

    idx = tid
    for i in S.range(GROUP_N * GROUP_K // VEC_SIZE // THREADS):
        row = idx // K_VEC
        col = idx % K_VEC
        reg_b[i] = g_b[tile_b_idx_row * GROUP_N + row, tile_b_idx_col * K_VEC + col]
        idx = idx + THREADS


@substrate.jit
def store_shm(
    shm_a: S.Tensor((SHM_A_U32,), S.u32),
    shm_b: S.Tensor((SHM_B_U32,), S.u32),
    reg_a: S.Tensor((GROUP_M * GROUP_K // VEC_SIZE // THREADS, 4,), S.u32,),
    reg_b: S.Tensor((GROUP_N * GROUP_K // VEC_SIZE // THREADS, 4,), S.u32,),
):
    REG_SIZE_A = GROUP_M * GROUP_K // VEC_SIZE // THREADS
    REG_SIZE_B = GROUP_N * GROUP_K // VEC_SIZE // THREADS
    K_VEC = GROUP_K // VEC_SIZE
    layout_sa = S.make_layout(
        ((GROUP_M // SHM_PAD_ROWS, SHM_PAD_ROWS), MATMUL_K_VEC, 4),
        ((ROW_STRIDE * SHM_PAD_ROWS + SHM_PAD_U32, ROW_STRIDE), 4, 1),
    )
    layout_sb = S.make_layout(
        ((GROUP_N // SHM_PAD_ROWS, SHM_PAD_ROWS), MATMUL_K_VEC, 4),
        ((ROW_STRIDE * SHM_PAD_ROWS + SHM_PAD_U32, ROW_STRIDE), 4, 1),
    )
    s_a = S.view(shm_a, S.u32, layout_sa)
    s_b = S.view(shm_b, S.u32, layout_sb)
    tid = S.thread_id(0)
    idx = tid
    for i in S.range(REG_SIZE_A):
        row = idx // K_VEC
        col = idx % K_VEC
        s_a[row, col] = reg_a[i]
        idx = idx + THREADS

    idx = tid
    for i in S.range(REG_SIZE_B):
        row = idx // K_VEC
        col = idx % K_VEC
        s_b[row, col] = reg_b[i]
        idx = idx + THREADS


@substrate.jit
def load_shm_to_regs(
    shm_a: S.Tensor((SHM_A_U32,), S.u32),
    shm_b: S.Tensor((SHM_B_U32,), S.u32),
    tile_m: S.u32,
    tile_n: S.u32,
    k_tile: S.u32,
    tile_a: S.Tensor((1, 4), S.u32),
    tile_b: S.Tensor((1, 4), S.u32),
):
    tid = S.thread_id(0)
    wid = tid // WARP_SIZE
    wtid = tid % WARP_SIZE
    warp_row = wid // WARP_PER_COL
    warp_col = wid % WARP_PER_COL
    mma_k = wtid % MMA_K
    batch_k = wtid // MMA_K

    layout_sa = S.make_layout(
        ((GROUP_M // SHM_PAD_ROWS, SHM_PAD_ROWS), MATMUL_K_VEC, 4),
        ((ROW_STRIDE * SHM_PAD_ROWS + SHM_PAD_U32, ROW_STRIDE), 4, 1),
    )
    layout_sb = S.make_layout(
        ((GROUP_N // SHM_PAD_ROWS, SHM_PAD_ROWS), MATMUL_K_VEC, 4),
        ((ROW_STRIDE * SHM_PAD_ROWS + SHM_PAD_U32, ROW_STRIDE), 4, 1),
    )
    s_a = S.view(shm_a, S.u32, layout_sa)
    s_b = S.view(shm_b, S.u32, layout_sb)

    row_a = warp_row * (M_TILES_PER_WARP * TILE_M) + tile_m * TILE_M + mma_k
    col_a = k_tile * BATCH_K + batch_k
    tile_a[0] = s_a[row_a, col_a]

    row_b = warp_col * (N_TILES_PER_WARP * TILE_N) + tile_n * TILE_N + mma_k
    col_b = k_tile * BATCH_K + batch_k
    tile_b[0] = s_b[row_b, col_b]


@substrate.jit
def matmul_from_regs(
    tile_a: S.Tensor((1, 4), S.u32),
    tile_b: S.Tensor((1, 4), S.u32),
    acc: S.Tensor((GROUP_M * GROUP_N // THREADS // 4, 4,), S.f32,),
    acc_idx: S.u32,
):
    r_a = tile_a[0]
    r_b = tile_b[0]
    r_a_bf16 = S.view(r_a, S.Tensor((2, 4, 1), S.bf16))
    r_b_bf16 = S.view(r_b, S.Tensor((2, 4, 1), S.bf16))
    acc[acc_idx] = S.amdgpu.mfma_f32_16x16x16_bf16(r_b_bf16[0], r_a_bf16[0], acc[acc_idx])
    acc[acc_idx] = S.amdgpu.mfma_f32_16x16x16_bf16(r_b_bf16[1], r_a_bf16[1], acc[acc_idx])


@substrate.jit
def write_results(
    id_m: S.u32,
    id_n: S.u32,
    acc: S.Tensor((GROUP_M * GROUP_N // THREADS // 4, 4,), S.f32),
    C: S.Pointer(S.u32),
    m: S.u32,
    n: S.u32,
):
    tid = S.thread_id(0)
    wid = tid // WARP_SIZE
    lane_id = tid % WARP_SIZE

    row_block = TILE_M // (WARP_SIZE // TILE_N)

    layout_c = S.make_layout(
        (
            m // GROUP_M, n // GROUP_N,
            (WARP_PER_COL, WARP_PER_ROW),
            (N_TILES_PER_WARP, M_TILES_PER_WARP),
            (TILE_N, TILE_N // 4),
            2,
        ),
        (
            WARP_PER_ROW * M_TILES_PER_WARP * TILE_M * (n // 2), GROUP_N // 2,
            (N_TILES_PER_WARP * (TILE_N // 2), M_TILES_PER_WARP * TILE_M * (n // 2)),
            (TILE_N // 2, TILE_M * (n // 2)),
            (n // 2, 2),
            1,
        ),
    )
    g_c = S.make_tensor(C, S.u32, layout_c)

    for acc_idx in S.range(M_TILES_PER_WARP * N_TILES_PER_WARP):
        acc_vec = acc[acc_idx]
        r = S.make_local((4,), S.bf16)
        for t in S.range(row_block):
            r[t] = S.convert(acc_vec[t], S.bf16)

        packed = S.view(r, S.Tensor((2,), S.u32))
        g_c[id_m, id_n, wid, acc_idx, lane_id] = packed


@substrate.jit
def wgm_mapping(m: S.u32, n: S.u32) -> (S.u32, S.u32):
    block_id_linear = S.block_id(0)
    linear_group_id = S.convert(block_id_linear, S.u32)
    group_m_size = S.convert(GROUP_M, S.u32)
    group_n_size = S.convert(GROUP_N, S.u32)
    m_groups = m // group_m_size
    n_groups = n // group_n_size

    total_groups = m_groups * n_groups

    cu_count = S.convert(38 * 8, S.u32)
    wgm_xcc = S.convert(8, S.u32)
    workgroup_mapping = S.convert(32, S.u32)

    linear_group_limit = (total_groups // wgm_xcc) * wgm_xcc
    cu_base = (linear_group_id // cu_count) * cu_count
    cu_xcc = (linear_group_id % cu_count) // wgm_xcc
    cu_base = cu_base + cu_xcc

    cu_tail_limit = (total_groups // cu_count) * cu_count
    active_cu = (total_groups % cu_count) if (linear_group_id > cu_tail_limit) else cu_count
    cu_xcc_stride = (active_cu // wgm_xcc) * (linear_group_id % wgm_xcc)
    linear_group_mapped = cu_base + cu_xcc_stride

    linear_group_id = (
        linear_group_mapped if (linear_group_id < linear_group_limit) else linear_group_id
    )

    group_m = linear_group_id // n_groups
    group_n = linear_group_id - group_m * n_groups

    mapping_block = group_m // workgroup_mapping
    mapping_linear = group_n + (group_m % workgroup_mapping) * n_groups
    mapping_groups = m_groups // workgroup_mapping
    mapping_tail = m_groups % workgroup_mapping
    mapping_tail = mapping_tail if (mapping_tail != 0) else workgroup_mapping
    mapping_span = mapping_tail if (mapping_block >= mapping_groups) else workgroup_mapping

    group_n = mapping_linear // mapping_span
    group_m = mapping_linear % mapping_span
    group_m = group_m + mapping_block * workgroup_mapping

    return group_m, group_n


@substrate.jit
def _gemm_1stage_pipeline_kernel(
    A: S.Pointer(S.u32),
    B: S.Pointer(S.u32),
    C: S.Pointer(S.u32),
    m: S.u32,
    n: S.u32,
    k: S.u32,
):
    id_m, id_n = wgm_mapping(m, n)

    shm_a = S.make_shared((SHM_A_U32,), S.u32)
    shm_b = S.make_shared((SHM_B_U32,), S.u32)
    acc = S.make_local((GROUP_M * GROUP_N // THREADS // 4, 4,), S.f32)
    for i in S.range(GROUP_M * GROUP_N // THREADS // 4):
        for j in S.range(4):
            acc[i, j] = 0

    k_tiles = k // GROUP_K
    k_tiles_u32 = S.convert(k_tiles, S.u32)
    k_stagger_mask = S.convert(0x7, S.u32)
    k_stagger_stride = S.convert(4, S.u32)
    stagger_data = id_n
    k_start = (stagger_data & k_stagger_mask) * k_stagger_stride
    k_start = k_start if (k_start < k_tiles_u32) else S.convert(0, S.u32)

    reg_a = S.make_local((GROUP_M * GROUP_K // VEC_SIZE // THREADS, 4,), S.u32,)
    reg_b = S.make_local((GROUP_N * GROUP_K // VEC_SIZE // THREADS, 4,), S.u32,)
    tile_a0 = S.make_local((1, 4), S.u32)
    tile_b0 = S.make_local((1, 4), S.u32)
    tile_a1 = S.make_local((1, 4), S.u32)
    tile_b1 = S.make_local((1, 4), S.u32)

    load_global(A, B, m, n, k, id_m, k_start, id_n, k_start, reg_a, reg_b)
    store_shm(shm_a, shm_b, reg_a, reg_b)
    S.syncthreads()

    for k_iter in S.range(k_tiles - 1):
        k_next = k_start + k_iter + 1
        k_next = k_next - (k_tiles_u32 if (k_next >= k_tiles_u32) else S.convert(0, S.u32))
        load_global(A, B, m, n, k, id_m, k_next, id_n, k_next, reg_a, reg_b)

        for i in S.range(M_TILES_PER_WARP):
            for j in S.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                load_shm_to_regs(shm_a, shm_b, i, j, 0, tile_a0, tile_b0)
                load_shm_to_regs(shm_a, shm_b, i, j, 1, tile_a1, tile_b1)
                matmul_from_regs(tile_a0, tile_b0, acc, acc_idx)
                matmul_from_regs(tile_a1, tile_b1, acc, acc_idx)
        S.syncthreads()

        store_shm(shm_a, shm_b, reg_a, reg_b)
        S.syncthreads()

    for i in S.range(M_TILES_PER_WARP):
        for j in S.range(N_TILES_PER_WARP):
            acc_idx = i * N_TILES_PER_WARP + j
            load_shm_to_regs(shm_a, shm_b, i, j, 0, tile_a0, tile_b0)
            load_shm_to_regs(shm_a, shm_b, i, j, 1, tile_a1, tile_b1)
            matmul_from_regs(tile_a0, tile_b0, acc, acc_idx)
            matmul_from_regs(tile_a1, tile_b1, acc, acc_idx)

    write_results(id_m, id_n, acc, C, m, n)


# ============================================================================
# Add Bias Kernel
# ============================================================================
BLOCK_SIZE_BIAS: S.constexpr = 256


@substrate.jit
def add_bias_kernel(
    output_ptr: S.Pointer(S.bf16),
    bias_ptr: S.Pointer(S.bf16),
    m: S.i32,
    n: S.i32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = m * n
    if idx < total:
        layout_out = S.make_layout((m, n), (n, 1))
        output = S.make_tensor(output_ptr, S.bf16, layout_out)
        layout_bias = S.make_layout((n,), (1,))
        bias = S.make_tensor(bias_ptr, S.bf16, layout_bias)

        row = idx // n
        col = idx % n
        val = output[row, col]
        b = bias[col]
        output[row, col] = val + b


# ============================================================================
# LogSumExp Kernel
# ============================================================================
BLOCK_SIZE_LOGSUMEXP: S.constexpr = 256
LOG2E = 1.4426950408889634  # log2(e)


@substrate.jit
def logsumexp_kernel(
    input_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    dim: S.i32,
):
    row_idx = S.block_id(0)

    if row_idx < batch_size:
        layout_in = S.make_layout((batch_size, dim), (dim, 1))
        input_tensor = S.make_tensor(input_ptr, S.bf16, layout_in)

        # Find max value in the row
        max_val = input_tensor[row_idx, 0]
        for i in S.range(1, dim):
            val = input_tensor[row_idx, i]
            max_val = val if val > max_val else max_val

        # Compute sum of exp(x - max) using exp2
        # exp(x) = exp2(x * log2(e))
        max_f32 = S.convert(max_val, S.f32)
        log2e = S.convert(LOG2E, S.f32)
        sum_exp = S.convert(0.0, S.f32)
        for i in S.range(dim):
            val = S.convert(input_tensor[row_idx, i], S.f32)
            shifted = val - max_f32
            exp_val = S.exp2(shifted * log2e)
            sum_exp = sum_exp + exp_val

        # Compute log(sum_exp) + max using log2
        # log(x) = log2(x) / log2(e)
        log_sum = S.log2(sum_exp) / log2e
        result = log_sum + max_f32

        layout_out = S.make_layout((batch_size, 1), (1, 1))
        output = S.make_tensor(output_ptr, S.bf16, layout_out)
        output[row_idx, 0] = S.convert(result, S.bf16)


# ============================================================================
# LeakyReLU Kernel (applied twice)
# ============================================================================
BLOCK_SIZE_ACTIVATION: S.constexpr = 256


@substrate.jit
def leaky_relu_twice_kernel(
    input_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    n: S.i32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if idx < n:
        layout = S.make_layout((n,), (1,))
        input_tensor = S.make_tensor(input_ptr, S.bf16, layout)
        output_tensor = S.make_tensor(output_ptr, S.bf16, layout)

        val = input_tensor[idx]
        val_f32 = S.convert(val, S.f32)

        # Apply LeakyReLU twice with slope=0.01
        zero = S.convert(0.0, S.f32)
        slope = S.convert(0.01, S.f32)
        slope_sq = S.convert(0.0001, S.f32)  # 0.01 * 0.01

        # First LeakyReLU
        val1 = val_f32 if val_f32 >= zero else val_f32 * slope
        # Second LeakyReLU
        result = val1 if val1 >= zero else val1 * slope

        output_tensor[idx] = S.convert(result, S.bf16)


# ============================================================================
# GELU Kernel (applied twice)
# ============================================================================
SQRT_2_OVER_PI = 0.7978845608028654
GELU_COEFF = 0.044715
LOG2E_GELU = 1.4426950408889634


@substrate.jit
def gelu_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    y_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    layout = S.make_layout((n,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    y = S.make_tensor(y_ptr, S.bf16, layout)

    if idx < n:
        xv = S.convert(x[idx], S.f32)

        sqrt_2_over_pi = S.convert(SQRT_2_OVER_PI, S.f32)
        coeff = S.convert(GELU_COEFF, S.f32)
        half = S.convert(0.5, S.f32)
        one = S.convert(1.0, S.f32)
        two = S.convert(2.0, S.f32)
        log2e = S.convert(LOG2E_GELU, S.f32)

        x3 = xv * xv * xv
        inner = sqrt_2_over_pi * (xv + coeff * x3)

        # Use exp2 instead of exp: exp(x) = exp2(x * log2(e))
        exp_arg = two * inner * log2e
        exp_val = S.exp2(exp_arg)

        tanh_inner = (exp_val - one) / (exp_val + one)

        result = half * xv * (one + tanh_inner)
        y[idx] = S.convert(result, S.bf16)


@substrate.jit
def gelu_twice_kernel(
    x_ptr: S.Pointer(S.bf16),
    y_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    layout = S.make_layout((n,), (1,))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    y = S.make_tensor(y_ptr, S.bf16, layout)

    if idx < n:
        xv = S.convert(x[idx], S.f32)

        sqrt_2_over_pi = S.convert(SQRT_2_OVER_PI, S.f32)
        coeff = S.convert(GELU_COEFF, S.f32)
        half = S.convert(0.5, S.f32)
        one = S.convert(1.0, S.f32)
        two = S.convert(2.0, S.f32)
        log2e = S.convert(LOG2E_GELU, S.f32)

        # First GELU
        x3 = xv * xv * xv
        inner = sqrt_2_over_pi * (xv + coeff * x3)
        exp_arg = two * inner * log2e
        exp_val = S.exp2(exp_arg)
        tanh_inner = (exp_val - one) / (exp_val + one)
        gelu1 = half * xv * (one + tanh_inner)

        # Second GELU
        x3_2 = gelu1 * gelu1 * gelu1
        inner2 = sqrt_2_over_pi * (gelu1 + coeff * x3_2)
        exp_arg2 = two * inner2 * log2e
        exp_val2 = S.exp2(exp_arg2)
        tanh_inner2 = (exp_val2 - one) / (exp_val2 + one)
        result = half * gelu1 * (one + tanh_inner2)

        y[idx] = S.convert(result, S.bf16)


# ============================================================================
# Host Wrappers
# ============================================================================
def substrate_gemm_add_bias(A: torch.Tensor, W_t: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """GEMM with bias: C = A @ W_t + bias"""
    m, k = A.shape
    n, _ = W_t.shape

    # Validate shapes
    assert m % GROUP_M == 0, f"M ({m}) must be multiple of {GROUP_M}"
    assert n % GROUP_N == 0, f"N ({n}) must be multiple of {GROUP_N}"
    assert k % GROUP_K == 0, f"K ({k}) must be multiple of {GROUP_K}"

    # Allocate output
    out = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)

    # Launch GEMM
    m_groups = (m + GROUP_M - 1) // GROUP_M
    n_groups = (n + GROUP_N - 1) // GROUP_N
    grid = (m_groups * n_groups, 1, 1)
    block = (NUM_WARPS * WARP_SIZE, 1, 1)

    _gemm_1stage_pipeline_kernel[lambda: (grid, block)](A, W_t, out, m, n, k)

    # Add bias
    total = m * n
    bias_grid = ((total + BLOCK_SIZE_BIAS - 1) // BLOCK_SIZE_BIAS, 1, 1)
    add_bias_kernel[lambda: (bias_grid, (BLOCK_SIZE_BIAS, 1, 1))](out, bias, m, n)

    return out


def substrate_logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute logsumexp along dimension"""
    assert dim == 1, "Only dim=1 is supported"

    batch_size = x.shape[0]
    dim_size = x.shape[1]

    output = torch.empty((batch_size, 1), dtype=torch.bfloat16, device=x.device)

    grid = (batch_size, 1, 1)
    block = (1, 1, 1)  # Each thread handles one row

    logsumexp_kernel[lambda: (grid, block)](x, output, batch_size, dim_size)

    return output


def substrate_leaky_relu_twice(x: torch.Tensor) -> torch.Tensor:
    """Apply LeakyReLU twice with slope=0.01"""
    n = x.numel()
    output = torch.empty_like(x)

    grid = ((n + BLOCK_SIZE_ACTIVATION - 1) // BLOCK_SIZE_ACTIVATION, 1, 1)
    leaky_relu_twice_kernel[lambda: (grid, (BLOCK_SIZE_ACTIVATION, 1, 1))](x, output, n)

    return output


def substrate_gelu_twice(x: torch.Tensor) -> torch.Tensor:
    """Apply GELU twice"""
    n = x.numel()
    output = torch.empty_like(x)

    grid = ((n + BLOCK_SIZE_ACTIVATION - 1) // BLOCK_SIZE_ACTIVATION, 1, 1)
    gelu_twice_kernel[lambda: (grid, (BLOCK_SIZE_ACTIVATION, 1, 1))](x, output, n)

    return output


# ============================================================================
# ModelNew
# ============================================================================
class ModelNew(nn.Module):
    """
    Optimized model using Substrate DSL kernels.
    Performs: Linear (GEMM+bias) -> LogSumExp -> LeakyReLU -> LeakyReLU -> GELU -> GELU
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to bfloat16
        x_bf16 = x.to(torch.bfloat16)
        weight_bf16 = self.weight.to(torch.bfloat16)
        bias_bf16 = self.bias.to(torch.bfloat16) if self.bias is not None else None

        # Ensure contiguous
        x_contig = x_bf16.contiguous()
        weight_t = weight_bf16.t().contiguous()  # Transpose for GEMM

        # GEMM + bias
        out = substrate_gemm_add_bias(x_contig, weight_t, bias_bf16)

        # LogSumExp over dim=1
        out = substrate_logsumexp(out, dim=1)

        # Two LeakyReLU applications
        out = substrate_leaky_relu_twice(out)

        # Two GELU applications
        out = substrate_gelu_twice(out)

        return out


batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
