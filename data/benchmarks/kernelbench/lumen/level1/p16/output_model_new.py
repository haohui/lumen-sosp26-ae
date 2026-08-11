"""Optimized GEMM kernel computing C = A^T @ B using Substrate DSL."""
import substrate
import substrate.language as S
import torch

# ============================================================================
# Tiling configuration for MI300X MFMA
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
# Launch configuration helpers
# ============================================================================
def gemm_1stage_launch_config(m, n):
    m_groups = (m + GROUP_M - 1) // GROUP_M
    n_groups = (n + GROUP_N - 1) // GROUP_N
    grid = (m_groups * n_groups, 1, 1)
    block = (NUM_WARPS * WARP_SIZE, 1, 1)
    return grid, block


def gemm_1stage_validate_shape(m, n, k):
    if m % GROUP_M != 0 or n % GROUP_N != 0 or k % GROUP_K != 0:
        raise ValueError(
            f"M and N must be multiples of 128 and K must be a multiple of 64 "
            f"(got m={m}, n={n}, k={k})."
        )


# ============================================================================
# GPU Kernels
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
    acc: S.Tensor((GROUP_M * GROUP_N // THREADS // 4, 4,), S.f32,),
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


def gemm_1stage_pipeline(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """GEMM kernel: C = A @ B^T where A is (M, K) and B is (N, K)."""
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("A and B must be torch.Tensor")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(
            f"A and B must be rank-2 tensors (got A.ndim={A.ndim}, B.ndim={B.ndim})"
        )
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        raise TypeError(
            f"A and B must be torch.bfloat16 (got A={A.dtype}, B={B.dtype})"
        )
    if A.device.type != "cuda" or B.device.type != "cuda":
        raise ValueError(
            f"A and B must be CUDA tensors (got A={A.device}, B={B.device})"
        )
    if A.device != B.device:
        raise ValueError(f"A and B must be on the same device (got {A.device} and {B.device})")

    m, k = A.shape
    n, b_k = B.shape

    if k != b_k:
        raise ValueError(
            f"K dimension mismatch: B must have shape (N, K) "
            f"(got A.shape={A.shape}, B.shape={B.shape})"
        )

    gemm_1stage_validate_shape(m, n, k)

    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be torch.Tensor")
    if out is None or out.numel() == 0:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)
    elif out.ndim != 2 or out.shape != (m, n):
        raise ValueError(f"out must have shape {(m, n)} (got {tuple(out.shape)})")
    elif out.dtype != torch.bfloat16:
        raise TypeError(f"out must be torch.bfloat16 (got {out.dtype})")
    elif out.device != A.device:
        raise ValueError(f"out must be on {A.device} (got {out.device})")

    grid, block = gemm_1stage_launch_config(m, n)
    _gemm_1stage_pipeline_kernel[lambda: (grid, block)](A, B, out, m, n, k)
    return out


# ============================================================================
# Model wrapper
# ============================================================================
class ModelNew(torch.nn.Module):
    """
    Optimized model that computes C = A^T @ B using Substrate GEMM kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Computes C = A^T @ B using optimized GPU kernel.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        # Ensure inputs are on GPU
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()

        # Convert to BF16 for optimal performance
        A_bf16 = A.to(torch.bfloat16)
        B_bf16 = B.to(torch.bfloat16)

        # Transpose inputs:
        # A^T has shape (M, K) - this is our first matrix for GEMM
        # B^T has shape (N, K) - this is the pre-transposed second matrix
        A_T = A_bf16.T.contiguous()
        B_T = B_bf16.T.contiguous()

        # Call GEMM kernel: C = A^T @ B = A^T @ (B^T)^T
        # Kernel computes: A_T @ B_T^T where A_T is (M, K) and B_T is (N, K)
        C = gemm_1stage_pipeline(A_T, B_T)

        return C
