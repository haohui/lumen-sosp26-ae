"""
Batched GEMM kernel for AMD MI300X using Substrate DSL.
Computes C = A @ B where A, B, C have the same batch dimension.
"""
import substrate
import substrate.language as S
import torch

# Kernel configuration constants
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


def batched_gemm_launch_config(batch_size, m, n):
    m_groups = (m + GROUP_M - 1) // GROUP_M
    n_groups = (n + GROUP_N - 1) // GROUP_N
    total_blocks = batch_size * m_groups * n_groups
    grid = (total_blocks, 1, 1)
    block = (NUM_WARPS * WARP_SIZE, 1, 1)
    return grid, block


def batched_gemm_validate_shape(m, n, k):
    if m % GROUP_M != 0 or n % GROUP_N != 0 or k % GROUP_K != 0:
        raise ValueError(
            f"M, N must be multiples of 128 and K must be a multiple of 64 "
            f"(got m={m}, n={n}, k={k})."
        )


@substrate.jit
def load_global_batched(
    A: S.Pointer(S.u32),
    B: S.Pointer(S.u32),
    batch_idx: S.u32,
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

    # A layout: (batch, m, k_vec, 4)
    layout_a = S.make_layout(
        (batch_idx + 1, m, k_vec, 4),
        (m * k_vec * 4, k_vec * 4, 4, 1),
    )
    g_a = S.make_tensor(A, S.u32, layout_a)

    # B layout: (batch, n, k_vec, 4) - B is transposed to (batch, n, k)
    layout_b = S.make_layout(
        (batch_idx + 1, n, k_vec, 4),
        (n * k_vec * 4, k_vec * 4, 4, 1),
    )
    g_b = S.make_tensor(B, S.u32, layout_b)

    tid = S.thread_id(0)
    idx = tid
    for i in S.range(GROUP_M * GROUP_K // VEC_SIZE // THREADS):
        row = idx // K_VEC
        col = idx % K_VEC
        reg_a[i] = g_a[batch_idx, tile_a_idx_row * GROUP_M + row, tile_a_idx_col * K_VEC + col]
        idx = idx + THREADS

    idx = tid
    for i in S.range(GROUP_N * GROUP_K // VEC_SIZE // THREADS):
        row = idx // K_VEC
        col = idx % K_VEC
        reg_b[i] = g_b[batch_idx, tile_b_idx_row * GROUP_N + row, tile_b_idx_col * K_VEC + col]
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
def write_results_batched(
    batch_idx: S.u32,
    id_m: S.u32,
    id_n: S.u32,
    acc: S.Tensor((GROUP_M * GROUP_N // THREADS // 4, 4,), S.f32,),
    C: S.Pointer(S.u32),
    batch_size: S.u32,
    m: S.u32,
    n: S.u32,
):
    tid = S.thread_id(0)
    wid = tid // WARP_SIZE
    lane_id = tid % WARP_SIZE

    row_block = TILE_M // (WARP_SIZE // TILE_N)

    # Full layout spanning all batches: (batch, m_group, n_group, warp, tile, lane, vec)
    m_groups = m // GROUP_M
    n_groups = n // GROUP_N
    layout_c = S.make_layout(
        (
            batch_size, m_groups, n_groups,
            (WARP_PER_COL, WARP_PER_ROW),
            (N_TILES_PER_WARP, M_TILES_PER_WARP),
            (TILE_N, TILE_N // 4),
            2,
        ),
        (
            m * n // 2,
            GROUP_M * n // 2,
            GROUP_N // 2,
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
        g_c[batch_idx, id_m, id_n, wid, acc_idx, lane_id] = packed


@substrate.jit
def wgm_mapping_batched(
    batch_size: S.u32,
    m: S.u32,
    n: S.u32,
) -> (S.u32, S.u32, S.u32):
    """Returns (batch_idx, group_m, group_n) from linear block_id."""
    block_id_linear = S.block_id(0)
    linear_group_id = S.convert(block_id_linear, S.u32)

    group_m_size = S.convert(GROUP_M, S.u32)
    group_n_size = S.convert(GROUP_N, S.u32)
    m_groups = m // group_m_size
    n_groups = n // group_n_size
    groups_per_batch = m_groups * n_groups

    # Compute batch index
    batch_idx = linear_group_id // groups_per_batch

    # Compute local group ID within the batch
    local_group_id = linear_group_id - batch_idx * groups_per_batch

    # Workgroup mapping for the local group ID (same as non-batched)
    total_groups = groups_per_batch

    cu_count = S.convert(38 * 8, S.u32)
    wgm_xcc = S.convert(8, S.u32)
    workgroup_mapping = S.convert(32, S.u32)

    linear_group_limit = (total_groups // wgm_xcc) * wgm_xcc
    cu_base = (local_group_id // cu_count) * cu_count
    cu_xcc = (local_group_id % cu_count) // wgm_xcc
    cu_base = cu_base + cu_xcc

    cu_tail_limit = (total_groups // cu_count) * cu_count
    active_cu = (total_groups % cu_count) if (local_group_id > cu_tail_limit) else cu_count
    cu_xcc_stride = (active_cu // wgm_xcc) * (local_group_id % wgm_xcc)
    linear_group_mapped = cu_base + cu_xcc_stride

    local_group_id = (
        linear_group_mapped if (local_group_id < linear_group_limit) else local_group_id
    )

    group_m = local_group_id // n_groups
    group_n = local_group_id - group_m * n_groups

    mapping_block = group_m // workgroup_mapping
    mapping_linear = group_n + (group_m % workgroup_mapping) * n_groups
    mapping_groups = m_groups // workgroup_mapping
    mapping_tail = m_groups % workgroup_mapping
    mapping_tail = mapping_tail if (mapping_tail != 0) else workgroup_mapping
    mapping_span = mapping_tail if (mapping_block >= mapping_groups) else workgroup_mapping

    group_n = mapping_linear // mapping_span
    group_m = mapping_linear % mapping_span
    group_m = group_m + mapping_block * workgroup_mapping

    return batch_idx, group_m, group_n


@substrate.jit
def _batched_gemm_1stage_pipeline_kernel(
    A: S.Pointer(S.u32),
    B: S.Pointer(S.u32),
    C: S.Pointer(S.u32),
    batch_size: S.u32,
    m: S.u32,
    n: S.u32,
    k: S.u32,
):
    batch_idx, id_m, id_n = wgm_mapping_batched(batch_size, m, n)

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

    # Prime the pipeline
    load_global_batched(A, B, batch_idx, m, n, k, id_m, k_start, id_n, k_start, reg_a, reg_b)
    store_shm(shm_a, shm_b, reg_a, reg_b)
    S.syncthreads()

    # Main pipeline loop
    for k_iter in S.range(k_tiles - 1):
        k_next = k_start + k_iter + 1
        k_next = k_next - (k_tiles_u32 if (k_next >= k_tiles_u32) else S.convert(0, S.u32))
        load_global_batched(A, B, batch_idx, m, n, k, id_m, k_next, id_n, k_next, reg_a, reg_b)

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

    # Compute the last tile
    for i in S.range(M_TILES_PER_WARP):
        for j in S.range(N_TILES_PER_WARP):
            acc_idx = i * N_TILES_PER_WARP + j
            load_shm_to_regs(shm_a, shm_b, i, j, 0, tile_a0, tile_b0)
            load_shm_to_regs(shm_a, shm_b, i, j, 1, tile_a1, tile_b1)
            matmul_from_regs(tile_a0, tile_b0, acc, acc_idx)
            matmul_from_regs(tile_a1, tile_b1, acc, acc_idx)

    write_results_batched(batch_idx, id_m, id_n, acc, C, batch_size, m, n)


def batched_gemm_1stage_pipeline(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes batched matrix multiplication C = A @ B.

    Args:
        A: Input tensor of shape (batch_size, m, k), dtype bfloat16.
        B: Input tensor of shape (batch_size, k, n), dtype bfloat16.
        out: Optional output tensor of shape (batch_size, m, n).

    Returns:
        C: Output tensor of shape (batch_size, m, n), dtype bfloat16.
    """
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("A and B must be torch.Tensor")
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError(
            f"A and B must be rank-3 tensors (got A.ndim={A.ndim}, B.ndim={B.ndim})"
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

    batch_size, m, k = A.shape
    b_batch, b_k, n = B.shape

    if batch_size != b_batch:
        raise ValueError(
            f"Batch size mismatch: A has batch_size={batch_size}, B has batch_size={b_batch}"
        )
    if k != b_k:
        raise ValueError(
            f"K dimension mismatch: A has k={k}, B has k={b_k}"
        )

    batched_gemm_validate_shape(m, n, k)

    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be torch.Tensor")
    if out is None or out.numel() == 0:
        out = torch.empty((batch_size, m, n), dtype=torch.bfloat16, device=A.device)
    elif out.ndim != 3 or out.shape != (batch_size, m, n):
        raise ValueError(f"out must have shape {(batch_size, m, n)} (got {tuple(out.shape)})")
    elif out.dtype != torch.bfloat16:
        raise TypeError(f"out must be torch.bfloat16 (got {out.dtype})")
    elif out.device != A.device:
        raise ValueError(f"out must be on {A.device} (got {out.device})")

    # Transpose B from (batch, k, n) to (batch, n, k) for the kernel
    B_t = B.transpose(1, 2).contiguous()

    grid, block = batched_gemm_launch_config(batch_size, m, n)
    _batched_gemm_1stage_pipeline_kernel[lambda: (grid, block)](
        A, B_t, out, batch_size, m, n, k
    )
    return out


class ModelNew(torch.nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) using optimized Substrate kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        # Ensure inputs are contiguous and on GPU
        A = A.contiguous()
        B = B.contiguous()

        # Convert to bfloat16 if needed
        if A.dtype != torch.bfloat16:
            A = A.to(torch.bfloat16)
        if B.dtype != torch.bfloat16:
            B = B.to(torch.bfloat16)

        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()

        return batched_gemm_1stage_pipeline(A, B)
