import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# Production-quality GEMM constants from amdgpu_gemm.py
# ---------------------------------------------------------------------------
WARP_SIZE = 64
NUM_WARPS = 4
GROUP_M = 128
GROUP_N = 128
GROUP_K = 64
WARP_PER_ROW = 2
WARP_PER_COL = 2
WARP_MAT_M = GROUP_M // WARP_PER_ROW
WARP_MAT_N = GROUP_N // WARP_PER_COL
M_TILES_PER_WARP = WARP_MAT_M // 16
N_TILES_PER_WARP = WARP_MAT_N // 16
VEC_SIZE = 8
THREADS = WARP_SIZE * NUM_WARPS
REG_ROWS_A = GROUP_M * GROUP_K // VEC_SIZE // THREADS
REG_ROWS_B = GROUP_N * GROUP_K // VEC_SIZE // THREADS
BF16_BYTES = 2
SHM_PAD_ROWS = 4
SHM_PAD_BF16 = 16
SHM_GROUPS_A = GROUP_M // SHM_PAD_ROWS
SHM_GROUPS_B = GROUP_N // SHM_PAD_ROWS
SHM_GROUP_BF16 = SHM_PAD_ROWS * GROUP_K + SHM_PAD_BF16
SHM_GROUP_WORDS = SHM_GROUP_BF16 // 2
SHM_TOTAL_BF16_A = SHM_GROUPS_A * SHM_GROUP_BF16
SHM_TOTAL_BF16_B = SHM_GROUPS_B * SHM_GROUP_BF16
SHM_CHUNKS_PER_ROW = GROUP_K // VEC_SIZE
MI300_CU_COUNT = 38 * 8
WGM_XCC = 8


@avelang.jit
def _load_global_a(
    src_rsrc: al.Tensor((4,), al.u32),
    k: al.u32,
    group_row: al.u32,
    k_idx: al.u32,
    tid: al.u32,
    reg: al.Tensor((REG_ROWS_A, VEC_SIZE), al.bf16),
):
    row = tid // SHM_CHUNKS_PER_ROW
    col = (tid - row * SHM_CHUNKS_PER_ROW) * VEC_SIZE
    thread_offset = (row * k + col) * BF16_BYTES
    tile_offset = (group_row * GROUP_M * k + k_idx * GROUP_K) * BF16_BYTES
    thread_offset_stride = (THREADS * VEC_SIZE // GROUP_K) * k * BF16_BYTES

    for i in al.range(REG_ROWS_A):
        packed = al.amdgpu.raw_buffer_load_x4(
            src_rsrc, thread_offset, tile_offset + i * thread_offset_stride, 0)
        frag = al.view(packed, al.Tensor((VEC_SIZE,), al.bf16))
        for v in al.range(VEC_SIZE):
            reg[i, v] = frag[v]


@avelang.jit
def _load_global_b(
    src_rsrc: al.Tensor((4,), al.u32),
    n: al.u32,
    group_col: al.u32,
    k_idx: al.u32,
    tid: al.u32,
    reg: al.Tensor((REG_ROWS_B, VEC_SIZE), al.bf16),
):
    row = tid // SHM_CHUNKS_PER_ROW
    col = (tid - row * SHM_CHUNKS_PER_ROW) * VEC_SIZE
    thread_offset = (row * n + col) * BF16_BYTES
    tile_offset = (group_col * GROUP_N * n + k_idx * GROUP_K) * BF16_BYTES
    thread_offset_stride = (THREADS * VEC_SIZE // GROUP_K) * n * BF16_BYTES

    for i in al.range(REG_ROWS_B):
        packed = al.amdgpu.raw_buffer_load_x4(
            src_rsrc, thread_offset, tile_offset + i * thread_offset_stride, 0)
        frag = al.view(packed, al.Tensor((VEC_SIZE,), al.bf16))
        for v in al.range(VEC_SIZE):
            reg[i, v] = frag[v]


@avelang.jit
def _store_shm_a(
    shm: al.Tensor((SHM_TOTAL_BF16_A,), al.bf16),
    reg: al.Tensor((REG_ROWS_A, VEC_SIZE), al.bf16),
    tid: al.u32,
):
    shm_vec = al.view(shm, al.u32,
        al.make_layout((SHM_TOTAL_BF16_A // VEC_SIZE, 4), (VEC_SIZE // 2, 1)))

    row = tid // SHM_CHUNKS_PER_ROW
    row_group = row // SHM_PAD_ROWS
    row_in_group = row - row_group * SHM_PAD_ROWS
    chunk = tid - row * SHM_CHUNKS_PER_ROW
    shm_chunk = (row_group * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
                 + row_in_group * SHM_CHUNKS_PER_ROW + chunk)
    shm_chunk_stride = ((THREADS * VEC_SIZE // GROUP_K // SHM_PAD_ROWS)
                        * (SHM_GROUP_WORDS // (VEC_SIZE // 2)))

    for i in al.range(REG_ROWS_A):
        packed = al.view(reg[i], al.Tensor((4,), al.u32))
        shm_vec[shm_chunk + i * shm_chunk_stride] = packed


@avelang.jit
def _store_shm_b(
    shm: al.Tensor((SHM_TOTAL_BF16_B,), al.bf16),
    reg: al.Tensor((REG_ROWS_B, VEC_SIZE), al.bf16),
    tid: al.u32,
):
    shm_vec = al.view(shm, al.u32,
        al.make_layout((SHM_TOTAL_BF16_B // VEC_SIZE, 4), (VEC_SIZE // 2, 1)))

    row = tid // SHM_CHUNKS_PER_ROW
    row_group = row // SHM_PAD_ROWS
    row_in_group = row - row_group * SHM_PAD_ROWS
    chunk = tid - row * SHM_CHUNKS_PER_ROW
    shm_chunk = (row_group * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
                 + row_in_group * SHM_CHUNKS_PER_ROW + chunk)
    shm_chunk_stride = ((THREADS * VEC_SIZE // GROUP_K // SHM_PAD_ROWS)
                        * (SHM_GROUP_WORDS // (VEC_SIZE // 2)))

    for i in al.range(REG_ROWS_B):
        packed = al.view(reg[i], al.Tensor((4,), al.u32))
        shm_vec[shm_chunk + i * shm_chunk_stride] = packed


@avelang.jit
def _load_shm_to_regs_batch_a(
    shm: al.Tensor((SHM_TOTAL_BF16_A,), al.bf16),
    row_base: al.u32,
    batch_id: al.u32,
    wtid: al.u32,
    data: al.Tensor((M_TILES_PER_WARP, 4), al.u32),
):
    shm_vec = al.view(shm, al.u32,
        al.make_layout((SHM_GROUPS_A, SHM_PAD_ROWS, SHM_CHUNKS_PER_ROW, 4),
                       (SHM_GROUP_WORDS, GROUP_K // 2, 4, 1)))

    row_start = row_base + (wtid % 16) * M_TILES_PER_WARP
    chunk_base = (wtid // 16) + batch_id * (32 // VEC_SIZE)

    for tile in al.range(M_TILES_PER_WARP):
        row = row_start + tile
        row_group = row // SHM_PAD_ROWS
        row_in_group = row - row_group * SHM_PAD_ROWS
        data[tile] = shm_vec[row_group, row_in_group, chunk_base]


@avelang.jit
def _load_shm_to_regs_batch_b(
    shm: al.Tensor((SHM_TOTAL_BF16_B,), al.bf16),
    row_base: al.u32,
    batch_id: al.u32,
    wtid: al.u32,
    data: al.Tensor((N_TILES_PER_WARP, 4), al.u32),
):
    shm_vec = al.view(shm, al.u32,
        al.make_layout((SHM_GROUPS_B, SHM_PAD_ROWS, SHM_CHUNKS_PER_ROW, 4),
                       (SHM_GROUP_WORDS, GROUP_K // 2, 4, 1)))

    row_start = row_base + (wtid % 16) * N_TILES_PER_WARP
    chunk_base = (wtid // 16) + batch_id * (32 // VEC_SIZE)

    for tile in al.range(N_TILES_PER_WARP):
        row = row_start + tile
        row_group = row // SHM_PAD_ROWS
        row_in_group = row - row_group * SHM_PAD_ROWS
        data[tile] = shm_vec[row_group, row_in_group, chunk_base]


@avelang.jit
def _matmul_from_regs_batch(
    data_a: al.Tensor((M_TILES_PER_WARP, 4), al.u32),
    data_b: al.Tensor((N_TILES_PER_WARP, 4), al.u32),
    acc: al.Tensor((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32),
):
    for tile_m in al.range(M_TILES_PER_WARP):
        for tile_n in al.range(N_TILES_PER_WARP):
            frag_a = al.view(data_a[tile_m], al.Tensor((2, 2, 1), al.u32))
            frag_b = al.view(data_b[tile_n], al.Tensor((2, 2, 1), al.u32))
            acc[tile_m, tile_n] = al.amdgpu.mfma_16x16x16_bf16_f32(
                frag_a[0], frag_b[0], acc[tile_m, tile_n])
            acc[tile_m, tile_n] = al.amdgpu.mfma_16x16x16_bf16_f32(
                frag_a[1], frag_b[1], acc[tile_m, tile_n])


# ---------------------------------------------------------------------------
# Main GEMM kernel: C = A @ B  where A=(M,K), B=(N,K) -> C=(M,N)
# Outputs fp32 for maximum reduction precision
# ---------------------------------------------------------------------------
@avelang.jit
def gemm_kernel(
    A: al.Pointer(al.bf16),
    B: al.Pointer(al.bf16),
    C: al.Pointer(al.f32),
    m: al.u32,
    n: al.u32,
    k: al.u32,
):
    tid = al.thread_id(0)
    wid = tid // WARP_SIZE
    wtid = tid % WARP_SIZE
    warp_row = wid // WARP_PER_COL
    warp_col = wid % WARP_PER_COL

    m_groups = m // GROUP_M
    n_groups = n // GROUP_N
    linear_group_id = al.block_id(0)
    group_m = linear_group_id // n_groups
    group_n = linear_group_id - group_m * n_groups

    a_tensor = al.make_tensor(A, al.bf16, al.make_layout((m, k), (k, 1)))
    b_tensor = al.make_tensor(B, al.bf16, al.make_layout((n, k), (k, 1)))
    c_tensor = al.make_tensor(C, al.f32, al.make_layout((m, n), (n, 1)))
    a_rsrc = al.amdgpu.make_rsrc(a_tensor, m * k * BF16_BYTES)
    b_rsrc = al.amdgpu.make_rsrc(b_tensor, n * k * BF16_BYTES)

    shm_a = al.make_shared((SHM_TOTAL_BF16_A,), al.bf16)
    shm_b = al.make_shared((SHM_TOTAL_BF16_B,), al.bf16)
    reg_a = al.make_local((REG_ROWS_A, VEC_SIZE), al.bf16)
    reg_b = al.make_local((REG_ROWS_B, VEC_SIZE), al.bf16)
    data_a0 = al.make_local((M_TILES_PER_WARP, 4), al.u32)
    data_a1 = al.make_local((M_TILES_PER_WARP, 4), al.u32)
    data_b0 = al.make_local((N_TILES_PER_WARP, 4), al.u32)
    data_b1 = al.make_local((N_TILES_PER_WARP, 4), al.u32)
    acc = al.make_local((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32)

    for tile_m in al.range(M_TILES_PER_WARP):
        for tile_n in al.range(N_TILES_PER_WARP):
            for acc_idx in al.range(4):
                acc[tile_m, tile_n, acc_idx] = al.convert(0.0, al.f32)

    k_total = k // GROUP_K

    _load_global_a(a_rsrc, k, group_m, 0, tid, reg_a)
    _load_global_b(b_rsrc, n, group_n, 0, tid, reg_b)
    _store_shm_a(shm_a, reg_a, tid)
    _store_shm_b(shm_b, reg_b, tid)
    al.syncthreads()

    _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 0, wtid, data_a0)
    _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 0, wtid, data_b0)
    _load_global_a(a_rsrc, k, group_m, 1, tid, reg_a)
    _load_global_b(b_rsrc, n, group_n, 1, tid, reg_b)

    for k_idx in al.range(0, k_total - 3, 2):
        _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 1, wtid, data_a1)
        _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 1, wtid, data_b1)
        _matmul_from_regs_batch(data_a0, data_b0, acc)
        al.syncthreads()

        _store_shm_a(shm_a, reg_a, tid)
        _store_shm_b(shm_b, reg_b, tid)
        _load_global_a(a_rsrc, k, group_m, k_idx + 2, tid, reg_a)
        _load_global_b(b_rsrc, n, group_n, k_idx + 2, tid, reg_b)
        al.syncthreads()

        _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 0, wtid, data_a0)
        _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 0, wtid, data_b0)
        _matmul_from_regs_batch(data_a1, data_b1, acc)

        _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 1, wtid, data_a1)
        _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 1, wtid, data_b1)
        _matmul_from_regs_batch(data_a0, data_b0, acc)
        al.syncthreads()

        _store_shm_a(shm_a, reg_a, tid)
        _store_shm_b(shm_b, reg_b, tid)
        _load_global_a(a_rsrc, k, group_m, k_idx + 3, tid, reg_a)
        _load_global_b(b_rsrc, n, group_n, k_idx + 3, tid, reg_b)
        al.syncthreads()

        _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 0, wtid, data_a0)
        _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 0, wtid, data_b0)
        _matmul_from_regs_batch(data_a1, data_b1, acc)

    _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 1, wtid, data_a1)
    _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 1, wtid, data_b1)
    _matmul_from_regs_batch(data_a0, data_b0, acc)
    al.syncthreads()

    _store_shm_a(shm_a, reg_a, tid)
    _store_shm_b(shm_b, reg_b, tid)
    al.syncthreads()

    _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 0, wtid, data_a0)
    _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 0, wtid, data_b0)
    _matmul_from_regs_batch(data_a1, data_b1, acc)

    _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 1, wtid, data_a1)
    _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 1, wtid, data_b1)
    _matmul_from_regs_batch(data_a0, data_b0, acc)
    _matmul_from_regs_batch(data_a1, data_b1, acc)

    lane_row_group = wtid // 16
    lane_col = wtid % 16
    warp_m_base = group_m * GROUP_M + warp_row * WARP_MAT_M
    warp_n_base = group_n * GROUP_N + warp_col * WARP_MAT_N

    for tile_m in al.range(M_TILES_PER_WARP):
        m_idx = warp_m_base + lane_row_group * (4 * M_TILES_PER_WARP) + tile_m
        for tile_n in al.range(N_TILES_PER_WARP):
            n_idx = warp_n_base + lane_col * N_TILES_PER_WARP + tile_n
            for acc_idx in al.range(4):
                val = acc[tile_m, tile_n, acc_idx]
                extra_m = acc_idx * M_TILES_PER_WARP
                out_row = m_idx + extra_m
                if out_row < m and n_idx < n:
                    c_tensor[out_row, n_idx] = val


# ---------------------------------------------------------------------------
# Host-side launcher
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_val = x.shape[0]
        K_val = x.shape[1]
        N_val = self.linear.out_features

        w_bf16 = self.linear.weight.data.to(device=x.device, dtype=torch.bfloat16).contiguous()
        bias = self.linear.bias.to(device=x.device, dtype=torch.bfloat16)
        x_bf16 = x.to(dtype=torch.bfloat16).contiguous()

        # GEMM: C = x @ W^T  using production kernel (W passed as (N,K))
        y_tmp = torch.empty((B_val, N_val), device=x.device, dtype=torch.float32)
        m_groups = B_val // GROUP_M
        n_groups = N_val // GROUP_N
        grid_size = m_groups * n_groups
        block_size = WARP_SIZE * NUM_WARPS

        gemm_kernel[lambda: ((grid_size, 1, 1), (block_size, 1, 1))](
            x_bf16, w_bf16, y_tmp,
            B_val, N_val, K_val,
        )

        # Full reference chain on (batch, 1) tensor: all post-sum ops are identity
        bias_f32 = bias.float()
        row_sums = y_tmp.sum(dim=1) + bias_f32.sum()
        x_out = row_sums.unsqueeze(1)
        x_out = torch.max(x_out, dim=1, keepdim=True)[0]
        x_out = torch.mean(x_out, dim=1, keepdim=True)
        x_out = torch.logsumexp(x_out, dim=1, keepdim=True)
        x_out = torch.logsumexp(x_out, dim=1, keepdim=True)
        return x_out.to(torch.bfloat16)
