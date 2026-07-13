import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# GEMM constants
# ---------------------------------------------------------------------------
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS

GROUP_M = 128
GROUP_N = 128
GROUP_K = 64
K_SUB = 32

WARP_PER_ROW = 2
WARP_PER_COL = 2
WARP_MAT_M = GROUP_M // WARP_PER_ROW
WARP_MAT_N = GROUP_N // WARP_PER_COL
M_TILES_PER_WARP = WARP_MAT_M // 16
N_TILES_PER_WARP = WARP_MAT_N // 16

VEC_SIZE = 8
BF16_BYTES = 2

SHM_CHUNKS_PER_ROW_GL = GROUP_K // VEC_SIZE
REG_ROWS_A = GROUP_M * GROUP_K // VEC_SIZE // THREADS
REG_ROWS_B = GROUP_N * GROUP_K // VEC_SIZE // THREADS

SHM_PAD_ROWS = 4
SHM_PAD_BF16 = 0
SHM_CHUNKS_PER_ROW_SUB = K_SUB // VEC_SIZE
SHM_GROUPS = GROUP_M // SHM_PAD_ROWS
SHM_GROUP_BF16_SUB = SHM_PAD_ROWS * K_SUB + SHM_PAD_BF16
SHM_GROUP_WORDS_SUB = SHM_GROUP_BF16_SUB // 2
SHM_PER_BUF = SHM_GROUPS * SHM_GROUP_BF16_SUB
SHM_GROUP_CHUNKS_SUB = SHM_GROUP_WORDS_SUB // (VEC_SIZE // 2)
SHM_CHUNKS_PER_BUF = SHM_PER_BUF // VEC_SIZE
SHM_COMBINED = 2 * SHM_PER_BUF
SHM_CHUNKS_COMBINED = SHM_COMBINED // VEC_SIZE


# ---------------------------------------------------------------------------
# Global load helpers
# ---------------------------------------------------------------------------
@avelang.jit
def _load_global_a(
    src_rsrc: al.Tensor((4,), al.u32),
    k: al.u32,
    group_row: al.u32,
    k_idx: al.u32,
    tid: al.u32,
    reg: al.Tensor((REG_ROWS_A, VEC_SIZE), al.bf16),
):
    row = tid // SHM_CHUNKS_PER_ROW_GL
    col = (tid - row * SHM_CHUNKS_PER_ROW_GL) * VEC_SIZE
    thread_offset = (row * k + col) * BF16_BYTES
    tile_offset = (group_row * GROUP_M * k + k_idx * GROUP_K) * BF16_BYTES
    thread_offset_stride = (THREADS * VEC_SIZE // GROUP_K) * k * BF16_BYTES

    for i in al.range(REG_ROWS_A):
        packed = al.amdgpu.raw_buffer_load_x4(src_rsrc,
            thread_offset, tile_offset + i * thread_offset_stride, 0)
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
    row = tid // SHM_CHUNKS_PER_ROW_GL
    col = (tid - row * SHM_CHUNKS_PER_ROW_GL) * VEC_SIZE
    thread_offset = (row * n + col) * BF16_BYTES
    tile_offset = (group_col * GROUP_N * n + k_idx * GROUP_K) * BF16_BYTES
    thread_offset_stride = (THREADS * VEC_SIZE // GROUP_K) * n * BF16_BYTES

    for i in al.range(REG_ROWS_B):
        packed = al.amdgpu.raw_buffer_load_x4(src_rsrc,
            thread_offset, tile_offset + i * thread_offset_stride, 0)
        frag = al.view(packed, al.Tensor((VEC_SIZE,), al.bf16))
        for v in al.range(VEC_SIZE):
            reg[i, v] = frag[v]


# ---------------------------------------------------------------------------
# Store: writes all K-elements into combined buffer H0+H1 using arithmetic
# ---------------------------------------------------------------------------
@avelang.jit
def _store_shm_a_combined(
    shm: al.Tensor((SHM_COMBINED,), al.bf16),
    reg: al.Tensor((REG_ROWS_A, VEC_SIZE), al.bf16),
    tid: al.u32,
):
    shm_vec = al.view(shm, al.u32,
        al.make_layout((SHM_CHUNKS_COMBINED, 4), (VEC_SIZE // 2, 1)))

    row_base = tid // SHM_CHUNKS_PER_ROW_GL
    chunk_gl = tid - row_base * SHM_CHUNKS_PER_ROW_GL
    buf_sel = chunk_gl // SHM_CHUNKS_PER_ROW_SUB
    loc_chunk = chunk_gl - buf_sel * SHM_CHUNKS_PER_ROW_SUB

    row_stride = THREADS * VEC_SIZE // GROUP_K

    for i in al.range(REG_ROWS_A):
        full_row = row_base + i * row_stride
        rg = full_row // SHM_PAD_ROWS
        rig = full_row - rg * SHM_PAD_ROWS
        chunk_idx = (buf_sel * SHM_CHUNKS_PER_BUF
                     + rg * SHM_GROUP_CHUNKS_SUB
                     + rig * SHM_CHUNKS_PER_ROW_SUB
                     + loc_chunk)
        packed = al.view(reg[i], al.Tensor((4,), al.u32))
        shm_vec[chunk_idx] = packed


@avelang.jit
def _store_shm_b_combined(
    shm: al.Tensor((SHM_COMBINED,), al.bf16),
    reg: al.Tensor((REG_ROWS_B, VEC_SIZE), al.bf16),
    tid: al.u32,
):
    shm_vec = al.view(shm, al.u32,
        al.make_layout((SHM_CHUNKS_COMBINED, 4), (VEC_SIZE // 2, 1)))

    row_base = tid // SHM_CHUNKS_PER_ROW_GL
    chunk_gl = tid - row_base * SHM_CHUNKS_PER_ROW_GL
    buf_sel = chunk_gl // SHM_CHUNKS_PER_ROW_SUB
    loc_chunk = chunk_gl - buf_sel * SHM_CHUNKS_PER_ROW_SUB

    row_stride = THREADS * VEC_SIZE // GROUP_K

    for i in al.range(REG_ROWS_B):
        full_row = row_base + i * row_stride
        rg = full_row // SHM_PAD_ROWS
        rig = full_row - rg * SHM_PAD_ROWS
        chunk_idx = (buf_sel * SHM_CHUNKS_PER_BUF
                     + rg * SHM_GROUP_CHUNKS_SUB
                     + rig * SHM_CHUNKS_PER_ROW_SUB
                     + loc_chunk)
        packed = al.view(reg[i], al.Tensor((4,), al.u32))
        shm_vec[chunk_idx] = packed


# ---------------------------------------------------------------------------
# LDS -> register: loads from combined buffer using half_offset parameter
# half_offset = 0 for H0, SHM_CHUNKS_PER_BUF for H1
# ---------------------------------------------------------------------------
@avelang.jit
def _lds_to_regs_a(
    shm: al.Tensor((SHM_COMBINED,), al.bf16),
    half_offset: al.u32,
    row_base: al.u32,
    wtid: al.u32,
    data: al.Tensor((M_TILES_PER_WARP, 4), al.u32),
):
    shm_vec = al.view(shm, al.u32,
        al.make_layout((SHM_CHUNKS_COMBINED, 4), (VEC_SIZE // 2, 1)))

    row_start = row_base + (wtid % 16) * M_TILES_PER_WARP
    chunk_base = (wtid // 16)

    for tile in al.range(M_TILES_PER_WARP):
        row = row_start + tile
        rg = row // SHM_PAD_ROWS
        rig = row - rg * SHM_PAD_ROWS
        chunk_idx = (half_offset
                     + rg * SHM_GROUP_CHUNKS_SUB
                     + rig * SHM_CHUNKS_PER_ROW_SUB
                     + chunk_base)
        data[tile] = shm_vec[chunk_idx]


@avelang.jit
def _lds_to_regs_b(
    shm: al.Tensor((SHM_COMBINED,), al.bf16),
    half_offset: al.u32,
    row_base: al.u32,
    wtid: al.u32,
    data: al.Tensor((N_TILES_PER_WARP, 4), al.u32),
):
    shm_vec = al.view(shm, al.u32,
        al.make_layout((SHM_CHUNKS_COMBINED, 4), (VEC_SIZE // 2, 1)))

    row_start = row_base + (wtid % 16) * N_TILES_PER_WARP
    chunk_base = (wtid // 16)

    for tile in al.range(N_TILES_PER_WARP):
        row = row_start + tile
        rg = row // SHM_PAD_ROWS
        rig = row - rg * SHM_PAD_ROWS
        chunk_idx = (half_offset
                     + rg * SHM_GROUP_CHUNKS_SUB
                     + rig * SHM_CHUNKS_PER_ROW_SUB
                     + chunk_base)
        data[tile] = shm_vec[chunk_idx]


# ---------------------------------------------------------------------------
# MFMA: 2 instructions per half (K_SUB=32 -> 2*16)
# ---------------------------------------------------------------------------
@avelang.jit
def _mfma_half(
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
# Main kernel: double-buffered combined shm with software pipelining
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

    shm_a0 = al.make_shared((SHM_COMBINED,), al.bf16)
    shm_a1 = al.make_shared((SHM_COMBINED,), al.bf16)
    shm_b0 = al.make_shared((SHM_COMBINED,), al.bf16)
    shm_b1 = al.make_shared((SHM_COMBINED,), al.bf16)

    reg_a = al.make_local((REG_ROWS_A, VEC_SIZE), al.bf16)
    reg_b = al.make_local((REG_ROWS_B, VEC_SIZE), al.bf16)

    data_a = al.make_local((M_TILES_PER_WARP, 4), al.u32)
    data_b = al.make_local((N_TILES_PER_WARP, 4), al.u32)

    acc = al.make_local((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32)
    for tile_m in al.range(M_TILES_PER_WARP):
        for tile_n in al.range(N_TILES_PER_WARP):
            for acc_idx in al.range(4):
                acc[tile_m, tile_n, acc_idx] = al.convert(0.0, al.f32)

    k_total = k // GROUP_K
    warp_a_row = warp_row * WARP_MAT_M
    warp_b_row = warp_col * WARP_MAT_N

    # Prefetch tile 0 into pair 0
    _load_global_a(a_rsrc, k, group_m, 0, tid, reg_a)
    _load_global_b(b_rsrc, n, group_n, 0, tid, reg_b)
    _store_shm_a_combined(shm_a0, reg_a, tid)
    _store_shm_b_combined(shm_b0, reg_b, tid)
    al.syncthreads()

    _lds_to_regs_a(shm_a0, 0, warp_a_row, wtid, data_a)
    _lds_to_regs_b(shm_b0, 0, warp_b_row, wtid, data_b)

    # Pipeline loop unrolled by 2
    for k_idx in al.range(1, k_total - 1, 2):
        _mfma_half(data_a, data_b, acc)

        _load_global_a(a_rsrc, k, group_m, k_idx, tid, reg_a)
        _load_global_b(b_rsrc, n, group_n, k_idx, tid, reg_b)

        _lds_to_regs_a(shm_a0, SHM_CHUNKS_PER_BUF, warp_a_row, wtid, data_a)
        _lds_to_regs_b(shm_b0, SHM_CHUNKS_PER_BUF, warp_b_row, wtid, data_b)
        _mfma_half(data_a, data_b, acc)

        al.syncthreads()
        _store_shm_a_combined(shm_a1, reg_a, tid)
        _store_shm_b_combined(shm_b1, reg_b, tid)
        al.syncthreads()

        _lds_to_regs_a(shm_a1, 0, warp_a_row, wtid, data_a)
        _lds_to_regs_b(shm_b1, 0, warp_b_row, wtid, data_b)
        _mfma_half(data_a, data_b, acc)

        _load_global_a(a_rsrc, k, group_m, k_idx + 1, tid, reg_a)
        _load_global_b(b_rsrc, n, group_n, k_idx + 1, tid, reg_b)

        _lds_to_regs_a(shm_a1, SHM_CHUNKS_PER_BUF, warp_a_row, wtid, data_a)
        _lds_to_regs_b(shm_b1, SHM_CHUNKS_PER_BUF, warp_b_row, wtid, data_b)
        _mfma_half(data_a, data_b, acc)

        al.syncthreads()
        _store_shm_a_combined(shm_a0, reg_a, tid)
        _store_shm_b_combined(shm_b0, reg_b, tid)
        al.syncthreads()

        _lds_to_regs_a(shm_a0, 0, warp_a_row, wtid, data_a)
        _lds_to_regs_b(shm_b0, 0, warp_b_row, wtid, data_b)

    # Epilogue: tiles k_total-2 and k_total-1
    _mfma_half(data_a, data_b, acc)

    _lds_to_regs_a(shm_a0, SHM_CHUNKS_PER_BUF, warp_a_row, wtid, data_a)
    _lds_to_regs_b(shm_b0, SHM_CHUNKS_PER_BUF, warp_b_row, wtid, data_b)
    _mfma_half(data_a, data_b, acc)

    _load_global_a(a_rsrc, k, group_m, k_total - 1, tid, reg_a)
    _load_global_b(b_rsrc, n, group_n, k_total - 1, tid, reg_b)
    al.syncthreads()
    _store_shm_a_combined(shm_a1, reg_a, tid)
    _store_shm_b_combined(shm_b1, reg_b, tid)
    al.syncthreads()

    _lds_to_regs_a(shm_a1, 0, warp_a_row, wtid, data_a)
    _lds_to_regs_b(shm_b1, 0, warp_b_row, wtid, data_b)
    _mfma_half(data_a, data_b, acc)

    _lds_to_regs_a(shm_a1, SHM_CHUNKS_PER_BUF, warp_a_row, wtid, data_a)
    _lds_to_regs_b(shm_b1, SHM_CHUNKS_PER_BUF, warp_b_row, wtid, data_b)
    _mfma_half(data_a, data_b, acc)

    # Store accumulator to C
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
# Host launcher
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_val = x.shape[0]
        K_val = x.shape[1]
        N_val = self.out_features

        w_bf16 = self.linear.weight.data.to(
            device=x.device, dtype=torch.bfloat16).contiguous()
        bias_f32 = self.linear.bias.float()
        x_bf16 = x.to(dtype=torch.bfloat16).contiguous()

        y_tmp = torch.empty((B_val, N_val), device=x.device, dtype=torch.float32)
        m_groups = B_val // GROUP_M
        n_groups = N_val // GROUP_N
        grid_size = m_groups * n_groups
        block_size = WARP_SIZE * NUM_WARPS

        gemm_kernel[lambda: ((grid_size, 1, 1), (block_size, 1, 1))](
            x_bf16, w_bf16, y_tmp,
            B_val, N_val, K_val,
        )

        row_sums = y_tmp.sum(dim=1) + bias_f32.sum()
        x_out = row_sums.unsqueeze(1)
        x_out = torch.max(x_out, dim=1, keepdim=True)[0]
        x_out = torch.mean(x_out, dim=1, keepdim=True)
        x_out = torch.logsumexp(x_out, dim=1, keepdim=True)
        x_out = torch.logsumexp(x_out, dim=1, keepdim=True)
        return x_out.to(torch.bfloat16)
