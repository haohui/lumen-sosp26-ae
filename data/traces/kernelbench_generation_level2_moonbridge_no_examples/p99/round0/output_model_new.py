import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# MFMA-based matmul + bias + GELU  (fused)
# Adapted from avelang_kernels/amdgpu_gemm.py
# ---------------------------------------------------------------------------

WARP_SIZE = 64
NUM_WARPS = 4
GROUP_M = 128
GROUP_N = 128
GROUP_K = 64
WARP_MAT_M = GROUP_M // 2
WARP_MAT_N = GROUP_N // 2
M_TILES = WARP_MAT_M // 16
N_TILES = WARP_MAT_N // 16
THREADS = WARP_SIZE * NUM_WARPS
VEC_SIZE = 8
REG_ROWS_A = GROUP_M * GROUP_K // VEC_SIZE // THREADS
REG_ROWS_B = GROUP_N * GROUP_K // VEC_SIZE // THREADS
BF16_BYTES = 2
SHM_PAD_ROWS = 4
SHM_CHUNKS_PER_ROW = GROUP_K // VEC_SIZE
SHM_GROUP_BF16 = SHM_PAD_ROWS * GROUP_K + 16
SHM_GROUP_WORDS = SHM_GROUP_BF16 // 2
SHM_GROUPS_A = GROUP_M // SHM_PAD_ROWS
SHM_GROUPS_B = GROUP_N // SHM_PAD_ROWS
SHM_TOTAL_BF16_A = SHM_GROUPS_A * SHM_GROUP_BF16
SHM_TOTAL_BF16_B = SHM_GROUPS_B * SHM_GROUP_BF16


@avelang.jit
def _load_global_a(
    x_rsrc: al.Tensor((4,), al.u32),
    K: al.u32,
    group_m: al.u32,
    k_idx: al.u32,
    tid: al.u32,
    reg: al.Tensor((REG_ROWS_A, VEC_SIZE), al.bf16),
):
    row = tid // SHM_CHUNKS_PER_ROW
    col = (tid - row * SHM_CHUNKS_PER_ROW) * VEC_SIZE
    t_off = (row * K + col) * BF16_BYTES
    tile_off = (group_m * GROUP_M * K + k_idx * GROUP_K) * BF16_BYTES
    stride = (THREADS * VEC_SIZE // GROUP_K) * K * BF16_BYTES
    for i in al.range(REG_ROWS_A):
        packed = al.amdgpu.raw_buffer_load_x4(x_rsrc, t_off, tile_off + i * stride, 0)
        frag = al.view(packed, al.Tensor((VEC_SIZE,), al.bf16))
        for v in al.range(VEC_SIZE):
            reg[i, v] = frag[v]


@avelang.jit
def _load_global_b(
    w_rsrc: al.Tensor((4,), al.u32),
    K: al.u32,
    group_n: al.u32,
    k_idx: al.u32,
    tid: al.u32,
    reg: al.Tensor((REG_ROWS_B, VEC_SIZE), al.bf16),
):
    row = tid // SHM_CHUNKS_PER_ROW
    col = (tid - row * SHM_CHUNKS_PER_ROW) * VEC_SIZE
    t_off = (row * K + col) * BF16_BYTES
    tile_off = (group_n * GROUP_N * K + k_idx * GROUP_K) * BF16_BYTES
    stride = (THREADS * VEC_SIZE // GROUP_K) * K * BF16_BYTES
    for i in al.range(REG_ROWS_B):
        packed = al.amdgpu.raw_buffer_load_x4(w_rsrc, t_off, tile_off + i * stride, 0)
        frag = al.view(packed, al.Tensor((VEC_SIZE,), al.bf16))
        for v in al.range(VEC_SIZE):
            reg[i, v] = frag[v]


@avelang.jit
def _store_shm_a(
    shm: al.Tensor((SHM_TOTAL_BF16_A,), al.bf16),
    reg: al.Tensor((REG_ROWS_A, VEC_SIZE), al.bf16),
    tid: al.u32,
):
    shm_vec = al.view(shm, al.u32, al.make_layout(
        (SHM_TOTAL_BF16_A // VEC_SIZE, 4), (VEC_SIZE // 2, 1),
    ))
    row = tid // SHM_CHUNKS_PER_ROW
    rg = row // SHM_PAD_ROWS
    ri = row - rg * SHM_PAD_ROWS
    chunk = tid - row * SHM_CHUNKS_PER_ROW
    shm_chunk = rg * (SHM_GROUP_WORDS // (VEC_SIZE // 2)) + ri * SHM_CHUNKS_PER_ROW + chunk
    c_stride = (THREADS * VEC_SIZE // GROUP_K // SHM_PAD_ROWS) * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
    for i in al.range(REG_ROWS_A):
        packed = al.view(reg[i], al.Tensor((4,), al.u32))
        shm_vec[shm_chunk + i * c_stride] = packed


@avelang.jit
def _store_shm_b(
    shm: al.Tensor((SHM_TOTAL_BF16_B,), al.bf16),
    reg: al.Tensor((REG_ROWS_B, VEC_SIZE), al.bf16),
    tid: al.u32,
):
    shm_vec = al.view(shm, al.u32, al.make_layout(
        (SHM_TOTAL_BF16_B // VEC_SIZE, 4), (VEC_SIZE // 2, 1),
    ))
    row = tid // SHM_CHUNKS_PER_ROW
    rg = row // SHM_PAD_ROWS
    ri = row - rg * SHM_PAD_ROWS
    chunk = tid - row * SHM_CHUNKS_PER_ROW
    shm_chunk = rg * (SHM_GROUP_WORDS // (VEC_SIZE // 2)) + ri * SHM_CHUNKS_PER_ROW + chunk
    c_stride = (THREADS * VEC_SIZE // GROUP_K // SHM_PAD_ROWS) * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
    for i in al.range(REG_ROWS_B):
        packed = al.view(reg[i], al.Tensor((4,), al.u32))
        shm_vec[shm_chunk + i * c_stride] = packed


@avelang.jit
def _load_shm_a_frags(
    shm: al.Tensor((SHM_TOTAL_BF16_A,), al.bf16),
    row_base: al.u32,
    batch_id: al.u32,
    wtid: al.u32,
    data: al.Tensor((M_TILES, 4), al.u32),
):
    shm_vec = al.view(shm, al.u32, al.make_layout(
        (SHM_GROUPS_A, SHM_PAD_ROWS, SHM_CHUNKS_PER_ROW, 4),
        (SHM_GROUP_WORDS, GROUP_K // 2, 4, 1),
    ))
    rs = row_base + (wtid % 16) * M_TILES
    cb = (wtid // 16) + batch_id * (32 // VEC_SIZE)
    for tile in al.range(M_TILES):
        r = rs + tile
        rg = r // SHM_PAD_ROWS
        ri = r - rg * SHM_PAD_ROWS
        data[tile] = shm_vec[rg, ri, cb]


@avelang.jit
def _load_shm_b_frags(
    shm: al.Tensor((SHM_TOTAL_BF16_B,), al.bf16),
    row_base: al.u32,
    batch_id: al.u32,
    wtid: al.u32,
    data: al.Tensor((N_TILES, 4), al.u32),
):
    shm_vec = al.view(shm, al.u32, al.make_layout(
        (SHM_GROUPS_B, SHM_PAD_ROWS, SHM_CHUNKS_PER_ROW, 4),
        (SHM_GROUP_WORDS, GROUP_K // 2, 4, 1),
    ))
    rs = row_base + (wtid % 16) * N_TILES
    cb = (wtid // 16) + batch_id * (32 // VEC_SIZE)
    for tile in al.range(N_TILES):
        r = rs + tile
        rg = r // SHM_PAD_ROWS
        ri = r - rg * SHM_PAD_ROWS
        data[tile] = shm_vec[rg, ri, cb]


@avelang.jit
def _mfma_compute(
    data_a: al.Tensor((M_TILES, 4), al.u32),
    data_b: al.Tensor((N_TILES, 4), al.u32),
    acc: al.Tensor((M_TILES, N_TILES, 4), al.f32),
):
    for tm in al.range(M_TILES):
        for tn in al.range(N_TILES):
            fa = al.view(data_a[tm], al.Tensor((2, 2, 1), al.u32))
            fb = al.view(data_b[tn], al.Tensor((2, 2, 1), al.u32))
            acc[tm, tn] = al.amdgpu.mfma_16x16x16_bf16_f32(fa[0], fb[0], acc[tm, tn])
            acc[tm, tn] = al.amdgpu.mfma_16x16x16_bf16_f32(fa[1], fb[1], acc[tm, tn])


@avelang.jit
def _gelu_f32(x: al.f32) -> al.f32:
    x3 = x * x * x
    inner = al.convert(0.7978845608028654, al.f32) * (x + al.convert(0.044715, al.f32) * x3)
    return al.convert(0.5, al.f32) * x * (al.convert(1.0, al.f32) + al.tanh(inner))


@avelang.jit
def _write_results_gelu_bias(
    dst_rsrc: al.Tensor((4,), al.u32),
    b_ptr: al.Pointer(al.bf16),
    N: al.u32,
    group_m: al.u32,
    group_n: al.u32,
    wtid: al.u32,
    warp_m: al.u32,
    warp_n: al.u32,
    acc: al.Tensor((M_TILES, N_TILES, 4), al.f32),
):
    one = al.convert(1, al.i32)
    b = al.make_tensor(b_ptr, al.bf16, al.make_layout((group_n * GROUP_N + GROUP_N,), (one,)))

    lane_grp = wtid // 16
    lane_col = wtid % 16
    warp_offset = (
        (group_m * GROUP_M + warp_m * WARP_MAT_M) * N
        + group_n * GROUP_N
        + warp_n * WARP_MAT_N
    ) * BF16_BYTES

    for tm in al.range(M_TILES):
        for ai in al.range(4):
            row_offset = (lane_grp * (4 * M_TILES) + ai * M_TILES + tm) * N
            col_base = lane_col * N_TILES
            t_off_base = (row_offset + col_base) * BF16_BYTES

            v0 = _gelu_f32(acc[tm, 0, ai] + al.convert(b[group_n * GROUP_N + warp_n * WARP_MAT_N + col_base + 0], al.f32))
            v1 = _gelu_f32(acc[tm, 1, ai] + al.convert(b[group_n * GROUP_N + warp_n * WARP_MAT_N + col_base + 1], al.f32))
            v2 = _gelu_f32(acc[tm, 2, ai] + al.convert(b[group_n * GROUP_N + warp_n * WARP_MAT_N + col_base + 2], al.f32))
            v3 = _gelu_f32(acc[tm, 3, ai] + al.convert(b[group_n * GROUP_N + warp_n * WARP_MAT_N + col_base + 3], al.f32))

            lo0 = al.bitcast(v0, al.u32)
            hi0 = al.bitcast(v1, al.u32)
            lo1 = al.bitcast(v2, al.u32)
            hi1 = al.bitcast(v3, al.u32)
            packed = al.full((2,), 0, al.u32)
            packed[0] = al.amdgpu.perm(hi0, lo0, 0x07060302)
            packed[1] = al.amdgpu.perm(hi1, lo1, 0x07060302)
            al.amdgpu.raw_buffer_store_x2(packed, dst_rsrc, t_off_base, warp_offset, 0)


@avelang.jit
def matmul_bias_gelu_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.u32,
    N: al.u32,
    K: al.u32,
):
    tid = al.thread_id(0)
    wid = tid // WARP_SIZE
    wtid = tid % WARP_SIZE
    warp_m = wid // 2
    warp_n = wid % 2

    one = al.convert(1, al.i32)
    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((M, K), (K, one)))
    w = al.make_tensor(w_ptr, al.bf16, al.make_layout((N, K), (K, one)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((M * N,), (one,)))

    x_rsrc = al.amdgpu.make_rsrc(x, M * K * BF16_BYTES)
    w_rsrc = al.amdgpu.make_rsrc(w, N * K * BF16_BYTES)
    out_rsrc = al.amdgpu.make_rsrc(out, M * N * BF16_BYTES)

    block = al.block_id(0)
    group_m = block // (N // GROUP_N)
    group_n = block % (N // GROUP_N)

    shm_a = al.make_shared((SHM_TOTAL_BF16_A,), al.bf16)
    shm_b = al.make_shared((SHM_TOTAL_BF16_B,), al.bf16)
    reg_a = al.make_local((REG_ROWS_A, VEC_SIZE), al.bf16)
    reg_b = al.make_local((REG_ROWS_B, VEC_SIZE), al.bf16)
    data_a = al.make_local((M_TILES, 4), al.u32)
    data_b = al.make_local((N_TILES, 4), al.u32)
    acc = al.make_local((M_TILES, N_TILES, 4), al.f32)

    for tm in al.range(M_TILES):
        for tn in al.range(N_TILES):
            acc[tm, tn, 0] = al.convert(0.0, al.f32)
            acc[tm, tn, 1] = al.convert(0.0, al.f32)
            acc[tm, tn, 2] = al.convert(0.0, al.f32)
            acc[tm, tn, 3] = al.convert(0.0, al.f32)

    k_total = K // GROUP_K

    for k_idx in al.range(k_total):
        _load_global_a(x_rsrc, K, group_m, k_idx, tid, reg_a)
        _load_global_b(w_rsrc, K, group_n, k_idx, tid, reg_b)
        _store_shm_a(shm_a, reg_a, tid)
        _store_shm_b(shm_b, reg_b, tid)
        al.syncthreads()

        for batch in al.range(2):
            _load_shm_a_frags(shm_a, warp_m * WARP_MAT_M, batch, wtid, data_a)
            _load_shm_b_frags(shm_b, warp_n * WARP_MAT_N, batch, wtid, data_b)
            _mfma_compute(data_a, data_b, acc)

        al.syncthreads()

    _write_results_gelu_bias(out_rsrc, b_ptr, N, group_m, group_n, wtid, warp_m, warp_n, acc)


# ---------------------------------------------------------------------------
# Softmax along dim=1
# ---------------------------------------------------------------------------
@avelang.jit
def softmax_kernel(
    in_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
):
    one = al.convert(1, al.i32)
    in_t = al.make_tensor(in_ptr, al.bf16, al.make_layout((M, N), (N, one)))
    out_t = al.make_tensor(out_ptr, al.bf16, al.make_layout((M, N), (N, one)))
    row = al.block_id(0)
    tid = al.thread_id(0)
    warp_id = tid // 64
    lane_id = tid % 64
    start_col = tid * 32

    local_max = al.convert(-1.0e12, al.f32)
    for i in al.range(32):
        v = al.convert(in_t[row, start_col + i], al.f32)
        if v > local_max:
            local_max = v

    v = al.shuffle_down(local_max, 32, 64)
    if v > local_max: local_max = v
    v = al.shuffle_down(local_max, 16, 64)
    if v > local_max: local_max = v
    v = al.shuffle_down(local_max, 8, 64)
    if v > local_max: local_max = v
    v = al.shuffle_down(local_max, 4, 64)
    if v > local_max: local_max = v
    v = al.shuffle_down(local_max, 2, 64)
    if v > local_max: local_max = v
    v = al.shuffle_down(local_max, 1, 64)
    if v > local_max: local_max = v

    smem = al.make_shared((4,), al.f32)
    if lane_id == 0:
        smem[warp_id] = local_max
    al.syncthreads()
    if tid == 0:
        w0 = smem[0]; w1 = smem[1]; w2 = smem[2]; w3 = smem[3]
        best01 = w0
        if w1 > best01: best01 = w1
        best23 = w2
        if w3 > best23: best23 = w3
        row_max = best01
        if best23 > row_max: row_max = best23
        smem[0] = row_max
    al.syncthreads()
    row_max = smem[0]

    local_sum = al.convert(0.0, al.f32)
    for i in al.range(32):
        v = al.convert(in_t[row, start_col + i], al.f32)
        local_sum = local_sum + al.exp(v - row_max)

    local_sum = local_sum + al.shuffle_down(local_sum, 32, 64)
    local_sum = local_sum + al.shuffle_down(local_sum, 16, 64)
    local_sum = local_sum + al.shuffle_down(local_sum, 8, 64)
    local_sum = local_sum + al.shuffle_down(local_sum, 4, 64)
    local_sum = local_sum + al.shuffle_down(local_sum, 2, 64)
    local_sum = local_sum + al.shuffle_down(local_sum, 1, 64)

    if lane_id == 0:
        smem[warp_id] = local_sum
    al.syncthreads()
    if tid == 0:
        row_sum = smem[0] + smem[1] + smem[2] + smem[3]
        smem[0] = row_sum
    al.syncthreads()
    row_sum = smem[0]
    inv_sum = al.convert(1.0, al.f32) / row_sum

    for i in al.range(32):
        v = al.convert(in_t[row, start_col + i], al.f32)
        out_t[row, start_col + i] = al.convert(al.exp(v - row_max) * inv_sum, al.bf16)


# ---------------------------------------------------------------------------
# Host wrappers
# ---------------------------------------------------------------------------

def _run_matmul_bias_gelu(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty(M * N, dtype=torch.bfloat16, device=x.device)
    m_groups = M // GROUP_M
    n_groups = N // GROUP_N
    grid = (m_groups * n_groups, 1, 1)
    matmul_bias_gelu_kernel[lambda: (grid, (THREADS, 1, 1))](
        x.contiguous(), w.contiguous(), b.contiguous(), out, M, N, K,
    )
    return out.view(M, N)


def _run_softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty_like(x)
    grid = (M, 1, 1)
    softmax_kernel[lambda: (grid, (256, 1, 1))](x.contiguous(), out, M, N)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.linear.weight.data
        b = self.linear.bias.data if self.linear.bias is not None else torch.zeros(
            self.out_features, device=x.device, dtype=x.dtype
        )
        x_bf16 = x.to(torch.bfloat16)
        w_bf16 = w.to(torch.bfloat16)
        b_bf16 = b.to(torch.bfloat16)

        y = _run_matmul_bias_gelu(x_bf16, w_bf16, b_bf16)
        y = _run_softmax(y)
        return y.to(x.dtype)
