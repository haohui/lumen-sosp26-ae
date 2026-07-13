import torch
import torch.nn as nn
import avelang
import avelang.language as al

EPS = 1e-05
BF16_BYTES = 2

# Tile config — must divide input dims
GROUP_M = 64
GROUP_N = 64
GROUP_K = 64
WARP_SIZE = 64
NUM_WARPS = 4
WARP_PER_ROW = 2
WARP_PER_COL = 2
WARP_MAT_M = GROUP_M // WARP_PER_ROW  # 32
WARP_MAT_N = GROUP_N // WARP_PER_COL  # 32
M_TILES_PER_WARP = WARP_MAT_M // 16  # 2
N_TILES_PER_WARP = WARP_MAT_N // 16  # 2
VEC_SIZE = 8
THREADS = WARP_SIZE * NUM_WARPS  # 256
REG_ROWS_A = GROUP_M * GROUP_K // VEC_SIZE // THREADS  # 2
REG_ROWS_B = GROUP_N * GROUP_K // VEC_SIZE // THREADS  # 2
SHM_PAD_ROWS = 4
SHM_PAD_BF16 = 16
SHM_GROUPS_A = GROUP_M // SHM_PAD_ROWS  # 16
SHM_GROUPS_B = GROUP_N // SHM_PAD_ROWS  # 16
SHM_GROUP_BF16 = SHM_PAD_ROWS * GROUP_K + SHM_PAD_BF16  # 4*64+16 = 272
SHM_GROUP_WORDS = SHM_GROUP_BF16 // 2  # 136
SHM_TOTAL_BF16_A = SHM_GROUPS_A * SHM_GROUP_BF16  # 4352
SHM_TOTAL_BF16_B = SHM_GROUPS_B * SHM_GROUP_BF16  # 4352
SHM_CHUNKS_PER_ROW = GROUP_K // VEC_SIZE  # 8


@avelang.jit
def _load_global_a(
    src_rsrc: al.Tensor((4,), al.u32),
    K_dim: al.i32,
    group_m: al.i32,
    k_idx: al.i32,
    tid: al.i32,
    reg: al.Tensor((REG_ROWS_A, VEC_SIZE), al.bf16),
):
    row = tid // SHM_CHUNKS_PER_ROW
    col = (tid - row * SHM_CHUNKS_PER_ROW) * VEC_SIZE
    thread_offset = (row * K_dim + col) * BF16_BYTES
    tile_offset = (group_m * GROUP_M * K_dim + k_idx * GROUP_K) * BF16_BYTES
    thread_stride = (THREADS * VEC_SIZE // GROUP_K) * K_dim * BF16_BYTES

    for i in al.range(REG_ROWS_A):
        packed = al.amdgpu.raw_buffer_load_x4(
            src_rsrc,
            thread_offset,
            tile_offset + i * thread_stride,
            0,
        )
        frag = al.view(packed, al.Tensor((VEC_SIZE,), al.bf16))
        for v in al.range(VEC_SIZE):
            reg[i, v] = frag[v]


@avelang.jit
def _load_global_b(
    src_rsrc: al.Tensor((4,), al.u32),
    K_dim: al.i32,
    group_n: al.i32,
    k_idx: al.i32,
    tid: al.i32,
    reg: al.Tensor((REG_ROWS_B, VEC_SIZE), al.bf16),
):
    row = tid // SHM_CHUNKS_PER_ROW
    col = (tid - row * SHM_CHUNKS_PER_ROW) * VEC_SIZE
    thread_offset = (row * K_dim + col) * BF16_BYTES
    tile_offset = (group_n * GROUP_N * K_dim + k_idx * GROUP_K) * BF16_BYTES
    thread_stride = (THREADS * VEC_SIZE // GROUP_K) * K_dim * BF16_BYTES

    for i in al.range(REG_ROWS_B):
        packed = al.amdgpu.raw_buffer_load_x4(
            src_rsrc,
            thread_offset,
            tile_offset + i * thread_stride,
            0,
        )
        frag = al.view(packed, al.Tensor((VEC_SIZE,), al.bf16))
        for v in al.range(VEC_SIZE):
            reg[i, v] = frag[v]


@avelang.jit
def _store_shm_a(
    shm: al.Tensor((SHM_TOTAL_BF16_A,), al.bf16),
    reg: al.Tensor((REG_ROWS_A, VEC_SIZE), al.bf16),
    tid: al.i32,
):
    row = tid // SHM_CHUNKS_PER_ROW
    row_group = row // SHM_PAD_ROWS
    row_in_group = row - row_group * SHM_PAD_ROWS
    chunk = tid - row * SHM_CHUNKS_PER_ROW
    shm_chunk = (
        row_group * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
        + row_in_group * SHM_CHUNKS_PER_ROW
        + chunk
    )
    shm_chunk_stride = (
        (THREADS * VEC_SIZE // GROUP_K // SHM_PAD_ROWS)
        * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
    )

    shm_vec = al.view(
        shm,
        al.u32,
        al.make_layout(
            (SHM_TOTAL_BF16_A // VEC_SIZE, 4),
            (VEC_SIZE // 2, 1),
        ),
    )

    for i in al.range(REG_ROWS_A):
        packed = al.view(reg[i], al.Tensor((4,), al.u32))
        shm_vec[shm_chunk + i * shm_chunk_stride] = packed


@avelang.jit
def _store_shm_b(
    shm: al.Tensor((SHM_TOTAL_BF16_B,), al.bf16),
    reg: al.Tensor((REG_ROWS_B, VEC_SIZE), al.bf16),
    tid: al.i32,
):
    row = tid // SHM_CHUNKS_PER_ROW
    row_group = row // SHM_PAD_ROWS
    row_in_group = row - row_group * SHM_PAD_ROWS
    chunk = tid - row * SHM_CHUNKS_PER_ROW
    shm_chunk = (
        row_group * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
        + row_in_group * SHM_CHUNKS_PER_ROW
        + chunk
    )
    shm_chunk_stride = (
        (THREADS * VEC_SIZE // GROUP_K // SHM_PAD_ROWS)
        * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
    )

    shm_vec = al.view(
        shm,
        al.u32,
        al.make_layout(
            (SHM_TOTAL_BF16_B // VEC_SIZE, 4),
            (VEC_SIZE // 2, 1),
        ),
    )

    for i in al.range(REG_ROWS_B):
        packed = al.view(reg[i], al.Tensor((4,), al.u32))
        shm_vec[shm_chunk + i * shm_chunk_stride] = packed


@avelang.jit
def _load_shm_to_regs_batch_a(
    shm: al.Tensor((SHM_TOTAL_BF16_A,), al.bf16),
    row_base: al.i32,
    batch_id: al.i32,
    wtid: al.i32,
    data: al.Tensor((M_TILES_PER_WARP, 4), al.u32),
):
    shm_vec = al.view(
        shm,
        al.u32,
        al.make_layout(
            (SHM_GROUPS_A, SHM_PAD_ROWS, SHM_CHUNKS_PER_ROW, 4),
            (SHM_GROUP_WORDS, GROUP_K // 2, 4, 1),
        ),
    )

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
    row_base: al.i32,
    batch_id: al.i32,
    wtid: al.i32,
    data: al.Tensor((N_TILES_PER_WARP, 4), al.u32),
):
    shm_vec = al.view(
        shm,
        al.u32,
        al.make_layout(
            (SHM_GROUPS_B, SHM_PAD_ROWS, SHM_CHUNKS_PER_ROW, 4),
            (SHM_GROUP_WORDS, GROUP_K // 2, 4, 1),
        ),
    )

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
                frag_a[0],
                frag_b[0],
                acc[tile_m, tile_n],
            )
            acc[tile_m, tile_n] = al.amdgpu.mfma_16x16x16_bf16_f32(
                frag_a[1],
                frag_b[1],
                acc[tile_m, tile_n],
            )


@avelang.jit
def gemm_scale_kernel(
    A: al.Pointer(al.bf16),
    B: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    scale_ptr: al.Pointer(al.bf16),
    C: al.Pointer(al.bf16),
    m: al.i32,
    k: al.i32,
    n: al.i32,
    m_groups: al.i32,
    n_groups: al.i32,
):
    tid = al.thread_id(0)
    wid = tid // WARP_SIZE
    wtid = tid % WARP_SIZE
    warp_row = wid // WARP_PER_COL
    warp_col = wid % WARP_PER_COL
    group_m = al.block_id(0)
    group_n = al.block_id(1)

    a_tensor = al.make_tensor(A, al.bf16, al.make_layout((m, k), (k, 1)))
    b_tensor = al.make_tensor(B, al.bf16, al.make_layout((n, k), (k, 1)))
    c_tensor = al.make_tensor(C, al.bf16, al.make_layout((m, n), (n, 1)))
    bias_t = al.make_tensor(bias_ptr, al.bf16, al.make_layout((n,), (1,)))
    scale_t = al.make_tensor(scale_ptr, al.bf16, al.make_layout((n,), (1,)))

    a_rsrc = al.amdgpu.make_rsrc(a_tensor, m * k * BF16_BYTES)
    b_rsrc = al.amdgpu.make_rsrc(b_tensor, k * n * BF16_BYTES)

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
                acc[tile_m, tile_n, acc_idx] = 0.0

    k_total = k // GROUP_K
    ki = al.convert(0, al.i32)
    gk = al.convert(GROUP_K, al.i32)

    _load_global_a(a_rsrc, k, group_m, ki, tid, reg_a)
    _load_global_b(b_rsrc, k, group_n, ki, tid, reg_b)
    _store_shm_a(shm_a, reg_a, tid)
    _store_shm_b(shm_b, reg_b, tid)
    al.syncthreads()

    _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 0, wtid, data_a0)
    _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 0, wtid, data_b0)
    _load_global_a(a_rsrc, k, group_m, ki + 1, tid, reg_a)
    _load_global_b(b_rsrc, k, group_n, ki + 1, tid, reg_b)

    for k_idx in al.range(0, k_total - 3, 2):
        _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 1, wtid, data_a1)
        _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 1, wtid, data_b1)
        _matmul_from_regs_batch(data_a0, data_b0, acc)
        al.syncthreads()

        _store_shm_a(shm_a, reg_a, tid)
        _store_shm_b(shm_b, reg_b, tid)
        _load_global_a(a_rsrc, k, group_m, ki + k_idx + 2, tid, reg_a)
        _load_global_b(b_rsrc, k, group_n, ki + k_idx + 2, tid, reg_b)
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
        _load_global_a(a_rsrc, k, group_m, ki + k_idx + 3, tid, reg_a)
        _load_global_b(b_rsrc, k, group_n, ki + k_idx + 3, tid, reg_b)
        al.syncthreads()

        _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 0, wtid, data_a0)
        _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 0, wtid, data_b0)
        _matmul_from_regs_batch(data_a1, data_b1, acc)

    # Epilogue
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

    # ── Write results with bias + scale ───────────────────────────────────
    lane_row_group = wtid // 16
    lane_col = wtid % 16

    for tile_m in al.range(M_TILES_PER_WARP):
        for acc_idx in al.range(4):
            row_offset = (
                lane_row_group * (4 * M_TILES_PER_WARP)
                + acc_idx * M_TILES_PER_WARP
                + tile_m
            )
            col_offset = lane_col * N_TILES_PER_WARP

            g_row = group_m * GROUP_M + warp_row * WARP_MAT_M + row_offset
            g_col_base = group_n * GROUP_N + warp_col * WARP_MAT_N + col_offset

            for tile_n in al.range(N_TILES_PER_WARP):
                g_col = g_col_base + tile_n
                val_f32 = acc[tile_m, tile_n, acc_idx]
                val_scaled = (val_f32 + al.convert(bias_t[g_col], al.f32)) * al.convert(scale_t[g_col], al.f32)
                c_tensor[g_row, g_col] = al.convert(val_scaled, al.bf16)


# ── Batch-norm kernel ────────────────────────────────────────────────────────
@avelang.jit
def batchnorm_kernel(
    Y_ptr: al.Pointer(al.bf16),
    bn_weight_ptr: al.Pointer(al.bf16),
    bn_bias_ptr: al.Pointer(al.bf16),
    running_mean_ptr: al.Pointer(al.bf16),
    running_var_ptr: al.Pointer(al.bf16),
    BATCH: al.i32,
    OUT_FEAT: al.i32,
    FEAT_PER_WG: al.i32,
):
    Y = al.make_tensor(Y_ptr, al.bf16,
                       al.make_layout((BATCH, OUT_FEAT), (OUT_FEAT, al.convert(1, al.i32))))
    bn_w = al.make_tensor(bn_weight_ptr, al.bf16,
                          al.make_layout((OUT_FEAT,), (al.convert(1, al.i32),)))
    bn_b = al.make_tensor(bn_bias_ptr, al.bf16,
                          al.make_layout((OUT_FEAT,), (al.convert(1, al.i32),)))
    r_mean = al.make_tensor(running_mean_ptr, al.bf16,
                            al.make_layout((OUT_FEAT,), (al.convert(1, al.i32),)))
    r_var = al.make_tensor(running_var_ptr, al.bf16,
                           al.make_layout((OUT_FEAT,), (al.convert(1, al.i32),)))

    tid = al.thread_id(0)
    wg_id = al.block_id(0)
    nthreads = al.convert(256, al.i32)
    one = al.convert(1, al.i32)

    feat_start = wg_id * FEAT_PER_WG
    feat_end = feat_start + FEAT_PER_WG

    for feat in al.range(feat_start, feat_end, one):
        m = al.convert(r_mean[feat], al.f32)
        v = al.convert(r_var[feat], al.f32)
        denom = al.sqrt(v + al.convert(1e-05, al.f32))
        w_val = al.convert(bn_w[feat], al.f32)
        b_val = al.convert(bn_b[feat], al.f32)
        inv_denom = al.convert(1.0, al.f32) / denom
        for i in al.range(tid, BATCH, nthreads):
            out = (al.convert(Y[i, feat], al.f32) - m) * inv_denom
            out = out * w_val + b_val
            Y[i, feat] = al.convert(out, al.bf16)


# ── Host ─────────────────────────────────────────────────────────────────────

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-05, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        B = x.shape[0]
        K_in = x.shape[1]
        N = self.scale.shape[0]

        if x.dtype != torch.bfloat16:
            raise RuntimeError('This fused kernel only supports bf16 input.')

        w_for_kernel = self.gemm.weight.to(device=x.device, dtype=x.dtype).contiguous()
        bias_val = self.gemm.bias.to(device=x.device, dtype=x.dtype).contiguous()
        scale_val = self.scale.to(device=x.device, dtype=x.dtype).contiguous()
        bn_w = self.bn.weight.to(device=x.device, dtype=x.dtype).contiguous()
        bn_b = self.bn.bias.to(device=x.device, dtype=x.dtype).contiguous()

        y = torch.empty((B, N), device=x.device, dtype=x.dtype)
        grid_m = (B + GROUP_M - 1) // GROUP_M
        grid_n = (N + GROUP_N - 1) // GROUP_N

        gemm_scale_kernel[lambda: ((grid_m, grid_n, 1), (256, 1, 1))](
            x.contiguous(), w_for_kernel, bias_val, scale_val, y,
            B, K_in, N, grid_m, grid_n,
        )

        FEAT_PER_WG = 64
        grid_bn = (N + FEAT_PER_WG - 1) // FEAT_PER_WG
        running_mean = self.bn.running_mean.to(device=x.device, dtype=x.dtype).contiguous()
        running_var = self.bn.running_var.to(device=x.device, dtype=x.dtype).contiguous()
        batchnorm_kernel[lambda: ((grid_bn, 1, 1), (256, 1, 1))](
            y, bn_w, bn_b, running_mean, running_var, B, N, FEAT_PER_WG,
        )

        return y
