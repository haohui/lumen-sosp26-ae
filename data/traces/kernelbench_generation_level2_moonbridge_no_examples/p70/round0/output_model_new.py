import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Tile configuration from reference MFMA GEMM
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


@avelang.jit
def _gload_a(
    src_rsrc: al.Tensor((4,), al.u32),
    k: al.i32,
    group_m: al.i32,
    k_idx: al.i32,
    tid: al.i32,
    reg: al.Tensor((REG_ROWS_A, VEC_SIZE), al.bf16),
):
    row = tid // SHM_CHUNKS_PER_ROW
    col = (tid - row * SHM_CHUNKS_PER_ROW) * VEC_SIZE
    thread_offset = (row * k + col) * BF16_BYTES
    tile_offset = (group_m * GROUP_M * k + k_idx * GROUP_K) * BF16_BYTES
    thread_stride = (THREADS * VEC_SIZE // GROUP_K) * k * BF16_BYTES

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
def _gload_b(
    src_rsrc: al.Tensor((4,), al.u32),
    k: al.i32,
    group_n: al.i32,
    k_idx: al.i32,
    tid: al.i32,
    reg: al.Tensor((REG_ROWS_B, VEC_SIZE), al.bf16),
):
    row = tid // SHM_CHUNKS_PER_ROW
    col = (tid - row * SHM_CHUNKS_PER_ROW) * VEC_SIZE
    thread_offset = (row * k + col) * BF16_BYTES
    tile_offset = (group_n * GROUP_N * k + k_idx * GROUP_K) * BF16_BYTES
    thread_stride = (THREADS * VEC_SIZE // GROUP_K) * k * BF16_BYTES

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
def _sshare_a(
    shm: al.Tensor((SHM_TOTAL_BF16_A,), al.bf16),
    reg: al.Tensor((REG_ROWS_A, VEC_SIZE), al.bf16),
    tid: al.i32,
):
    shm_vec = al.view(
        shm,
        al.u32,
        al.make_layout(
            (SHM_TOTAL_BF16_A // VEC_SIZE, 4),
            (VEC_SIZE // 2, 1),
        ),
    )
    row = tid // SHM_CHUNKS_PER_ROW
    row_group = row // SHM_PAD_ROWS
    row_in_group = row - row_group * SHM_PAD_ROWS
    chunk = tid - row * SHM_CHUNKS_PER_ROW
    shm_chunk = (
        row_group * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
        + row_in_group * SHM_CHUNKS_PER_ROW
        + chunk
    )
    shm_stride = (
        (THREADS * VEC_SIZE // GROUP_K // SHM_PAD_ROWS)
        * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
    )
    for i in al.range(REG_ROWS_A):
        packed = al.view(reg[i], al.Tensor((4,), al.u32))
        shm_vec[shm_chunk + i * shm_stride] = packed


@avelang.jit
def _sshare_b(
    shm: al.Tensor((SHM_TOTAL_BF16_B,), al.bf16),
    reg: al.Tensor((REG_ROWS_B, VEC_SIZE), al.bf16),
    tid: al.i32,
):
    shm_vec = al.view(
        shm,
        al.u32,
        al.make_layout(
            (SHM_TOTAL_BF16_B // VEC_SIZE, 4),
            (VEC_SIZE // 2, 1),
        ),
    )
    row = tid // SHM_CHUNKS_PER_ROW
    row_group = row // SHM_PAD_ROWS
    row_in_group = row - row_group * SHM_PAD_ROWS
    chunk = tid - row * SHM_CHUNKS_PER_ROW
    shm_chunk = (
        row_group * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
        + row_in_group * SHM_CHUNKS_PER_ROW
        + chunk
    )
    shm_stride = (
        (THREADS * VEC_SIZE // GROUP_K // SHM_PAD_ROWS)
        * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
    )
    for i in al.range(REG_ROWS_B):
        packed = al.view(reg[i], al.Tensor((4,), al.u32))
        shm_vec[shm_chunk + i * shm_stride] = packed


@avelang.jit
def _lshare_a(
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
def _lshare_b(
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
def _mfma_step(
    data_a: al.Tensor((M_TILES_PER_WARP, 4), al.u32),
    data_b: al.Tensor((N_TILES_PER_WARP, 4), al.u32),
    acc: al.Tensor((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32),
):
    for tm in al.range(M_TILES_PER_WARP):
        for tn in al.range(N_TILES_PER_WARP):
            frag_a = al.view(data_a[tm], al.Tensor((2, 2, 1), al.u32))
            frag_b = al.view(data_b[tn], al.Tensor((2, 2, 1), al.u32))
            acc[tm, tn] = al.amdgpu.mfma_16x16x16_bf16_f32(
                frag_a[0],
                frag_b[0],
                acc[tm, tn],
            )
            acc[tm, tn] = al.amdgpu.mfma_16x16x16_bf16_f32(
                frag_a[1],
                frag_b[1],
                acc[tm, tn],
            )


@avelang.jit
def gemm_sigmoid_scale_residual_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
    stride_a_m: al.i32,
    stride_a_k: al.i32,
    stride_b_n: al.i32,
    stride_b_k: al.i32,
    stride_c_m: al.i32,
    stride_c_n: al.i32,
    SCALE: al.constexpr,
):
    tid = al.thread_id(0)
    wid = tid // WARP_SIZE
    wtid = tid % WARP_SIZE
    warp_row = wid // WARP_PER_COL
    warp_col = wid % WARP_PER_COL

    block_m = al.block_id(1)
    block_n = al.block_id(0)

    start_m = block_m * GROUP_M
    start_n = block_n * GROUP_N

    layout_a = al.make_layout((M, K), (stride_a_m, stride_a_k))
    a = al.make_tensor(a_ptr, al.bf16, layout_a)
    layout_b = al.make_layout((N, K), (stride_b_n, stride_b_k))
    b = al.make_tensor(b_ptr, al.bf16, layout_b)
    layout_bias = al.make_layout((N,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, layout_bias)
    layout_c = al.make_layout((M, N), (stride_c_m, stride_c_n))
    c = al.make_tensor(c_ptr, al.bf16, layout_c)

    a_rsrc = al.amdgpu.make_rsrc(a, M * K * BF16_BYTES)
    b_rsrc = al.amdgpu.make_rsrc(b, N * K * BF16_BYTES)

    shm_a = al.make_shared((SHM_TOTAL_BF16_A,), al.bf16)
    shm_b = al.make_shared((SHM_TOTAL_BF16_B,), al.bf16)

    reg_a = al.make_local((REG_ROWS_A, VEC_SIZE), al.bf16)
    reg_b = al.make_local((REG_ROWS_B, VEC_SIZE), al.bf16)
    data_a = al.make_local((M_TILES_PER_WARP, 4), al.u32)
    data_b = al.make_local((N_TILES_PER_WARP, 4), al.u32)
    acc = al.make_local((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32)

    for tm in al.range(M_TILES_PER_WARP):
        for tn in al.range(N_TILES_PER_WARP):
            for ai in al.range(4):
                acc[tm, tn, ai] = al.convert(0.0, al.f32)

    k_total = K // GROUP_K

    # Load first tile
    _gload_a(a_rsrc, K, block_m, 0, tid, reg_a)
    _gload_b(b_rsrc, K, block_n, 0, tid, reg_b)
    _sshare_a(shm_a, reg_a, tid)
    _sshare_b(shm_b, reg_b, tid)
    al.syncthreads()

    _lshare_a(shm_a, warp_row * WARP_MAT_M, 0, wtid, data_a)
    _lshare_b(shm_b, warp_col * WARP_MAT_N, 0, wtid, data_b)

    for k_idx in al.range(1, k_total + 1):
        if k_idx < k_total:
            _gload_a(a_rsrc, K, block_m, k_idx, tid, reg_a)
            _gload_b(b_rsrc, K, block_n, k_idx, tid, reg_b)

        # MFMA on batch 0 (K = 0..31 within the GROUP_K tile)
        _mfma_step(data_a, data_b, acc)

        # Load batch 1 (K = 32..63) from same shared memory tile
        _lshare_a(shm_a, warp_row * WARP_MAT_M, 1, wtid, data_a)
        _lshare_b(shm_b, warp_col * WARP_MAT_N, 1, wtid, data_b)
        _mfma_step(data_a, data_b, acc)

        if k_idx < k_total:
            al.syncthreads()
            _sshare_a(shm_a, reg_a, tid)
            _sshare_b(shm_b, reg_b, tid)
            al.syncthreads()
            # Pre-load batch 0 for next tile
            _lshare_a(shm_a, warp_row * WARP_MAT_M, 0, wtid, data_a)
            _lshare_b(shm_b, warp_col * WARP_MAT_N, 0, wtid, data_b)

    # Epilogue: bias add, sigmoid, scale, residual add - per element
    lane_row_group = wtid // 16
    lane_col = wtid % 16

    for tm in al.range(M_TILES_PER_WARP):
        for tn in al.range(N_TILES_PER_WARP):
            for ai in al.range(4):
                row = (
                    start_m
                    + warp_row * WARP_MAT_M
                    + lane_row_group * (4 * M_TILES_PER_WARP)
                    + ai * M_TILES_PER_WARP
                    + tm
                )
                col = (
                    start_n
                    + warp_col * WARP_MAT_N
                    + lane_col * N_TILES_PER_WARP
                    + tn
                )
                if row < M and col < N:
                    val = acc[tm, tn, ai] + al.convert(bias[col], al.f32)
                    orig = val
                    zero = al.convert(0.0, al.f32)
                    one = al.convert(1.0, al.f32)
                    neg_val = zero - val
                    exp_neg = al.exp(neg_val)
                    sig = al.amdgpu.rcp(one + exp_neg)
                    result = sig * al.convert(SCALE, al.f32) + orig
                    c[row, col] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    """AveLang-accelerated Gemm_Sigmoid_Scaling_ResidualAdd with MFMA."""

    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = x.contiguous()
        batch_size, input_size = x.shape
        hidden_size = self.gemm.weight.shape[0]

        weight = self.gemm.weight.data.contiguous()
        bias = self.gemm.bias.data.contiguous()

        x_bf16 = x.to(torch.bfloat16)
        weight_bf16 = weight.to(torch.bfloat16)
        bias_bf16 = bias.to(torch.bfloat16)

        out = torch.empty(batch_size, hidden_size, dtype=torch.bfloat16, device=x.device)

        m_groups = (batch_size + GROUP_M - 1) // GROUP_M
        n_groups = (hidden_size + GROUP_N - 1) // GROUP_N

        gemm_sigmoid_scale_residual_kernel[lambda: ((n_groups, m_groups, 1), (THREADS, 1, 1))](
            x_bf16,
            weight_bf16,
            bias_bf16,
            out,
            batch_size,
            hidden_size,
            input_size,
            x_bf16.stride(0),
            x_bf16.stride(1),
            weight_bf16.stride(0),
            weight_bf16.stride(1),
            out.stride(0),
            out.stride(1),
            float(self.scaling_factor),
        )

        return out
