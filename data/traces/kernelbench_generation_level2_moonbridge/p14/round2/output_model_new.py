import torch
import torch.nn as nn
import avelang
import avelang.language as al


batch_size   = 1024
input_size   = 8192
hidden_size  = 8192
scaling_factor = 1.5

WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS
GROUP_M = 128
GROUP_N = 128
GROUP_K = 64
VEC_SIZE = 8
BF16_BYTES = 2
F32_BYTES = 4
SHM_PAD_ROWS = 4
SHM_PAD_BF16 = 16
SHM_GROUPS_A = GROUP_M // SHM_PAD_ROWS
SHM_GROUPS_B = GROUP_N // SHM_PAD_ROWS
SHM_GROUP_BF16 = SHM_PAD_ROWS * GROUP_K + SHM_PAD_BF16
SHM_GROUP_WORDS = SHM_GROUP_BF16 // 2
SHM_TOTAL_BF16_A = SHM_GROUPS_A * SHM_GROUP_BF16
SHM_TOTAL_BF16_B = SHM_GROUPS_B * SHM_GROUP_BF16
SHM_CHUNKS_PER_ROW = GROUP_K // VEC_SIZE
REG_ROWS_A = GROUP_M * GROUP_K // VEC_SIZE // THREADS
REG_ROWS_B = GROUP_N * GROUP_K // VEC_SIZE // THREADS
WARP_PER_ROW = 2
WARP_PER_COL = 2
WARP_MAT_M = GROUP_M // WARP_PER_ROW
WARP_MAT_N = GROUP_N // WARP_PER_COL
M_TILES_PER_WARP = WARP_MAT_M // 16
N_TILES_PER_WARP = WARP_MAT_N // 16


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
            src_rsrc,
            thread_offset,
            tile_offset + i * thread_offset_stride,
            0,
        )
        frag = al.view(packed, al.Tensor((VEC_SIZE,), al.bf16))
        for v in al.range(VEC_SIZE):
            reg[i, v] = frag[v]


@avelang.jit
def _load_global_b(
    src_rsrc: al.Tensor((4,), al.u32),
    k: al.u32,
    group_row: al.u32,
    k_idx: al.u32,
    tid: al.u32,
    reg: al.Tensor((REG_ROWS_B, VEC_SIZE), al.bf16),
):
    row = tid // SHM_CHUNKS_PER_ROW
    col = (tid - row * SHM_CHUNKS_PER_ROW) * VEC_SIZE
    thread_offset = (row * k + col) * BF16_BYTES
    tile_offset = (group_row * GROUP_N * k + k_idx * GROUP_K) * BF16_BYTES
    thread_offset_stride = (THREADS * VEC_SIZE // GROUP_K) * k * BF16_BYTES

    for i in al.range(REG_ROWS_B):
        packed = al.amdgpu.raw_buffer_load_x4(
            src_rsrc,
            thread_offset,
            tile_offset + i * thread_offset_stride,
            0,
        )
        frag = al.view(packed, al.Tensor((VEC_SIZE,), al.bf16))
        for v in al.range(VEC_SIZE):
            reg[i, v] = frag[v]


@avelang.jit
def _store_shm_a(
    shm: al.Tensor((SHM_TOTAL_BF16_A,), al.bf16),
    reg: al.Tensor((REG_ROWS_A, VEC_SIZE), al.bf16),
    tid: al.u32,
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
    shm_chunk_stride = (
        (THREADS * VEC_SIZE // GROUP_K // SHM_PAD_ROWS)
        * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
    )

    for i in al.range(REG_ROWS_A):
        packed = al.view(reg[i], al.Tensor((4,), al.u32))
        shm_vec[shm_chunk + i * shm_chunk_stride] = packed


@avelang.jit
def _store_shm_b(
    shm: al.Tensor((SHM_TOTAL_BF16_B,), al.bf16),
    reg: al.Tensor((REG_ROWS_B, VEC_SIZE), al.bf16),
    tid: al.u32,
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
    shm_chunk_stride = (
        (THREADS * VEC_SIZE // GROUP_K // SHM_PAD_ROWS)
        * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
    )

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
    row_base: al.u32,
    batch_id: al.u32,
    wtid: al.u32,
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
def _write_results_div2_f32(
    dst_rsrc: al.Tensor((4,), al.u32),
    n: al.u32,
    group_m: al.u32,
    group_n: al.u32,
    wtid: al.u32,
    warp_row: al.u32,
    warp_col: al.u32,
    acc: al.Tensor((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32),
):
    half = al.convert(0.5, al.f32)
    lane_row_group = wtid // 16
    lane_col = wtid % 16
    warp_offset = (
        (group_m * GROUP_M + warp_row * WARP_MAT_M) * n
        + group_n * GROUP_N
        + warp_col * WARP_MAT_N
    ) * F32_BYTES

    for tile_m in al.range(M_TILES_PER_WARP):
        for acc_idx in al.range(4):
            row_offset = (
                lane_row_group * (4 * M_TILES_PER_WARP)
                + acc_idx * M_TILES_PER_WARP
                + tile_m
            ) * n
            col_offset = lane_col * N_TILES_PER_WARP
            thread_offset = (row_offset + col_offset) * F32_BYTES

            val0 = acc[tile_m, 0, acc_idx] * half
            val1 = acc[tile_m, 1, acc_idx] * half
            val2 = acc[tile_m, 2, acc_idx] * half
            val3 = acc[tile_m, 3, acc_idx] * half

            packed = al.full((4,), 0, al.u32)
            packed[0] = al.bitcast(val0, al.u32)
            packed[1] = al.bitcast(val1, al.u32)
            packed[2] = al.bitcast(val2, al.u32)
            packed[3] = al.bitcast(val3, al.u32)
            al.amdgpu.raw_buffer_store_x4(packed, dst_rsrc, thread_offset, warp_offset, 0)


@avelang.jit
def gemm_div2_f32_kernel(
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

    group_m = al.block_id(1)
    group_n = al.block_id(0)

    a_tensor = al.make_tensor(A, al.bf16, al.make_layout((m, k), (k, 1)))
    b_tensor = al.make_tensor(B, al.bf16, al.make_layout((n, k), (k, 1)))
    c_tensor = al.make_tensor(C, al.f32, al.make_layout((m * n,), (1,)))
    a_rsrc = al.amdgpu.make_rsrc(a_tensor, m * k * BF16_BYTES)
    b_rsrc = al.amdgpu.make_rsrc(b_tensor, n * k * BF16_BYTES)
    c_rsrc = al.amdgpu.make_rsrc(c_tensor, m * n * F32_BYTES)

    shm_a = al.make_shared((SHM_TOTAL_BF16_A,), al.bf16)
    shm_b = al.make_shared((SHM_TOTAL_BF16_B,), al.bf16)
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
    for k_idx in al.range(k_total):
        _load_global_a(a_rsrc, k, group_m, k_idx, tid, reg_a)
        _load_global_b(b_rsrc, k, group_n, k_idx, tid, reg_b)
        _store_shm_a(shm_a, reg_a, tid)
        _store_shm_b(shm_b, reg_b, tid)
        al.syncthreads()

        _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 0, wtid, data_a)
        _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 0, wtid, data_b)
        _matmul_from_regs_batch(data_a, data_b, acc)

        _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 1, wtid, data_a)
        _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 1, wtid, data_b)
        _matmul_from_regs_batch(data_a, data_b, acc)

        al.syncthreads()

    _write_results_div2_f32(c_rsrc, n, group_m, group_n, wtid, warp_row, warp_col, acc)


@avelang.jit
def row_sum_reduce_bf16_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    scale: al.f32,
):
    tid = al.thread_id(0)
    row = al.block_id(0)
    if row >= m:
        return

    layout_in = al.make_layout((m, n), (n, 1))
    input_tensor = al.make_tensor(input_ptr, al.bf16, layout_in)

    acc = al.convert(0.0, al.f32)
    for col in al.range(tid, n, al.block_dim(0)):
        val = input_tensor[row, col]
        acc = acc + al.convert(val, al.f32)

    shm = al.make_shared((256,), al.f32)
    shm[tid] = acc
    al.syncthreads()

    if tid < 128:
        shm[tid] = shm[tid] + shm[tid + 128]
    al.syncthreads()
    if tid < 64:
        shm[tid] = shm[tid] + shm[tid + 64]
    al.syncthreads()
    if tid < 32:
        shm[tid] = shm[tid] + shm[tid + 32]
    al.syncthreads()
    if tid < 16:
        shm[tid] = shm[tid] + shm[tid + 16]
    al.syncthreads()
    if tid < 8:
        shm[tid] = shm[tid] + shm[tid + 8]
    al.syncthreads()
    if tid < 4:
        shm[tid] = shm[tid] + shm[tid + 4]
    al.syncthreads()
    if tid < 2:
        shm[tid] = shm[tid] + shm[tid + 2]
    al.syncthreads()
    if tid == 0:
        shm[0] = shm[0] + shm[1]
    al.syncthreads()

    if tid == 0:
        result = shm[0] * scale
        layout_out = al.make_layout((m, 1), (1, 1))
        output_tensor = al.make_tensor(output_ptr, al.bf16, layout_out)
        output_tensor[row, 0] = al.convert(result, al.bf16)


def avelang_gemm_div2_sum_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    scaling_factor: float,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    input_dtype = x.dtype
    device = x.device

    x_bf16 = x.contiguous()
    w_bf16 = weight.contiguous()

    m, k = x_bf16.shape
    n = w_bf16.shape[0]
    weight_k = w_bf16.shape[1]
    if weight_k != k:
        raise ValueError(
            f"Weight/input K mismatch: x has K={k}, weight has K={weight_k}"
        )

    # MFMA GEMM in BF16, f32 output with div-by-2 fused
    gemm_out = torch.empty((m, n), device=device, dtype=torch.float32)
    grid = (n // GROUP_N, m // GROUP_M, 1)
    gemm_div2_f32_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, w_bf16, gemm_out, m, n, k
    )

    # Round to BF16 to match reference's intermediate rounding
    gemm_out_bf16 = gemm_out.to(dtype=torch.bfloat16).contiguous()

    # Reduction: sum rows + scale
    final_out = torch.empty((m, 1), device=device, dtype=torch.bfloat16)
    REDUCE_THREADS = 256
    row_sum_reduce_bf16_kernel[lambda: ((m, 1, 1), (REDUCE_THREADS, 1, 1))](
        gemm_out_bf16, final_out, m, n, scaling_factor
    )

    return final_out.to(dtype=input_dtype)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        return avelang_gemm_div2_sum_scale(x, self.weight, self.scaling_factor)


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
