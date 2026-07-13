import torch
import torch.nn as nn
import torch.nn.functional as F
import avelang
import avelang.language as al

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
NUM_GROUPS = 256
GROUP_SIZE = OUT_FEATURES // NUM_GROUPS
EPS = 1e-05

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
def _fetch_mfma_operand_half(
    ret: al.Tensor((4,), al.bf16),
    shm: al.Tensor((SHM_A_VECS, 4), al.u32),
    tile_idx: al.u32,
    half: al.u32,
    lane: al.u32,
):
    ret_u32 = al.view(ret, al.Tensor((2,), al.u32))
    shm_u32 = al.view(shm, al.Tensor((SHM_A_VECS * 4,), al.u32))
    row = tile_idx * MMA_M + (lane % MMA_M)
    k_group_u32 = (lane // MMA_M) * al.convert(2, al.u32)
    half_offset = half * al.convert(4, al.u32)
    row_base = row * ROW_U32
    ret_u32[0] = shm_u32[row_base + half_offset + k_group_u32]
    ret_u32[1] = shm_u32[row_base + half_offset + k_group_u32 + al.convert(1, al.u32)]


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

    # Double-buffered shared memory
    shm_a0 = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b0 = al.make_shared((SHM_B_VECS, 4), al.u32)
    shm_a1 = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b1 = al.make_shared((SHM_B_VECS, 4), al.u32)

    # Fine-grained operand registers: one half at a time
    a_half = al.make_local((4,), al.bf16)
    b_half = al.make_local((4,), al.bf16)

    # Accumulator
    acc = al.make_local((M_TILES_PER_WARP * N_TILES_PER_WARP, ACC_SIZE), al.f32)
    for i in al.range(M_TILES_PER_WARP * N_TILES_PER_WARP):
        for j in al.range(ACC_SIZE):
            acc[i, j] = al.convert(0.0, al.f32)

    k_tiles = k // GROUP_K
    ktwo = al.convert(2, al.u32)
    kzero = al.convert(0, al.u32)
    kone = al.convert(1, al.u32)
    half0 = al.convert(0, al.u32)
    half1 = al.convert(1, al.u32)

    # Prologue: load tile k=0 into buffer 0
    _load_global_a_to_shm(shm_a0, x_rsrc, block_m, kzero, k, tid)
    _load_global_b_to_shm(shm_b0, w_rsrc, block_n, kzero, k, tid)
    al.syncthreads()

    # Main loop: unrolled by 2, double buffering, fine-grained LDS/MFMA overlap
    for kt in al.range(kzero, k_tiles, ktwo):
        # Prefetch tile kt+1 into buffer 1 (overlapped with compute below)
        k_next = (kt + kone) * GROUP_K
        _load_global_a_to_shm(shm_a1, x_rsrc, block_m, k_next, k, tid)
        _load_global_b_to_shm(shm_b1, w_rsrc, block_n, k_next, k, tid)

        # Compute tile kt from buffer 0 (fine-grained: fetch + MFMA per half)
        for i in al.range(M_TILES_PER_WARP):
            tile_m = warp_row * M_TILES_PER_WARP + i
            for j in al.range(N_TILES_PER_WARP):
                tile_n = warp_col * N_TILES_PER_WARP + j
                acc_idx = i * N_TILES_PER_WARP + j

                _fetch_mfma_operand_half(a_half, shm_a0, tile_m, half0, lane)
                _fetch_mfma_operand_half(b_half, shm_b0, tile_n, half0, lane)
                a0 = al.view(a_half, al.Tensor((2,), al.u32))
                b0 = al.view(b_half, al.Tensor((2,), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a0, b0, acc[acc_idx])

                _fetch_mfma_operand_half(a_half, shm_a0, tile_m, half1, lane)
                _fetch_mfma_operand_half(b_half, shm_b0, tile_n, half1, lane)
                a1 = al.view(a_half, al.Tensor((2,), al.u32))
                b1 = al.view(b_half, al.Tensor((2,), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a1, b1, acc[acc_idx])

        al.syncthreads()

        # Prefetch tile kt+2 into buffer 0
        kt2 = kt + ktwo
        if kt2 < k_tiles:
            k_next2 = kt2 * GROUP_K
            _load_global_a_to_shm(shm_a0, x_rsrc, block_m, k_next2, k, tid)
            _load_global_b_to_shm(shm_b0, w_rsrc, block_n, k_next2, k, tid)

        # Compute tile kt+1 from buffer 1 (fine-grained)
        for i in al.range(M_TILES_PER_WARP):
            tile_m = warp_row * M_TILES_PER_WARP + i
            for j in al.range(N_TILES_PER_WARP):
                tile_n = warp_col * N_TILES_PER_WARP + j
                acc_idx = i * N_TILES_PER_WARP + j

                _fetch_mfma_operand_half(a_half, shm_a1, tile_m, half0, lane)
                _fetch_mfma_operand_half(b_half, shm_b1, tile_n, half0, lane)
                a0 = al.view(a_half, al.Tensor((2,), al.u32))
                b0 = al.view(b_half, al.Tensor((2,), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a0, b0, acc[acc_idx])

                _fetch_mfma_operand_half(a_half, shm_a1, tile_m, half1, lane)
                _fetch_mfma_operand_half(b_half, shm_b1, tile_n, half1, lane)
                a1 = al.view(a_half, al.Tensor((2,), al.u32))
                b1 = al.view(b_half, al.Tensor((2,), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a1, b1, acc[acc_idx])

        al.syncthreads()

    # Store results with bias
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


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_FEATURES)]


def get_init_inputs():
    return [IN_FEATURES, OUT_FEATURES, NUM_GROUPS, (OUT_FEATURES,)]


class ModelNew(nn.Module):

    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

    def forward(self, x):
        m_val = BATCH_SIZE
        n_val = OUT_FEATURES
        k_val = IN_FEATURES

        x_bf16 = x.contiguous()
        w_t = self.gemm.weight.contiguous()
        bias = self.gemm.bias.contiguous()

        gemm_out = torch.empty(
            (BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=torch.bfloat16
        )

        grid_gemm = (n_val // GROUP_N, m_val // GROUP_M, 1)
        gemm_kernel[lambda: (grid_gemm, (THREADS, 1, 1))](
            x_bf16, w_t, bias, gemm_out, m_val, n_val, k_val
        )

        x = self.group_norm(gemm_out)
        x = F.silu(x)
        x = x * self.multiply_weight
        x = F.silu(x)
        return x
