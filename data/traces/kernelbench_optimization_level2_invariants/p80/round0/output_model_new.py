import torch
import torch.nn as nn
import avelang
import avelang.language as al

SQRT_2 = 1.4142135623730951

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
MAX_DIM = 1

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
M_TILES_PER_WARP = GROUP_M // (NUM_WARPS // 2 * MMA_M)
N_TILES_PER_WARP = GROUP_N // (NUM_WARPS // 2 * MMA_N)
A_VECS_PER_ROW = GROUP_K // VEC_ELEMS
B_VECS_PER_ROW = GROUP_K // VEC_ELEMS
SHM_A_VECS = GROUP_M * A_VECS_PER_ROW
SHM_B_VECS = GROUP_N * B_VECS_PER_ROW
GLOBAL_LOADS_A = SHM_A_VECS // THREADS
GLOBAL_LOADS_B = SHM_B_VECS // THREADS

WARPS_M = 2
WARPS_N = 2
WARP_M_TILES = M_TILES_PER_WARP
WARP_N_TILES = N_TILES_PER_WARP
WARP_M_ROWS = WARP_M_TILES * MMA_M


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
def fused_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    y_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    k: al.u32,
):
    tid = al.thread_id(0)
    block_m = al.block_id(0)
    wid = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    warp_row = wid // WARPS_N
    warp_col = wid % WARPS_N

    x_memref = al.make_tensor(x_ptr, al.bf16, al.make_layout((m * k,), (1,)))
    w_memref = al.make_tensor(w_ptr, al.bf16, al.make_layout((n * k,), (1,)))
    g_bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((n,), (1,)))
    g_out = al.make_tensor(y_ptr, al.bf16, al.make_layout((m, 1), (1, 1)))

    x_rsrc = al.amdgpu.make_rsrc(x_memref, m * k * BF16_BYTES)
    w_rsrc = al.amdgpu.make_rsrc(w_memref, n * k * BF16_BYTES)

    shm_a = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_B_VECS, 4), al.u32)
    acc = al.make_local((WARP_M_TILES * WARP_N_TILES, ACC_SIZE), al.f32)

    neg_inf = al.convert(-3.402823466e+38, al.f32)
    row_max_shm = al.make_shared((GROUP_M,), al.f32)
    tile_max = al.make_shared((GROUP_M, WARPS_N), al.f32)

    if tid < GROUP_M:
        row_max_shm[tid] = neg_inf
    al.syncthreads()

    k_tiles = k // GROUP_K
    n_tiles = n // GROUP_N

    for nt in al.range(n_tiles):
        block_n = nt

        for ti in al.range(WARP_M_TILES * WARP_N_TILES):
            for ai in al.range(ACC_SIZE):
                acc[ti, ai] = al.convert(0.0, al.f32)

        for kt in al.range(k_tiles):
            k_base = kt * GROUP_K
            _load_global_a_to_shm(shm_a, x_rsrc, block_m, k_base, k, tid)
            _load_global_b_to_shm(shm_b, w_rsrc, block_n, k_base, k, tid)
            al.syncthreads()

            k_half = lane // MMA_N

            for i in al.range(WARP_M_TILES):
                lds_row_a = (warp_row * WARP_M_TILES + i) * MMA_M * A_VECS_PER_ROW + (lane % MMA_M) * A_VECS_PER_ROW + k_half
                a_words = shm_a[lds_row_a]
                a_frag = al.view(a_words, al.Tensor((2, 2, 1), al.u32))

                for j in al.range(WARP_N_TILES):
                    lds_row_b = (warp_col * WARP_N_TILES + j) * MMA_N * B_VECS_PER_ROW + (lane % MMA_N) * B_VECS_PER_ROW + k_half
                    b_words = shm_b[lds_row_b]
                    b_frag = al.view(b_words, al.Tensor((2, 2, 1), al.u32))

                    tile_idx = i * WARP_N_TILES + j
                    acc[tile_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(
                        b_frag[0], a_frag[0], acc[tile_idx]
                    )
                    acc[tile_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(
                        b_frag[1], a_frag[1], acc[tile_idx]
                    )

            al.syncthreads()

        for i in al.range(WARP_M_TILES):
            for j in al.range(WARP_N_TILES):
                tile_idx = i * WARP_N_TILES + j
                col = block_n * GROUP_N + warp_col * WARP_N_TILES * MMA_N + j * MMA_N + lane % MMA_N
                bval = al.convert(g_bias[col], al.f32)
                for t in al.range(ACC_SIZE):
                    acc[tile_idx, t] = acc[tile_idx, t] + bval

        al.syncthreads()

        if tid < GROUP_M:
            for ci in al.range(WARPS_N):
                tile_max[tid, ci] = neg_inf
        al.syncthreads()

        subgroup = lane // MMA_N

        for i in al.range(WARP_M_TILES):
            for j in al.range(WARP_N_TILES):
                tile_idx = i * WARP_N_TILES + j
                for t in al.range(ACC_SIZE):
                    row_in_tile = (t // 4) * 8 + subgroup * 4 + (t % 4)
                    val = acc[tile_idx, t]

                    other = al.shuffle_xor(val, al.convert(16, al.u32), 32)
                    if other > val:
                        val = other
                    other = al.shuffle_xor(val, al.convert(8, al.u32), 32)
                    if other > val:
                        val = other
                    other = al.shuffle_xor(val, al.convert(4, al.u32), 32)
                    if other > val:
                        val = other
                    other = al.shuffle_xor(val, al.convert(2, al.u32), 32)
                    if other > val:
                        val = other
                    other = al.shuffle_xor(val, al.convert(1, al.u32), 32)
                    if other > val:
                        val = other

                    if lane % 32 == 0:
                        warp_local_row = i * MMA_M + row_in_tile
                        block_row = warp_row * WARP_M_ROWS + warp_local_row
                        prev = tile_max[block_row, warp_col]
                        if val > prev:
                            tile_max[block_row, warp_col] = val

        al.syncthreads()

        if warp_col == 0:
            for ri in al.range(WARP_M_ROWS):
                global_row = warp_row * WARP_M_ROWS + ri
                v0 = tile_max[global_row, 0]
                v1 = tile_max[global_row, 1]
                combined = v0
                if v1 > combined:
                    combined = v1
                prev = row_max_shm[global_row]
                if combined > prev:
                    row_max_shm[global_row] = combined
        al.syncthreads()

    block_row_base = block_m * GROUP_M
    zero = al.convert(0.0, al.f32)
    half = al.convert(0.5, al.f32)
    one = al.convert(1.0, al.f32)
    sqrt2 = al.convert(SQRT_2, al.f32)
    if tid < GROUP_M:
        row_idx = block_row_base + al.convert(tid, al.u32)
        max_v = row_max_shm[tid]
        diff = max_v - max_v
        erf_arg = diff / sqrt2
        gelu_v = half * diff * (one + al.erf(erf_arg))
        g_out[row_idx, 0] = al.convert(gelu_v + zero, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES) or x.dtype != torch.bfloat16 or self.max_dim != MAX_DIM:
            raise RuntimeError('This fused kernel only supports the benchmark input shape and dtype.')

        x_bf16 = _prepare_bf16_cuda_contiguous(x)
        w = _prepare_bf16_cuda_contiguous(self.gemm.weight)
        bias = _prepare_bf16_cuda_contiguous(self.gemm.bias)

        m_val = BATCH_SIZE
        n_val = OUT_FEATURES
        k_val = IN_FEATURES

        if m_val % GROUP_M != 0 or n_val % GROUP_N != 0 or k_val % GROUP_K != 0:
            raise ValueError(
                f"Expected m % {GROUP_M} == 0, n % {GROUP_N} == 0, k % {GROUP_K} == 0 "
                f"(got m={m_val}, n={n_val}, k={k_val})"
            )

        y = torch.empty((m_val, 1), device=x.device, dtype=torch.bfloat16)
        grid = (m_val // GROUP_M, 1, 1)
        fused_kernel[lambda: (grid, (THREADS, 1, 1))](
            x_bf16, w, bias, y, m_val, n_val, k_val
        )
        return y
