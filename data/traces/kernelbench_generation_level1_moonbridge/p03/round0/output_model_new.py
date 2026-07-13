import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Tile dimensions
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 16

# Thread organization
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS  # 256

# MFMA instruction properties
MMA_M = 32
MMA_N = 32

# Vectorized loads
VEC_ELEMS = 8
BF16_BYTES = 2

# Accumulator
ACC_SIZE = 16  # 32*32/64 lanes

# Warp grid
WARPS_M = 2
WARPS_N = 2

# Tiles per warp
M_TILES_PER_WARP = BLOCK_M // (WARPS_M * MMA_M)  # 2
N_TILES_PER_WARP = BLOCK_N // (WARPS_N * MMA_N)  # 2

# Shared memory layout
A_VECS_PER_ROW = BLOCK_K // VEC_ELEMS  # 2
B_VECS_PER_ROW = BLOCK_K // VEC_ELEMS  # 2
SHM_A_VECS = BLOCK_M * A_VECS_PER_ROW  # 256
SHM_B_VECS = BLOCK_N * B_VECS_PER_ROW  # 256
GLOBAL_LOADS_A = SHM_A_VECS // THREADS  # 1
GLOBAL_LOADS_B = SHM_B_VECS // THREADS  # 1
ROW_U32 = A_VECS_PER_ROW * 4  # 8


@avelang.jit
def _load_global_a_to_shm(
    shm_a: al.Tensor((SHM_A_VECS, 4), al.u32),
    a_rsrc: al.Tensor((4,), al.u32),
    batch: al.u32,
    block_m: al.u32,
    k_base: al.u32,
    m: al.u32,
    k: al.u32,
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    idx = tid
    for _ in al.range(GLOBAL_LOADS_A):
        row = idx // A_VECS_PER_ROW
        col_vec = idx % A_VECS_PER_ROW
        off = (batch * m * k + (block_m * BLOCK_M + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
        shm_a[idx] = al.amdgpu.raw_buffer_load_x4(a_rsrc, zero, off, 0)
        idx += THREADS


@avelang.jit
def _load_global_b_to_shm(
    shm_b: al.Tensor((SHM_B_VECS, 4), al.u32),
    b_rsrc: al.Tensor((4,), al.u32),
    batch: al.u32,
    block_n: al.u32,
    k_base: al.u32,
    n: al.u32,
    k: al.u32,
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    idx = tid
    for _ in al.range(GLOBAL_LOADS_B):
        row = idx // B_VECS_PER_ROW
        col_vec = idx % B_VECS_PER_ROW
        off = (batch * n * k + (block_n * BLOCK_N + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
        shm_b[idx] = al.amdgpu.raw_buffer_load_x4(b_rsrc, zero, off, 0)
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
def bmm_bf16_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
    batch_size: al.u32,
    m: al.u32,
    n: al.u32,
    k: al.u32,
):
    tid = al.thread_id(0)
    block_n = al.block_id(0)
    block_m = al.block_id(1)
    batch = al.block_id(2)
    wid = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    warp_row = wid // WARPS_N
    warp_col = wid % WARPS_N

    a_memref = al.make_tensor(a_ptr, al.bf16, al.make_layout((batch_size * m * k,), (1,)))
    b_memref = al.make_tensor(b_ptr, al.bf16, al.make_layout((batch_size * n * k,), (1,)))
    g_out = al.make_tensor(c_ptr, al.bf16, al.make_layout((batch_size * m, n), (n, 1)))

    a_rsrc = al.amdgpu.make_rsrc(a_memref, batch_size * m * k * BF16_BYTES)
    b_rsrc = al.amdgpu.make_rsrc(b_memref, batch_size * n * k * BF16_BYTES)

    shm_a = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_B_VECS, 4), al.u32)
    a_reg = al.make_local((M_TILES_PER_WARP, 2, 4), al.bf16)
    b_reg = al.make_local((N_TILES_PER_WARP, 2, 4), al.bf16)
    acc = al.make_local((M_TILES_PER_WARP * N_TILES_PER_WARP, ACC_SIZE), al.f32)

    for i in al.range(M_TILES_PER_WARP * N_TILES_PER_WARP):
        for j in al.range(ACC_SIZE):
            acc[i, j] = 0

    k_tiles = k // BLOCK_K
    for kt in al.range(k_tiles):
        k_base = kt * BLOCK_K
        _load_global_a_to_shm(shm_a, a_rsrc, batch, block_m, k_base, m, k, tid)
        _load_global_b_to_shm(shm_b, b_rsrc, batch, block_n, k_base, n, k, tid)
        al.syncthreads()

        for i in al.range(M_TILES_PER_WARP):
            _fetch_mfma_operand_32x32x16(a_reg[i], shm_a, warp_row * M_TILES_PER_WARP + i, lane)
        for j in al.range(N_TILES_PER_WARP):
            _fetch_mfma_operand_32x32x16(b_reg[j], shm_b, warp_col * N_TILES_PER_WARP + j, lane)

        for i in al.range(M_TILES_PER_WARP):
            for j in al.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                a_op = al.view(a_reg[i, 0], al.Tensor((2,), al.u32))
                b_op = al.view(b_reg[j, 0], al.Tensor((2,), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_op, b_op, acc[acc_idx])
                a_op2 = al.view(a_reg[i, 1], al.Tensor((2,), al.u32))
                b_op2 = al.view(b_reg[j, 1], al.Tensor((2,), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_op2, b_op2, acc[acc_idx])

        al.syncthreads()

    lane_group = lane // MMA_N
    lane_col = lane % MMA_N
    block_row_base = block_m * BLOCK_M
    block_col_base = block_n * BLOCK_N
    batch_row_off = batch * m

    for j in al.range(N_TILES_PER_WARP):
        col = block_col_base + (warp_col * N_TILES_PER_WARP + j) * MMA_N + lane_col
        for i in al.range(M_TILES_PER_WARP):
            acc_idx = i * N_TILES_PER_WARP + j
            row_base = block_row_base + (warp_row * M_TILES_PER_WARP + i) * MMA_M
            for t in al.range(ACC_SIZE):
                row = row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
                result = acc[acc_idx, t]
                g_out[batch_row_off + row, col] = al.convert(result, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_bmm(
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    A_bf16 = _prepare_bf16_cuda_contiguous(A)
    B_bf16 = _prepare_bf16_cuda_contiguous(B)

    batch_size, m, k = A_bf16.shape
    b_batch, b_k, b_n = B_bf16.shape

    if batch_size != b_batch or k != b_k:
        raise ValueError(f"Shape mismatch: A={A_bf16.shape}, B={B_bf16.shape}")

    if m % BLOCK_M != 0 or b_n % BLOCK_N != 0 or k % BLOCK_K != 0:
        raise ValueError(
            f"Expected m % {BLOCK_M} == 0, n % {BLOCK_N} == 0, k % {BLOCK_K} == 0 "
            f"(got m={m}, n={b_n}, k={k})"
        )

    B_t = B_bf16.transpose(1, 2).contiguous()

    out = torch.empty((batch_size, m, b_n), device=A_bf16.device, dtype=torch.bfloat16)
    grid = (b_n // BLOCK_N, m // BLOCK_M, batch_size)
    bmm_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](
        A_bf16, B_t, out, batch_size, m, b_n, k
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return avelang_bmm(A, B)
