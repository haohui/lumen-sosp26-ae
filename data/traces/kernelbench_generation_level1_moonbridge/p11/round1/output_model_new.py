import torch
import torch.nn as nn
import avelang
import avelang.language as al

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
    a_rsrc: al.Tensor((4,), al.u32),
    block_m: al.u32,
    k_base: al.u32,
    K: al.u32,
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    idx = tid
    for _ in al.range(GLOBAL_LOADS_A):
        row = idx // A_VECS_PER_ROW
        col_vec = idx % A_VECS_PER_ROW
        off = ((block_m * GROUP_M + row) * K + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
        shm_a[idx] = al.amdgpu.raw_buffer_load_x4(a_rsrc, zero, off, 0)
        idx += THREADS


@avelang.jit
def _load_global_b_to_shm(
    shm_b: al.Tensor((SHM_B_VECS, 4), al.u32),
    b_rsrc: al.Tensor((4,), al.u32),
    block_n: al.u32,
    k_base: al.u32,
    K: al.u32,
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    idx = tid
    for _ in al.range(GLOBAL_LOADS_B):
        row = idx // B_VECS_PER_ROW
        col_vec = idx % B_VECS_PER_ROW
        off = ((block_n * GROUP_N + row) * K + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
        shm_b[idx] = al.amdgpu.raw_buffer_load_x4(b_rsrc, zero, off, 0)
        idx += THREADS


@avelang.jit
def _fetch_operand_u32(
    ret: al.Tensor((4,), al.u32),
    shm: al.Tensor((SHM_A_VECS, 4), al.u32),
    tile_idx: al.u32,
    lane: al.u32,
):
    shm_u32 = al.view(shm, al.Tensor((SHM_A_VECS * 4,), al.u32))
    row = tile_idx * MMA_M + (lane % MMA_M)
    k_group_u32 = (lane // MMA_M) * 2
    row_base = row * ROW_U32

    ret[0] = shm_u32[row_base + k_group_u32]
    ret[1] = shm_u32[row_base + k_group_u32 + 1]
    ret[2] = shm_u32[row_base + 4 + k_group_u32]
    ret[3] = shm_u32[row_base + 5 + k_group_u32]


@avelang.jit
def tensor_matmul_bf16_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
    M: al.u32,
    N: al.u32,
    K: al.u32,
):
    tid = al.thread_id(0)
    block_n = al.block_id(0)
    block_m = al.block_id(1)
    wid = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    warp_row = wid // WARPS_N
    warp_col = wid % WARPS_N

    a_flat = al.make_tensor(a_ptr, al.bf16, al.make_layout((M * K,), (1,)))
    b_flat = al.make_tensor(b_ptr, al.bf16, al.make_layout((N * K,), (1,)))
    c_out = al.make_tensor(c_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    a_rsrc = al.amdgpu.make_rsrc(a_flat, M * K * BF16_BYTES)
    b_rsrc = al.amdgpu.make_rsrc(b_flat, N * K * BF16_BYTES)

    shm_a = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_B_VECS, 4), al.u32)
    a_data = al.make_local((M_TILES_PER_WARP, 4), al.u32)
    b_data = al.make_local((N_TILES_PER_WARP, 4), al.u32)
    acc = al.make_local((M_TILES_PER_WARP, N_TILES_PER_WARP, ACC_SIZE), al.f32)

    for i in al.range(M_TILES_PER_WARP):
        for j in al.range(N_TILES_PER_WARP):
            for k in al.range(ACC_SIZE):
                acc[i, j, k] = al.convert(0.0, al.f32)

    k_tiles = K // GROUP_K
    for kt in al.range(k_tiles):
        k_base = kt * GROUP_K

        _load_global_a_to_shm(shm_a, a_rsrc, block_m, k_base, K, tid)
        _load_global_b_to_shm(shm_b, b_rsrc, block_n, k_base, K, tid)
        al.syncthreads()

        for i in al.range(M_TILES_PER_WARP):
            _fetch_operand_u32(
                a_data[i], shm_a, warp_row * M_TILES_PER_WARP + i, lane
            )
        for j in al.range(N_TILES_PER_WARP):
            _fetch_operand_u32(
                b_data[j], shm_b, warp_col * N_TILES_PER_WARP + j, lane
            )

        for i in al.range(M_TILES_PER_WARP):
            frag_a = al.view(a_data[i], al.Tensor((2, 2, 1), al.u32))
            for j in al.range(N_TILES_PER_WARP):
                frag_b = al.view(b_data[j], al.Tensor((2, 2, 1), al.u32))
                acc[i, j] = al.amdgpu.mfma_32x32x8_bf16_f32(
                    frag_a[0], frag_b[0], acc[i, j]
                )
                acc[i, j] = al.amdgpu.mfma_32x32x8_bf16_f32(
                    frag_a[1], frag_b[1], acc[i, j]
                )

        al.syncthreads()

    lane_group = lane // MMA_N
    lane_col = lane % MMA_N
    block_row_base = block_m * GROUP_M
    block_col_base = block_n * GROUP_N

    for j in al.range(N_TILES_PER_WARP):
        col = block_col_base + (warp_col * N_TILES_PER_WARP + j) * MMA_N + lane_col
        for i in al.range(M_TILES_PER_WARP):
            row_base = block_row_base + (warp_row * M_TILES_PER_WARP + i) * MMA_M
            for t in al.range(ACC_SIZE):
                row = row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
                c_out[row, col] = al.convert(acc[i, j, t], al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def _avelang_tensor_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    A_bf16 = _prepare_bf16_cuda_contiguous(A)
    B_bf16 = _prepare_bf16_cuda_contiguous(B)

    b_dim, i_dim, j_dim, l_dim = A_bf16.shape
    l_dim2, k_dim = B_bf16.shape

    M_val = b_dim * i_dim * j_dim
    K_val = l_dim
    N_val = k_dim

    A_2d = A_bf16.reshape(M_val, K_val).contiguous()
    B_t = B_bf16.T.contiguous()

    out = torch.empty((M_val, N_val), device=A_bf16.device, dtype=torch.bfloat16)
    grid = (N_val // GROUP_N, M_val // GROUP_M, 1)
    tensor_matmul_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](
        A_2d, B_t, out, M_val, N_val, K_val
    )
    return out.reshape(b_dim, i_dim, j_dim, N_val)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return _avelang_tensor_matmul(A, B)
