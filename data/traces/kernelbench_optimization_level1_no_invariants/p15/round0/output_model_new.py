import torch
import torch.nn as nn
import avelang
import avelang.language as al


GROUP_M = 64
GROUP_N = 64
GROUP_K = 32
MMA_M = 32
MMA_N = 32
VEC_ELEMS = 8
BF16_BYTES = 2
ACC_SIZE = 16
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS
WARPS_M = 2
WARPS_N = 2
A_VECS_PER_ROW = GROUP_K // VEC_ELEMS
B_VECS_PER_ROW = GROUP_K // VEC_ELEMS
SHM_A_VECS = GROUP_M * A_VECS_PER_ROW
SHM_B_VECS = GROUP_N * B_VECS_PER_ROW
ROW_U32 = A_VECS_PER_ROW * 4
NUM_MFMA_STEPS = GROUP_K // 8


@avelang.jit
def _fetch_mfma_operand_4(
    ret: al.Tensor((4, 4), al.bf16),
    shm: al.Tensor((SHM_A_VECS, 4), al.u32),
    tile_idx: al.u32,
    lane: al.u32,
):
    ret_u32 = al.view(ret, al.Tensor((8,), al.u32))
    shm_u32 = al.view(shm, al.Tensor((SHM_A_VECS * 4,), al.u32))
    row = tile_idx * MMA_M + (lane % MMA_M)
    kg = (lane // MMA_M) * 2
    row_base = row * ROW_U32
    for s in al.range(4):
        ret_u32[s * 2] = shm_u32[row_base + s * 4 + kg]
        ret_u32[s * 2 + 1] = shm_u32[row_base + s * 4 + kg + 1]


@avelang.jit
def tri_gemm_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M: al.u32,
    N: al.u32,
    K: al.u32,
):
    a_flat = al.make_tensor(A_ptr, al.bf16, al.make_layout((M * K,), (1,)))
    b_flat = al.make_tensor(B_ptr, al.bf16, al.make_layout((N * K,), (1,)))
    out_2d = al.make_tensor(C_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    a_rsrc = al.amdgpu.make_rsrc(a_flat, M * K * BF16_BYTES)
    b_rsrc = al.amdgpu.make_rsrc(b_flat, N * K * BF16_BYTES)

    shm_a = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_B_VECS, 4), al.u32)

    a_reg = al.make_local((4, 4), al.bf16)
    b_reg = al.make_local((4, 4), al.bf16)
    acc = al.make_local((ACC_SIZE,), al.f32)
    for i in al.range(ACC_SIZE):
        acc[i] = al.convert(0.0, al.f32)

    tid = al.thread_id(0)
    block_m = al.block_id(1)
    block_n = al.block_id(0)
    wid = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    warp_row = wid // WARPS_N
    warp_col = wid % WARPS_N

    zero_u32 = al.convert(0, al.u32)

    k_tiles = K // GROUP_K
    for kt in al.range(k_tiles):
        k_base = kt * GROUP_K

        # Global-to-LDS: A
        for idx in al.range(tid, SHM_A_VECS, THREADS):
            row = idx // A_VECS_PER_ROW
            col_vec = idx % A_VECS_PER_ROW
            off = ((block_m * GROUP_M + row) * K + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
            shm_a[idx] = al.amdgpu.raw_buffer_load_x4(a_rsrc, zero_u32, off, 0)

        # Global-to-LDS: B (transposed)
        for idx in al.range(tid, SHM_B_VECS, THREADS):
            row = idx // B_VECS_PER_ROW
            col_vec = idx % B_VECS_PER_ROW
            off = ((block_n * GROUP_N + row) * K + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
            shm_b[idx] = al.amdgpu.raw_buffer_load_x4(b_rsrc, zero_u32, off, 0)

        al.syncthreads()

        # Fetch MFMA operands (4 steps)
        _fetch_mfma_operand_4(a_reg, shm_a, warp_row, lane)
        _fetch_mfma_operand_4(b_reg, shm_b, warp_col, lane)

        # MFMA: 4 steps
        for s in al.range(NUM_MFMA_STEPS):
            a_op = al.view(a_reg[s], al.Tensor((2,), al.u32))
            b_op = al.view(b_reg[s], al.Tensor((2,), al.u32))
            acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_op, b_op, acc)


    # Writeback with tril
    lane_group = lane // MMA_N
    lane_col = lane % MMA_N
    block_row_base = block_m * GROUP_M
    block_col_base = block_n * GROUP_N

    col = block_col_base + warp_col * MMA_N + lane_col
    row_base = block_row_base + warp_row * MMA_M
    for t in al.range(ACC_SIZE):
        r = row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
        if col <= r:
            out_2d[r, col] = al.convert(acc[t], al.bf16)
        else:
            out_2d[r, col] = al.convert(0.0, al.bf16)


def _round_up(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        A = A.contiguous()
        B = B.contiguous()
        m_orig = A.shape[0]
        k_orig = A.shape[1]
        n_orig = B.shape[1]

        m_pad = _round_up(m_orig, GROUP_M)
        n_pad = _round_up(n_orig, GROUP_N)
        k_pad = _round_up(k_orig, GROUP_K)

        A_pad = torch.zeros((m_pad, k_pad), dtype=A.dtype, device=A.device)
        A_pad[:m_orig, :k_orig] = A

        B_t = B.T.contiguous()
        B_pad = torch.zeros((n_pad, k_pad), dtype=B.dtype, device=B.device)
        B_pad[:n_orig, :k_orig] = B_t

        C_pad = torch.empty((m_pad, n_pad), dtype=A.dtype, device=A.device)

        grid = (n_pad // GROUP_N, m_pad // GROUP_M, 1)
        tri_gemm_kernel[lambda: (grid, (THREADS, 1, 1))](
            A_pad, B_pad, C_pad,
            m_pad, n_pad, k_pad,
        )
        return C_pad[:m_orig, :n_orig]
