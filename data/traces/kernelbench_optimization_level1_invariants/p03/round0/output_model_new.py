import torch
import torch.nn as nn
import avelang
import avelang.language as al


# ── compile-time constants ──────────────────────────────────────────────
BATCH = 128
M = 512
K = 1024
N = 2048
BF16_BYTES = 2

WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS          # 256
GROUP_M = 128
GROUP_N = 128
GROUP_K = 32
MMA = 32
WARPS_M = 2
WARPS_N = 2
M_TILES_PER_WARP = GROUP_M // (WARPS_M * MMA)   # 2
N_TILES_PER_WARP = GROUP_N // (WARPS_N * MMA)   # 2
VEC_ELEMS = 8
A_VECS_PER_ROW = GROUP_K // VEC_ELEMS   # 4
B_VECS_PER_ROW = GROUP_K // VEC_ELEMS   # 4
SHM_A_VECS = GROUP_M * A_VECS_PER_ROW   # 512
SHM_B_VECS = GROUP_N * B_VECS_PER_ROW   # 512
ROW_U32 = A_VECS_PER_ROW * 4            # 16
ACC_SIZE = 16

M_TILES = M // GROUP_M    # 4
N_TILES = N // GROUP_N    # 16
K_TILES = K // GROUP_K    # 32

STRIDE_A = M * K           # 524288
STRIDE_B = N * K           # 2097152  (B is pre-transposed to N×K on host)


# ── main BMM kernel ─────────────────────────────────────────────────────
@avelang.jit
def bmm_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
):
    tid = al.thread_id(0)
    n_block = al.block_id(0)
    m_block = al.block_id(1)
    batch = al.block_id(2)

    lane = tid % WARP_SIZE
    wid = tid // WARP_SIZE
    warp_row = wid // WARPS_N      # 0 or 1
    warp_col = wid % WARPS_N       # 0 or 1

    # --- buffer resources ---
    a_1d = al.make_tensor(a_ptr, al.bf16, al.make_layout((BATCH * STRIDE_A,), (1,)))
    b_1d = al.make_tensor(b_ptr, al.bf16, al.make_layout((BATCH * STRIDE_B,), (1,)))
    a_rsrc = al.amdgpu.make_rsrc(a_1d, BATCH * STRIDE_A * BF16_BYTES)
    b_rsrc = al.amdgpu.make_rsrc(b_1d, BATCH * STRIDE_B * BF16_BYTES)
    c_out = al.make_tensor(c_ptr, al.bf16, al.make_layout((BATCH * M, N), (N, 1)))

    # --- shared memory ---
    shm_a = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_B_VECS, 4), al.u32)

    # --- registers (2 M tiles × 2 N tiles per warp = 4 accumulators) ---
    num_acc = M_TILES_PER_WARP * N_TILES_PER_WARP  # 4
    a_reg = al.make_local((M_TILES_PER_WARP, 2, 2), al.u32)
    b_reg = al.make_local((N_TILES_PER_WARP, 2, 2), al.u32)
    a_reg2 = al.make_local((M_TILES_PER_WARP, 2, 2), al.u32)
    b_reg2 = al.make_local((N_TILES_PER_WARP, 2, 2), al.u32)

    acc = al.make_local((num_acc, ACC_SIZE), al.f32)
    for i in al.range(num_acc):
        for j in al.range(ACC_SIZE):
            acc[i, j] = 0

    # --- K loop ---
    zero = al.convert(0, al.i32)
    for kt in al.range(K_TILES):
        k_base = kt * GROUP_K

        # load A (512 vectors, 256 threads → 2 loads per thread)
        idx = tid
        for _ in al.range(2):
            row = idx // A_VECS_PER_ROW
            col_vec = idx % A_VECS_PER_ROW
            off = (batch * STRIDE_A + (m_block * GROUP_M + row) * K + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
            shm_a[idx] = al.amdgpu.raw_buffer_load_x4(a_rsrc, zero, off, 0)
            idx = idx + THREADS

        # load B
        idx = tid
        for _ in al.range(2):
            row = idx // B_VECS_PER_ROW
            col_vec = idx % B_VECS_PER_ROW
            off = (batch * STRIDE_B + (n_block * GROUP_N + row) * K + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
            shm_b[idx] = al.amdgpu.raw_buffer_load_x4(b_rsrc, zero, off, 0)
            idx = idx + THREADS

        al.syncthreads()

        shm_a_u32 = al.view(shm_a, al.Tensor((SHM_A_VECS * 4,), al.u32))
        shm_b_u32 = al.view(shm_b, al.Tensor((SHM_B_VECS * 4,), al.u32))

        k_group = (lane // MMA) * 2
        base8 = al.convert(8, al.i32)

        # fetch A
        for i in al.range(M_TILES_PER_WARP):
            ti = warp_row * M_TILES_PER_WARP + i
            a_row = ti * MMA + (lane % MMA)
            a_rb = a_row * ROW_U32
            af = al.view(a_reg[i], al.Tensor((4,), al.u32))
            af[0] = shm_a_u32[a_rb + k_group]
            af[1] = shm_a_u32[a_rb + k_group + 1]
            af[2] = shm_a_u32[a_rb + 4 + k_group]
            af[3] = shm_a_u32[a_rb + 5 + k_group]
            af2 = al.view(a_reg2[i], al.Tensor((4,), al.u32))
            af2[0] = shm_a_u32[a_rb + base8 + k_group]
            af2[1] = shm_a_u32[a_rb + base8 + k_group + 1]
            af2[2] = shm_a_u32[a_rb + base8 + 4 + k_group]
            af2[3] = shm_a_u32[a_rb + base8 + 5 + k_group]

        # fetch B
        for j in al.range(N_TILES_PER_WARP):
            tj = warp_col * N_TILES_PER_WARP + j
            b_row = tj * MMA + (lane % MMA)
            b_rb = b_row * ROW_U32
            bf = al.view(b_reg[j], al.Tensor((4,), al.u32))
            bf[0] = shm_b_u32[b_rb + k_group]
            bf[1] = shm_b_u32[b_rb + k_group + 1]
            bf[2] = shm_b_u32[b_rb + 4 + k_group]
            bf[3] = shm_b_u32[b_rb + 5 + k_group]
            bf2 = al.view(b_reg2[j], al.Tensor((4,), al.u32))
            bf2[0] = shm_b_u32[b_rb + base8 + k_group]
            bf2[1] = shm_b_u32[b_rb + base8 + k_group + 1]
            bf2[2] = shm_b_u32[b_rb + base8 + 4 + k_group]
            bf2[3] = shm_b_u32[b_rb + base8 + 5 + k_group]

        # 4 MFMA instructions per (M,N) tile pair
        for i in al.range(M_TILES_PER_WARP):
            for j in al.range(N_TILES_PER_WARP):
                ai = i * N_TILES_PER_WARP + j
                acc[ai] = al.amdgpu.mfma_32x32x8_bf16_f32(a_reg[i, 0], b_reg[j, 0], acc[ai])
                acc[ai] = al.amdgpu.mfma_32x32x8_bf16_f32(a_reg[i, 1], b_reg[j, 1], acc[ai])
                acc[ai] = al.amdgpu.mfma_32x32x8_bf16_f32(a_reg2[i, 0], b_reg2[j, 0], acc[ai])
                acc[ai] = al.amdgpu.mfma_32x32x8_bf16_f32(a_reg2[i, 1], b_reg2[j, 1], acc[ai])

        al.syncthreads()

    # --- writeback ---
    lane_col = lane % 32
    lane_grp = lane // 32
    for j in al.range(N_TILES_PER_WARP):
        col = (n_block * GROUP_N) + (warp_col * N_TILES_PER_WARP + j) * MMA + lane_col
        for i in al.range(M_TILES_PER_WARP):
            ai = i * N_TILES_PER_WARP + j
            row_base = (m_block * GROUP_M) + (warp_row * M_TILES_PER_WARP + i) * MMA
            for t in al.range(ACC_SIZE):
                row = row_base + 8 * (t // 4) + 4 * lane_grp + (t % 4)
                out_row = batch * M + row
                c_out[out_row, col] = al.convert(acc[ai, t], al.bf16)


# ── host wrapper ────────────────────────────────────────────────────────
def avelang_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA/HIP device"

    if a.dtype != torch.bfloat16:
        a = a.to(torch.bfloat16)
    if b.dtype != torch.bfloat16:
        b = b.to(torch.bfloat16)
    a = a.contiguous()
    b_t = b.transpose(1, 2).contiguous()

    bsz, m_in, k_in = a.shape
    _, n_in, k_in_b = b_t.shape
    if bsz != BATCH or m_in != M or k_in != K or k_in_b != K or n_in != N:
        raise RuntimeError(
            f"Expected shapes ({BATCH},{M},{K}) and ({BATCH},{K},{N}); "
            f"got ({bsz},{m_in},{k_in}) and b_t ({bsz},{n_in},{k_in_b})"
        )

    c = torch.empty((BATCH, M, N), device=a.device, dtype=torch.bfloat16)
    grid = (N_TILES, M_TILES, BATCH)
    bmm_kernel[lambda: (grid, (THREADS, 1, 1))](a, b_t, c)
    return c


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return avelang_bmm(A, B)
