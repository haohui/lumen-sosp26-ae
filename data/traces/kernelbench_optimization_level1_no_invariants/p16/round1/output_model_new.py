import torch
import torch.nn as nn

import avelang
import avelang.language as al


@avelang.jit
def _mfma16(a: al.Tensor((2,), al.u32), b: al.Tensor((2,), al.u32), c: al.Tensor((4,), al.f32)) -> al.Tensor((4,), al.f32):
    return al.amdgpu.mfma_16x16x16_bf16_f32(a, b, c)


@avelang.jit
def _load_g2r_a_k16(
    rsrc: al.Tensor((4,), al.u32),
    m_base: al.u32,
    kk: al.u32,
    tid: al.u32,
    M: al.u32,
    reg: al.Tensor((8,), al.bf16),
):
    warp_tid = tid % 64
    k_off = warp_tid % 16
    m_group = warp_tid // 16
    m_off = m_group * 8
    gk = kk + k_off
    gm = m_base + m_off
    data = al.amdgpu.raw_buffer_load_x4(rsrc, (gk * M + gm) * 2, 0, 0)
    words = al.view(data, al.Tensor((8,), al.bf16))
    reg[0] = words[0]
    reg[1] = words[1]
    reg[2] = words[2]
    reg[3] = words[3]
    reg[4] = words[4]
    reg[5] = words[5]
    reg[6] = words[6]
    reg[7] = words[7]


@avelang.jit
def _store_r2s_a(
    shm: al.Tensor((64, 16), al.bf16),
    reg: al.Tensor((8,), al.bf16),
    warp_dim: al.u32,
    tid: al.u32,
):
    warp_tid = tid % 64
    k_off = warp_tid % 16
    m_group = warp_tid // 16
    m_off = m_group * 8
    row0 = warp_dim * 32 + m_off
    shm[row0 + 0, k_off] = reg[0]
    shm[row0 + 1, k_off] = reg[1]
    shm[row0 + 2, k_off] = reg[2]
    shm[row0 + 3, k_off] = reg[3]
    shm[row0 + 4, k_off] = reg[4]
    shm[row0 + 5, k_off] = reg[5]
    shm[row0 + 6, k_off] = reg[6]
    shm[row0 + 7, k_off] = reg[7]


@avelang.jit
def _load_s2r_a_tile(
    shm: al.Tensor((64, 16), al.bf16),
    row_base: al.u32,
    lane: al.u32,
    data: al.Tensor((2,), al.u32),
):
    r = row_base + lane % 16
    c = (lane // 16) * 4
    bf = al.make_local((4,), al.bf16)
    bf[0] = shm[r, c + 0]
    bf[1] = shm[r, c + 1]
    bf[2] = shm[r, c + 2]
    bf[3] = shm[r, c + 3]
    u32v = al.view(bf, al.Tensor((2,), al.u32))
    data[0] = u32v[0]
    data[1] = u32v[1]


@avelang.jit
def gemm_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M: al.u32,
    K: al.u32,
    N: al.u32,
):
    tid = al.thread_id(0)
    lane_id = tid % 64
    warp_id = tid // 64
    warp_row = warp_id // 2
    warp_col = warp_id % 2

    block_m = al.block_id(0)
    block_n = al.block_id(1)

    m_base = block_m * 64 + warp_row * 32
    n_base = block_n * 64 + warp_col * 32

    layout_A = al.make_layout((K, M), (M, 1))
    A = al.make_tensor(A_ptr, al.bf16, layout_A)
    layout_B = al.make_layout((K, N), (N, 1))
    B = al.make_tensor(B_ptr, al.bf16, layout_B)
    layout_C = al.make_layout((M, N), (N, 1))
    C = al.make_tensor(C_ptr, al.bf16, layout_C)

    rsrc_A = al.amdgpu.make_rsrc(A, K * M * 2)
    rsrc_B = al.amdgpu.make_rsrc(B, K * N * 2)

    # Double-buffered shared memory
    A_shm0 = al.make_shared((64, 16), al.bf16)
    A_shm1 = al.make_shared((64, 16), al.bf16)
    B_shm0 = al.make_shared((64, 16), al.bf16)
    B_shm1 = al.make_shared((64, 16), al.bf16)

    reg_a = al.make_local((8,), al.bf16)
    reg_b = al.make_local((8,), al.bf16)

    da00 = al.make_local((2,), al.u32)
    da01 = al.make_local((2,), al.u32)
    da10 = al.make_local((2,), al.u32)
    da11 = al.make_local((2,), al.u32)
    db00 = al.make_local((2,), al.u32)
    db01 = al.make_local((2,), al.u32)
    db10 = al.make_local((2,), al.u32)
    db11 = al.make_local((2,), al.u32)

    acc00 = al.make_local((4,), al.f32)
    acc01 = al.make_local((4,), al.f32)
    acc10 = al.make_local((4,), al.f32)
    acc11 = al.make_local((4,), al.f32)
    for i in al.range(4):
        acc00[i] = al.convert(0.0, al.f32)
        acc01[i] = al.convert(0.0, al.f32)
        acc10[i] = al.convert(0.0, al.f32)
        acc11[i] = al.convert(0.0, al.f32)

    a_row_off = warp_row * 32
    b_row_off = warp_col * 32

    zero = al.convert(0, al.u32)
    one = al.convert(1, al.u32)
    sixteen = al.convert(16, al.u32)

    # Preload tile 0 into buffer 0
    _load_g2r_a_k16(rsrc_A, m_base, zero, tid, M, reg_a)
    _load_g2r_a_k16(rsrc_B, n_base, zero, tid, N, reg_b)
    _store_r2s_a(A_shm0, reg_a, warp_row, tid)
    _store_r2s_a(B_shm0, reg_b, warp_col, tid)
    al.syncthreads()

    # toggle=1: next tile goes to buffer 1, compute from buffer 0
    toggle = one

    for kk in al.range(sixteen, K, sixteen):
        _load_g2r_a_k16(rsrc_A, m_base, kk, tid, M, reg_a)
        _load_g2r_a_k16(rsrc_B, n_base, kk, tid, N, reg_b)

        if toggle == one:
            # Compute from buffer 0, store to buffer 1
            _load_s2r_a_tile(A_shm0, a_row_off, lane_id, da00)
            _load_s2r_a_tile(B_shm0, b_row_off, lane_id, db00)
            acc00 = _mfma16(da00, db00, acc00)

            _load_s2r_a_tile(A_shm0, a_row_off, lane_id, da01)
            _load_s2r_a_tile(B_shm0, b_row_off + 16, lane_id, db01)
            acc01 = _mfma16(da01, db01, acc01)

            _load_s2r_a_tile(A_shm0, a_row_off + 16, lane_id, da10)
            _load_s2r_a_tile(B_shm0, b_row_off, lane_id, db10)
            acc10 = _mfma16(da10, db10, acc10)

            _load_s2r_a_tile(A_shm0, a_row_off + 16, lane_id, da11)
            _load_s2r_a_tile(B_shm0, b_row_off + 16, lane_id, db11)
            acc11 = _mfma16(da11, db11, acc11)

            _store_r2s_a(A_shm1, reg_a, warp_row, tid)
            _store_r2s_a(B_shm1, reg_b, warp_col, tid)
            toggle = zero
        else:
            # Compute from buffer 1, store to buffer 0
            _load_s2r_a_tile(A_shm1, a_row_off, lane_id, da00)
            _load_s2r_a_tile(B_shm1, b_row_off, lane_id, db00)
            acc00 = _mfma16(da00, db00, acc00)

            _load_s2r_a_tile(A_shm1, a_row_off, lane_id, da01)
            _load_s2r_a_tile(B_shm1, b_row_off + 16, lane_id, db01)
            acc01 = _mfma16(da01, db01, acc01)

            _load_s2r_a_tile(A_shm1, a_row_off + 16, lane_id, da10)
            _load_s2r_a_tile(B_shm1, b_row_off, lane_id, db10)
            acc10 = _mfma16(da10, db10, acc10)

            _load_s2r_a_tile(A_shm1, a_row_off + 16, lane_id, da11)
            _load_s2r_a_tile(B_shm1, b_row_off + 16, lane_id, db11)
            acc11 = _mfma16(da11, db11, acc11)

            _store_r2s_a(A_shm0, reg_a, warp_row, tid)
            _store_r2s_a(B_shm0, reg_b, warp_col, tid)
            toggle = one

        al.syncthreads()

    # Epilogue: compute the last tile from the buffer that has it
    if toggle == zero:
        _load_s2r_a_tile(A_shm1, a_row_off, lane_id, da00)
        _load_s2r_a_tile(B_shm1, b_row_off, lane_id, db00)
        acc00 = _mfma16(da00, db00, acc00)

        _load_s2r_a_tile(A_shm1, a_row_off, lane_id, da01)
        _load_s2r_a_tile(B_shm1, b_row_off + 16, lane_id, db01)
        acc01 = _mfma16(da01, db01, acc01)

        _load_s2r_a_tile(A_shm1, a_row_off + 16, lane_id, da10)
        _load_s2r_a_tile(B_shm1, b_row_off, lane_id, db10)
        acc10 = _mfma16(da10, db10, acc10)

        _load_s2r_a_tile(A_shm1, a_row_off + 16, lane_id, da11)
        _load_s2r_a_tile(B_shm1, b_row_off + 16, lane_id, db11)
        acc11 = _mfma16(da11, db11, acc11)
    else:
        _load_s2r_a_tile(A_shm0, a_row_off, lane_id, da00)
        _load_s2r_a_tile(B_shm0, b_row_off, lane_id, db00)
        acc00 = _mfma16(da00, db00, acc00)

        _load_s2r_a_tile(A_shm0, a_row_off, lane_id, da01)
        _load_s2r_a_tile(B_shm0, b_row_off + 16, lane_id, db01)
        acc01 = _mfma16(da01, db01, acc01)

        _load_s2r_a_tile(A_shm0, a_row_off + 16, lane_id, da10)
        _load_s2r_a_tile(B_shm0, b_row_off, lane_id, db10)
        acc10 = _mfma16(da10, db10, acc10)

        _load_s2r_a_tile(A_shm0, a_row_off + 16, lane_id, da11)
        _load_s2r_a_tile(B_shm0, b_row_off + 16, lane_id, db11)
        acc11 = _mfma16(da11, db11, acc11)

    # Writeback
    tr = lane_id // 16
    tc = lane_id % 16
    C[m_base + tr * 4 + 0, n_base + tc] = al.convert(acc00[0], al.bf16)
    C[m_base + tr * 4 + 1, n_base + tc] = al.convert(acc00[1], al.bf16)
    C[m_base + tr * 4 + 2, n_base + tc] = al.convert(acc00[2], al.bf16)
    C[m_base + tr * 4 + 3, n_base + tc] = al.convert(acc00[3], al.bf16)
    C[m_base + tr * 4 + 0, n_base + 16 + tc] = al.convert(acc01[0], al.bf16)
    C[m_base + tr * 4 + 1, n_base + 16 + tc] = al.convert(acc01[1], al.bf16)
    C[m_base + tr * 4 + 2, n_base + 16 + tc] = al.convert(acc01[2], al.bf16)
    C[m_base + tr * 4 + 3, n_base + 16 + tc] = al.convert(acc01[3], al.bf16)
    C[m_base + 16 + tr * 4 + 0, n_base + tc] = al.convert(acc10[0], al.bf16)
    C[m_base + 16 + tr * 4 + 1, n_base + tc] = al.convert(acc10[1], al.bf16)
    C[m_base + 16 + tr * 4 + 2, n_base + tc] = al.convert(acc10[2], al.bf16)
    C[m_base + 16 + tr * 4 + 3, n_base + tc] = al.convert(acc10[3], al.bf16)
    C[m_base + 16 + tr * 4 + 0, n_base + 16 + tc] = al.convert(acc11[0], al.bf16)
    C[m_base + 16 + tr * 4 + 1, n_base + 16 + tc] = al.convert(acc11[1], al.bf16)
    C[m_base + 16 + tr * 4 + 2, n_base + 16 + tc] = al.convert(acc11[2], al.bf16)
    C[m_base + 16 + tr * 4 + 3, n_base + 16 + tc] = al.convert(acc11[3], al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        M_val = A.shape[1]
        K_val = A.shape[0]
        N_val = B.shape[1]

        if K_val != B.shape[0]:
            raise RuntimeError("Inner dimension mismatch")

        A_contig = A.contiguous()
        B_contig = B.contiguous()
        C = torch.empty((M_val, N_val), device=A.device, dtype=A.dtype)

        grid_m = (M_val + 63) // 64
        grid_n = (N_val + 63) // 64

        gemm_kernel[lambda: ((grid_m, grid_n, 1), (256, 1, 1))](
            A_contig, B_contig, C, M_val, K_val, N_val,
        )
        return C
