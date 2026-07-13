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
    k_off = (tid // 8) % 16
    m_group = tid % 8
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
    m_base: al.u32,
    tid: al.u32,
):
    k_off = (tid // 8) % 16
    m_group = tid % 8
    m_off = m_group * 8
    shm[m_off + 0, k_off] = reg[0]
    shm[m_off + 1, k_off] = reg[1]
    shm[m_off + 2, k_off] = reg[2]
    shm[m_off + 3, k_off] = reg[3]
    shm[m_off + 4, k_off] = reg[4]
    shm[m_off + 5, k_off] = reg[5]
    shm[m_off + 6, k_off] = reg[6]
    shm[m_off + 7, k_off] = reg[7]


@avelang.jit
def _load_s2r_a_tile(
    shm: al.Tensor((64, 16), al.bf16),
    row_base: al.u32,
    lane: al.u32,
    data: al.Tensor((4,), al.u32),
):
    r = row_base + lane // 4
    c = (lane % 4) * 4
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

    A_shared = al.make_shared((64, 16), al.bf16)
    B_shared = al.make_shared((64, 16), al.bf16)

    reg_a = al.make_local((8,), al.bf16)
    reg_b = al.make_local((8,), al.bf16)

    da0 = al.make_local((4,), al.u32)
    da1 = al.make_local((4,), al.u32)
    db0 = al.make_local((4,), al.u32)
    db1 = al.make_local((4,), al.u32)

    acc0 = al.make_local((4,), al.f32)
    acc1 = al.make_local((4,), al.f32)
    acc2 = al.make_local((4,), al.f32)
    acc3 = al.make_local((4,), al.f32)
    for i in al.range(4):
        acc0[i] = al.convert(0.0, al.f32)
        acc1[i] = al.convert(0.0, al.f32)
        acc2[i] = al.convert(0.0, al.f32)
        acc3[i] = al.convert(0.0, al.f32)

    a_row_off = warp_row * 32
    b_row_off = warp_col * 32

    for kk in al.range(0, K, 16):
        _load_g2r_a_k16(rsrc_A, m_base, kk, tid, M, reg_a)
        _load_g2r_a_k16(rsrc_B, n_base, kk, tid, N, reg_b)
        _store_r2s_a(A_shared, reg_a, m_base, tid)
        _store_r2s_a(B_shared, reg_b, n_base, tid)
        al.syncthreads()

        # Tile (0,0)
        _load_s2r_a_tile(A_shared, a_row_off, lane_id, da0)
        _load_s2r_a_tile(B_shared, b_row_off, lane_id, db0)
        frag_a = al.view(da0, al.Tensor((2, 2), al.u32))
        frag_b = al.view(db0, al.Tensor((2, 2), al.u32))
        acc0 = _mfma16(frag_a[0], frag_b[0], acc0)
        acc0 = _mfma16(frag_a[1], frag_b[1], acc0)

        # Tile (0,1)
        _load_s2r_a_tile(A_shared, a_row_off, lane_id, da0)
        _load_s2r_a_tile(B_shared, b_row_off + 16, lane_id, db0)
        frag_a2 = al.view(da0, al.Tensor((2, 2), al.u32))
        frag_b2 = al.view(db0, al.Tensor((2, 2), al.u32))
        acc1 = _mfma16(frag_a2[0], frag_b2[0], acc1)
        acc1 = _mfma16(frag_a2[1], frag_b2[1], acc1)

        # Tile (1,0)
        _load_s2r_a_tile(A_shared, a_row_off + 16, lane_id, da1)
        _load_s2r_a_tile(B_shared, b_row_off, lane_id, db1)
        frag_a3 = al.view(da1, al.Tensor((2, 2), al.u32))
        frag_b3 = al.view(db1, al.Tensor((2, 2), al.u32))
        acc2 = _mfma16(frag_a3[0], frag_b3[0], acc2)
        acc2 = _mfma16(frag_a3[1], frag_b3[1], acc2)

        # Tile (1,1)
        _load_s2r_a_tile(A_shared, a_row_off + 16, lane_id, da1)
        _load_s2r_a_tile(B_shared, b_row_off + 16, lane_id, db1)
        frag_a4 = al.view(da1, al.Tensor((2, 2), al.u32))
        frag_b4 = al.view(db1, al.Tensor((2, 2), al.u32))
        acc3 = _mfma16(frag_a4[0], frag_b4[0], acc3)
        acc3 = _mfma16(frag_a4[1], frag_b4[1], acc3)

        al.syncthreads()

    # Writeback: each tile is 16x16, thread has 4 f32 in a 4x1 strip
    tr = lane_id // 16
    tc = lane_id % 16
    # Tile (0,0) at (m_base, n_base)
    C[m_base + tr * 4 + 0, n_base + tc] = al.convert(acc0[0], al.bf16)
    C[m_base + tr * 4 + 1, n_base + tc] = al.convert(acc0[1], al.bf16)
    C[m_base + tr * 4 + 2, n_base + tc] = al.convert(acc0[2], al.bf16)
    C[m_base + tr * 4 + 3, n_base + tc] = al.convert(acc0[3], al.bf16)
    # Tile (0,1) at (m_base, n_base+16)
    C[m_base + tr * 4 + 0, n_base + 16 + tc] = al.convert(acc1[0], al.bf16)
    C[m_base + tr * 4 + 1, n_base + 16 + tc] = al.convert(acc1[1], al.bf16)
    C[m_base + tr * 4 + 2, n_base + 16 + tc] = al.convert(acc1[2], al.bf16)
    C[m_base + tr * 4 + 3, n_base + 16 + tc] = al.convert(acc1[3], al.bf16)
    # Tile (1,0) at (m_base+16, n_base)
    C[m_base + 16 + tr * 4 + 0, n_base + tc] = al.convert(acc2[0], al.bf16)
    C[m_base + 16 + tr * 4 + 1, n_base + tc] = al.convert(acc2[1], al.bf16)
    C[m_base + 16 + tr * 4 + 2, n_base + tc] = al.convert(acc2[2], al.bf16)
    C[m_base + 16 + tr * 4 + 3, n_base + tc] = al.convert(acc2[3], al.bf16)
    # Tile (1,1) at (m_base+16, n_base+16)
    C[m_base + 16 + tr * 4 + 0, n_base + 16 + tc] = al.convert(acc3[0], al.bf16)
    C[m_base + 16 + tr * 4 + 1, n_base + 16 + tc] = al.convert(acc3[1], al.bf16)
    C[m_base + 16 + tr * 4 + 2, n_base + 16 + tc] = al.convert(acc3[2], al.bf16)
    C[m_base + 16 + tr * 4 + 3, n_base + 16 + tc] = al.convert(acc3[3], al.bf16)


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
