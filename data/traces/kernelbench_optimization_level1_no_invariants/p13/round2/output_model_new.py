import torch
import torch.nn as nn

import avelang
import avelang.language as al


WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS
TILE_M = 64
TILE_N = 64
K_STEP = 8
WARP_TILE_M = 32
WARP_TILE_N = 32
BF16_BYTES = 2


@avelang.jit
def gemm_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    m: al.i32,
    n: al.i32,
    k: al.i32,
):
    tid = al.thread_id(0)
    wid = tid // WARP_SIZE
    wtid = tid % WARP_SIZE
    warp_row = wid // 2
    warp_col = wid % 2

    block_m = al.block_id(0)
    block_n = al.block_id(1)

    stride_a = k
    stride_b = n
    a_tensor = al.make_tensor(A_ptr, al.bf16, al.make_layout((m * k,), (1,)))
    b_tensor = al.make_tensor(B_ptr, al.bf16, al.make_layout((k * n,), (1,)))
    c_tensor = al.make_tensor(C_ptr, al.bf16, al.make_layout((m * n,), (1,)))
    a_rsrc = al.amdgpu.make_rsrc(a_tensor, m * k * BF16_BYTES)
    b_rsrc = al.amdgpu.make_rsrc(b_tensor, k * n * BF16_BYTES)
    c_rsrc = al.amdgpu.make_rsrc(c_tensor, m * n * BF16_BYTES)

    # Double-buffered shared memory for software pipelining
    As0 = al.make_shared((TILE_M, K_STEP), al.bf16)
    Bs0 = al.make_shared((K_STEP, TILE_N), al.bf16)
    As1 = al.make_shared((TILE_M, K_STEP), al.bf16)
    Bs1 = al.make_shared((K_STEP, TILE_N), al.bf16)

    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = al.convert(0.0, al.f32)

    # === Prologue ===
    if tid < TILE_M:
        gm_row = block_m * TILE_M + tid
        goff = gm_row * stride_a + 0
        packed = al.amdgpu.raw_buffer_load_x4(a_rsrc, goff * BF16_BYTES, 0, 0)
        frag = al.view(packed, al.Tensor((8,), al.bf16))
        for c in al.range(8):
            As0[tid, c] = frag[c]

    if tid < TILE_M:
        b_row = tid // 8
        b_col_chunk = tid % 8
        gm_row = 0 + b_row
        gm_col = block_n * TILE_N + b_col_chunk * 8
        goff = gm_row * stride_b + gm_col
        packed = al.amdgpu.raw_buffer_load_x4(b_rsrc, goff * BF16_BYTES, 0, 0)
        frag = al.view(packed, al.Tensor((8,), al.bf16))
        for c in al.range(8):
            Bs0[b_row, b_col_chunk * 8 + c] = frag[c]

    al.syncthreads()

    # === Main loop ===
    for kk in al.range(K_STEP, k + K_STEP, 2 * K_STEP):
        # --- Phase 1 ---
        if tid < TILE_M:
            gm_row = block_m * TILE_M + tid
            goff = gm_row * stride_a + kk
            packed = al.amdgpu.raw_buffer_load_x4(a_rsrc, goff * BF16_BYTES, 0, 0)
            frag = al.view(packed, al.Tensor((8,), al.bf16))
            for c in al.range(8):
                As1[tid, c] = frag[c]

        if tid < TILE_M:
            b_row = tid // 8
            b_col_chunk = tid % 8
            gm_row = kk + b_row
            gm_col = block_n * TILE_N + b_col_chunk * 8
            goff = gm_row * stride_b + gm_col
            packed = al.amdgpu.raw_buffer_load_x4(b_rsrc, goff * BF16_BYTES, 0, 0)
            frag = al.view(packed, al.Tensor((8,), al.bf16))
            for c in al.range(8):
                Bs1[b_row, b_col_chunk * 8 + c] = frag[c]

        a_m = warp_row * WARP_TILE_M + (wtid % WARP_TILE_M)
        a_k = (wtid // WARP_TILE_M) * 4
        a_v0 = al.convert(As0[a_m, a_k + 0], al.f32)
        a_v1 = al.convert(As0[a_m, a_k + 1], al.f32)
        a_v2 = al.convert(As0[a_m, a_k + 2], al.f32)
        a_v3 = al.convert(As0[a_m, a_k + 3], al.f32)
        a_pack = al.make_local((2,), al.u32)
        a_pack[0] = al.amdgpu.perm(
            al.bitcast(a_v1, al.u32), al.bitcast(a_v0, al.u32),
            al.convert(0x07060302, al.u32))
        a_pack[1] = al.amdgpu.perm(
            al.bitcast(a_v3, al.u32), al.bitcast(a_v2, al.u32),
            al.convert(0x07060302, al.u32))

        b_k = (wtid // WARP_TILE_N) * 4
        b_n = warp_col * WARP_TILE_N + (wtid % WARP_TILE_N)
        b_v0 = al.convert(Bs0[b_k + 0, b_n], al.f32)
        b_v1 = al.convert(Bs0[b_k + 1, b_n], al.f32)
        b_v2 = al.convert(Bs0[b_k + 2, b_n], al.f32)
        b_v3 = al.convert(Bs0[b_k + 3, b_n], al.f32)
        b_pack = al.make_local((2,), al.u32)
        b_pack[0] = al.amdgpu.perm(
            al.bitcast(b_v1, al.u32), al.bitcast(b_v0, al.u32),
            al.convert(0x07060302, al.u32))
        b_pack[1] = al.amdgpu.perm(
            al.bitcast(b_v3, al.u32), al.bitcast(b_v2, al.u32),
            al.convert(0x07060302, al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_pack, b_pack, acc)

        al.syncthreads()

        # --- Phase 2 ---
        if tid < TILE_M:
            gm_row = block_m * TILE_M + tid
            goff = gm_row * stride_a + kk + K_STEP
            packed = al.amdgpu.raw_buffer_load_x4(a_rsrc, goff * BF16_BYTES, 0, 0)
            frag = al.view(packed, al.Tensor((8,), al.bf16))
            for c in al.range(8):
                As0[tid, c] = frag[c]

        if tid < TILE_M:
            b_row = tid // 8
            b_col_chunk = tid % 8
            gm_row = kk + K_STEP + b_row
            gm_col = block_n * TILE_N + b_col_chunk * 8
            goff = gm_row * stride_b + gm_col
            packed = al.amdgpu.raw_buffer_load_x4(b_rsrc, goff * BF16_BYTES, 0, 0)
            frag = al.view(packed, al.Tensor((8,), al.bf16))
            for c in al.range(8):
                Bs0[b_row, b_col_chunk * 8 + c] = frag[c]

        a_v0 = al.convert(As1[a_m, a_k + 0], al.f32)
        a_v1 = al.convert(As1[a_m, a_k + 1], al.f32)
        a_v2 = al.convert(As1[a_m, a_k + 2], al.f32)
        a_v3 = al.convert(As1[a_m, a_k + 3], al.f32)
        a_pack2 = al.make_local((2,), al.u32)
        a_pack2[0] = al.amdgpu.perm(
            al.bitcast(a_v1, al.u32), al.bitcast(a_v0, al.u32),
            al.convert(0x07060302, al.u32))
        a_pack2[1] = al.amdgpu.perm(
            al.bitcast(a_v3, al.u32), al.bitcast(a_v2, al.u32),
            al.convert(0x07060302, al.u32))

        b_v0 = al.convert(Bs1[b_k + 0, b_n], al.f32)
        b_v1 = al.convert(Bs1[b_k + 1, b_n], al.f32)
        b_v2 = al.convert(Bs1[b_k + 2, b_n], al.f32)
        b_v3 = al.convert(Bs1[b_k + 3, b_n], al.f32)
        b_pack2 = al.make_local((2,), al.u32)
        b_pack2[0] = al.amdgpu.perm(
            al.bitcast(b_v1, al.u32), al.bitcast(b_v0, al.u32),
            al.convert(0x07060302, al.u32))
        b_pack2[1] = al.amdgpu.perm(
            al.bitcast(b_v3, al.u32), al.bitcast(b_v2, al.u32),
            al.convert(0x07060302, al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_pack2, b_pack2, acc)

        al.syncthreads()

    # === Store output via shared memory ===
    # CDNA3 VGPR layout: each thread produces 16 values at 16 distinct rows
    # and one column. Store to shared memory, then use row-wise packing
    # to write 8 consecutive BF16 values (from the same row) to global memory.
    Cshared = al.make_shared((TILE_M, TILE_N), al.bf16)

    # Each thread writes its 16 accumulator values to Cshared
    col_out = warp_col * WARP_TILE_N + wtid % WARP_TILE_N
    row_grp = wtid // WARP_TILE_N
    for m in al.range(16):
        row = warp_row * WARP_TILE_M + row_grp * 4 + (m // 4) * 8 + (m % 4)
        bf_val = al.convert(acc[m], al.bf16)
        Cshared[row, col_out] = bf_val

    al.syncthreads()

    # Store Cshared to global memory: each of 64 threads handles one row,
    # writing 8 consecutive BF16 values at a time (using x4 loads/stores on
    # the shared memory side, x4 raw stores to global).
    if tid < TILE_M:
        row = block_m * TILE_M + tid
        for col8 in al.range(0, TILE_N, 8):
            gm_col = block_n * TILE_N + col8
            # Pack 8 BF16 values into 4 u32
            bf0 = Cshared[tid, col8 + 0]
            bf1 = Cshared[tid, col8 + 1]
            bf2 = Cshared[tid, col8 + 2]
            bf3 = Cshared[tid, col8 + 3]
            bf4 = Cshared[tid, col8 + 4]
            bf5 = Cshared[tid, col8 + 5]
            bf6 = Cshared[tid, col8 + 6]
            bf7 = Cshared[tid, col8 + 7]
            u0 = al.bitcast(bf0, al.u16)
            u1 = al.bitcast(bf1, al.u16)
            u2 = al.bitcast(bf2, al.u16)
            u3 = al.bitcast(bf3, al.u16)
            u4 = al.bitcast(bf4, al.u16)
            u5 = al.bitcast(bf5, al.u16)
            u6 = al.bitcast(bf6, al.u16)
            u7 = al.bitcast(bf7, al.u16)
            w0 = al.convert(u0, al.u32)
            w1 = al.convert(u1, al.u32)
            w2 = al.convert(u2, al.u32)
            w3 = al.convert(u3, al.u32)
            w4 = al.convert(u4, al.u32)
            w5 = al.convert(u5, al.u32)
            w6 = al.convert(u6, al.u32)
            w7 = al.convert(u7, al.u32)
            packed = al.make_local((4,), al.u32)
            packed[0] = al.amdgpu.perm(w1, w0, al.convert(0x05040100, al.u32))
            packed[1] = al.amdgpu.perm(w3, w2, al.convert(0x05040100, al.u32))
            packed[2] = al.amdgpu.perm(w5, w4, al.convert(0x05040100, al.u32))
            packed[3] = al.amdgpu.perm(w7, w6, al.convert(0x05040100, al.u32))
            goff = (row * n + gm_col) * BF16_BYTES
            al.amdgpu.raw_buffer_store_x4(packed, c_rsrc, goff, 0, 0)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        A = A.contiguous()
        B = B.contiguous()
        m = A.shape[0]
        k_in = A.shape[1]
        n = B.shape[1]
        C = torch.empty((m, n), device=A.device, dtype=A.dtype)
        m_groups = m // TILE_M
        n_groups = n // TILE_N
        gemm_kernel[lambda: ((m_groups, n_groups, 1), (THREADS, 1, 1))](
            A, B, C,
            m, n, k_in,
        )
        return C
