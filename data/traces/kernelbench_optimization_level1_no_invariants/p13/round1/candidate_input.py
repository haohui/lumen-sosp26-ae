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
    a_tensor = al.make_tensor(A_ptr, al.bf16, al.make_layout((m, k), (stride_a, 1)))
    b_tensor = al.make_tensor(B_ptr, al.bf16, al.make_layout((k, n), (stride_b, 1)))
    c_tensor = al.make_tensor(C_ptr, al.bf16, al.make_layout((m * n,), (1,)))
    a_rsrc = al.amdgpu.make_rsrc(a_tensor, m * k * BF16_BYTES)
    b_rsrc = al.amdgpu.make_rsrc(b_tensor, k * n * BF16_BYTES)
    c_rsrc = al.amdgpu.make_rsrc(c_tensor, m * n * BF16_BYTES)

    As = al.make_shared((TILE_M, K_STEP), al.bf16)
    Bs = al.make_shared((K_STEP, TILE_N), al.bf16)

    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = al.convert(0.0, al.f32)

    for kk in al.range(0, k, K_STEP):
        if tid < 64:
            row = tid
            goff = (block_m * TILE_M + row) * stride_a + kk
            packed = al.amdgpu.raw_buffer_load_x4(a_rsrc, goff * BF16_BYTES, 0, 0)
            frag = al.view(packed, al.Tensor((8,), al.bf16))
            for c in al.range(8):
                As[row, c] = frag[c]

        if tid < 64:
            b_row = tid // 8
            b_col_chunk = tid % 8
            goff = (kk + b_row) * stride_b + block_n * TILE_N + b_col_chunk * 8
            packed = al.amdgpu.raw_buffer_load_x4(b_rsrc, goff * BF16_BYTES, 0, 0)
            frag = al.view(packed, al.Tensor((8,), al.bf16))
            for c in al.range(8):
                Bs[b_row, b_col_chunk * 8 + c] = frag[c]

        al.syncthreads()

        a_row = warp_row * WARP_TILE_M + (wtid % 32)
        a_col = (wtid // 32) * 4

        a_f32_0 = al.convert(As[a_row, a_col + 0], al.f32)
        a_f32_1 = al.convert(As[a_row, a_col + 1], al.f32)
        a_f32_2 = al.convert(As[a_row, a_col + 2], al.f32)
        a_f32_3 = al.convert(As[a_row, a_col + 3], al.f32)

        a_packed = al.make_local((2,), al.u32)
        a_packed[0] = al.amdgpu.perm(
            al.bitcast(a_f32_1, al.u32), al.bitcast(a_f32_0, al.u32),
            al.convert(0x07060302, al.u32))
        a_packed[1] = al.amdgpu.perm(
            al.bitcast(a_f32_3, al.u32), al.bitcast(a_f32_2, al.u32),
            al.convert(0x07060302, al.u32))

        b_k_start = (wtid // 32) * 4
        b_n_col = warp_col * WARP_TILE_N + (wtid % 32)

        b_f32_0 = al.convert(Bs[b_k_start + 0, b_n_col], al.f32)
        b_f32_1 = al.convert(Bs[b_k_start + 1, b_n_col], al.f32)
        b_f32_2 = al.convert(Bs[b_k_start + 2, b_n_col], al.f32)
        b_f32_3 = al.convert(Bs[b_k_start + 3, b_n_col], al.f32)

        b_packed = al.make_local((2,), al.u32)
        b_packed[0] = al.amdgpu.perm(
            al.bitcast(b_f32_1, al.u32), al.bitcast(b_f32_0, al.u32),
            al.convert(0x07060302, al.u32))
        b_packed[1] = al.amdgpu.perm(
            al.bitcast(b_f32_3, al.u32), al.bitcast(b_f32_2, al.u32),
            al.convert(0x07060302, al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_packed, b_packed, acc)

        al.syncthreads()

    lane_row = wtid // 8
    lane_col = wtid % 8
    warp_offset = (
        (block_m * TILE_M + warp_row * WARP_TILE_M) * n
        + block_n * TILE_N
        + warp_col * WARP_TILE_N
    ) * BF16_BYTES

    for r in al.range(4):
        row_out = lane_row * 4 + r
        col_base = lane_col * 4
        a0 = al.bitcast(acc[r * 4 + 0], al.u32)
        a1 = al.bitcast(acc[r * 4 + 1], al.u32)
        a2 = al.bitcast(acc[r * 4 + 2], al.u32)
        a3 = al.bitcast(acc[r * 4 + 3], al.u32)
        packed = al.make_local((2,), al.u32)
        packed[0] = al.amdgpu.perm(a1, a0, al.convert(0x07060302, al.u32))
        packed[1] = al.amdgpu.perm(a3, a2, al.convert(0x07060302, al.u32))
        thread_offset = (row_out * n + col_base) * BF16_BYTES
        al.amdgpu.raw_buffer_store_x2(packed, c_rsrc, thread_offset, warp_offset, 0)


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
