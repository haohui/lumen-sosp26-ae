import torch
import torch.nn as nn

import substrate
import substrate.language as S

M = 4096
K = 4096
N = 4096

BLOCK_M = 64
BLOCK_N = 64
WARP_M = 32
WARP_N = 32
NUM_WARPS = 4

A_RANGE = M * 1024 * 2 * 4  # 33554432 bytes
B_RANGE = N * 1024 * 2 * 4  # 33554432 bytes


@substrate.jit
def tri_gemm_kernel(
    A: S.Tensor((4096, 1024, 2), S.u32),
    B: S.Tensor((4096, 1024, 2), S.u32),
    C: S.Tensor((4096, 4096), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % 64
    warp_id = tid // 64
    warp_row = warp_id // 2
    warp_col = warp_id % 2

    block_m = S.block_id(0)
    block_n = S.block_id(1)

    row_base = block_m * BLOCK_M + warp_row * WARP_M
    col_base = block_n * BLOCK_N + warp_col * WARP_N

    acc = S.full((16,), 0.0, S.f32)

    # Resource descriptors with range for OOB-safe loads
    rsrc_A = S.amdgpu.make_rsrc(A, A_RANGE)
    rsrc_B = S.amdgpu.make_rsrc(B, B_RANGE)

    # Double-buffered LDS for A and B
    lds_A0 = S.make_shared((256, 2), S.u32)
    lds_A1 = S.make_shared((256, 2), S.u32)
    lds_B0 = S.make_shared((256, 2), S.u32)
    lds_B1 = S.make_shared((256, 2), S.u32)

    a_row = row_base + lane % 32
    b_row = col_base + lane % 32

    # --- Prologue: Load K=0 to buf0 using raw_buffer_load_x4 with range ---
    # Byte offset for A[a_row, lane//32, :] = (a_row * 2048 + (lane//32) * 2) * 4
    offset_A = (a_row * 2048 + (lane // 32) * 2) * 4
    offset_B = (b_row * 2048 + (lane // 32) * 2) * 4
    lds_A0[tid] = S.amdgpu.raw_buffer_load_x2(rsrc_A, offset_A, 0, 0)
    lds_B0[tid] = S.amdgpu.raw_buffer_load_x2(rsrc_B, offset_B, 0, 0)
    S.syncthreads()

    # --- Main loop: K unrolled by 2 ---
    for k_pack in S.range(255):
        k1 = k_pack * 2 + 1
        k2 = k_pack * 2 + 2
        kg1 = k1 * 2 + lane // 32
        kg2 = k2 * 2 + lane // 32

        offset_A1 = (a_row * 2048 + kg1 * 2) * 4
        offset_A2 = (a_row * 2048 + kg2 * 2) * 4
        offset_B1 = (b_row * 2048 + kg1 * 2) * 4
        offset_B2 = (b_row * 2048 + kg2 * 2) * 4

        # == Half-step A: read buf0, MFMA, load next to buf1 ==
        a_pair = lds_A0[tid]
        lds_A1[tid] = S.amdgpu.raw_buffer_load_x2(rsrc_A, offset_A1, 0, 0)
        b_pair = lds_B0[tid]
        lds_B1[tid] = S.amdgpu.raw_buffer_load_x2(rsrc_B, offset_B1, 0, 0)

        a_bf16 = S.view(a_pair, S.Tensor((1, 4, 1), S.bf16))
        b_bf16 = S.view(b_pair, S.Tensor((1, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_bf16[0], b_bf16[0], acc)

        S.syncthreads()

        # == Half-step B: read buf1, MFMA, load next to buf0 ==
        a_pair = lds_A1[tid]
        lds_A0[tid] = S.amdgpu.raw_buffer_load_x2(rsrc_A, offset_A2, 0, 0)
        b_pair = lds_B1[tid]
        lds_B0[tid] = S.amdgpu.raw_buffer_load_x2(rsrc_B, offset_B2, 0, 0)

        a_bf16 = S.view(a_pair, S.Tensor((1, 4, 1), S.bf16))
        b_bf16 = S.view(b_pair, S.Tensor((1, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_bf16[0], b_bf16[0], acc)

        S.syncthreads()

    # --- Epilogue: K=510 from buf0, K=511 from buf1 ---
    a_pair = lds_A0[tid]
    b_pair = lds_B0[tid]
    a_bf16 = S.view(a_pair, S.Tensor((1, 4, 1), S.bf16))
    b_bf16 = S.view(b_pair, S.Tensor((1, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_bf16[0], b_bf16[0], acc)

    offset_A_ep = (a_row * 2048 + (1022 + lane // 32) * 2) * 4
    offset_B_ep = (b_row * 2048 + (1022 + lane // 32) * 2) * 4
    lds_A1[tid] = S.amdgpu.raw_buffer_load_x2(rsrc_A, offset_A_ep, 0, 0)
    lds_B1[tid] = S.amdgpu.raw_buffer_load_x2(rsrc_B, offset_B_ep, 0, 0)
    S.syncthreads()

    a_pair = lds_A1[tid]
    b_pair = lds_B1[tid]
    a_bf16 = S.view(a_pair, S.Tensor((1, 4, 1), S.bf16))
    b_bf16 = S.view(b_pair, S.Tensor((1, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_bf16[0], b_bf16[0], acc)

    # --- Write back accumulator with triangular mask (branch removed) ---
    for acc_idx in S.range(16):
        col = col_base + (lane % 32)
        row = row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        # Branchless: comparison sign-extends to -1 for true, negate to get 1.0
        mask = 0.0 - S.convert(col >= row, S.f32)
        C[row, col] = S.convert(acc[acc_idx] * mask, S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._A_u32 = None
        self._A_ptr = None
        self._B_u32 = None
        self._B_ptr = None

    def _ensure_u32_view(self, t, cache_attr, ptr_attr):
        """Create/cache a u32 view of a bf16 tensor for cudagraph safety."""
        ptr = t.data_ptr()
        cached = getattr(self, cache_attr)
        cached_ptr = getattr(self, ptr_attr)
        if cached is None or cached_ptr != ptr:
            flat_u32 = t.reshape(-1).view(torch.int32)
            view = flat_u32.reshape(t.shape[0], t.shape[1] // 4, 2)
            setattr(self, cache_attr, view)
            setattr(self, ptr_attr, ptr)
        return getattr(self, cache_attr)

    def forward(self, A, B):
        A = A.contiguous()
        B_t = B.t().contiguous()

        A_u32 = self._ensure_u32_view(A, '_A_u32', '_A_ptr')
        B_u32 = self._ensure_u32_view(B_t, '_B_u32', '_B_ptr')

        C = torch.zeros((M, N), device=A.device, dtype=A.dtype)
        tri_gemm_kernel[lambda: ((M // BLOCK_M, N // BLOCK_N, 1), (NUM_WARPS * 64, 1, 1))](
            A_u32, B_u32, C
        )
        return C
