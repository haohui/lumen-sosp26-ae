import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 2048
K = 8192
N = 4096

WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
GRID_TILES_N = N // BLOCK_N

# RAW_BUFFER_WORD3: Format control word for buffer descriptor
# Bit 17 (0x20000) enables bounds checking using range field
RAW_BUFFER_WORD3 = 0x00020000


@substrate.jit
def gemm_kernel(
    A_desc: S.Tensor((4,), S.u32),
    B_desc: S.Tensor((4,), S.u32),
    C: S.Tensor((2048, 4096), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WARP_SIZE
    warp = tid // WARP_SIZE
    warp_m = warp // 2
    warp_n = warp % 2

    block = S.block_id(0)
    tile_m = block // GRID_TILES_N
    tile_n = block % GRID_TILES_N
    block_row = tile_m * BLOCK_M
    block_col = tile_n * BLOCK_N

    # Double-buffered shared memory
    a_shared = S.make_shared((2, 2, 64, 8), S.bf16)
    b_shared = S.make_shared((2, 2, 64, 8), S.bf16)
    a_shared_u32 = S.view(a_shared, S.Tensor((2, 2, 64, 4), S.u32))
    b_shared_u32 = S.view(b_shared, S.Tensor((2, 2, 64, 4), S.u32))

    acc = S.full((16,), 0.0, S.f32)

    NUM_K_TILES = K // BLOCK_K

    # Prologue: load first K tile into buffer 0
    # Range in buffer descriptors (word 2) enables OOB handling:
    # - raw_buffer_load_x4 returns 0 for OOB accesses
    # - raw_buffer_store discards OOB writes
    # This eliminates need for explicit OOB guards on global memory
    kk0 = 0
    buf = 0
    if tid < 128:
        load_group = tid // 64
        load_id = tid % 64
        row = load_id % 32
        k_chunk = load_id // 32
        k_base = kk0 + k_chunk * 8
        global_row = block_row + load_group * 32 + row
        soffset = S.convert((global_row * K + k_base) * 2, S.i32)
        vec = S.amdgpu.raw_buffer_load_x4(A_desc, 0, soffset, 0)
        frag = S.view(vec, S.Tensor((2, 4, 1), S.bf16))
        elem_base = k_chunk * 4
        for elem in S.range(4):
            a_shared[buf, load_group, row, elem_base + elem] = frag[0, elem, 0]
            a_shared[buf, load_group, row + 32, elem_base + elem] = frag[1, elem, 0]
    else:
        load_id = tid - 128
        load_group = load_id // 64
        local_id = load_id % 64
        k_local = local_id // 4
        col_chunk = local_id % 4
        col_base = block_col + load_group * 32 + col_chunk * 8
        global_row = kk0 + k_local
        soffset = S.convert((global_row * N + col_base) * 2, S.i32)
        vec = S.amdgpu.raw_buffer_load_x4(B_desc, 0, soffset, 0)
        frag = S.view(vec, S.Tensor((2, 4, 1), S.bf16))
        lane_base = ((k_local % 8) // 4) * 32 + col_chunk * 8
        elem_idx = (k_local // 8) * 4 + (k_local % 4)
        for elem in S.range(4):
            b_shared[buf, load_group, lane_base + elem, elem_idx] = frag[0, elem, 0]
            b_shared[buf, load_group, lane_base + 4 + elem, elem_idx] = frag[1, elem, 0]

    S.syncthreads()

    # Main loop with double buffering - unroll by 2 for software pipelining
    for kt in S.range(0, NUM_K_TILES - 1, 2):
        cur_buf = 0
        next_buf = 1

        # ---- Compute on buffer 0 (tile kt) ----
        a_frag = S.view(a_shared_u32[cur_buf, warp_m, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_shared_u32[cur_buf, warp_n, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        # ---- Load tile kt+1 into buffer 1 ----
        kk0 = (kt + 1) * BLOCK_K
        if tid < 128:
            load_group = tid // 64
            load_id = tid % 64
            row = load_id % 32
            k_chunk = load_id // 32
            k_base = kk0 + k_chunk * 8
            global_row = block_row + load_group * 32 + row
            soffset = S.convert((global_row * K + k_base) * 2, S.i32)
            vec = S.amdgpu.raw_buffer_load_x4(A_desc, 0, soffset, 0)
            frag = S.view(vec, S.Tensor((2, 4, 1), S.bf16))
            elem_base = k_chunk * 4
            for elem in S.range(4):
                a_shared[next_buf, load_group, row, elem_base + elem] = frag[0, elem, 0]
                a_shared[next_buf, load_group, row + 32, elem_base + elem] = frag[1, elem, 0]
        else:
            load_id = tid - 128
            load_group = load_id // 64
            local_id = load_id % 64
            k_local = local_id // 4
            col_chunk = local_id % 4
            col_base = block_col + load_group * 32 + col_chunk * 8
            global_row = kk0 + k_local
            soffset = S.convert((global_row * N + col_base) * 2, S.i32)
            vec = S.amdgpu.raw_buffer_load_x4(B_desc, 0, soffset, 0)
            frag = S.view(vec, S.Tensor((2, 4, 1), S.bf16))
            lane_base = ((k_local % 8) // 4) * 32 + col_chunk * 8
            elem_idx = (k_local // 8) * 4 + (k_local % 4)
            for elem in S.range(4):
                b_shared[next_buf, load_group, lane_base + elem, elem_idx] = frag[0, elem, 0]
                b_shared[next_buf, load_group, lane_base + 4 + elem, elem_idx] = frag[1, elem, 0]

        S.syncthreads()

        # ---- Compute on buffer 1 (tile kt+1) ----
        a_frag = S.view(a_shared_u32[next_buf, warp_m, lane], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_shared_u32[next_buf, warp_n, lane], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        # ---- Load tile kt+2 into buffer 0 ----
        kk0 = (kt + 2) * BLOCK_K
        if tid < 128:
            load_group = tid // 64
            load_id = tid % 64
            row = load_id % 32
            k_chunk = load_id // 32
            k_base = kk0 + k_chunk * 8
            global_row = block_row + load_group * 32 + row
            soffset = S.convert((global_row * K + k_base) * 2, S.i32)
            vec = S.amdgpu.raw_buffer_load_x4(A_desc, 0, soffset, 0)
            frag = S.view(vec, S.Tensor((2, 4, 1), S.bf16))
            elem_base = k_chunk * 4
            for elem in S.range(4):
                a_shared[cur_buf, load_group, row, elem_base + elem] = frag[0, elem, 0]
                a_shared[cur_buf, load_group, row + 32, elem_base + elem] = frag[1, elem, 0]
        else:
            load_id = tid - 128
            load_group = load_id // 64
            local_id = load_id % 64
            k_local = local_id // 4
            col_chunk = local_id % 4
            col_base = block_col + load_group * 32 + col_chunk * 8
            global_row = kk0 + k_local
            soffset = S.convert((global_row * N + col_base) * 2, S.i32)
            vec = S.amdgpu.raw_buffer_load_x4(B_desc, 0, soffset, 0)
            frag = S.view(vec, S.Tensor((2, 4, 1), S.bf16))
            lane_base = ((k_local % 8) // 4) * 32 + col_chunk * 8
            elem_idx = (k_local // 8) * 4 + (k_local % 4)
            for elem in S.range(4):
                b_shared[cur_buf, load_group, lane_base + elem, elem_idx] = frag[0, elem, 0]
                b_shared[cur_buf, load_group, lane_base + 4 + elem, elem_idx] = frag[1, elem, 0]

        S.syncthreads()

    # Compute on the last tile
    last_buf = (NUM_K_TILES - 1) % 2
    a_frag = S.view(a_shared_u32[last_buf, warp_m, lane], S.Tensor((2, 4, 1), S.bf16))
    b_frag = S.view(b_shared_u32[last_buf, warp_n, lane], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

    # Store results
    tile_row_base = block_row + warp_m * 32
    tile_col_base = block_col + warp_n * 32
    col = tile_col_base + (lane % 32)
    row_quad = 4 * (lane // 32)
    for acc_idx in S.range(16):
        row = tile_row_base + 8 * (acc_idx // 4) + row_quad + (acc_idx % 4)
        C[row, col] = S.convert(acc[acc_idx], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._desc_cache = {}

    def _get_raw_buffer_desc(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Create a raw buffer descriptor for AMDGPU buffer operations.

        The descriptor contains 4 words:
        - Word 0-1: Base address (64-bit pointer split into low/high 32 bits)
        - Word 2: Range in bytes (numel * element_size)
        - Word 3: Format control (RAW_BUFFER_WORD3 with bounds checking enabled)

        When range is set, the hardware handles OOB access:
        - raw_buffer_load returns 0 for OOB elements
        - raw_buffer_store discards OOB writes

        This eliminates the need for explicit OOB guard branches.
        """
        key = (
            tensor.device,
            tensor.data_ptr(),
            tensor.numel(),
            tensor.element_size(),
        )
        desc = self._desc_cache.get(key)
        if desc is None:
            ptr = tensor.data_ptr()
            desc = torch.tensor(
                [
                    ptr & 0xFFFFFFFF,                      # Word 0: Address low
                    (ptr >> 32) & 0xFFFFFFFF,              # Word 1: Address high
                    tensor.numel() * tensor.element_size(), # Word 2: Range in bytes
                    RAW_BUFFER_WORD3,                       # Word 3: Format control
                ],
                dtype=torch.uint32,
                device=tensor.device,
            )
            self._desc_cache = {key: desc}
        return desc

    def forward(self, A, B):
        if (
            tuple(A.shape) != (M, K)
            or tuple(B.shape) != (K, N)
            or A.dtype != torch.bfloat16
            or B.dtype != torch.bfloat16
            or not A.is_cuda
            or not B.is_cuda
        ):
            return torch.matmul(A, B)
        if torch.cuda.is_current_stream_capturing():
            return torch.matmul(A, B)

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        A_desc = self._get_raw_buffer_desc(A)
        B_desc = self._get_raw_buffer_desc(B)
        gemm_kernel[lambda: (((M // BLOCK_M) * (N // BLOCK_N), 1, 1), (THREADS, 1, 1))](
            A_desc, B_desc, C
        )
        return C
