import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 8205
K = 2949
N = 5921

WAVE_SIZE = 64
WAVES_M = 2
WAVES_N = 2
WAVES_PER_BLOCK = WAVES_M * WAVES_N
THREADS_PER_BLOCK = WAVES_PER_BLOCK * WAVE_SIZE

TILE_M = 32
TILE_N = 32
TILE_K = 16
BLOCK_M = WAVES_M * TILE_M
BLOCK_N = WAVES_N * TILE_N

M_PAD = ((M + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
N_PAD = ((N + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
K_PAD = ((K + (2 * TILE_K) - 1) // (2 * TILE_K)) * (2 * TILE_K)
K_TILES = K_PAD // TILE_K
K_PAIRS = K_TILES // 2


@substrate.jit
def bf16_gemm_mfma_pipelined(
    A_pack: S.Tensor((M_PAD, K_TILES, 2, 8), S.bf16),
    B_pack: S.Tensor((N_PAD, K_TILES, 2, 8), S.bf16),
    C: S.Tensor((M_PAD, N_PAD), S.bf16),
    a_range_bytes: S.i32,
    b_range_bytes: S.i32,
):
    tid = S.thread_id(0)
    lane = tid & (WAVE_SIZE - 1)
    wave = tid >> 6
    wave_m = wave >> 1
    wave_n = wave & 1

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N

    zero = S.convert(0, S.i32)
    a_rsrc = S.amdgpu.make_rsrc(A_pack, a_range_bytes)
    b_rsrc = S.amdgpu.make_rsrc(B_pack, b_range_bytes)

    a_smem = S.make_shared((2, 128, 4), S.u32)
    b_smem = S.make_shared((2, 128, 4), S.u32)

    acc = S.full((16,), 0.0, S.f32)
    if tid < 128:
        frag = tid
        row = block_row + (frag >> 1)
        group = frag & 1
        byte_offset = S.convert((((row * K_TILES) * 2 + group) * 8) * 2, S.i32)
        a_smem[0, frag] = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, byte_offset, 0)
        byte_offset = S.convert(((((row * K_TILES) + 1) * 2 + group) * 8) * 2, S.i32)
        a_smem[1, frag] = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, byte_offset, 0)
    else:
        frag = tid - 128
        col = block_col + (frag >> 1)
        group = frag & 1
        byte_offset = S.convert((((col * K_TILES) * 2 + group) * 8) * 2, S.i32)
        b_smem[0, frag] = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, byte_offset, 0)
        byte_offset = S.convert(((((col * K_TILES) + 1) * 2 + group) * 8) * 2, S.i32)
        b_smem[1, frag] = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, byte_offset, 0)
    S.syncthreads()

    a_index = wave_m * 64 + ((lane & 31) << 1) + (lane >> 5)
    b_index = wave_n * 64 + ((lane & 31) << 1) + (lane >> 5)

    for pair in S.range(K_PAIRS - 1):
        a_frag = S.view(a_smem[0, a_index], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_smem[0, b_index], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        if tid < 128:
            frag = tid
            row = block_row + (frag >> 1)
            group = frag & 1
            k_tile = pair * 2 + 2
            byte_offset = S.convert((((row * K_TILES + k_tile) * 2 + group) * 8) * 2, S.i32)
            a_smem[0, frag] = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, byte_offset, 0)
        else:
            frag = tid - 128
            col = block_col + (frag >> 1)
            group = frag & 1
            k_tile = pair * 2 + 2
            byte_offset = S.convert((((col * K_TILES + k_tile) * 2 + group) * 8) * 2, S.i32)
            b_smem[0, frag] = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, byte_offset, 0)

        a_frag = S.view(a_smem[1, a_index], S.Tensor((2, 4, 1), S.bf16))
        b_frag = S.view(b_smem[1, b_index], S.Tensor((2, 4, 1), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

        if tid < 128:
            frag = tid
            row = block_row + (frag >> 1)
            group = frag & 1
            k_tile = pair * 2 + 3
            byte_offset = S.convert((((row * K_TILES + k_tile) * 2 + group) * 8) * 2, S.i32)
            a_smem[1, frag] = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, byte_offset, 0)
        else:
            frag = tid - 128
            col = block_col + (frag >> 1)
            group = frag & 1
            k_tile = pair * 2 + 3
            byte_offset = S.convert((((col * K_TILES + k_tile) * 2 + group) * 8) * 2, S.i32)
            b_smem[1, frag] = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, byte_offset, 0)

        S.syncthreads()

    a_frag = S.view(a_smem[0, a_index], S.Tensor((2, 4, 1), S.bf16))
    b_frag = S.view(b_smem[0, b_index], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

    a_frag = S.view(a_smem[1, a_index], S.Tensor((2, 4, 1), S.bf16))
    b_frag = S.view(b_smem[1, b_index], S.Tensor((2, 4, 1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], acc)

    out_col = block_col + wave_n * TILE_N + (lane & 31)
    row_base = block_row + wave_m * TILE_M + ((lane >> 5) * 4)
    for i in S.range(16):
        out_row = row_base + ((i >> 2) * 8) + (i & 3)
        C[out_row, out_col] = S.convert(acc[i], S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._device = None
        self._a_pad = None
        self._b_pad = None
        self._a_pack = None
        self._b_pack = None
        self._c = None
        self._a_bytes = None
        self._b_bytes = None

    def _ensure_buffers(self, device):
        if self._device == device:
            return
        self._device = device
        self._a_pad = torch.zeros((M_PAD, K_PAD), device=device, dtype=torch.bfloat16)
        self._b_pad = torch.zeros((K_PAD, N_PAD), device=device, dtype=torch.bfloat16)
        self._a_pack = torch.empty((M_PAD, K_TILES, 2, 8), device=device, dtype=torch.bfloat16)
        self._b_pack = torch.empty((N_PAD, K_TILES, 2, 8), device=device, dtype=torch.bfloat16)
        self._c = torch.empty((M_PAD, N_PAD), device=device, dtype=torch.bfloat16)
        self._a_bytes = self._a_pack.numel() * self._a_pack.element_size()
        self._b_bytes = self._b_pack.numel() * self._b_pack.element_size()

    def forward(self, A, B):
        if tuple(A.shape) != (M, K) or tuple(B.shape) != (K, N):
            raise ValueError(f"expected A=({M}, {K}) and B=({K}, {N})")

        A = A.contiguous()
        B = B.contiguous()

        self._ensure_buffers(A.device)

        self._a_pad.zero_()
        self._b_pad.zero_()
        self._a_pad[:M, :K].copy_(A)
        self._b_pad[:K, :N].copy_(B)

        a_view = self._a_pad.view(M_PAD, K_TILES, TILE_K)
        self._a_pack[:, :, 0, :4].copy_(a_view[:, :, :4])
        self._a_pack[:, :, 0, 4:].copy_(a_view[:, :, 8:12])
        self._a_pack[:, :, 1, :4].copy_(a_view[:, :, 4:8])
        self._a_pack[:, :, 1, 4:].copy_(a_view[:, :, 12:16])

        b_view = self._b_pad.view(K_TILES, TILE_K, N_PAD).permute(2, 0, 1).contiguous()
        self._b_pack[:, :, 0, :4].copy_(b_view[:, :, :4])
        self._b_pack[:, :, 0, 4:].copy_(b_view[:, :, 8:12])
        self._b_pack[:, :, 1, :4].copy_(b_view[:, :, 4:8])
        self._b_pack[:, :, 1, 4:].copy_(b_view[:, :, 12:16])

        grid = (N_PAD // BLOCK_N, M_PAD // BLOCK_M, 1)
        block = (THREADS_PER_BLOCK, 1, 1)
        bf16_gemm_mfma_pipelined[lambda: (grid, block)](
            self._a_pack,
            self._b_pack,
            self._c,
            self._a_bytes,
            self._b_bytes,
        )
        return self._c[:M, :N]
