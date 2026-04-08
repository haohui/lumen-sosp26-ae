"""AMDGPU Conv2D kernels and helpers."""

import math

import substrate
import substrate.language as S
import torch
import torch.nn as nn

WARP_SIZE = 64
NUM_WARPS = 4
GROUP_M = 128
GROUP_N = 128
GROUP_K = 32
THREADS = WARP_SIZE * NUM_WARPS
TRANSPOSE_THREADS = 256

# MFMA 32x32x8 configuration
MFMA_M = 32
MFMA_N = 32
MFMA_K = 8
MFMA_K_U32 = MFMA_K // 2

WARP_PER_ROW = 2
WARP_PER_COL = 2

M_TILES = GROUP_M // MFMA_M
N_TILES = GROUP_N // MFMA_N
K_TILES = GROUP_K // MFMA_K

# Wave repeat: each wave computes multiple tiles
WAVE_REPEAT_M = M_TILES // WARP_PER_ROW
WAVE_REPEAT_N = N_TILES // WARP_PER_COL
WAVE_TILE_COUNT = WAVE_REPEAT_M * WAVE_REPEAT_N
ACC_SIZE = WAVE_TILE_COUNT * 16

SHM_A_U32 = GROUP_M * MFMA_K_U32
SHM_B_U32 = GROUP_N * MFMA_K_U32
SHM_STAGE_U32 = SHM_A_U32 + SHM_B_U32
SHM_STAGE_U4 = SHM_STAGE_U32 // 4


@substrate.jit
def _compute_input_addr(
    a_row: S.u32,
    k_base: S.u32,
    batch_size: S.u32,
    hw_out: S.u32,
    out_w: S.u32,
    in_h: S.u32,
    in_w: S.u32,
    in_channels: S.u32,
    kernel_h: S.u32,
    kernel_w: S.u32,
    pad_h: S.u32,
    pad_w: S.u32,
    stride_h: S.u32,
    stride_w: S.u32,
    dilation_h: S.u32,
    dilation_w: S.u32,
) -> (S.u32, S.u32):
    """Compute the base address and offset for loading input tile."""
    batch = a_row // hw_out
    hw_idx = a_row % hw_out
    h_out_idx = hw_idx // out_w
    w_out_idx = hw_idx % out_w
    h_in_base = h_out_idx * stride_h - pad_h
    w_in_base = w_out_idx * stride_w - pad_w
    k_spatial = k_base // in_channels
    channel_base = k_base % in_channels
    kh = k_spatial // kernel_w
    kw = k_spatial % kernel_w
    h = h_in_base + kh * dilation_h
    w = w_in_base + kw * dilation_w
    input_base = (((batch * in_h + h) * in_w + w) * in_channels) * 2
    input_off = channel_base * 2
    return input_base, input_off


def _igemm_launch_config(batch_size, out_channels, out_h, out_w):
    gemm_m = batch_size * out_h * out_w
    gemm_n = out_channels

    m_groups = (gemm_m + GROUP_M - 1) // GROUP_M
    n_groups = (gemm_n + GROUP_N - 1) // GROUP_N

    grid = (m_groups, n_groups, 1)
    block = (THREADS, 1, 1)

    return grid, block


def _transpose_launch_config(total_elems):
    num_blocks = (total_elems + TRANSPOSE_THREADS - 1) // TRANSPOSE_THREADS
    return (num_blocks, 1, 1), (TRANSPOSE_THREADS, 1, 1)


@substrate.jit
def _transpose_input_nchw_to_nhwc_kernel(
    src: S.Pointer(S.bf16),
    dst: S.Pointer(S.bf16),
    batch_size: S.u32,
    in_channels: S.u32,
    in_h: S.u32,
    in_w: S.u32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = batch_size * in_channels * in_h * in_w
    if idx >= total:
        return

    c = idx % in_channels
    tmp = idx // in_channels
    w = tmp % in_w
    tmp = tmp // in_w
    h = tmp % in_h
    n = tmp // in_h

    src_tensor = S.make_tensor(
        src,
        S.bf16,
        S.make_layout(
            (batch_size, in_channels, in_h, in_w),
            (in_channels * in_h * in_w, in_h * in_w, in_w, 1),
        ),
    )
    dst_tensor = S.make_tensor(
        dst,
        S.bf16,
        S.make_layout(
            (batch_size, in_h, in_w, in_channels),
            (in_h * in_w * in_channels, in_w * in_channels, in_channels, 1),
        ),
    )
    dst_tensor[n, h, w, c] = src_tensor[n, c, h, w]


@substrate.jit
def _transpose_weight_oihw_to_ohwi_kernel(
    src: S.Pointer(S.bf16),
    dst: S.Pointer(S.bf16),
    out_channels: S.u32,
    in_channels: S.u32,
    kernel_h: S.u32,
    kernel_w: S.u32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = out_channels * in_channels * kernel_h * kernel_w
    if idx >= total:
        return

    c = idx % in_channels
    tmp = idx // in_channels
    kw = tmp % kernel_w
    tmp = tmp // kernel_w
    kh = tmp % kernel_h
    o = tmp // kernel_h

    src_tensor = S.make_tensor(
        src,
        S.bf16,
        S.make_layout(
            (out_channels, in_channels, kernel_h, kernel_w),
            (in_channels * kernel_h * kernel_w, kernel_h * kernel_w, kernel_w, 1),
        ),
    )
    dst_tensor = S.make_tensor(
        dst,
        S.bf16,
        S.make_layout(
            (out_channels, kernel_h, kernel_w, in_channels),
            (kernel_h * kernel_w * in_channels, kernel_w * in_channels, in_channels, 1),
        ),
    )
    dst_tensor[o, kh, kw, c] = src_tensor[o, c, kh, kw]


@substrate.jit
def _igemm_kernel(
    input_nhwc: S.Pointer(S.bf16),
    weight_ohwi: S.Pointer(S.bf16),
    out: S.Pointer(S.bf16),
    in_h: S.u32,
    in_w: S.u32,
    batch_size: S.u32,
    out_channels: S.u32,
    in_channels: S.u32,
    out_h: S.u32,
    out_w: S.u32,
    kernel_h: S.u32,
    kernel_w: S.u32,
    pad_h: S.u32,
    pad_w: S.u32,
    stride_h: S.u32,
    stride_w: S.u32,
    dilation_h: S.u32,
    dilation_w: S.u32,
    groups: S.u32,
):
    """igemm kernel using MFMA 32x32x8_bf16_f32 instructions.

    4x4 MFMA tile arrangement:
    - Each wave (64 lanes) computes a 32x32x8 matmul per MFMA call
    - GROUP_M=128, GROUP_N=128: 4x4=16 total MFMA tiles
    - 4 warps, each handles a 2x2 subtile via WAVE_REPEAT_M=WAVE_REPEAT_N=2

    Warp arrangement:
    - Warp 0: tile (0,0)
    - Warp 1: tile (0,1)
    - Warp 2: tile (1,0)
    - Warp 3: tile (1,1)

    MFMA swizzle invariants:
      For A, i in [0,32), j in [0,8): A(i, j) -> (lane_id = i + (j / 4) * 32, element = j % 4)
      For B, j in [0,8), i in [0,32): B(j, i) -> (lane_id = j + (i / 4) * 32, element = i % 4)

    Accumulator invariant for C: for each lane in [0, 64) and acc_idx in [0, 16):
      - col = tile_col_base + (lane % 32)
      - row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
    """
    gemm_m = batch_size * out_h * out_w
    gemm_n = out_channels
    gemm_k = in_channels * kernel_h * kernel_w
    hw_out = out_h * out_w

    # 2D block ID for better spatial locality
    group_m = S.block_id(0)
    group_n = S.block_id(1)

    tid = S.thread_id(0)
    wid = tid // WARP_SIZE  # 0-3
    lane = tid % WARP_SIZE  # 0-63

    # Warp-to-tile mapping for a WARP_PER_ROW x WARP_PER_COL arrangement.
    warp_m_base = wid // WARP_PER_COL
    warp_n_base = wid % WARP_PER_COL

    input_u32_layout = S.make_layout(
        (batch_size, in_h, in_w, in_channels // 2),
        (in_h * in_w * in_channels // 2, in_w * in_channels // 2, in_channels // 2, 1),
    )
    input_u32_tensor = S.make_tensor(input_nhwc, S.u32, input_u32_layout)
    rsrc_input = S.amdgpu.make_rsrc(
        input_u32_tensor, batch_size * in_h * in_w * in_channels * 2
    )

    # u32x4 view for vectorized global loads of one k=8 slice.
    weight_u32_layout = S.make_layout(
        (gemm_n, gemm_k // MFMA_K, 4), (gemm_k // 2, 4, 1)
    )
    weight_u32_tensor = S.make_tensor(weight_ohwi, S.u32, weight_u32_layout)
    rsrc_weight = S.amdgpu.make_rsrc(weight_u32_tensor, gemm_n * gemm_k * 2)

    out_layout = S.make_layout(
        (batch_size, out_channels, hw_out), (out_channels * hw_out, hw_out, 1)
    )
    out_tensor = S.make_tensor(out, S.bf16, out_layout)

    # Shared-memory staging buffers (double buffer) for one MFMA_K slice per stage:
    # A tile: [GROUP_M, MFMA_K], B tile: [GROUP_N, MFMA_K].
    shm = S.make_shared((2 * SHM_STAGE_U32,), S.u32)
    shm_u4 = S.view(shm, S.Tensor((2 * SHM_STAGE_U4, 4), S.u32))
    shm_u2 = S.view(shm, S.Tensor((SHM_STAGE_U32, 2), S.u32))

    # One 16-lane accumulator vector per tile handled by this warp.
    acc00 = S.make_local((16,), S.f32)
    acc01 = S.make_local((16,), S.f32)
    acc10 = S.make_local((16,), S.f32)
    acc11 = S.make_local((16,), S.f32)
    for i in S.range(16):
        acc00[i] = 0
        acc01[i] = 0
        acc10[i] = 0
        acc11[i] = 0

    lane_row = lane % 32
    lane_half = lane // 32
    zero_u4 = S.make_local((4,), S.u32)
    for i in S.range(4):
        zero_u4[i] = 0

    # Load fragments and perform MFMA across K tiles with software pipelining.
    k_steps = gemm_k // MFMA_K

    # Prime Stage: Load k=0 into shm stage 0
    k_base = S.convert(0, S.u32)
    if tid < GROUP_M:
        a_row = group_m * GROUP_M + tid
        if a_row < gemm_m:
            input_base, input_off = _compute_input_addr(
                a_row,
                k_base,
                batch_size,
                hw_out,
                out_w,
                in_h,
                in_w,
                in_channels,
                kernel_h,
                kernel_w,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
            )
            shm_u4[tid] = S.amdgpu.raw_buffer_load_x4(rsrc_input, input_base, input_off, 0)
        else:
            shm_u4[tid] = zero_u4
    else:
        b_idx = tid - GROUP_M
        if b_idx < GROUP_N:
            b_col = group_n * GROUP_N + b_idx
            shm_u4[GROUP_M + b_idx] = S.amdgpu.raw_buffer_load_x4(
                rsrc_weight, b_col * gemm_k * 2, 0, 0
            )
    S.syncthreads()

    k_idx = S.convert(1, S.u32)
    while k_idx + 1 < k_steps:
        # MFMA from stage 0
        for repeat_m in S.range(WAVE_REPEAT_M):
            for repeat_n in S.range(WAVE_REPEAT_N):
                m_tile = warp_m_base * WAVE_REPEAT_M + repeat_m
                n_tile = warp_n_base * WAVE_REPEAT_N + repeat_n
                row_local = m_tile * MFMA_M + lane_row
                col_local = n_tile * MFMA_N + lane_row
                a_pair_u32 = shm_u2[row_local * 2 + lane_half]
                b_pair_u32 = shm_u2[(GROUP_M + col_local) * 2 + lane_half]
                frag_a = S.view(a_pair_u32, S.Tensor((4,), S.bf16))
                frag_b = S.view(b_pair_u32, S.Tensor((4,), S.bf16))
                if repeat_m == 0:
                    if repeat_n == 0:
                        acc00 = S.amdgpu.mfma_32x32x8_bf16_f32(frag_a, frag_b, acc00)
                    else:
                        acc01 = S.amdgpu.mfma_32x32x8_bf16_f32(frag_a, frag_b, acc01)
                else:
                    if repeat_n == 0:
                        acc10 = S.amdgpu.mfma_32x32x8_bf16_f32(frag_a, frag_b, acc10)
                    else:
                        acc11 = S.amdgpu.mfma_32x32x8_bf16_f32(frag_a, frag_b, acc11)

        # Load next slice into stage 1
        k_base = k_idx * MFMA_K
        if tid < GROUP_M:
            a_row = group_m * GROUP_M + tid
            if a_row < gemm_m:
                input_base, input_off = _compute_input_addr(
                    a_row,
                    k_base,
                    batch_size,
                    hw_out,
                    out_w,
                    in_h,
                    in_w,
                    in_channels,
                    kernel_h,
                    kernel_w,
                    pad_h,
                    pad_w,
                    stride_h,
                    stride_w,
                    dilation_h,
                    dilation_w,
                )
                shm_u4[SHM_STAGE_U4 + tid] = S.amdgpu.raw_buffer_load_x4(
                    rsrc_input, input_base, input_off, 0
                )
            else:
                shm_u4[SHM_STAGE_U4 + tid] = zero_u4
        else:
            b_idx = tid - GROUP_M
            if b_idx < GROUP_N:
                b_col = group_n * GROUP_N + b_idx
                shm_u4[SHM_STAGE_U4 + GROUP_M + b_idx] = S.amdgpu.raw_buffer_load_x4(
                    rsrc_weight, b_col * gemm_k * 2, k_base * 2, 0
                )
        S.syncthreads()

        # MFMA from stage 1
        for repeat_m in S.range(WAVE_REPEAT_M):
            for repeat_n in S.range(WAVE_REPEAT_N):
                m_tile = warp_m_base * WAVE_REPEAT_M + repeat_m
                n_tile = warp_n_base * WAVE_REPEAT_N + repeat_n
                row_local = m_tile * MFMA_M + lane_row
                col_local = n_tile * MFMA_N + lane_row
                base_u2 = SHM_STAGE_U4 * 2
                a_pair_u32 = shm_u2[base_u2 + row_local * 2 + lane_half]
                b_pair_u32 = shm_u2[base_u2 + (GROUP_M + col_local) * 2 + lane_half]
                frag_a = S.view(a_pair_u32, S.Tensor((4,), S.bf16))
                frag_b = S.view(b_pair_u32, S.Tensor((4,), S.bf16))
                if repeat_m == 0:
                    if repeat_n == 0:
                        acc00 = S.amdgpu.mfma_32x32x8_bf16_f32(frag_a, frag_b, acc00)
                    else:
                        acc01 = S.amdgpu.mfma_32x32x8_bf16_f32(frag_a, frag_b, acc01)
                else:
                    if repeat_n == 0:
                        acc10 = S.amdgpu.mfma_32x32x8_bf16_f32(frag_a, frag_b, acc10)
                    else:
                        acc11 = S.amdgpu.mfma_32x32x8_bf16_f32(frag_a, frag_b, acc11)

        # Load next slice back into stage 0
        k_base = (k_idx + 1) * MFMA_K
        if tid < GROUP_M:
            a_row = group_m * GROUP_M + tid
            if a_row < gemm_m:
                input_base, input_off = _compute_input_addr(
                    a_row,
                    k_base,
                    batch_size,
                    hw_out,
                    out_w,
                    in_h,
                    in_w,
                    in_channels,
                    kernel_h,
                    kernel_w,
                    pad_h,
                    pad_w,
                    stride_h,
                    stride_w,
                    dilation_h,
                    dilation_w,
                )
                shm_u4[tid] = S.amdgpu.raw_buffer_load_x4(
                    rsrc_input, input_base, input_off, 0
                )
            else:
                shm_u4[tid] = zero_u4
        else:
            b_idx = tid - GROUP_M
            if b_idx < GROUP_N:
                b_col = group_n * GROUP_N + b_idx
                shm_u4[GROUP_M + b_idx] = S.amdgpu.raw_buffer_load_x4(
                    rsrc_weight, b_col * gemm_k * 2, k_base * 2, 0
                )
        S.syncthreads()

        k_idx = k_idx + 2

    # Tail: one remaining K-slice after x2-unrolled body for k_steps multiple of 2.
    for repeat_m in S.range(WAVE_REPEAT_M):
        for repeat_n in S.range(WAVE_REPEAT_N):
            m_tile = warp_m_base * WAVE_REPEAT_M + repeat_m
            n_tile = warp_n_base * WAVE_REPEAT_N + repeat_n
            row_local = m_tile * MFMA_M + lane_row
            col_local = n_tile * MFMA_N + lane_row
            a_pair_u32 = shm_u2[row_local * 2 + lane_half]
            b_pair_u32 = shm_u2[(GROUP_M + col_local) * 2 + lane_half]
            frag_a = S.view(a_pair_u32, S.Tensor((4,), S.bf16))
            frag_b = S.view(b_pair_u32, S.Tensor((4,), S.bf16))
            if repeat_m == 0:
                if repeat_n == 0:
                    acc00 = S.amdgpu.mfma_32x32x8_bf16_f32(frag_a, frag_b, acc00)
                else:
                    acc01 = S.amdgpu.mfma_32x32x8_bf16_f32(frag_a, frag_b, acc01)
            else:
                if repeat_n == 0:
                    acc10 = S.amdgpu.mfma_32x32x8_bf16_f32(frag_a, frag_b, acc10)
                else:
                    acc11 = S.amdgpu.mfma_32x32x8_bf16_f32(frag_a, frag_b, acc11)

    # Load final slice into stage 1
    k_base = k_idx * MFMA_K
    if tid < GROUP_M:
        a_row = group_m * GROUP_M + tid
        if a_row < gemm_m:
            input_base, input_off = _compute_input_addr(
                a_row,
                k_base,
                batch_size,
                hw_out,
                out_w,
                in_h,
                in_w,
                in_channels,
                kernel_h,
                kernel_w,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
            )
            shm_u4[SHM_STAGE_U4 + tid] = S.amdgpu.raw_buffer_load_x4(
                rsrc_input, input_base, input_off, 0
            )
        else:
            shm_u4[SHM_STAGE_U4 + tid] = zero_u4
    else:
        b_idx = tid - GROUP_M
        if b_idx < GROUP_N:
            b_col = group_n * GROUP_N + b_idx
            shm_u4[SHM_STAGE_U4 + GROUP_M + b_idx] = S.amdgpu.raw_buffer_load_x4(
                rsrc_weight, b_col * gemm_k * 2, k_base * 2, 0
            )
    S.syncthreads()

    # Final MFMA from stage 1
    for repeat_m in S.range(WAVE_REPEAT_M):
        for repeat_n in S.range(WAVE_REPEAT_N):
            m_tile = warp_m_base * WAVE_REPEAT_M + repeat_m
            n_tile = warp_n_base * WAVE_REPEAT_N + repeat_n
            row_local = m_tile * MFMA_M + lane_row
            col_local = n_tile * MFMA_N + lane_row
            base_u2 = SHM_STAGE_U4 * 2
            a_pair_u32 = shm_u2[base_u2 + row_local * 2 + lane_half]
            b_pair_u32 = shm_u2[base_u2 + (GROUP_M + col_local) * 2 + lane_half]
            frag_a = S.view(a_pair_u32, S.Tensor((4,), S.bf16))
            frag_b = S.view(b_pair_u32, S.Tensor((4,), S.bf16))
            if repeat_m == 0:
                if repeat_n == 0:
                    acc00 = S.amdgpu.mfma_32x32x8_bf16_f32(frag_a, frag_b, acc00)
                else:
                    acc01 = S.amdgpu.mfma_32x32x8_bf16_f32(frag_a, frag_b, acc01)
            else:
                if repeat_n == 0:
                    acc10 = S.amdgpu.mfma_32x32x8_bf16_f32(frag_a, frag_b, acc10)
                else:
                    acc11 = S.amdgpu.mfma_32x32x8_bf16_f32(frag_a, frag_b, acc11)

    # Write results using accumulator invariant with wave repeat
    for repeat_m in S.range(WAVE_REPEAT_M):
        for repeat_n in S.range(WAVE_REPEAT_N):
            m_tile = warp_m_base * WAVE_REPEAT_M + repeat_m
            n_tile = warp_n_base * WAVE_REPEAT_N + repeat_n

            tile_row_base = group_m * GROUP_M + m_tile * MFMA_M
            tile_col_base = group_n * GROUP_N + n_tile * MFMA_N

            for acc_idx in S.range(16):
                col_out = tile_col_base + (lane % 32)
                row_out = (
                    tile_row_base
                    + 8 * (acc_idx // 4)
                    + 4 * (lane // 32)
                    + (acc_idx % 4)
                )

                if row_out < gemm_m:
                    batch = row_out // hw_out
                    hw_idx = row_out % hw_out
                    if repeat_m == 0:
                        if repeat_n == 0:
                            out_tensor[batch, col_out, hw_idx] = S.convert(
                                acc00[acc_idx], S.bf16
                            )
                        else:
                            out_tensor[batch, col_out, hw_idx] = S.convert(
                                acc01[acc_idx], S.bf16
                            )
                    else:
                        if repeat_n == 0:
                            out_tensor[batch, col_out, hw_idx] = S.convert(
                                acc10[acc_idx], S.bf16
                            )
                        else:
                            out_tensor[batch, col_out, hw_idx] = S.convert(
                                acc11[acc_idx], S.bf16
                            )


def conv2d_naive(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 0,
    groups: int = 1,
) -> torch.Tensor:
    """
    Conv2D using implicit GEMM.

    Args:
        input: Input tensor of shape (batch_size, in_channels, in_h, in_w)
        weight: Weight tensor of shape (out_channels, in_channels // groups, kernel_h, kernel_w)
        out: Optional output tensor. If None, a new one is allocated.

    Returns:
        Output tensor of shape (batch_size, out_channels, out_h, out_w)
    """
    if not isinstance(input, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("input and weight must be torch.Tensor")
    if input.ndim != 4 or weight.ndim != 4:
        raise ValueError(
            f"input must be rank-4 and weight must be rank-4 "
            f"(got input.ndim={input.ndim}, weight.ndim={weight.ndim})"
        )
    if input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError(
            f"input and weight must be torch.bfloat16 "
            f"(got input={input.dtype}, weight={weight.dtype})"
        )
    if input.device.type != "cuda" or weight.device.type != "cuda":
        raise ValueError(
            f"input and weight must be CUDA tensors "
            f"(got input={input.device}, weight={weight.device})"
        )
    if input.device != weight.device:
        raise ValueError(
            f"input and weight must be on the same device "
            f"(got {input.device} and {weight.device})"
        )

    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    # Compute output dimensions
    out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1
    gemm_m = batch_size * out_h * out_w
    gemm_k = in_channels * kernel_h * kernel_w

    # Kernel assumes dense groups=1/padding=0 and aligned K slices for MFMA.
    if groups != 1:
        raise ValueError(f"only groups=1 is supported (got groups={groups})")
    if padding != 0:
        raise ValueError(f"only padding=0 is supported (got padding={padding})")
    if out_channels % GROUP_N != 0:
        raise ValueError(
            f"out_channels must be a multiple of {GROUP_N} for vectorized B-tile loads (got {out_channels})"
        )
    if gemm_k % (2 * MFMA_K) != 0:
        raise ValueError(
            f"in_channels*kernel_h*kernel_w must be a multiple of {2 * MFMA_K} (got {gemm_k})"
        )
    if in_channels % MFMA_K != 0:
        raise ValueError(
            f"in_channels must be a multiple of {MFMA_K} for contiguous x4 A-loads (got {in_channels})"
        )

    # Allocate output tensor if not provided
    if out is None or out.numel() == 0:
        out = torch.empty(
            (batch_size, out_channels, out_h, out_w),
            dtype=torch.bfloat16,
            device=input.device,
        )
    elif out.ndim != 4 or out.shape != (batch_size, out_channels, out_h, out_w):
        raise ValueError(
            f"out must have shape {(batch_size, out_channels, out_h, out_w)} (got {out.shape})"
        )
    elif out.dtype != torch.bfloat16:
        raise TypeError(f"out must be torch.bfloat16 (got {out.dtype})")
    elif out.device != input.device:
        raise ValueError(f"out must be on {input.device} (got {out.device})")

    # Pre-transpose to contiguous layouts for igemm global loads.
    input_nhwc = torch.empty(
        (batch_size, in_h, in_w, in_channels), dtype=torch.bfloat16, device=input.device
    )
    weight_ohwi = torch.empty(
        (out_channels, kernel_h, kernel_w, in_channels),
        dtype=torch.bfloat16,
        device=input.device,
    )

    grid_transpose_in, block_transpose_in = _transpose_launch_config(
        batch_size * in_channels * in_h * in_w
    )
    _transpose_input_nchw_to_nhwc_kernel[
        lambda: (grid_transpose_in, block_transpose_in)
    ](
        input,
        input_nhwc,
        batch_size,
        in_channels,
        in_h,
        in_w,
    )

    grid_transpose_w, block_transpose_w = _transpose_launch_config(
        out_channels * in_channels * kernel_h * kernel_w
    )
    _transpose_weight_oihw_to_ohwi_kernel[
        lambda: (grid_transpose_w, block_transpose_w)
    ](
        weight,
        weight_ohwi,
        out_channels,
        in_channels,
        kernel_h,
        kernel_w,
    )

    # Fused implicit GEMM over transposed layouts.
    grid_matmul, block_matmul = _igemm_launch_config(
        batch_size, out_channels, out_h, out_w
    )
    _igemm_kernel[lambda: (grid_matmul, block_matmul)](
        input_nhwc,
        weight_ohwi,
        out,
        in_h,
        in_w,
        batch_size,
        out_channels,
        in_channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        padding,
        padding,
        stride,
        stride,
        dilation,
        dilation,
        groups,
    )
    return out


class ModelNew(nn.Module):
    """
    KernelBench-style Conv2d wrapper around the inlined `conv2d_naive` kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias_param", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        original_dtype = x.dtype

        x_bf16 = x.to(dtype=torch.bfloat16)
        weight_bf16 = self.weight.detach().to(device=x.device, dtype=torch.bfloat16).contiguous()

        out = conv2d_naive(
            x_bf16,
            weight_bf16,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            groups=self.groups,
        )

        if self.bias_param is not None:
            bias = self.bias_param.detach().to(device=x.device, dtype=torch.bfloat16)
            out = out + bias.view(1, -1, 1, 1)

        if original_dtype != torch.bfloat16:
            out = out.to(original_dtype)
        return out
