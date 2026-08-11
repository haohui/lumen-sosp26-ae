"""
AMDGPU Conv1D kernel using implicit GEMM with MFMA instructions.
Adapted from the Conv2D igemm template for 1D convolution.
"""

import math

import substrate
import substrate.language as S
import torch
import torch.nn as nn

# Thread and tiling configuration
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
def _compute_input_addr_1d(
    a_row: S.u32,
    k_base: S.u32,
    batch_size: S.u32,
    out_length: S.u32,
    in_length: S.u32,
    in_channels: S.u32,
    kernel_size: S.u32,
    stride: S.u32,
    dilation: S.u32,
) -> (S.u32, S.u32):
    """Compute the base address and offset for loading input tile for Conv1D."""
    # Decode a_row into (batch, out_pos)
    batch = a_row // out_length
    ol = a_row % out_length

    # Compute base input position
    in_base = ol * stride

    # Decode k_base into (kernel_pos, channel)
    k_spatial = k_base // in_channels
    channel_base = k_base % in_channels
    k = k_spatial % kernel_size

    # Compute input position with dilation
    in_pos = in_base + k * dilation

    # Compute address in NHWC-like layout: (batch, in_length, in_channels)
    # Input is stored as (batch, in_length, in_channels)
    input_base = ((batch * in_length + in_pos) * in_channels) * 2
    input_off = channel_base * 2
    return input_base, input_off


def _igemm_launch_config_1d(batch_size, out_channels, out_length):
    gemm_m = batch_size * out_length
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
def _transpose_input_ncl_to_nlc_kernel(
    src: S.Pointer(S.bf16),
    dst: S.Pointer(S.bf16),
    batch_size: S.u32,
    in_channels: S.u32,
    in_length: S.u32,
):
    """Transpose input from (N, C, L) to (N, L, C) layout."""
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = batch_size * in_channels * in_length
    if idx >= total:
        return

    c = idx % in_channels
    tmp = idx // in_channels
    l = tmp % in_length
    n = tmp // in_length

    src_tensor = S.make_tensor(
        src,
        S.bf16,
        S.make_layout(
            (batch_size, in_channels, in_length),
            (in_channels * in_length, in_length, 1),
        ),
    )
    dst_tensor = S.make_tensor(
        dst,
        S.bf16,
        S.make_layout(
            (batch_size, in_length, in_channels),
            (in_length * in_channels, in_channels, 1),
        ),
    )
    dst_tensor[n, l, c] = src_tensor[n, c, l]


@substrate.jit
def _transpose_weight_oik_to_oki_kernel(
    src: S.Pointer(S.bf16),
    dst: S.Pointer(S.bf16),
    out_channels: S.u32,
    in_channels: S.u32,
    kernel_size: S.u32,
):
    """Transpose weight from (O, I, K) to (O, K, I) layout."""
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    total = out_channels * in_channels * kernel_size
    if idx >= total:
        return

    c = idx % in_channels
    tmp = idx // in_channels
    k = tmp % kernel_size
    o = tmp // kernel_size

    src_tensor = S.make_tensor(
        src,
        S.bf16,
        S.make_layout(
            (out_channels, in_channels, kernel_size),
            (in_channels * kernel_size, kernel_size, 1),
        ),
    )
    dst_tensor = S.make_tensor(
        dst,
        S.bf16,
        S.make_layout(
            (out_channels, kernel_size, in_channels),
            (kernel_size * in_channels, in_channels, 1),
        ),
    )
    dst_tensor[o, k, c] = src_tensor[o, c, k]


@substrate.jit
def _igemm_conv1d_kernel(
    input_nlc: S.Pointer(S.bf16),
    weight_oki: S.Pointer(S.bf16),
    out: S.Pointer(S.bf16),
    in_length: S.u32,
    batch_size: S.u32,
    out_channels: S.u32,
    in_channels: S.u32,
    out_length: S.u32,
    kernel_size: S.u32,
    stride: S.u32,
    dilation: S.u32,
):
    """
    igemm kernel for Conv1D using MFMA 32x32x8_bf16_f32 instructions.

    Conv1D as implicit GEMM:
    - M = batch_size * out_length (output spatial positions)
    - N = out_channels
    - K = in_channels * kernel_size

    4x4 MFMA tile arrangement with 4 warps handling 2x2 sub-tiles each.
    """
    gemm_m = batch_size * out_length
    gemm_n = out_channels
    gemm_k = in_channels * kernel_size

    # 2D block ID for better spatial locality
    group_m = S.block_id(0)
    group_n = S.block_id(1)

    tid = S.thread_id(0)
    wid = tid // WARP_SIZE  # 0-3
    lane = tid % WARP_SIZE  # 0-63

    # Warp-to-tile mapping
    warp_m_base = wid // WARP_PER_COL
    warp_n_base = wid % WARP_PER_COL

    # Input tensor in NLC layout as u32 pairs for vectorized loads
    input_u32_layout = S.make_layout(
        (batch_size, in_length, in_channels // 2),
        (in_length * in_channels // 2, in_channels // 2, 1),
    )
    input_u32_tensor = S.make_tensor(input_nlc, S.u32, input_u32_layout)
    rsrc_input = S.amdgpu.make_rsrc(
        input_u32_tensor, batch_size * in_length * in_channels * 2
    )

    # Weight tensor in OKI layout as u32 pairs
    weight_u32_layout = S.make_layout(
        (gemm_n, gemm_k // MFMA_K, 4), (gemm_k // 2, 4, 1)
    )
    weight_u32_tensor = S.make_tensor(weight_oki, S.u32, weight_u32_layout)
    rsrc_weight = S.amdgpu.make_rsrc(weight_u32_tensor, gemm_n * gemm_k * 2)

    # Output tensor in (batch, out_channels, out_length) layout
    out_layout = S.make_layout(
        (batch_size, out_channels, out_length), (out_channels * out_length, out_length, 1)
    )
    out_tensor = S.make_tensor(out, S.bf16, out_layout)

    # Shared-memory staging buffers (double buffer)
    shm = S.make_shared((2 * SHM_STAGE_U32,), S.u32)
    shm_u4 = S.view(shm, S.Tensor((2 * SHM_STAGE_U4, 4), S.u32))
    shm_u2 = S.view(shm, S.Tensor((SHM_STAGE_U32, 2), S.u32))

    # Accumulator vectors for each tile
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

    # Load fragments and perform MFMA across K tiles
    k_steps = gemm_k // MFMA_K

    # Prime Stage: Load k=0 into shm stage 0
    k_base = S.convert(0, S.u32)
    if tid < GROUP_M:
        a_row = group_m * GROUP_M + tid
        if a_row < gemm_m:
            input_base, input_off = _compute_input_addr_1d(
                a_row,
                k_base,
                batch_size,
                out_length,
                in_length,
                in_channels,
                kernel_size,
                stride,
                dilation,
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
                input_base, input_off = _compute_input_addr_1d(
                    a_row,
                    k_base,
                    batch_size,
                    out_length,
                    in_length,
                    in_channels,
                    kernel_size,
                    stride,
                    dilation,
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
                input_base, input_off = _compute_input_addr_1d(
                    a_row,
                    k_base,
                    batch_size,
                    out_length,
                    in_length,
                    in_channels,
                    kernel_size,
                    stride,
                    dilation,
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

    # Tail: one remaining K-slice after unrolled body
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
            input_base, input_off = _compute_input_addr_1d(
                a_row,
                k_base,
                batch_size,
                out_length,
                in_length,
                in_channels,
                kernel_size,
                stride,
                dilation,
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

    # Write results
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
                    batch = row_out // out_length
                    ol = row_out % out_length
                    if repeat_m == 0:
                        if repeat_n == 0:
                            out_tensor[batch, col_out, ol] = S.convert(
                                acc00[acc_idx], S.bf16
                            )
                        else:
                            out_tensor[batch, col_out, ol] = S.convert(
                                acc01[acc_idx], S.bf16
                            )
                    else:
                        if repeat_n == 0:
                            out_tensor[batch, col_out, ol] = S.convert(
                                acc10[acc_idx], S.bf16
                            )
                        else:
                            out_tensor[batch, col_out, ol] = S.convert(
                                acc11[acc_idx], S.bf16
                            )


def conv1d_igemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    stride: int = 1,
    dilation: int = 1,
) -> torch.Tensor:
    """
    Conv1D using implicit GEMM.

    Args:
        input: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_size)
        out: Optional output tensor.

    Returns:
        Output tensor of shape (batch_size, out_channels, out_length)
    """
    if not isinstance(input, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("input and weight must be torch.Tensor")
    if input.ndim != 3 or weight.ndim != 3:
        raise ValueError(
            f"input must be rank-3 and weight must be rank-3 "
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

    batch_size, in_channels, in_length = input.shape
    out_channels, _, kernel_size = weight.shape

    # Compute output length (padding=0)
    out_length = (in_length - dilation * (kernel_size - 1) - 1) // stride + 1
    gemm_m = batch_size * out_length
    gemm_k = in_channels * kernel_size

    # Validate constraints for MFMA-based kernel
    if out_channels % GROUP_N != 0:
        raise ValueError(
            f"out_channels must be a multiple of {GROUP_N} (got {out_channels})"
        )
    if gemm_k % (2 * MFMA_K) != 0:
        raise ValueError(
            f"in_channels*kernel_size must be a multiple of {2 * MFMA_K} (got {gemm_k})"
        )
    if in_channels % MFMA_K != 0:
        raise ValueError(
            f"in_channels must be a multiple of {MFMA_K} (got {in_channels})"
        )

    # Allocate output tensor if not provided
    if out is None or out.numel() == 0:
        out = torch.empty(
            (batch_size, out_channels, out_length),
            dtype=torch.bfloat16,
            device=input.device,
        )

    # Pre-transpose to contiguous layouts for igemm global loads
    # Input: NCL -> NLC
    input_nlc = torch.empty(
        (batch_size, in_length, in_channels), dtype=torch.bfloat16, device=input.device
    )
    # Weight: OIK -> OKI
    weight_oki = torch.empty(
        (out_channels, kernel_size, in_channels),
        dtype=torch.bfloat16,
        device=input.device,
    )

    grid_transpose_in, block_transpose_in = _transpose_launch_config(
        batch_size * in_channels * in_length
    )
    _transpose_input_ncl_to_nlc_kernel[
        lambda: (grid_transpose_in, block_transpose_in)
    ](
        input,
        input_nlc,
        batch_size,
        in_channels,
        in_length,
    )

    grid_transpose_w, block_transpose_w = _transpose_launch_config(
        out_channels * in_channels * kernel_size
    )
    _transpose_weight_oik_to_oki_kernel[
        lambda: (grid_transpose_w, block_transpose_w)
    ](
        weight,
        weight_oki,
        out_channels,
        in_channels,
        kernel_size,
    )

    # Launch implicit GEMM kernel
    grid_matmul, block_matmul = _igemm_launch_config_1d(
        batch_size, out_channels, out_length
    )
    _igemm_conv1d_kernel[lambda: (grid_matmul, block_matmul)](
        input_nlc,
        weight_oki,
        out,
        in_length,
        batch_size,
        out_channels,
        in_channels,
        out_length,
        kernel_size,
        stride,
        dilation,
    )
    return out


class ModelNew(nn.Module):
    """
    KernelBench-style Conv1d wrapper using implicit GEMM with MFMA.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()

        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size)
        )
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias_param", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = self.weight.shape[1] * self.weight.shape[2]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        original_dtype = x.dtype

        x_bf16 = x.to(dtype=torch.bfloat16)
        weight_bf16 = self.weight.detach().to(device=x.device, dtype=torch.bfloat16).contiguous()

        out = conv1d_igemm(
            x_bf16,
            weight_bf16,
            stride=self.stride,
            dilation=self.dilation,
        )

        if self.bias_param is not None:
            bias = self.bias_param.detach().to(device=x.device, dtype=torch.bfloat16)
            out = out + bias.view(1, -1, 1)

        if original_dtype != torch.bfloat16:
            out = out.to(original_dtype)
        return out
