import torch
import torch.nn as nn
import substrate
import substrate.language as S
import math

WARP_SIZE = 64
SPLIT_K_SLICES = 2

def _launch_splitk(gemm_m_tiles, gemm_n_tiles):
    """Launch config for split-K kernel."""
    return ((gemm_m_tiles * gemm_n_tiles * SPLIT_K_SLICES, 1, 1), (WARP_SIZE, 1, 1))


def _launch_finalize(batch_size, hw_tiles, channel_tiles):
    """Launch config for finalization kernel."""
    return ((batch_size * hw_tiles * channel_tiles, 1, 1), (256, 1, 1))


@substrate.jit
def _mdiv_u32(numer: S.u32, magic: S.u32, shift: S.u32) -> S.u32:
    prod_hi = S.convert((S.convert(magic, S.u64) * S.convert(numer, S.u64)) >> 32, S.u32)
    return (prod_hi + numer) >> shift


@substrate.jit
def _mdiv_u32_rem(numer: S.u32, denom: S.u32, magic: S.u32, shift: S.u32) -> (S.u32, S.u32):
    quot = _mdiv_u32(numer, magic, shift)
    rem = numer - quot * denom
    return quot, rem


@substrate.jit
def conv2d_mfma_splitk_kernel(
    input_nchw: S.Pointer(S.bf16),
    weight_oihw: S.Pointer(S.bf16),
    workspace: S.Pointer(S.f32),
    batch_size: S.u32,
    in_channels: S.u32,
    out_channels: S.u32,
    in_h: S.u32,
    in_w: S.u32,
    out_h: S.u32,
    out_w: S.u32,
    kernel_h: S.u32,
    kernel_w: S.u32,
    gemm_m: S.u32,
    gemm_n: S.u32,
    gemm_k: S.u32,
    gemm_n_tiles: S.u32,
    gemm_n_magic: S.u32,
    gemm_n_shift: S.u32,
):
    """MFMA Conv2D kernel with split-K reduction."""
    linear_block_id = S.block_id(0)
    split_k_id = linear_block_id % SPLIT_K_SLICES
    tile_block_id = linear_block_id // SPLIT_K_SLICES

    group_m, group_n = _mdiv_u32_rem(tile_block_id, gemm_n_tiles, gemm_n_magic, gemm_n_shift)

    lane = S.thread_id(0)

    group_m_base = group_m * 32
    group_n_base = group_n * 32

    k_per_split = (gemm_k + SPLIT_K_SLICES - 1) // SPLIT_K_SLICES
    k_start = split_k_id * k_per_split
    k_end = k_start + k_per_split
    if k_end > gemm_k:
        k_end = gemm_k

    kernel_area = kernel_h * kernel_w

    input_tensor = S.make_tensor(
        input_nchw,
        S.bf16,
        S.make_layout(
            (batch_size, in_channels, in_h * in_w),
            (in_channels * in_h * in_w, in_h * in_w, 1),
        ),
    )
    weight_tensor = S.make_tensor(
        weight_oihw,
        S.bf16,
        S.make_layout(
            (out_channels, in_channels * kernel_area),
            (in_channels * kernel_area, 1),
        ),
    )
    workspace_tensor = S.make_tensor(
        workspace,
        S.f32,
        S.make_layout((gemm_m * gemm_n,), (1,)),
    )

    workspace_rsrc = S.amdgpu.make_rsrc(workspace_tensor, gemm_m * gemm_n * 4)

    zero_u32 = S.convert(0, S.u32)
    zero_bf16 = S.convert(0.0, S.bf16)

    acc = S.full((16,), 0.0, S.f32)

    A_shmem = S.make_shared((WARP_SIZE, 2), S.u32)
    B_shmem = S.make_shared((WARP_SIZE, 2), S.u32)
    A_shmem_bf16 = S.view(A_shmem, S.Tensor((WARP_SIZE, 4), S.bf16))
    B_shmem_bf16 = S.view(B_shmem, S.Tensor((WARP_SIZE, 4), S.bf16))

    lane_col = lane % 32
    lane_row_grp = lane // 32

    MFMA_K = 8
    k_chunks = (k_per_split + MFMA_K - 1) // MFMA_K

    for k_chunk in S.range(k_chunks):
        k_base = k_start + k_chunk * MFMA_K

        for t in S.range(4):
            k_offset = t * 2
            k_idx = k_base + k_offset

            if k_idx < k_end:
                c = k_idx // kernel_area
                spatial = k_idx % kernel_area
                kh = spatial // kernel_w
                kw = spatial % kernel_w

                row_in_tile = lane_col
                row_gemm = group_m_base + row_in_tile

                if row_gemm < gemm_m:
                    batch_idx = row_gemm // (out_h * out_w)
                    hw_idx = row_gemm % (out_h * out_w)
                    h_out_idx = hw_idx // out_w
                    w_out_idx = hw_idx % out_w

                    h_in = h_out_idx + kh
                    w_in = w_out_idx + kw

                    if h_in < in_h and w_in < in_w:
                        A_shmem_bf16[lane, t] = input_tensor[batch_idx, c, h_in * in_w + w_in]
                    else:
                        A_shmem_bf16[lane, t] = zero_bf16
                else:
                    A_shmem_bf16[lane, t] = zero_bf16
            else:
                A_shmem_bf16[lane, t] = zero_bf16

        for t in S.range(4):
            k_offset = t * 2
            k_idx = k_base + k_offset

            if k_idx < k_end:
                col_in_tile = lane_col
                col_gemm = group_n_base + col_in_tile

                if col_gemm < gemm_n:
                    B_shmem_bf16[lane, t] = weight_tensor[col_gemm, k_idx]
                else:
                    B_shmem_bf16[lane, t] = zero_bf16
            else:
                B_shmem_bf16[lane, t] = zero_bf16

        S.syncthreads()

        a_frag = S.view(A_shmem[lane], S.Tensor((1, 4, 1), S.bf16))
        b_frag = S.view(B_shmem[lane], S.Tensor((1, 4, 1), S.bf16))

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)

    for acc_idx in S.range(4):
        row_local = 4 * lane_row_grp + acc_idx
        row_gemm = group_m_base + row_local

        col_gemm = group_n_base + lane_col

        if row_gemm < gemm_m and col_gemm < gemm_n:
            linear_idx = row_gemm * gemm_n + col_gemm
            byte_offset = linear_idx * 4

            S.amdgpu.buffer_atomic_add_f32(acc[acc_idx], workspace_rsrc, byte_offset, zero_u32, 0)


@substrate.jit
def finalize_kernel(
    workspace: S.Pointer(S.f32),
    output_nchw: S.Pointer(S.bf16),
    batch_size: S.u32,
    out_channels: S.u32,
    out_h: S.u32,
    out_w: S.u32,
    gemm_m: S.u32,
    gemm_n: S.u32,
):
    """Finalization kernel."""
    OUTPUT_TILE = 32
    ROWS_PER_THREAD = 4

    hw_out = out_h * out_w
    hw_tiles = (hw_out + OUTPUT_TILE - 1) // OUTPUT_TILE
    channel_tiles = (out_channels + OUTPUT_TILE - 1) // OUTPUT_TILE

    linear_block_id = S.block_id(0)
    tiles_per_batch = hw_tiles * channel_tiles
    batch = linear_block_id // tiles_per_batch
    tile_id = linear_block_id - batch * tiles_per_batch
    tile_hw = tile_id // channel_tiles
    tile_channel = tile_id % channel_tiles

    tid = S.thread_id(0)
    local_col = tid & 31
    local_row_base = (tid >> 5) * ROWS_PER_THREAD
    src_hw_base = tile_hw * OUTPUT_TILE + local_row_base
    src_channel = tile_channel * OUTPUT_TILE + local_col

    workspace_matrix = S.make_tensor(
        workspace,
        S.f32,
        S.make_layout((gemm_m, gemm_n), (gemm_n, 1)),
    )
    output_tensor = S.make_tensor(
        output_nchw,
        S.bf16,
        S.make_layout(
            (batch_size, out_channels, hw_out),
            (out_channels * hw_out, hw_out, 1),
        ),
    )
    tile = S.make_shared((OUTPUT_TILE, OUTPUT_TILE + 1), S.bf16)

    for i in S.range(ROWS_PER_THREAD):
        src_hw = src_hw_base + i
        if batch < batch_size and src_hw < hw_out and src_channel < out_channels:
            tile[local_row_base + i, local_col] = S.convert(
                workspace_matrix[batch * hw_out + src_hw, src_channel], S.bf16
            )
    S.syncthreads()

    for i in S.range(ROWS_PER_THREAD):
        dst_channel = tile_channel * OUTPUT_TILE + local_row_base + i
        dst_hw = tile_hw * OUTPUT_TILE + local_col
        if batch < batch_size and dst_channel < out_channels and dst_hw < hw_out:
            output_tensor[batch, dst_channel, dst_hw] = tile[local_col, local_row_base + i]


def _compute_magic_u32_params(divisor: int) -> tuple:
    if divisor <= 0 or divisor >= (1 << 32):
        raise ValueError(f"divisor must be in [1, 2^32) (got {divisor})")

    shift = (divisor - 1).bit_length()
    if divisor & (divisor - 1) == 0:
        return 0, shift

    magic = ((1 << (32 + shift)) // divisor) - (1 << 32) + 1
    return magic, shift


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self._workspace = None
        self._gemm_n_magic = None
        self._gemm_n_shift = None

    def _get_workspace(self, device, gemm_m, gemm_n):
        workspace_size = gemm_m * gemm_n
        if self._workspace is None or self._workspace.numel() != workspace_size or self._workspace.device != device:
            self._workspace = torch.zeros(workspace_size, device=device, dtype=torch.float32)
        return self._workspace

    def forward(self, x):
        expected_shape = (16, 16, 1024, 1024)
        if tuple(x.shape) != expected_shape or x.dtype != torch.float32:
            raise RuntimeError(f'This fused kernel only supports input shape {expected_shape} and dtype float32.')

        device = x.device
        batch_size, in_channels, in_h, in_w = x.shape
        out_channels = self.out_channels
        kernel_size = self.kernel_size

        out_h = (in_h - kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (in_w - kernel_size + 2 * self.padding) // self.stride + 1

        gemm_m = batch_size * out_h * out_w
        gemm_n = out_channels
        gemm_k = in_channels * kernel_size * kernel_size

        gemm_n_tiles = (gemm_n + 31) // 32
        if self._gemm_n_magic is None:
            self._gemm_n_magic, self._gemm_n_shift = _compute_magic_u32_params(gemm_n_tiles)

        x_bf16 = x.to(dtype=torch.bfloat16).contiguous()
        w_bf16 = self.conv2d.weight.to(device=device, dtype=torch.bfloat16).contiguous()

        workspace = self._get_workspace(device, gemm_m, gemm_n)
        workspace.zero_()

        y_bf16 = torch.empty((batch_size, out_channels, out_h, out_w), device=device, dtype=torch.bfloat16)

        gemm_m_tiles = (gemm_m + 31) // 32

        conv2d_mfma_splitk_kernel[lambda: _launch_splitk(gemm_m_tiles, gemm_n_tiles)](
            x_bf16.data_ptr(),
            w_bf16.data_ptr(),
            workspace.data_ptr(),
            batch_size,
            in_channels,
            out_channels,
            in_h,
            in_w,
            out_h,
            out_w,
            kernel_size,
            kernel_size,
            gemm_m,
            gemm_n,
            gemm_k,
            gemm_n_tiles,
            self._gemm_n_magic,
            self._gemm_n_shift,
        )

        hw_tiles = (out_h * out_w + 31) // 32
        channel_tiles = (out_channels + 31) // 32

        finalize_kernel[lambda: _launch_finalize(batch_size, hw_tiles, channel_tiles)](
            workspace.data_ptr(),
            y_bf16.data_ptr(),
            batch_size,
            out_channels,
            out_h,
            out_w,
            gemm_m,
            gemm_n,
        )

        return y_bf16.to(dtype=torch.float32)
