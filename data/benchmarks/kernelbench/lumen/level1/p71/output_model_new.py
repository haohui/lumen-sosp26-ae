import math

import substrate
import substrate.language as S
import torch
import torch.nn as nn

WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS

GROUP_M = 128
GROUP_N = 32

MFMA_M = 32
MFMA_N = 32
MFMA_K = 8
MFMA_K_U32 = MFMA_K // 2

TRANSPOSE_TILE = 32
TRANSPOSE_THREADS = 256
TRANSPOSE_ROWS_PER_THREAD = 4

SHM_A_U32 = GROUP_M * MFMA_K_U32
SHM_B_U32 = GROUP_N * MFMA_K_U32
SHM_STAGE_U32 = SHM_A_U32 + SHM_B_U32
SHM_STAGE_U4 = SHM_STAGE_U32 // 4

batch_size = 8
in_channels = 32
out_channels = 32
kernel_size = 3
height_in = 512
width_in = 1024


def _transpose_input_launch_config(batch_size: int, in_channels: int, in_h: int, in_w: int):
    hw_in = in_h * in_w
    h_tiles = (in_channels + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE
    w_tiles = (hw_in + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE
    grid = (batch_size * h_tiles * w_tiles, 1, 1)
    block = (TRANSPOSE_THREADS, 1, 1)
    return grid, block


def _igemm_launch_config(batch_size: int, out_channels: int, out_h: int, out_w: int):
    gemm_m = batch_size * out_h * out_w
    m_groups = (gemm_m + GROUP_M - 1) // GROUP_M
    n_groups = (out_channels + GROUP_N - 1) // GROUP_N
    return (m_groups, n_groups, 1), (THREADS, 1, 1)


@substrate.jit
def _transpose_input_nchw_to_nhwc_kernel(
    input_nchw: S.Pointer(S.bf16),
    input_nhwc: S.Pointer(S.bf16),
    batch_size: S.u32,
    in_channels: S.u32,
    in_h: S.u32,
    in_w: S.u32,
):
    hw_in = in_h * in_w
    h_tiles = (in_channels + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE
    w_tiles = (hw_in + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE

    linear_block_id = S.block_id(0)
    tiles_per_batch = h_tiles * w_tiles
    batch = linear_block_id // tiles_per_batch
    tile_id = linear_block_id - batch * tiles_per_batch
    tile_h = tile_id // w_tiles
    tile_w = tile_id % w_tiles

    tid = S.thread_id(0)
    local_col = tid & 31
    local_row_base = (tid >> 5) * TRANSPOSE_ROWS_PER_THREAD
    global_row_base = tile_h * TRANSPOSE_TILE + local_row_base
    global_col = tile_w * TRANSPOSE_TILE + local_col

    src = S.make_tensor(
        input_nchw,
        S.bf16,
        S.make_layout(
            (batch_size, in_channels, hw_in),
            (in_channels * hw_in, hw_in, 1),
        ),
    )
    dst = S.make_tensor(
        input_nhwc,
        S.bf16,
        S.make_layout(
            (batch_size, hw_in, in_channels),
            (hw_in * in_channels, in_channels, 1),
        ),
    )
    tile = S.make_shared((TRANSPOSE_TILE, TRANSPOSE_TILE + 1), S.bf16)

    for i in S.range(TRANSPOSE_ROWS_PER_THREAD):
        src_row = global_row_base + i
        if batch < batch_size and src_row < in_channels and global_col < hw_in:
            tile[local_row_base + i, local_col] = src[batch, src_row, global_col]
    S.syncthreads()

    for i in S.range(TRANSPOSE_ROWS_PER_THREAD):
        dst_row = tile_w * TRANSPOSE_TILE + local_row_base + i
        dst_col = tile_h * TRANSPOSE_TILE + local_col
        if batch < batch_size and dst_row < hw_in and dst_col < in_channels:
            dst[batch, dst_row, dst_col] = tile[local_col, local_row_base + i]


@substrate.jit
def _compute_input_addr_conv_t(
    a_row: S.u32,
    k_base: S.u32,
    hw_out: S.u32,
    out_w: S.u32,
    in_h: S.u32,
    in_w: S.u32,
    in_channels: S.u32,
    kernel_w: S.u32,
    pad_h: S.u32,
    pad_w: S.u32,
    stride_h: S.u32,
    stride_w: S.u32,
    dilation_h: S.u32,
    dilation_w: S.u32,
) -> (S.u32, S.u32, S.u32):
    batch = a_row // hw_out
    hw_idx = a_row % hw_out
    h_out_idx = hw_idx // out_w
    w_out_idx = hw_idx % out_w

    k_spatial = k_base // in_channels
    channel_base = k_base % in_channels
    kh = k_spatial // kernel_w
    kw = k_spatial % kernel_w

    h_coord = h_out_idx * stride_h + kh * dilation_h
    w_coord = w_out_idx * stride_w + kw * dilation_w

    valid = S.convert(1, S.u32)
    if h_coord < pad_h:
        valid = S.convert(0, S.u32)
    if h_coord >= pad_h + in_h:
        valid = S.convert(0, S.u32)
    if w_coord < pad_w:
        valid = S.convert(0, S.u32)
    if w_coord >= pad_w + in_w:
        valid = S.convert(0, S.u32)

    h = h_coord - pad_h
    w = w_coord - pad_w
    input_base = (((batch * in_h + h) * in_w + w) * in_channels) * 2
    input_off = channel_base * 2
    return input_base, input_off, valid


@substrate.jit
def _conv_transpose2d_igemm_kernel(
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
):
    gemm_m = batch_size * out_h * out_w
    gemm_k = in_channels * kernel_h * kernel_w
    hw_out = out_h * out_w

    group_m = S.block_id(0)
    group_n = S.block_id(1)

    tid = S.thread_id(0)
    wid = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    lane_row = lane % 32
    lane_half = lane // 32

    input_u32_layout = S.make_layout(
        (batch_size, in_h, in_w, in_channels // 2),
        (in_h * in_w * in_channels // 2, in_w * in_channels // 2, in_channels // 2, 1),
    )
    input_u32_tensor = S.make_tensor(input_nhwc, S.u32, input_u32_layout)
    rsrc_input = S.amdgpu.make_rsrc(
        input_u32_tensor, batch_size * in_h * in_w * in_channels * 2
    )

    weight_u32_layout = S.make_layout(
        (out_channels, gemm_k // MFMA_K, 4),
        (gemm_k // 2, 4, 1),
    )
    weight_u32_tensor = S.make_tensor(weight_ohwi, S.u32, weight_u32_layout)
    rsrc_weight = S.amdgpu.make_rsrc(weight_u32_tensor, out_channels * gemm_k * 2)

    out_layout = S.make_layout(
        (batch_size, out_channels, hw_out),
        (out_channels * hw_out, hw_out, 1),
    )
    out_tensor = S.make_tensor(out, S.bf16, out_layout)

    shm = S.make_shared((SHM_STAGE_U32,), S.u32)
    shm_u4 = S.view(shm, S.Tensor((SHM_STAGE_U4, 4), S.u32))
    shm_u2 = S.view(shm, S.Tensor((SHM_STAGE_U32 // 2, 2), S.u32))

    zero_u4 = S.make_local((4,), S.u32)
    acc = S.make_local((16,), S.f32)
    for i in S.range(4):
        zero_u4[i] = 0
    for i in S.range(16):
        acc[i] = 0

    k_steps = gemm_k // MFMA_K
    for k_idx in S.range(k_steps):
        k_base = k_idx * MFMA_K

        if tid < GROUP_M:
            a_row = group_m * GROUP_M + tid
            if a_row < gemm_m:
                input_base, input_off, valid = _compute_input_addr_conv_t(
                    a_row,
                    k_base,
                    hw_out,
                    out_w,
                    in_h,
                    in_w,
                    in_channels,
                    kernel_w,
                    pad_h,
                    pad_w,
                    stride_h,
                    stride_w,
                    dilation_h,
                    dilation_w,
                )
                if valid != 0:
                    shm_u4[tid] = S.amdgpu.raw_buffer_load_x4(
                        rsrc_input, input_base, input_off, 0
                    )
                else:
                    shm_u4[tid] = zero_u4
            else:
                shm_u4[tid] = zero_u4
        else:
            b_idx = tid - GROUP_M
            if b_idx < GROUP_N:
                b_col = group_n * GROUP_N + b_idx
                if b_col < out_channels:
                    shm_u4[GROUP_M + b_idx] = S.amdgpu.raw_buffer_load_x4(
                        rsrc_weight, b_col * gemm_k * 2, k_base * 2, 0
                    )
                else:
                    shm_u4[GROUP_M + b_idx] = zero_u4
        S.syncthreads()

        row_local = wid * MFMA_M + lane_row
        col_local = lane_row
        a_pair_u32 = shm_u2[row_local * 2 + lane_half]
        b_pair_u32 = shm_u2[(GROUP_M + col_local) * 2 + lane_half]
        frag_a = S.view(a_pair_u32, S.Tensor((4,), S.bf16))
        frag_b = S.view(b_pair_u32, S.Tensor((4,), S.bf16))
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(frag_a, frag_b, acc)
        S.syncthreads()

    tile_row_base = group_m * GROUP_M + wid * MFMA_M
    tile_col_base = group_n * GROUP_N
    for acc_idx in S.range(16):
        col_out = tile_col_base + (lane % 32)
        row_out = (
            tile_row_base
            + 8 * (acc_idx // 4)
            + 4 * (lane // 32)
            + (acc_idx % 4)
        )
        if row_out < gemm_m and col_out < out_channels:
            batch = row_out // hw_out
            hw_idx = row_out % hw_out
            out_tensor[batch, col_out, hw_idx] = S.convert(acc[acc_idx], S.bf16)


def conv_transpose2d_igemm(
    input: torch.Tensor,
    weight_ohwi: torch.Tensor,
    input_nhwc: torch.Tensor,
    out: torch.Tensor,
    *,
    stride_h: int = 1,
    stride_w: int = 1,
    dilation_h: int = 1,
    dilation_w: int = 1,
):
    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, kernel_h, kernel_w, weight_in_channels = weight_ohwi.shape

    if weight_in_channels != in_channels:
        raise ValueError(
            f"weight in_channels mismatch: expected {in_channels}, got {weight_in_channels}"
        )
    if input.dtype != torch.bfloat16 or weight_ohwi.dtype != torch.bfloat16 or input_nhwc.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise TypeError("all tensors must be bf16")
    if not input.is_cuda or not weight_ohwi.is_cuda or not input_nhwc.is_cuda or not out.is_cuda:
        raise ValueError("all tensors must be CUDA tensors")

    out_h = (in_h - 1) * stride_h + dilation_h * (kernel_h - 1) + 1
    out_w = (in_w - 1) * stride_w + dilation_w * (kernel_w - 1) + 1
    if tuple(out.shape) != (batch_size, out_channels, out_h, out_w):
        raise ValueError(
            f"out must have shape {(batch_size, out_channels, out_h, out_w)} (got {tuple(out.shape)})"
        )
    if tuple(input_nhwc.shape) != (batch_size, in_h, in_w, in_channels):
        raise ValueError(
            f"input_nhwc must have shape {(batch_size, in_h, in_w, in_channels)} (got {tuple(input_nhwc.shape)})"
        )
    if in_channels % MFMA_K != 0:
        raise ValueError(f"in_channels must be divisible by {MFMA_K} (got {in_channels})")
    gemm_k = in_channels * kernel_h * kernel_w
    if gemm_k % MFMA_K != 0:
        raise ValueError(f"gemm_k must be divisible by {MFMA_K} (got {gemm_k})")

    grid_transpose, block_transpose = _transpose_input_launch_config(
        batch_size, in_channels, in_h, in_w
    )
    _transpose_input_nchw_to_nhwc_kernel[lambda: (grid_transpose, block_transpose)](
        input,
        input_nhwc,
        batch_size,
        in_channels,
        in_h,
        in_w,
    )

    grid, block = _igemm_launch_config(batch_size, out_channels, out_h, out_w)
    _conv_transpose2d_igemm_kernel[lambda: (grid, block)](
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
        kernel_h - 1,
        kernel_w - 1,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
    )
    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if stride != 1 or padding != 0 or output_padding != 0 or groups != 1:
            raise NotImplementedError(
                "This optimized kernel supports only stride=1, padding=0, output_padding=0, groups=1."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias_param", None)

        self.register_buffer(
            "_input_nhwc_workspace", torch.empty(0, dtype=torch.bfloat16), persistent=False
        )
        self.register_buffer(
            "_output_workspace", torch.empty(0, dtype=torch.bfloat16), persistent=False
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_param, -bound, bound)

    def _ensure_workspace(self, x_bf16: torch.Tensor):
        batch_size, in_channels, in_h, in_w = x_bf16.shape
        nhwc_shape = (batch_size, in_h, in_w, in_channels)
        out_h = in_h + self.kernel_size - 1
        out_w = in_w + self.kernel_size - 1
        out_shape = (batch_size, self.out_channels, out_h, out_w)

        if (
            self._input_nhwc_workspace.device != x_bf16.device
            or tuple(self._input_nhwc_workspace.shape) != nhwc_shape
        ):
            self._input_nhwc_workspace = torch.empty(
                nhwc_shape, dtype=torch.bfloat16, device=x_bf16.device
            )
        if (
            self._output_workspace.device != x_bf16.device
            or tuple(self._output_workspace.shape) != out_shape
        ):
            self._output_workspace = torch.empty(
                out_shape, dtype=torch.bfloat16, device=x_bf16.device
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bf16 = x.contiguous()
        original_dtype = x_bf16.dtype
        if x_bf16.dtype != torch.bfloat16:
            x_bf16 = x_bf16.to(torch.bfloat16)

        self._ensure_workspace(x_bf16)
        weight_ohwi = (
            self.weight
            .to(dtype=torch.bfloat16)
            .permute(1, 2, 3, 0)
            .flip(1, 2)
            .contiguous()
        )
        out = conv_transpose2d_igemm(
            x_bf16,
            weight_ohwi,
            self._input_nhwc_workspace,
            self._output_workspace,
        )

        if self.bias_param is not None:
            out = out + self.bias_param.to(dtype=torch.bfloat16).view(1, -1, 1, 1)
        if original_dtype != torch.bfloat16:
            out = out.to(original_dtype)
        return out


def get_inputs():
    x = torch.rand(batch_size, in_channels, height_in, width_in)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
