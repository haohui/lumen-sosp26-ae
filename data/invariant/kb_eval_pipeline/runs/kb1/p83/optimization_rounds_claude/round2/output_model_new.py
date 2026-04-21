import torch
import torch.nn as nn
import substrate
import substrate.language as S


BATCH = 64
BATCH_SIZE = BATCH
IN_CHANNELS = 8
OUT_CHANNELS = 8
IN_H = 512
IN_W = 512
KERNEL_H = 3
KERNEL_W = 1
STRIDE = 1
PADDING = 0
DILATION = 1
GROUPS = 8
OUT_H = 510
OUT_W = 512

BLOCK_M = 128
BLOCK_N = 128
WARP_SIZE = 64
NUM_WARPS = 4
THREADS_PER_BLOCK = WARP_SIZE * NUM_WARPS

SPLIT_K_SLICES = 2
MFMA_ACC_SIZE = 16

GEMM_M = BATCH * OUT_CHANNELS * OUT_H
GEMM_N = OUT_W

INPUT0_SHAPE = (BATCH, IN_CHANNELS, IN_H, IN_W)
WEIGHT_SHAPE = (OUT_CHANNELS, 1, KERNEL_H, KERNEL_W)
OUTPUT_SHAPE = (BATCH, OUT_CHANNELS, OUT_H, OUT_W)
OUTPUT_TORCH_DTYPE = torch.bfloat16

KERNEL_AREA = KERNEL_H * KERNEL_W


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _launch_split_k():
    num_tiles_n = _ceil_div(GEMM_N, BLOCK_N)
    num_tiles_m = _ceil_div(GEMM_M, BLOCK_M)
    return lambda: ((num_tiles_n * SPLIT_K_SLICES, num_tiles_m, 1), (THREADS_PER_BLOCK, 1, 1))


def _launch_finalize():
    num_tiles_n = _ceil_div(GEMM_N, BLOCK_N)
    num_tiles_m = _ceil_div(GEMM_M, BLOCK_M)
    return lambda: ((num_tiles_n, num_tiles_m, 1), (THREADS_PER_BLOCK, 1, 1))


@substrate.jit
def split_k_conv2d_kernel(
    X: S.Pointer(S.bf16),
    W: S.Pointer(S.bf16),
    workspace: S.Pointer(S.f32),
):
    """Split-K Conv2D kernel with MFMA accumulator structure.

    For depthwise convolution with KERNEL_AREA=3, we use the MFMA accumulator
    layout (16 f32 values per thread) but compute element-wise since K=3 is
    too small for efficient MFMA_32x32x8 (which requires K>=8).
    """
    # Grid is (num_tiles_n * SPLIT_K_SLICES, num_tiles_m, 1)
    # blockIdx.x encodes (group_n, split_k_id), blockIdx.y = group_m
    bx = S.block_id(0)
    by = S.block_id(1)

    n_groups = GEMM_N // BLOCK_N
    group_n = bx // SPLIT_K_SLICES
    split_k_id = bx % SPLIT_K_SLICES
    group_m = by

    tid = S.thread_id(0)

    group_m_base = group_m * BLOCK_M
    group_n_base = group_n * BLOCK_N

    # Split-K partition over kernel spatial positions
    k_total = KERNEL_AREA
    k_per_split = (k_total + SPLIT_K_SLICES - 1) // SPLIT_K_SLICES
    k_start = split_k_id * k_per_split
    k_end = k_total if split_k_id == SPLIT_K_SLICES - 1 else (split_k_id + 1) * k_per_split

    # Create tensor views for NCHW layout
    X_tensor = S.make_tensor(X, S.bf16, S.make_layout((BATCH, IN_CHANNELS, IN_H, IN_W), (IN_CHANNELS * IN_H * IN_W, IN_H * IN_W, IN_W, 1)))
    W_tensor = S.make_tensor(W, S.bf16, S.make_layout((OUT_CHANNELS, 1, KERNEL_H, KERNEL_W), (KERNEL_H * KERNEL_W, KERNEL_W, KERNEL_W, 1)))

    workspace_size = GEMM_M * GEMM_N
    workspace_tensor = S.make_tensor(workspace, S.f32, S.make_layout((workspace_size,), (1,)))
    workspace_rsrc = S.amdgpu.make_rsrc(workspace_tensor, workspace_size * 4)

    # MFMA lane/warp layout
    lane = tid % 64
    lane_col = lane % 32
    lane_row = lane // 32

    wid = tid // 64
    warp_row = wid // 2
    warp_col = wid % 2

    # MFMA-style accumulators: 2x2 array of 32x32 tiles
    # Each acc[tm, tn] holds 16 f32 values (MFMA_ACC_SIZE)
    # Accumulator layout: row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
    acc = S.make_local((2, 2, MFMA_ACC_SIZE), S.f32)
    zero_f32 = S.convert(0.0, S.f32)
    zero_u32 = S.convert(0, S.u32)

    for tm in S.range(2):
        for tn in S.range(2):
            for acc_idx in S.range(MFMA_ACC_SIZE):
                acc[tm, tn, acc_idx] = zero_f32

    # Process K iterations (kernel positions)
    for k_iter in S.range(k_end - k_start):
        k_idx = k_start + k_iter
        kh = k_idx // KERNEL_W
        kw = k_idx % KERNEL_W

        # For each accumulator position, compute contribution
        for tm in S.range(2):
            tile_row_base = group_m_base + warp_row * 64 + tm * 32
            for tn in S.range(2):
                tile_col_base = group_n_base + warp_col * 64 + tn * 32

                for acc_idx in S.range(MFMA_ACC_SIZE):
                    # MFMA accumulator layout
                    row_local = 8 * (acc_idx // 4) + 4 * lane_row + (acc_idx % 4)
                    col_local = lane_col

                    row = tile_row_base + row_local
                    col = tile_col_base + col_local

                    if row < GEMM_M and col < GEMM_N:
                        # Decode row to (batch, out_channel, out_row)
                        batch = row // (OUT_CHANNELS * OUT_H)
                        rem = row % (OUT_CHANNELS * OUT_H)
                        out_channel = rem // OUT_H
                        out_row = rem % OUT_H
                        out_col = col

                        in_row = out_row + kh
                        in_channel = out_channel  # depthwise

                        if in_row >= 0 and in_row < IN_H:
                            x_val = X_tensor[batch, in_channel, in_row, out_col]
                            w_val = W_tensor[out_channel, 0, kh, kw]
                            x_f32 = S.convert(x_val, S.f32)
                            w_f32 = S.convert(w_val, S.f32)
                            acc[tm, tn, acc_idx] = acc[tm, tn, acc_idx] + x_f32 * w_f32

    # Writeback with buffer_atomic_add_f32 for split-K reduction
    for tm in S.range(2):
        tile_row_base = group_m_base + warp_row * 64 + tm * 32
        for tn in S.range(2):
            tile_col_base = group_n_base + warp_col * 64 + tn * 32

            for acc_idx in S.range(MFMA_ACC_SIZE):
                row_local = 8 * (acc_idx // 4) + 4 * lane_row + (acc_idx % 4)
                col_local = lane_col

                row = tile_row_base + row_local
                col = tile_col_base + col_local

                if row < GEMM_M and col < GEMM_N:
                    linear_idx = row * GEMM_N + col
                    byte_offset = linear_idx * 4
                    S.amdgpu.buffer_atomic_add_f32(acc[tm, tn, acc_idx], workspace_rsrc, byte_offset, zero_u32, 0)


@substrate.jit
def finalize_kernel(
    workspace: S.Pointer(S.f32),
    Y: S.Pointer(S.bf16),
):
    """Convert fp32 workspace to bf16 NCHW output."""
    # Grid is (num_tiles_n, num_tiles_m, 1)
    # blockIdx.x = group_n, blockIdx.y = group_m
    group_n = S.block_id(0)
    group_m = S.block_id(1)

    tid = S.thread_id(0)
    lane = tid % 64
    lane_col = lane % 32
    lane_row = lane // 32
    wid = tid // 64
    warp_row = wid // 2
    warp_col = wid % 2

    group_m_base = group_m * BLOCK_M
    group_n_base = group_n * BLOCK_N

    workspace_tensor = S.make_tensor(workspace, S.f32, S.make_layout((GEMM_M, GEMM_N), (GEMM_N, 1)))
    Y_tensor = S.make_tensor(Y, S.bf16, S.make_layout((BATCH, OUT_CHANNELS, OUT_H, OUT_W), (OUT_CHANNELS * OUT_H * OUT_W, OUT_H * OUT_W, OUT_W, 1)))

    for tm in S.range(2):
        tile_row_base = group_m_base + warp_row * 64 + tm * 32
        for tn in S.range(2):
            tile_col_base = group_n_base + warp_col * 64 + tn * 32

            for acc_idx in S.range(MFMA_ACC_SIZE):
                row_local = 8 * (acc_idx // 4) + 4 * lane_row + (acc_idx % 4)
                col_local = lane_col

                row = tile_row_base + row_local
                col = tile_col_base + col_local

                if row < GEMM_M and col < GEMM_N:
                    val = workspace_tensor[row, col]

                    batch = row // (OUT_CHANNELS * OUT_H)
                    rem = row % (OUT_CHANNELS * OUT_H)
                    out_channel = rem // OUT_H
                    out_row = rem % OUT_H
                    out_col = col

                    Y_tensor[batch, out_channel, out_row, out_col] = S.convert(val, S.bf16)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        kernel_size: int = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if kernel_size is None:
            kernel_size = KERNEL_H
        if out_channels is None:
            out_channels = in_channels

        self.conv2d = nn.Conv2d(
            IN_CHANNELS,
            OUT_CHANNELS,
            kernel_size=(KERNEL_H, KERNEL_W),
            stride=STRIDE,
            padding=PADDING,
            dilation=DILATION,
            groups=GROUPS,
            bias=bias,
        )
        self._weight_cache = None
        self._weight_cache_key = None

    def _supports_optimized_path(self, x: torch.Tensor) -> bool:
        return (
            tuple(x.shape) == INPUT0_SHAPE
            and self.conv2d.in_channels == IN_CHANNELS
            and self.conv2d.out_channels == IN_CHANNELS
            and self.conv2d.kernel_size == (KERNEL_H, KERNEL_W)
            and self.conv2d.stride == (STRIDE, STRIDE)
            and self.conv2d.padding == (PADDING, PADDING)
            and self.conv2d.dilation == (DILATION, DILATION)
            and self.conv2d.groups == GROUPS
            and self.conv2d.bias is None
        )

    def _get_cached_weight(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv2d.weight.detach()
        key = (weight.data_ptr(), x.device.type, x.device.index, x.dtype, weight.is_contiguous())
        if self._weight_cache_key != key:
            self._weight_cache = weight.to(device=x.device, dtype=torch.bfloat16).contiguous()
            self._weight_cache_key = key
        return self._weight_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._supports_optimized_path(x):
            raise RuntimeError(
                f"This optimized kernel only supports input shape {INPUT0_SHAPE}, "
                f"Conv2d({IN_CHANNELS}, {IN_CHANNELS}, ({KERNEL_H}, {KERNEL_W}), "
                f"stride={STRIDE}, padding={PADDING}, dilation={DILATION}, groups={GROUPS}, bias=False)."
            )

        x0 = x.to(dtype=torch.bfloat16).contiguous()
        w = self._get_cached_weight(x0)

        workspace = torch.zeros((GEMM_M, GEMM_N), device=x0.device, dtype=torch.float32)

        split_k_conv2d_kernel[_launch_split_k()](x0, w, workspace)

        y = torch.empty(OUTPUT_SHAPE, device=x0.device, dtype=OUTPUT_TORCH_DTYPE)
        finalize_kernel[_launch_finalize()](workspace, y)

        return y
