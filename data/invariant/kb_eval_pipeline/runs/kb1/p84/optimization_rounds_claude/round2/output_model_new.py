import torch
import torch.nn as nn
import substrate
import substrate.language as S

# MFMA configuration for MI300
WARP_SIZE = 64
NUM_WARPS = 4
MFMA_M = 32
MFMA_N = 32
MFMA_K = 8
MFMA_ACC_SIZE = 16

# Block tile configuration: 128x128 with 2x2 warp grid
# Each warp handles 64x64 tile as 2x2 array of 32x32 MFMA tiles
BLOCK_M = 128
BLOCK_N = 128
THREADS = WARP_SIZE * NUM_WARPS

# Wave repeat for MFMA tiles within warp (2x2)
WAVE_REPEAT_M = 2
WAVE_REPEAT_N = 2
WARP_TILE_M = WAVE_REPEAT_M * MFMA_M  # 64
WARP_TILE_N = WAVE_REPEAT_N * MFMA_N  # 64

# Split-K configuration
SPLIT_K_SLICES = 2

# Problem dimensions (will be overridden by ModelNew parameters)
BATCH_SIZE = 4
IN_CHANNELS = 32
IN_H = 64
IN_W = 64
OUT_CHANNELS = 64
KERNEL_H = 3
KERNEL_W = 3
KERNEL_SIZE = KERNEL_H * KERNEL_W  # 9

# Compute output dimensions and workspace size
OUT_H = IN_H - KERNEL_H + 1  # 62
OUT_W = IN_W - KERNEL_W + 1  # 62
GEMM_M = BATCH_SIZE * OUT_H * OUT_W
GEMM_N = OUT_CHANNELS
WORKSPACE_SIZE = GEMM_M * GEMM_N


@substrate.jit
def conv2d_splitk_kernel(
    X: S.Tensor((BATCH_SIZE, IN_CHANNELS, IN_H, IN_W), S.bf16),
    W: S.Tensor((OUT_CHANNELS, IN_CHANNELS, KERNEL_H, KERNEL_W), S.bf16),
    Workspace: S.Tensor((WORKSPACE_SIZE,), S.f32),
    gemm_m: S.i32,
    gemm_n: S.i32,
    in_channels: S.i32,
    kernel_h: S.i32,
    kernel_w: S.i32,
    out_h: S.i32,
    out_w: S.i32,
    c_per_split: S.i32,
    workspace_range_bytes: S.i64,
):
    """Split-K Conv2D kernel with MFMA tiling.

    Block decomposition: linear_block_id = tile_block_id * SPLIT_K_SLICES + split_k_id
    - tile_block_id selects the same (group_m, group_n) output tile
    - split_k_id only chooses which K-slice to process
    """
    # Block and thread identification
    linear_block_id = S.block_id(0)

    # Block decomposition invariant
    tile_block_id = linear_block_id // SPLIT_K_SLICES
    split_k_id = linear_block_id % SPLIT_K_SLICES

    n_blocks = (gemm_n + BLOCK_N - 1) // BLOCK_N
    block_m = tile_block_id // n_blocks
    block_n = tile_block_id % n_blocks

    tid = S.thread_id(0)
    wid = tid // WARP_SIZE
    lane = tid % WARP_SIZE

    # Warp position in 2x2 grid
    warp_row = wid // 2
    warp_col = wid % 2

    # MFMA fragment layout invariants
    lane_col = lane % MFMA_N
    lane_k_base = (lane // MFMA_N) * 4  # Each lane holds 4 bf16 elements

    # Block base coordinates
    group_m_base = block_m * BLOCK_M
    group_n_base = block_n * BLOCK_N

    # Warp base coordinates (64x64 tile)
    warp_m_base = group_m_base + warp_row * WARP_TILE_M
    warp_n_base = group_n_base + warp_col * WARP_TILE_N

    # Channel partition for split-K
    c_start = split_k_id * c_per_split
    c_end = S.min(in_channels, c_start + c_per_split)

    kernel_area = kernel_h * kernel_w

    # Accumulators for 2x2 MFMA tiles (each 32x32)
    # acc[tm, tn] corresponds to subtile at (tm*32, tn*32) within the 64x64 warp tile
    acc = S.make_local((WAVE_REPEAT_M, WAVE_REPEAT_N, MFMA_ACC_SIZE), S.f32)

    # Initialize accumulators to zero
    for tm in S.range(WAVE_REPEAT_M):
        for tn in S.range(WAVE_REPEAT_N):
            for acc_idx in S.range(MFMA_ACC_SIZE):
                acc[tm, tn, acc_idx] = S.convert(0.0, S.f32)

    # Process K dimension in MFMA-sized chunks
    # K = sum over (c, kh, kw) where c is in the split's channel range
    # MFMA_K = 8: we need 8 K elements per instruction, with each lane holding 4 bf16
    k_total = (c_end - c_start) * kernel_area

    # Process K in chunks of 8 (for bf16 MFMA)
    # For Conv2D, K = in_channels * kernel_h * kernel_w
    # We flatten to K-major order: (c, kh, kw) -> k_idx = c * kernel_area + kh * kernel_w + kw
    for k_base in S.range(0, k_total, MFMA_K):
        # Build A fragment (input) - each lane holds 4 bf16 elements
        # For A: lane_col determines row, lane_k determines K element
        a_frag = S.make_local((WAVE_REPEAT_M, 4), S.bf16)
        for tm in S.range(WAVE_REPEAT_M):
            for e in S.range(4):
                # K index for this element (within the chunk of 8)
                k_offset = lane_k_base + e
                k_idx = k_base + k_offset

                # M coordinate within the 64x64 warp tile
                m_local = tm * MFMA_M + lane_col

                # Global M coordinate
                m = warp_m_base + m_local

                if m < gemm_m and k_idx < k_total:
                    # Convert linear M to (batch, oh, ow)
                    batch_idx = m // (out_h * out_w)
                    hw_rem = m % (out_h * out_w)
                    oh_idx = hw_rem // out_w
                    ow_idx = hw_rem % out_w

                    # Convert linear K to (c, kh, kw)
                    c = c_start + k_idx // kernel_area
                    k_rem = k_idx % kernel_area
                    kh = k_rem // kernel_w
                    kw = k_rem % kernel_w

                    # Input spatial position
                    ih = oh_idx + kh
                    iw = ow_idx + kw

                    if ih < IN_H and iw < IN_W and c < in_channels:
                        a_frag[tm, e] = X[batch_idx, c, ih, iw]
                    else:
                        a_frag[tm, e] = S.convert(0.0, S.bf16)
                else:
                    a_frag[tm, e] = S.convert(0.0, S.bf16)

        # Build B fragment (weights) - each lane holds 4 bf16 elements
        # For B: lane_col determines column, lane_k determines K element
        b_frag = S.make_local((WAVE_REPEAT_N, 4), S.bf16)
        for tn in S.range(WAVE_REPEAT_N):
            for e in S.range(4):
                k_offset = lane_k_base + e
                k_idx = k_base + k_offset

                # N coordinate within the 64x64 warp tile
                n_local = tn * MFMA_N + lane_col

                # Global N coordinate
                n = warp_n_base + n_local

                if n < gemm_n and k_idx < k_total:
                    # Convert linear K to (c, kh, kw)
                    c = c_start + k_idx // kernel_area
                    k_rem = k_idx % kernel_area
                    kh = k_rem // kernel_w
                    kw = k_rem % kernel_w

                    b_frag[tn, e] = W[n, c, kh, kw]
                else:
                    b_frag[tn, e] = S.convert(0.0, S.bf16)

        # Execute MFMA for each subtile pair
        for tm in S.range(WAVE_REPEAT_M):
            for tn in S.range(WAVE_REPEAT_N):
                # Pack fragments for MFMA
                a_packed = S.view(a_frag[tm], S.Tensor((1, 4, 1), S.bf16))
                b_packed = S.view(b_frag[tn], S.Tensor((1, 4, 1), S.bf16))

                acc_view = S.view(acc[tm, tn], S.Tensor((MFMA_ACC_SIZE,), S.f32))
                acc_view = S.amdgpu.mfma_32x32x8_bf16_f32(a_packed[0], b_packed[0], acc_view)

                # Copy back to accumulator
                for acc_idx in S.range(MFMA_ACC_SIZE):
                    acc[tm, tn, acc_idx] = acc_view[acc_idx]

    # Create buffer resource for atomic operations
    workspace_rsrc = S.amdgpu.make_rsrc(Workspace, workspace_range_bytes)
    zero_u32 = S.convert(0, S.i32)

    # Writeback with atomic add to workspace
    # Workspace is in GEMM-major layout: linear_idx = row * gemm_n + col
    for tm in S.range(WAVE_REPEAT_M):
        for tn in S.range(WAVE_REPEAT_N):
            tile_row_base = group_m_base + warp_row * WARP_TILE_M + tm * MFMA_M
            tile_col_base = group_n_base + warp_col * WARP_TILE_N + tn * MFMA_N

            for acc_idx in S.range(MFMA_ACC_SIZE):
                # MFMA accumulator layout:
                # col = tile_col_base + (lane % 32)
                # row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
                col = tile_col_base + lane_col
                row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // MFMA_N) + (acc_idx % 4)

                if row < gemm_m and col < gemm_n:
                    # Linear index in GEMM-major workspace
                    linear_idx = row * gemm_n + col

                    # Byte offset for f32 elements (4 bytes each)
                    byte_offset = linear_idx * 4

                    # Atomic add to workspace
                    S.amdgpu.buffer_atomic_add_f32(acc[tm, tn, acc_idx], workspace_rsrc, byte_offset, zero_u32, 0)


@substrate.jit
def finalize_kernel(
    Workspace: S.Tensor((WORKSPACE_SIZE,), S.f32),
    Output: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_H, OUT_W), S.bf16),
    gemm_m: S.i32,
    gemm_n: S.i32,
    out_h: S.i32,
    out_w: S.i32,
    total_elements: S.i32,
    grid_stride: S.i32,
):
    """Convert fp32 workspace to bf16 NCHW output using grid-stride loop."""
    tid = S.thread_id(0)
    block_id = S.block_id(0)

    # Grid-stride loop: each thread handles indices [start, start+stride, start+2*stride, ...)
    start_idx = block_id * THREADS + tid

    idx = start_idx
    for _ in S.range((total_elements + grid_stride - 1) // grid_stride):
        if idx < total_elements:
            # GEMM-major to NCHW
            row = idx // gemm_n
            col = idx % gemm_n

            # row = batch * out_h * out_w + hw_idx
            batch_idx = row // (out_h * out_w)
            hw_rem = row % (out_h * out_w)
            oh_idx = hw_rem // out_w
            ow_idx = hw_rem % out_w

            # col = out_channel
            out_channel = col

            if batch_idx < BATCH_SIZE and out_channel < OUT_CHANNELS:
                val = Workspace[idx]
                Output[batch_idx, out_channel, oh_idx, ow_idx] = S.convert(val, S.bf16)

        idx = idx + grid_stride


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                                stride=stride, padding=padding, bias=bias)
        self._workspace = None
        self._workspace_ptr = None

    def forward(self, x):
        batch_size, in_channels, in_h, in_w = x.shape
        out_channels = self.conv2d.out_channels
        kernel_h, kernel_w = self.conv2d.kernel_size
        stride = self.conv2d.stride[0]
        padding = self.conv2d.padding[0]

        # Compute output dimensions
        out_h = (in_h + 2 * padding - kernel_h) // stride + 1
        out_w = (in_w + 2 * padding - kernel_w) // stride + 1

        # GEMM dimensions
        gemm_m = batch_size * out_h * out_w
        gemm_n = out_channels

        # Convert to bf16 for computation
        x_bf16 = x.to(dtype=torch.bfloat16)
        w_bf16 = self.conv2d.weight.to(dtype=torch.bfloat16)

        # Allocate workspace for fp32 partial sums
        workspace_size = gemm_m * gemm_n
        if self._workspace is None or self._workspace.size(0) < workspace_size:
            self._workspace = torch.zeros(workspace_size, device=x.device, dtype=torch.float32)
            self._workspace_ptr = self._workspace.data_ptr()
        elif self._workspace.data_ptr() != self._workspace_ptr:
            self._workspace_ptr = self._workspace.data_ptr()

        # Clear workspace
        self._workspace.zero_()

        # Compute launch configuration
        m_groups = (gemm_m + BLOCK_M - 1) // BLOCK_M
        n_groups = (gemm_n + BLOCK_N - 1) // BLOCK_N
        tile_blocks = m_groups * n_groups

        grid = (tile_blocks * SPLIT_K_SLICES, 1, 1)
        block = (THREADS, 1, 1)

        # Channel partition for split-K
        c_per_split = (in_channels + SPLIT_K_SLICES - 1) // SPLIT_K_SLICES

        # Compute workspace range in bytes
        workspace_range_bytes = workspace_size * 4  # f32 = 4 bytes

        # Launch split-K kernel
        conv2d_splitk_kernel[lambda: (grid, block)](
            x_bf16, w_bf16, self._workspace,
            gemm_m, gemm_n, in_channels, kernel_h, kernel_w, out_h, out_w,
            c_per_split, workspace_range_bytes
        )

        # Allocate output tensor
        y_bf16 = torch.empty((batch_size, out_channels, out_h, out_w),
                            device=x.device, dtype=torch.bfloat16)

        # Launch finalize kernel with grid-stride loop
        # Use enough blocks to saturate the GPU
        num_blocks = min((workspace_size + THREADS - 1) // THREADS, 256)
        finalize_grid = (num_blocks, 1, 1)
        grid_stride = num_blocks * THREADS
        finalize_kernel[lambda: (finalize_grid, block)](
            self._workspace, y_bf16, gemm_m, gemm_n, out_h, out_w,
            workspace_size, grid_stride
        )

        return y_bf16.to(dtype=torch.float32)
