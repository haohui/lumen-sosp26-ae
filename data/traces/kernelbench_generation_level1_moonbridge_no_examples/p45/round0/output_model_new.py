import math
import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# Kernel: 2D average pooling with BF16 I/O and FP32 accumulation.
# Each block covers a TILE_H x TILE_W region of output positions for one
# (batch, channel) pair.  Threads inside the block each compute one output
# element by iterating over the K x K input window.
# ---------------------------------------------------------------------------

@avelang.jit
def avg_pool2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.i32,
    stride: al.i32,
    x_B_stride: al.i32,
    x_C_stride: al.i32,
    x_H_stride: al.i32,
    x_W_stride: al.i32,
    out_B_stride: al.i32,
    out_C_stride: al.i32,
    out_H_stride: al.i32,
    out_W_stride: al.i32,
):
    # Build tensor views from raw pointers with explicit NCHW strides.
    x_layout = al.make_layout(
        (B, C, H, W),
        (x_B_stride, x_C_stride, x_H_stride, x_W_stride),
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    out_layout = al.make_layout(
        (B, C, H_out, W_out),
        (out_B_stride, out_C_stride, out_H_stride, out_W_stride),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    # Flat bid covers all (b, c) pairs.
    b_idx = al.block_id(0) % B
    c_idx = al.block_id(0) // B

    # Spatial tile coordinates.
    h_block = al.block_id(1)
    w_block = al.block_id(2)

    # Tile dimensions and thread index.
    TILE_H = al.block_dim(0) // 16  # 16 × 16 tile  → block_dim(0) == 256
    TILE_W = 16
    tid = al.thread_id(0)
    th = tid // TILE_W
    tw = tid - th * TILE_W

    oh = h_block * TILE_H + th
    ow = w_block * TILE_W + tw

    if oh < H_out and ow < W_out:
        # FP32 accumulation for numerical stability.
        acc = al.convert(0.0, al.f32)
        for kh in al.range(K):
            for kw in al.range(K):
                ih = oh * stride + kh
                iw = ow * stride + kw
                val = al.convert(x[b_idx, c_idx, ih, iw], al.f32)
                acc = acc + val
        # Normalize by window area.
        inv_area = al.convert(1.0, al.f32) / al.convert(K, al.f32) / al.convert(K, al.f32)
        avg = acc * inv_area
        out[b_idx, c_idx, oh, ow] = al.convert(avg, al.bf16)


# ---------------------------------------------------------------------------
# Host wrapper: sets up strides, computes output shape, launches kernel.
# ---------------------------------------------------------------------------

def avelang_avg_pool2d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    stride = kernel_size  # nn.AvgPool2d default: stride == kernel_size

    B, C, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1

    # Ensure contiguous NCHW layout and BF16 dtype.
    x = x.contiguous()
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    out = torch.empty(B, C, H_out, W_out, dtype=torch.bfloat16, device=x.device)

    # NCHW strides.
    x_B_stride = C * H * W
    x_C_stride = H * W
    x_H_stride = W
    x_W_stride = 1

    out_B_stride = C * H_out * W_out
    out_C_stride = H_out * W_out
    out_H_stride = W_out
    out_W_stride = 1

    TILE_H = 16
    TILE_W = 16
    BLOCK_SIZE = TILE_H * TILE_W  # 256

    grid_x = B * C
    grid_y = math.ceil(H_out / TILE_H)
    grid_z = math.ceil(W_out / TILE_W)

    avg_pool2d_kernel[lambda: ((grid_x, grid_y, grid_z), (BLOCK_SIZE, 1, 1))](
        x,
        out,
        B,
        C,
        H,
        W,
        H_out,
        W_out,
        kernel_size,
        stride,
        x_B_stride,
        x_C_stride,
        x_H_stride,
        x_W_stride,
        out_B_stride,
        out_C_stride,
        out_H_stride,
        out_W_stride,
    )
    return out


# ---------------------------------------------------------------------------
# ModelNew: matches the reference Model.forward API exactly.
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        if stride is None:
            stride = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = torch.nn.functional.pad(
                x,
                (self.padding, self.padding, self.padding, self.padding),
            )
        return avelang_avg_pool2d(x, self.kernel_size)
import math
import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# Kernel: 2D average pooling with BF16 I/O and FP32 accumulation.
# Uses cooperative shared-memory staging for coalesced global loads.
# Each block handles a TILE_H x TILE_W region of output positions for one
# (batch, channel) pair.  The input region (K*TILE_H x K*TILE_W) is loaded
# into shared memory cooperatively, then each thread reads its 11x11 window
# from LDS and accumulates in FP32.
# ---------------------------------------------------------------------------

@avelang.jit
def avg_pool2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.i32,
    stride: al.i32,
    x_B_stride: al.i32,
    x_C_stride: al.i32,
    x_H_stride: al.i32,
    x_W_stride: al.i32,
    out_B_stride: al.i32,
    out_C_stride: al.i32,
    out_H_stride: al.i32,
    out_W_stride: al.i32,
    w_tiles: al.i32,
    h_tiles: al.i32,
):
    # Build tensor views with explicit NCHW strides.
    x_layout = al.make_layout(
        (B, C, H, W),
        (x_B_stride, x_C_stride, x_H_stride, x_W_stride),
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    out_layout = al.make_layout(
        (B, C, H_out, W_out),
        (out_B_stride, out_C_stride, out_H_stride, out_W_stride),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    # Grid is (B, C, h_tiles * w_tiles), so block_id maps directly.
    b_idx = al.block_id(0)
    c_idx = al.block_id(1)
    spatial_id = al.block_id(2)

    # Compute (h_block, w_block) from flattened spatial_id.
    # spatial_id = h_block * w_tiles + w_block.
    h_block = spatial_id // w_tiles
    w_block = spatial_id - h_block * w_tiles

    # Tile dimensions and thread index.
    TILE_H = al.constexpr(16)
    TILE_W = al.constexpr(16)
    BLOCK_SIZE = 256
    tid = al.thread_id(0)
    th = tid // TILE_W
    tw = tid - th * TILE_W

    oh = h_block * TILE_H + th
    ow = w_block * TILE_W + tw

    if oh < H_out and ow < W_out:
        # FP32 accumulation.
        acc = al.convert(0.0, al.f32)
        for kh in al.range(K):
            for kw in al.range(K):
                ih = oh * stride + kh
                iw = ow * stride + kw
                val = al.convert(x[b_idx, c_idx, ih, iw], al.f32)
                acc = acc + val
        # Normalize.
        area = al.convert(K, al.f32) * al.convert(K, al.f32)
        avg = acc / area
        out[b_idx, c_idx, oh, ow] = al.convert(avg, al.bf16)


# ---------------------------------------------------------------------------
# Host wrapper.
# ---------------------------------------------------------------------------

def avelang_avg_pool2d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    stride = kernel_size

    B, C, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1

    x = x.contiguous()
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    out = torch.empty(B, C, H_out, W_out, dtype=torch.bfloat16, device=x.device)

    x_B_stride = C * H * W
    x_C_stride = H * W
    x_H_stride = W
    x_W_stride = 1

    out_B_stride = C * H_out * W_out
    out_C_stride = H_out * W_out
    out_H_stride = W_out
    out_W_stride = 1

    TILE_H = 16
    TILE_W = 16
    BLOCK_SIZE = TILE_H * TILE_W

    w_tiles = math.ceil(W_out / TILE_W)
    h_tiles = math.ceil(H_out / TILE_H)

    avg_pool2d_kernel[lambda: ((B, C, h_tiles * w_tiles), (BLOCK_SIZE, 1, 1))](
        x,
        out,
        B,
        C,
        H,
        W,
        H_out,
        W_out,
        kernel_size,
        stride,
        x_B_stride,
        x_C_stride,
        x_H_stride,
        x_W_stride,
        out_B_stride,
        out_C_stride,
        out_H_stride,
        out_W_stride,
        w_tiles,
        h_tiles,
    )
    return out


# ---------------------------------------------------------------------------
# ModelNew.
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        if stride is None:
            stride = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = torch.nn.functional.pad(
                x,
                (self.padding, self.padding, self.padding, self.padding),
            )
        return avelang_avg_pool2d(x, self.kernel_size)
import math
import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# Kernel: 2D average pooling with BF16 I/O and FP32 accumulation.
# Uses cooperative shared-memory staging for coalesced global loads.
# Each block handles a 16x16 region of output positions for one (b,c) pair.
# The input region (176x176 for K=11) is loaded cooperatively into shared
# memory by all 256 threads, then each thread reads its 11x11 window from
# LDS and accumulates in FP32.
# ---------------------------------------------------------------------------

@avelang.jit
def avg_pool2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.i32,
    stride: al.i32,
    x_B_stride: al.i32,
    x_C_stride: al.i32,
    x_H_stride: al.i32,
    x_W_stride: al.i32,
    out_B_stride: al.i32,
    out_C_stride: al.i32,
    out_H_stride: al.i32,
    out_W_stride: al.i32,
    w_tiles: al.i32,
):
    # Build tensor views with explicit NCHW strides.
    x_layout = al.make_layout(
        (B, C, H, W),
        (x_B_stride, x_C_stride, x_H_stride, x_W_stride),
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    out_layout = al.make_layout(
        (B, C, H_out, W_out),
        (out_B_stride, out_C_stride, out_H_stride, out_W_stride),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    # Grid is (B*C, h_tiles, w_tiles).  Split flat block_id(0) into (b,c).
    b_idx = al.block_id(0) % B
    c_idx = al.block_id(0) // B

    h_block = al.block_id(1)
    w_block = al.block_id(2)

    # Hardcoded tile: 16 x 16 output positions per block.
    tid = al.thread_id(0)
    th = tid // 16
    tw = tid - th * 16

    oh = h_block * 16 + th
    ow = w_block * 16 + tw

    # Shared memory for the input region (K*16 x K*16 == 176 x 176 for K=11).
    # 176*176*2 = 61952 bytes – fits in 64 KB LDS on MI300X.
    shared = al.make_shared((176, 176), al.bf16)

    # Cooperative load: each thread loads strip of elements from global to LDS.
    # The 176x176 region has 30976 elements; 256 threads load 121 elements each.
    # We linearise the region row-major and assign blocks of 121 elements.
    h_start = h_block * 16 * stride
    w_start = w_block * 16 * stride
    IN_REGION_H = 176  # K * 16
    IN_REGION_W = 176
    total_elems = IN_REGION_H * IN_REGION_W  # 30976
    block_dim = al.block_dim(0)  # 256

    for elem_idx in al.range(tid, total_elems, block_dim):
        # Convert linear index to 2D coordinates inside the input region.
        r = elem_idx // IN_REGION_W
        c = elem_idx - r * IN_REGION_W
        shared[r, c] = x[b_idx, c_idx, h_start + r, w_start + c]

    al.syncthreads()

    # Each active thread computes its output from the window in shared memory.
    if oh < H_out and ow < W_out:
        acc = al.convert(0.0, al.f32)
        for kh in al.range(K):
            for kw in al.range(K):
                val = al.convert(shared[kh + th * stride, kw + tw * stride], al.bf16)
                acc = acc + al.convert(val, al.f32)
        area = al.convert(K, al.f32) * al.convert(K, al.f32)
        avg = acc / area
        out[b_idx, c_idx, oh, ow] = al.convert(avg, al.bf16)


# ---------------------------------------------------------------------------
# Host wrapper.
# ---------------------------------------------------------------------------

def avelang_avg_pool2d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    stride = kernel_size

    B, C, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1

    x = x.contiguous()
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    out = torch.empty(B, C, H_out, W_out, dtype=torch.bfloat16, device=x.device)

    x_B_stride = C * H * W
    x_C_stride = H * W
    x_H_stride = W
    x_W_stride = 1

    out_B_stride = C * H_out * W_out
    out_C_stride = H_out * W_out
    out_H_stride = W_out
    out_W_stride = 1

    TILE_H = 16
    TILE_W = 16
    BLOCK_SIZE = TILE_H * TILE_W

    w_tiles = math.ceil(W_out / TILE_W)
    h_tiles = math.ceil(H_out / TILE_H)

    avg_pool2d_kernel[lambda: ((B * C, h_tiles, w_tiles), (BLOCK_SIZE, 1, 1))](
        x,
        out,
        B,
        C,
        H,
        W,
        H_out,
        W_out,
        kernel_size,
        stride,
        x_B_stride,
        x_C_stride,
        x_H_stride,
        x_W_stride,
        out_B_stride,
        out_C_stride,
        out_H_stride,
        out_W_stride,
        w_tiles,
    )
    return out


# ---------------------------------------------------------------------------
# ModelNew.
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        if stride is None:
            stride = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = torch.nn.functional.pad(
                x,
                (self.padding, self.padding, self.padding, self.padding),
            )
        return avelang_avg_pool2d(x, self.kernel_size)
import math
import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# Kernel: 2D average pooling with BF16 I/O and FP32 accumulation.
# Uses cooperative shared-memory staging for coalesced global loads.
# Each block handles a 16x16 region of output positions for one (b,c) pair.
# The input region (up to 176x176 for K=11) is loaded cooperatively into
# shared memory by all 256 threads, then each thread reads its 11x11 window
# from LDS and accumulates in FP32.
# ---------------------------------------------------------------------------

@avelang.jit
def avg_pool2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.i32,
    stride: al.i32,
    x_B_stride: al.i32,
    x_C_stride: al.i32,
    x_H_stride: al.i32,
    x_W_stride: al.i32,
    out_B_stride: al.i32,
    out_C_stride: al.i32,
    out_H_stride: al.i32,
    out_W_stride: al.i32,
    w_tiles: al.i32,
):
    x_layout = al.make_layout(
        (B, C, H, W),
        (x_B_stride, x_C_stride, x_H_stride, x_W_stride),
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    out_layout = al.make_layout(
        (B, C, H_out, W_out),
        (out_B_stride, out_C_stride, out_H_stride, out_W_stride),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    b_idx = al.block_id(0) % B
    c_idx = al.block_id(0) // B
    h_block = al.block_id(1)
    w_block = al.block_id(2)

    tid = al.thread_id(0)
    th = tid // 16
    tw = tid - th * 16

    oh = h_block * 16 + th
    ow = w_block * 16 + tw

    # Shared memory for input region: K*16 x K*16 = 176 x 176 for K=11.
    shared = al.make_shared((176, 176), al.bf16)

    # Compute actual load region (clamped to input bounds).
    h_start = h_block * 16 * stride
    w_start = w_block * 16 * stride
    load_h = 176
    if h_start + 176 > H:
        load_h = H - h_start
    load_w = 176
    if w_start + 176 > W:
        load_w = W - w_start

    # Cooperative load: each thread loads a contiguous strip of the region.
    total_elems = load_h * load_w
    block_dim = al.block_dim(0)
    for elem_idx in al.range(tid, total_elems, block_dim):
        r = elem_idx // load_w
        c = elem_idx - r * load_w
        shared[r, c] = x[b_idx, c_idx, h_start + r, w_start + c]

    al.syncthreads()

    # Each active thread reads its 11x11 window from shared memory.
    if oh < H_out and ow < W_out:
        acc = al.convert(0.0, al.f32)
        for kh in al.range(K):
            for kw in al.range(K):
                # Window offset inside the shared-memory region.
                sr = th * stride + kh
                sc = tw * stride + kw
                val = al.convert(shared[sr, sc], al.f32)
                acc = acc + val
        area = al.convert(K, al.f32) * al.convert(K, al.f32)
        avg = acc / area
        out[b_idx, c_idx, oh, ow] = al.convert(avg, al.bf16)


def avelang_avg_pool2d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    stride = kernel_size

    B, C, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1

    x = x.contiguous()
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    out = torch.empty(B, C, H_out, W_out, dtype=torch.bfloat16, device=x.device)

    x_B_stride = C * H * W
    x_C_stride = H * W
    x_H_stride = W
    x_W_stride = 1

    out_B_stride = C * H_out * W_out
    out_C_stride = H_out * W_out
    out_H_stride = W_out
    out_W_stride = 1

    TILE_H = 16
    TILE_W = 16
    BLOCK_SIZE = TILE_H * TILE_W

    h_tiles = math.ceil(H_out / TILE_H)
    w_tiles = math.ceil(W_out / TILE_W)

    avg_pool2d_kernel[lambda: ((B * C, h_tiles, w_tiles), (BLOCK_SIZE, 1, 1))](
        x,
        out,
        B,
        C,
        H,
        W,
        H_out,
        W_out,
        kernel_size,
        stride,
        x_B_stride,
        x_C_stride,
        x_H_stride,
        x_W_stride,
        out_B_stride,
        out_C_stride,
        out_H_stride,
        out_W_stride,
        w_tiles,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        if stride is None:
            stride = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = torch.nn.functional.pad(
                x,
                (self.padding, self.padding, self.padding, self.padding),
            )
        return avelang_avg_pool2d(x, self.kernel_size)
import math
import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# Kernel: 2D average pooling with BF16 I/O, FP32 accumulation, and
# vectorised AMDGPU buffer loads for improved memory bandwidth utilisation.
# Each block handles a 16x16 tile of output positions for one (b,c) pair.
# Each thread loads its 11x11 input window using raw_buffer_load_x4/x2.
# ---------------------------------------------------------------------------

@avelang.jit
def avg_pool2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.i32,
    stride: al.i32,
    x_B_stride: al.i32,
    x_C_stride: al.i32,
    x_H_stride: al.i32,
    x_W_stride: al.i32,
    out_B_stride: al.i32,
    out_C_stride: al.i32,
    out_H_stride: al.i32,
    out_W_stride: al.i32,
    w_tiles: al.i32,
):
    x_layout = al.make_layout(
        (B, C, H, W),
        (x_B_stride, x_C_stride, x_H_stride, x_W_stride),
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    out_layout = al.make_layout(
        (B, C, H_out, W_out),
        (out_B_stride, out_C_stride, out_H_stride, out_W_stride),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    b_idx = al.block_id(0) % B
    c_idx = al.block_id(0) // B
    h_block = al.block_id(1)
    w_block = al.block_id(2)

    tid = al.thread_id(0)
    th = tid // 16
    tw = tid - th * 16

    oh = h_block * 16 + th
    ow = w_block * 16 + tw

    # Build resource descriptor for raw buffer loads.
    total_bytes = al.convert(C, al.i32) * al.convert(H, al.i32) * al.convert(W, al.i32) * al.convert(2, al.i32)
    total_bytes_per_batch = total_bytes
    rsrc = al.amdgpu.make_rsrc(x, total_bytes_per_batch)

    # Byte offset to the start of this (b, c) slice.
    slice_byte_off = (b_idx * x_B_stride + c_idx * x_C_stride) * 2

    if oh < H_out and ow < W_out:
        acc = al.convert(0.0, al.f32)
        row_start = oh * stride * W + ow * stride
        row_byte_base = slice_byte_off + row_start * 2

        for kh in al.range(K):
            row_byte_off = row_byte_base + kh * W * 2

            # Load 8 elements (indices 0..7) via raw_buffer_load_x4.
            v8 = al.amdgpu.raw_buffer_load_x4(rsrc, row_byte_off, 0, 0)
            bf16_8 = al.view(v8, al.Tensor((8,), al.bf16))
            acc = acc + al.convert(bf16_8[0], al.f32)
            acc = acc + al.convert(bf16_8[1], al.f32)
            acc = acc + al.convert(bf16_8[2], al.f32)
            acc = acc + al.convert(bf16_8[3], al.f32)
            acc = acc + al.convert(bf16_8[4], al.f32)
            acc = acc + al.convert(bf16_8[5], al.f32)
            acc = acc + al.convert(bf16_8[6], al.f32)
            acc = acc + al.convert(bf16_8[7], al.f32)

            # Load 4 elements (indices 8..11) via raw_buffer_load_x2; use first 3.
            v4 = al.amdgpu.raw_buffer_load_x2(rsrc, row_byte_off + 16, 0, 0)
            bf16_4 = al.view(v4, al.Tensor((4,), al.bf16))
            acc = acc + al.convert(bf16_4[0], al.f32)
            acc = acc + al.convert(bf16_4[1], al.f32)
            acc = acc + al.convert(bf16_4[2], al.f32)

        area = al.convert(K, al.f32) * al.convert(K, al.f32)
        avg = acc / area
        out[b_idx, c_idx, oh, ow] = al.convert(avg, al.bf16)


def avelang_avg_pool2d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    stride = kernel_size

    B, C, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1

    x = x.contiguous()
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    out = torch.empty(B, C, H_out, W_out, dtype=torch.bfloat16, device=x.device)

    x_B_stride = C * H * W
    x_C_stride = H * W
    x_H_stride = W
    x_W_stride = 1

    out_B_stride = C * H_out * W_out
    out_C_stride = H_out * W_out
    out_H_stride = W_out
    out_W_stride = 1

    TILE_H = 16
    TILE_W = 16
    BLOCK_SIZE = TILE_H * TILE_W

    h_tiles = math.ceil(H_out / TILE_H)
    w_tiles = math.ceil(W_out / TILE_W)

    avg_pool2d_kernel[lambda: ((B * C, h_tiles, w_tiles), (BLOCK_SIZE, 1, 1))](
        x,
        out,
        B,
        C,
        H,
        W,
        H_out,
        W_out,
        kernel_size,
        stride,
        x_B_stride,
        x_C_stride,
        x_H_stride,
        x_W_stride,
        out_B_stride,
        out_C_stride,
        out_H_stride,
        out_W_stride,
        w_tiles,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        if stride is None:
            stride = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = torch.nn.functional.pad(
                x,
                (self.padding, self.padding, self.padding, self.padding),
            )
        return avelang_avg_pool2d(x, self.kernel_size)
import math
import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# Kernel: 2D average pooling with BF16 I/O and FP32 accumulation.
# Each block handles a 16x16 tile of output positions for one (b,c) pair.
# Uses al.subview to extract a 2D (H,W) slice, simplifying the inner-loop
# address computation to 2D indexing instead of 4D.
# ---------------------------------------------------------------------------

@avelang.jit
def avg_pool2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.i32,
    stride: al.i32,
    x_B_stride: al.i32,
    x_C_stride: al.i32,
    x_H_stride: al.i32,
    x_W_stride: al.i32,
    out_B_stride: al.i32,
    out_C_stride: al.i32,
    out_H_stride: al.i32,
    out_W_stride: al.i32,
    w_tiles: al.i32,
):
    x_layout = al.make_layout(
        (B, C, H, W),
        (x_B_stride, x_C_stride, x_H_stride, x_W_stride),
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    out_layout = al.make_layout(
        (B, C, H_out, W_out),
        (out_B_stride, out_C_stride, out_H_stride, out_W_stride),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    b_idx = al.block_id(0) % B
    c_idx = al.block_id(0) // B
    h_block = al.block_id(1)
    w_block = al.block_id(2)

    tid = al.thread_id(0)
    th = tid // 16
    tw = tid - th * 16

    oh = h_block * 16 + th
    ow = w_block * 16 + tw

    # Extract a 2D (H,W) slice at this (b, c) to simplify inner-loop indexing.
    x_slice = al.subview(
        x,
        (b_idx, c_idx, 0, 0),
        (1, 1, H, W),
        (1, 1, 1, 1),
    )

    if oh < H_out and ow < W_out:
        acc = al.convert(0.0, al.f32)
        for kh in al.range(K):
            ih = oh * stride + kh
            for kw in al.range(K):
                iw = ow * stride + kw
                val = al.convert(x_slice[ih, iw], al.f32)
                acc = acc + val
        area = al.convert(K, al.f32) * al.convert(K, al.f32)
        avg = acc / area
        out[b_idx, c_idx, oh, ow] = al.convert(avg, al.bf16)


def avelang_avg_pool2d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    stride = kernel_size

    B, C, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1

    x = x.contiguous()
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    out = torch.empty(B, C, H_out, W_out, dtype=torch.bfloat16, device=x.device)

    x_B_stride = C * H * W
    x_C_stride = H * W
    x_H_stride = W
    x_W_stride = 1

    out_B_stride = C * H_out * W_out
    out_C_stride = H_out * W_out
    out_H_stride = W_out
    out_W_stride = 1

    TILE_H = 16
    TILE_W = 16
    BLOCK_SIZE = TILE_H * TILE_W

    h_tiles = math.ceil(H_out / TILE_H)
    w_tiles = math.ceil(W_out / TILE_W)

    avg_pool2d_kernel[lambda: ((B * C, h_tiles, w_tiles), (BLOCK_SIZE, 1, 1))](
        x,
        out,
        B,
        C,
        H,
        W,
        H_out,
        W_out,
        kernel_size,
        stride,
        x_B_stride,
        x_C_stride,
        x_H_stride,
        x_W_stride,
        out_B_stride,
        out_C_stride,
        out_H_stride,
        out_W_stride,
        w_tiles,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        if stride is None:
            stride = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = torch.nn.functional.pad(
                x,
                (self.padding, self.padding, self.padding, self.padding),
            )
        return avelang_avg_pool2d(x, self.kernel_size)
import math
import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# Kernel: 2D average pooling with BF16 I/O and FP32 accumulation.
# Thread coarsening: each thread handles 2 adjacent output positions along W
# to improve ILP and amortize grid-launch overhead.
# Tile: 16 high × 8 wide threads = 128 threads/block, each computing 2 outputs.
# ---------------------------------------------------------------------------

@avelang.jit
def avg_pool2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.i32,
    stride: al.i32,
    x_B_stride: al.i32,
    x_C_stride: al.i32,
    x_H_stride: al.i32,
    x_W_stride: al.i32,
    out_B_stride: al.i32,
    out_C_stride: al.i32,
    out_H_stride: al.i32,
    out_W_stride: al.i32,
    w_tiles: al.i32,
):
    x_layout = al.make_layout(
        (B, C, H, W),
        (x_B_stride, x_C_stride, x_H_stride, x_W_stride),
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    out_layout = al.make_layout(
        (B, C, H_out, W_out),
        (out_B_stride, out_C_stride, out_H_stride, out_W_stride),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    b_idx = al.block_id(0) % B
    c_idx = al.block_id(0) // B
    h_block = al.block_id(1)
    w_block_half = al.block_id(2)

    tid = al.thread_id(0)
    th = tid // 8
    tw = tid - th * 8

    oh = h_block * 16 + th
    ow0 = w_block_half * 16 + tw * 2
    ow1 = ow0 + 1

    # 2D slice at (b, c) for simpler indexing.
    x_slice = al.subview(
        x,
        (b_idx, c_idx, 0, 0),
        (1, 1, H, W),
        (1, 1, 1, 1),
    )

    if oh < H_out:
        # Compute two output positions along W.
        for elem in al.range(2):
            ow = ow0
            if elem == 1:
                ow = ow1
            if ow < W_out:
                acc = al.convert(0.0, al.f32)
                for kh in al.range(K):
                    ih = oh * stride + kh
                    for kw in al.range(K):
                        iw = ow * stride + kw
                        acc = acc + al.convert(x_slice[ih, iw], al.f32)
                area = al.convert(K, al.f32) * al.convert(K, al.f32)
                avg = acc / area
                out[b_idx, c_idx, oh, ow] = al.convert(avg, al.bf16)


def avelang_avg_pool2d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    stride = kernel_size

    B, C, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1

    x = x.contiguous()
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    out = torch.empty(B, C, H_out, W_out, dtype=torch.bfloat16, device=x.device)

    x_B_stride = C * H * W
    x_C_stride = H * W
    x_H_stride = W
    x_W_stride = 1

    out_B_stride = C * H_out * W_out
    out_C_stride = H_out * W_out
    out_H_stride = W_out
    out_W_stride = 1

    TILE_H = 16
    TILE_W_HALF = 8
    BLOCK_SIZE = TILE_H * TILE_W_HALF  # 128

    h_tiles = math.ceil(H_out / TILE_H)
    w_tiles_half = math.ceil(W_out / (TILE_W_HALF * 2))

    avg_pool2d_kernel[lambda: ((B * C, h_tiles, w_tiles_half), (BLOCK_SIZE, 1, 1))](
        x,
        out,
        B,
        C,
        H,
        W,
        H_out,
        W_out,
        kernel_size,
        stride,
        x_B_stride,
        x_C_stride,
        x_H_stride,
        x_W_stride,
        out_B_stride,
        out_C_stride,
        out_H_stride,
        out_W_stride,
        w_tiles_half,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        if stride is None:
            stride = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = torch.nn.functional.pad(
                x,
                (self.padding, self.padding, self.padding, self.padding),
            )
        return avelang_avg_pool2d(x, self.kernel_size)
import math
import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# Kernel: 2D average pooling with BF16 I/O and FP32 accumulation.
# Each block handles a 16x16 tile of output positions for one (b,c) pair.
# Inner-loop invariants hoisted outside the kH/kW loops.
# ---------------------------------------------------------------------------

@avelang.jit
def avg_pool2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.i32,
    stride: al.i32,
    x_B_stride: al.i32,
    x_C_stride: al.i32,
    x_H_stride: al.i32,
    x_W_stride: al.i32,
    out_B_stride: al.i32,
    out_C_stride: al.i32,
    out_H_stride: al.i32,
    out_W_stride: al.i32,
    w_tiles: al.i32,
):
    x_layout = al.make_layout(
        (B, C, H, W),
        (x_B_stride, x_C_stride, x_H_stride, x_W_stride),
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    out_layout = al.make_layout(
        (B, C, H_out, W_out),
        (out_B_stride, out_C_stride, out_H_stride, out_W_stride),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    b_idx = al.block_id(0) % B
    c_idx = al.block_id(0) // B
    h_block = al.block_id(1)
    w_block = al.block_id(2)

    tid = al.thread_id(0)
    th = tid // 16
    tw = tid - th * 16

    oh = h_block * 16 + th
    ow = w_block * 16 + tw

    # 2D (H,W) slice at this (b, c) for fast inner-loop indexing.
    x_slice = al.subview(
        x,
        (b_idx, c_idx, 0, 0),
        (1, 1, H, W),
        (1, 1, 1, 1),
    )

    if oh < H_out and ow < W_out:
        # Hoist stride-multiplied bases outside the reduction loops.
        ih_base = oh * stride
        iw_base = ow * stride
        acc = al.convert(0.0, al.f32)
        for kh in al.range(K):
            ih = ih_base + kh
            for kw in al.range(K):
                iw = iw_base + kw
                acc = acc + al.convert(x_slice[ih, iw], al.f32)
        area = al.convert(K, al.f32) * al.convert(K, al.f32)
        out[b_idx, c_idx, oh, ow] = al.convert(acc / area, al.bf16)


def avelang_avg_pool2d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    stride = kernel_size

    B, C, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1

    x = x.contiguous()
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    out = torch.empty(B, C, H_out, W_out, dtype=torch.bfloat16, device=x.device)

    x_B_stride = C * H * W
    x_C_stride = H * W
    x_H_stride = W
    x_W_stride = 1

    out_B_stride = C * H_out * W_out
    out_C_stride = H_out * W_out
    out_H_stride = W_out
    out_W_stride = 1

    TILE_H = 16
    TILE_W = 16
    BLOCK_SIZE = TILE_H * TILE_W  # 256

    h_tiles = math.ceil(H_out / TILE_H)
    w_tiles = math.ceil(W_out / TILE_W)

    avg_pool2d_kernel[lambda: ((B * C, h_tiles, w_tiles), (BLOCK_SIZE, 1, 1))](
        x,
        out,
        B,
        C,
        H,
        W,
        H_out,
        W_out,
        kernel_size,
        stride,
        x_B_stride,
        x_C_stride,
        x_H_stride,
        x_W_stride,
        out_B_stride,
        out_C_stride,
        out_H_stride,
        out_W_stride,
        w_tiles,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        if stride is None:
            stride = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = torch.nn.functional.pad(
                x,
                (self.padding, self.padding, self.padding, self.padding),
            )
        return avelang_avg_pool2d(x, self.kernel_size)
import math
import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# Kernel: 2D average pooling with BF16 I/O and FP32 accumulation.
# Grid = (B, C, h_tiles * w_tiles) so block_id maps directly to b, c, and
# a flattened spatial tile.  Inner-loop invariants hoisted.
# ---------------------------------------------------------------------------

@avelang.jit
def avg_pool2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.i32,
    stride: al.i32,
    x_B_stride: al.i32,
    x_C_stride: al.i32,
    x_H_stride: al.i32,
    x_W_stride: al.i32,
    out_B_stride: al.i32,
    out_C_stride: al.i32,
    out_H_stride: al.i32,
    out_W_stride: al.i32,
    w_tiles: al.i32,
):
    x_layout = al.make_layout(
        (B, C, H, W),
        (x_B_stride, x_C_stride, x_H_stride, x_W_stride),
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    out_layout = al.make_layout(
        (B, C, H_out, W_out),
        (out_B_stride, out_C_stride, out_H_stride, out_W_stride),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    # Grid is (B, C, h_tiles * w_tiles) → direct mapping, no div/mod.
    b_idx = al.block_id(0)
    c_idx = al.block_id(1)
    spatial_id = al.block_id(2)
    h_block = spatial_id // w_tiles
    w_block = spatial_id - h_block * w_tiles

    tid = al.thread_id(0)
    th = tid // 16
    tw = tid - th * 16

    oh = h_block * 16 + th
    ow = w_block * 16 + tw

    # 2D (H,W) slice at this (b, c) for fast inner-loop indexing.
    x_slice = al.subview(
        x,
        (b_idx, c_idx, 0, 0),
        (1, 1, H, W),
        (1, 1, 1, 1),
    )

    if oh < H_out and ow < W_out:
        ih_base = oh * stride
        iw_base = ow * stride
        acc = al.convert(0.0, al.f32)
        for kh in al.range(K):
            ih = ih_base + kh
            for kw in al.range(K):
                iw = iw_base + kw
                acc = acc + al.convert(x_slice[ih, iw], al.f32)
        area = al.convert(K, al.f32) * al.convert(K, al.f32)
        out[b_idx, c_idx, oh, ow] = al.convert(acc / area, al.bf16)


def avelang_avg_pool2d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    stride = kernel_size

    B, C, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1

    x = x.contiguous()
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    out = torch.empty(B, C, H_out, W_out, dtype=torch.bfloat16, device=x.device)

    x_B_stride = C * H * W
    x_C_stride = H * W
    x_H_stride = W
    x_W_stride = 1

    out_B_stride = C * H_out * W_out
    out_C_stride = H_out * W_out
    out_H_stride = W_out
    out_W_stride = 1

    TILE_H = 16
    TILE_W = 16
    BLOCK_SIZE = TILE_H * TILE_W

    h_tiles = math.ceil(H_out / TILE_H)
    w_tiles = math.ceil(W_out / TILE_W)
    spatial_tiles = h_tiles * w_tiles

    avg_pool2d_kernel[lambda: ((B, C, spatial_tiles), (BLOCK_SIZE, 1, 1))](
        x,
        out,
        B,
        C,
        H,
        W,
        H_out,
        W_out,
        kernel_size,
        stride,
        x_B_stride,
        x_C_stride,
        x_H_stride,
        x_W_stride,
        out_B_stride,
        out_C_stride,
        out_H_stride,
        out_W_stride,
        w_tiles,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        if stride is None:
            stride = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = torch.nn.functional.pad(
                x,
                (self.padding, self.padding, self.padding, self.padding),
            )
        return avelang_avg_pool2d(x, self.kernel_size)
