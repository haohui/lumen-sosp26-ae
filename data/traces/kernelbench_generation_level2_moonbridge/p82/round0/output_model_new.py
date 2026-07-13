import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# Compile-time constants (match problem specification)
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
IN_CHANNELS = 8
OUT_CHANNELS = 64
H_IN = 256
W_IN = 256
KERNEL_SIZE = 3
SCALING_FACTOR = 2.0
POOL_SIZE = 4

# Derived spatial dimensions
H_OUT_CONV = H_IN - KERNEL_SIZE + 1   # 254
W_OUT_CONV = W_IN - KERNEL_SIZE + 1   # 254
H_OUT_POOL = H_OUT_CONV // POOL_SIZE   # 63
W_OUT_POOL = W_OUT_CONV // POOL_SIZE   # 63

# Tile parameters for the convolution kernel
TILE_H = 16
TILE_W = 16
THREADS = TILE_H * TILE_W               # 256
IN_TILE_H = TILE_H + KERNEL_SIZE - 1    # 18
IN_TILE_W = TILE_W + KERNEL_SIZE - 1    # 18
IN_TILE_FLAT = IN_TILE_H * IN_TILE_W * IN_CHANNELS   # 2592
W_FLAT = OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE  # 4608
W_INNER = IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE       # 72
KSQ = KERNEL_SIZE * KERNEL_SIZE                         # 9

NUM_TILES_H = (H_OUT_CONV + TILE_H - 1) // TILE_H       # 16
NUM_TILES_W = (W_OUT_CONV + TILE_W - 1) // TILE_W       # 16

# MaxPool tile parameters
POOL_THREADS = 256


# ---------------------------------------------------------------------------
# Kernel 1: Conv2d forward pass with fused tanh, scale, and bias epilogue.
# Output is FP32 to preserve precision through the subsequent maxpool.
# ---------------------------------------------------------------------------
@avelang.jit
def conv2d_tanh_scale_bias_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    conv_bias_ptr: al.Pointer(al.bf16),
    ext_bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.f32),
    B: al.i32,
    num_tiles_h: al.i32,
    num_tiles_w: al.i32,
):
    tid = al.thread_id(0)
    tile_idx = al.block_id(0)
    b = al.block_id(1)

    h_tile = tile_idx // num_tiles_w
    w_tile = tile_idx - h_tile * num_tiles_w

    ty = tid // TILE_W
    tx = tid - ty * TILE_W

    oh = h_tile * TILE_H + ty
    ow = w_tile * TILE_W + tx

    # Global tensor views
    x_4d = al.make_tensor(
        x_ptr, al.bf16,
        al.make_layout(
            (B, IN_CHANNELS, H_IN, W_IN),
            (IN_CHANNELS * H_IN * W_IN, H_IN * W_IN, W_IN, 1),
        ),
    )
    w_1d = al.make_tensor(w_ptr, al.bf16, al.make_layout((W_FLAT,), (1,)))
    cb_1d = al.make_tensor(conv_bias_ptr, al.bf16, al.make_layout((OUT_CHANNELS,), (1,)))
    eb_1d = al.make_tensor(ext_bias_ptr, al.bf16, al.make_layout((OUT_CHANNELS,), (1,)))
    out_fp32_4d = al.make_tensor(
        out_ptr, al.f32,
        al.make_layout(
            (B, OUT_CHANNELS, H_OUT_CONV, W_OUT_CONV),
            (OUT_CHANNELS * H_OUT_CONV * W_OUT_CONV, H_OUT_CONV * W_OUT_CONV, W_OUT_CONV, 1),
        ),
    )

    # Shared memory
    smem_in = al.make_shared((IN_TILE_FLAT,), al.bf16)
    smem_w = al.make_shared((W_FLAT,), al.bf16)

    # Cooperative load: weights
    for i in al.range(tid, W_FLAT, THREADS):
        smem_w[i] = w_1d[i]

    # Cooperative load: input tile
    in_h_start = h_tile * TILE_H
    in_w_start = w_tile * TILE_W
    for i in al.range(tid, IN_TILE_FLAT, THREADS):
        ih = i // (IN_TILE_W * IN_CHANNELS)
        rest = i - ih * (IN_TILE_W * IN_CHANNELS)
        iw = rest // IN_CHANNELS
        ic = rest - iw * IN_CHANNELS

        gbl_h = in_h_start + ih
        gbl_w = in_w_start + iw

        if gbl_h < H_IN and gbl_w < W_IN:
            smem_in[i] = x_4d[b, ic, gbl_h, gbl_w]
        else:
            smem_in[i] = al.convert(0.0, al.bf16)

    al.syncthreads()

    # Compute: convolution + epilogue
    if oh < H_OUT_CONV and ow < W_OUT_CONV:
        zero_f32 = al.convert(0.0, al.f32)
        scale_f32 = al.convert(SCALING_FACTOR, al.f32)
        for oc in al.range(OUT_CHANNELS):
            acc = zero_f32
            w_oc_base = oc * W_INNER
            for kh in al.range(KERNEL_SIZE):
                in_kh_base = (ty + kh) * IN_TILE_W
                w_kh_base = w_oc_base + kh * KERNEL_SIZE
                for kw in al.range(KERNEL_SIZE):
                    in_base = (in_kh_base + (tx + kw)) * IN_CHANNELS
                    w_kw_base = w_kh_base + kw

                    w0 = al.convert(smem_w[w_kw_base + 0 * KSQ], al.f32)
                    i0 = al.convert(smem_in[in_base + 0], al.f32)
                    acc = acc + i0 * w0

                    w1 = al.convert(smem_w[w_kw_base + 1 * KSQ], al.f32)
                    i1 = al.convert(smem_in[in_base + 1], al.f32)
                    acc = acc + i1 * w1

                    w2 = al.convert(smem_w[w_kw_base + 2 * KSQ], al.f32)
                    i2 = al.convert(smem_in[in_base + 2], al.f32)
                    acc = acc + i2 * w2

                    w3 = al.convert(smem_w[w_kw_base + 3 * KSQ], al.f32)
                    i3 = al.convert(smem_in[in_base + 3], al.f32)
                    acc = acc + i3 * w3

                    w4 = al.convert(smem_w[w_kw_base + 4 * KSQ], al.f32)
                    i4 = al.convert(smem_in[in_base + 4], al.f32)
                    acc = acc + i4 * w4

                    w5 = al.convert(smem_w[w_kw_base + 5 * KSQ], al.f32)
                    i5 = al.convert(smem_in[in_base + 5], al.f32)
                    acc = acc + i5 * w5

                    w6 = al.convert(smem_w[w_kw_base + 6 * KSQ], al.f32)
                    i6 = al.convert(smem_in[in_base + 6], al.f32)
                    acc = acc + i6 * w6

                    w7 = al.convert(smem_w[w_kw_base + 7 * KSQ], al.f32)
                    i7 = al.convert(smem_in[in_base + 7], al.f32)
                    acc = acc + i7 * w7

            acc = acc + al.convert(cb_1d[oc], al.f32)
            acc = al.tanh(acc)
            acc = acc * scale_f32
            acc = acc + al.convert(eb_1d[oc], al.f32)
            out_fp32_4d[b, oc, oh, ow] = acc


# ---------------------------------------------------------------------------
# Kernel 2: MaxPool 4x4 -- simple 1D grid, one thread per output element.
# ---------------------------------------------------------------------------
@avelang.jit
def maxpool2d_kernel(
    x_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    total_outputs: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    gid = bid * POOL_THREADS + tid

    if gid < total_outputs:
        ow = gid % W_OUT_POOL
        rest = gid // W_OUT_POOL
        oh = rest % H_OUT_POOL
        rest = rest // H_OUT_POOL
        oc = rest % OUT_CHANNELS
        b = rest // OUT_CHANNELS

        x_fp32 = al.make_tensor(
            x_ptr, al.f32,
            al.make_layout(
                (B, OUT_CHANNELS, H_OUT_CONV, W_OUT_CONV),
                (OUT_CHANNELS * H_OUT_CONV * W_OUT_CONV, H_OUT_CONV * W_OUT_CONV, W_OUT_CONV, 1),
            ),
        )
        out_4d = al.make_tensor(
            out_ptr, al.bf16,
            al.make_layout(
                (B, OUT_CHANNELS, H_OUT_POOL, W_OUT_POOL),
                (OUT_CHANNELS * H_OUT_POOL * W_OUT_POOL, H_OUT_POOL * W_OUT_POOL, W_OUT_POOL, 1),
            ),
        )

        in_h_start = oh * POOL_SIZE
        in_w_start = ow * POOL_SIZE

        max_val = x_fp32[b, oc, in_h_start, in_w_start]
        for kh in al.range(POOL_SIZE):
            ih = in_h_start + kh
            if ih < H_OUT_CONV:
                for kw in al.range(POOL_SIZE):
                    iw = in_w_start + kw
                    if iw < W_OUT_CONV:
                        if kh > 0 or kw > 0:
                            val = x_fp32[b, oc, ih, iw]
                            if val > max_val:
                                max_val = val

        out_4d[b, oc, oh, ow] = al.convert(max_val, al.bf16)


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------
def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv2d_maxpool(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_bias: torch.Tensor,
    ext_bias: torch.Tensor,
) -> torch.Tensor:
    """Full pipeline: Conv2d -> tanh -> scale -> bias -> MaxPool4x4."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)
    cb_bf16 = _to_bf16_contiguous(conv_bias)
    eb_bf16 = _to_bf16_contiguous(ext_bias)

    B, IC, H, W = x_bf16.shape

    conv_out_fp32 = torch.empty(
        (B, OUT_CHANNELS, H_OUT_CONV, W_OUT_CONV),
        device=x_bf16.device,
        dtype=torch.float32,
    )

    grid_conv = (NUM_TILES_H * NUM_TILES_W, B, 1)
    conv2d_tanh_scale_bias_kernel[lambda: (grid_conv, (THREADS, 1, 1))](
        x_bf16, w_bf16, cb_bf16, eb_bf16, conv_out_fp32,
        B, NUM_TILES_H, NUM_TILES_W,
    )

    # MaxPool
    total_pool = B * OUT_CHANNELS * H_OUT_POOL * W_OUT_POOL
    num_pool_blocks = (total_pool + POOL_THREADS - 1) // POOL_THREADS

    pool_out = torch.empty(
        (B, OUT_CHANNELS, H_OUT_POOL, W_OUT_POOL),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )

    maxpool2d_kernel[lambda: ((num_pool_blocks, 1, 1), (POOL_THREADS, 1, 1))](
        conv_out_fp32, pool_out, B, total_pool,
    )

    return pool_out


# ---------------------------------------------------------------------------
# ModelNew entrypoint
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    A model that performs a convolution, applies tanh, scaling, adds a bias term,
    and then max-pools -- executed via AveLang GPU kernels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        scaling_factor: float,
        bias_shape: tuple,
        pool_kernel_size: int,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv.weight.data
        conv_bias = self.conv.bias.data
        ext_bias = self.bias.data.reshape(-1).contiguous()
        return avelang_conv2d_maxpool(x, weight, conv_bias, ext_bias)


# ---------------------------------------------------------------------------
# Module-level contract
# ---------------------------------------------------------------------------
batch_size = BATCH_SIZE
in_channels = IN_CHANNELS
out_channels = OUT_CHANNELS
height = H_IN
width = W_IN
kernel_size = KERNEL_SIZE
scaling_factor = SCALING_FACTOR
bias_shape = (OUT_CHANNELS, 1, 1)
pool_kernel_size = POOL_SIZE


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_CHANNELS, H_IN, W_IN)]


def get_init_inputs():
    return [
        IN_CHANNELS,
        OUT_CHANNELS,
        KERNEL_SIZE,
        SCALING_FACTOR,
        (OUT_CHANNELS, 1, 1),
        POOL_SIZE,
    ]
