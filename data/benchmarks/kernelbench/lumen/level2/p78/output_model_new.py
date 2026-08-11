import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Problem shape constants
BATCH_SIZE = 16
IN_CHANNELS = 32
OUT_CHANNELS = 64
IN_D = 32
IN_H = 32
IN_W = 32
KERNEL_SIZE = 5
STRIDE = 2
PADDING = 2

# ConvTranspose3d output dimensions
# output = (input - 1) * stride - 2 * padding + kernel_size
CONV_OUT_D = (IN_D - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE  # 63
CONV_OUT_H = (IN_H - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE  # 63
CONV_OUT_W = (IN_W - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE  # 63

# After MaxPool3d(kernel_size=2)
POOL1_OUT_D = CONV_OUT_D // 2  # 31
POOL1_OUT_H = CONV_OUT_H // 2  # 31
POOL1_OUT_W = CONV_OUT_W // 2  # 31

# After MaxPool3d(kernel_size=3)
POOL2_OUT_D = POOL1_OUT_D // 3  # 10
POOL2_OUT_H = POOL1_OUT_H // 3  # 10
POOL2_OUT_W = POOL1_OUT_W // 3  # 10

NEG_INF_F32 = -3.402823466e38


# ========== 3D Max Pooling Kernels ==========

@substrate.jit
def maxpool3d_k2_f32_kernel(
    x: S.Pointer(S.f32),
    out: S.Pointer(S.f32),
    total_out: S.u32,
):
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if tid < total_out:
        # Unpack output coordinates: (n, c, od, oh, ow)
        spatial = POOL1_OUT_D * POOL1_OUT_H * POOL1_OUT_W
        nc = tid // spatial
        s = tid - nc * spatial
        n = nc // OUT_CHANNELS
        c = nc - n * OUT_CHANNELS
        od = s // (POOL1_OUT_H * POOL1_OUT_W)
        s2 = s - od * (POOL1_OUT_H * POOL1_OUT_W)
        oh = s2 // POOL1_OUT_W
        ow = s2 - oh * POOL1_OUT_W

        # 5D layout for input
        x_layout = S.make_layout(
            (BATCH_SIZE, OUT_CHANNELS, CONV_OUT_D, CONV_OUT_H, CONV_OUT_W),
            (OUT_CHANNELS * CONV_OUT_D * CONV_OUT_H * CONV_OUT_W,
             CONV_OUT_D * CONV_OUT_H * CONV_OUT_W,
             CONV_OUT_H * CONV_OUT_W,
             CONV_OUT_W,
             1)
        )
        g_x = S.make_tensor(x, S.f32, x_layout)

        out_layout = S.make_layout(
            (BATCH_SIZE, OUT_CHANNELS, POOL1_OUT_D, POOL1_OUT_H, POOL1_OUT_W),
            (OUT_CHANNELS * POOL1_OUT_D * POOL1_OUT_H * POOL1_OUT_W,
             POOL1_OUT_D * POOL1_OUT_H * POOL1_OUT_W,
             POOL1_OUT_H * POOL1_OUT_W,
             POOL1_OUT_W,
             1)
        )
        g_out = S.make_tensor(out, S.f32, out_layout)

        m = S.convert(NEG_INF_F32, S.f32)
        for kd in S.range(2):
            for kh in S.range(2):
                for kw in S.range(2):
                    id = od * 2 + kd
                    ih = oh * 2 + kh
                    iw = ow * 2 + kw
                    val = g_x[n, c, id, ih, iw]
                    m = val if val > m else m
        g_out[n, c, od, oh, ow] = m


@substrate.jit
def maxpool3d_k2_bf16_kernel(
    x: S.Pointer(S.bf16),
    out: S.Pointer(S.bf16),
    total_out: S.u32,
):
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if tid < total_out:
        spatial = POOL1_OUT_D * POOL1_OUT_H * POOL1_OUT_W
        nc = tid // spatial
        s = tid - nc * spatial
        n = nc // OUT_CHANNELS
        c = nc - n * OUT_CHANNELS
        od = s // (POOL1_OUT_H * POOL1_OUT_W)
        s2 = s - od * (POOL1_OUT_H * POOL1_OUT_W)
        oh = s2 // POOL1_OUT_W
        ow = s2 - oh * POOL1_OUT_W

        x_layout = S.make_layout(
            (BATCH_SIZE, OUT_CHANNELS, CONV_OUT_D, CONV_OUT_H, CONV_OUT_W),
            (OUT_CHANNELS * CONV_OUT_D * CONV_OUT_H * CONV_OUT_W,
             CONV_OUT_D * CONV_OUT_H * CONV_OUT_W,
             CONV_OUT_H * CONV_OUT_W,
             CONV_OUT_W,
             1)
        )
        g_x = S.make_tensor(x, S.bf16, x_layout)

        out_layout = S.make_layout(
            (BATCH_SIZE, OUT_CHANNELS, POOL1_OUT_D, POOL1_OUT_H, POOL1_OUT_W),
            (OUT_CHANNELS * POOL1_OUT_D * POOL1_OUT_H * POOL1_OUT_W,
             POOL1_OUT_D * POOL1_OUT_H * POOL1_OUT_W,
             POOL1_OUT_H * POOL1_OUT_W,
             POOL1_OUT_W,
             1)
        )
        g_out = S.make_tensor(out, S.bf16, out_layout)

        m = S.convert(NEG_INF_F32, S.f32)
        for kd in S.range(2):
            for kh in S.range(2):
                for kw in S.range(2):
                    id = od * 2 + kd
                    ih = oh * 2 + kh
                    iw = ow * 2 + kw
                    val = S.convert(g_x[n, c, id, ih, iw], S.f32)
                    m = val if val > m else m
        g_out[n, c, od, oh, ow] = S.convert(m, S.bf16)


@substrate.jit
def maxpool3d_k3_f32_kernel(
    x: S.Pointer(S.f32),
    out: S.Pointer(S.f32),
    total_out: S.u32,
):
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if tid < total_out:
        spatial = POOL2_OUT_D * POOL2_OUT_H * POOL2_OUT_W
        nc = tid // spatial
        s = tid - nc * spatial
        n = nc // OUT_CHANNELS
        c = nc - n * OUT_CHANNELS
        od = s // (POOL2_OUT_H * POOL2_OUT_W)
        s2 = s - od * (POOL2_OUT_H * POOL2_OUT_W)
        oh = s2 // POOL2_OUT_W
        ow = s2 - oh * POOL2_OUT_W

        x_layout = S.make_layout(
            (BATCH_SIZE, OUT_CHANNELS, POOL1_OUT_D, POOL1_OUT_H, POOL1_OUT_W),
            (OUT_CHANNELS * POOL1_OUT_D * POOL1_OUT_H * POOL1_OUT_W,
             POOL1_OUT_D * POOL1_OUT_H * POOL1_OUT_W,
             POOL1_OUT_H * POOL1_OUT_W,
             POOL1_OUT_W,
             1)
        )
        g_x = S.make_tensor(x, S.f32, x_layout)

        out_layout = S.make_layout(
            (BATCH_SIZE, OUT_CHANNELS, POOL2_OUT_D, POOL2_OUT_H, POOL2_OUT_W),
            (OUT_CHANNELS * POOL2_OUT_D * POOL2_OUT_H * POOL2_OUT_W,
             POOL2_OUT_D * POOL2_OUT_H * POOL2_OUT_W,
             POOL2_OUT_H * POOL2_OUT_W,
             POOL2_OUT_W,
             1)
        )
        g_out = S.make_tensor(out, S.f32, out_layout)

        m = S.convert(NEG_INF_F32, S.f32)
        for kd in S.range(3):
            for kh in S.range(3):
                for kw in S.range(3):
                    id = od * 3 + kd
                    ih = oh * 3 + kh
                    iw = ow * 3 + kw
                    val = g_x[n, c, id, ih, iw]
                    m = val if val > m else m
        g_out[n, c, od, oh, ow] = m


@substrate.jit
def maxpool3d_k3_bf16_kernel(
    x: S.Pointer(S.bf16),
    out: S.Pointer(S.bf16),
    total_out: S.u32,
):
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if tid < total_out:
        spatial = POOL2_OUT_D * POOL2_OUT_H * POOL2_OUT_W
        nc = tid // spatial
        s = tid - nc * spatial
        n = nc // OUT_CHANNELS
        c = nc - n * OUT_CHANNELS
        od = s // (POOL2_OUT_H * POOL2_OUT_W)
        s2 = s - od * (POOL2_OUT_H * POOL2_OUT_W)
        oh = s2 // POOL2_OUT_W
        ow = s2 - oh * POOL2_OUT_W

        x_layout = S.make_layout(
            (BATCH_SIZE, OUT_CHANNELS, POOL1_OUT_D, POOL1_OUT_H, POOL1_OUT_W),
            (OUT_CHANNELS * POOL1_OUT_D * POOL1_OUT_H * POOL1_OUT_W,
             POOL1_OUT_D * POOL1_OUT_H * POOL1_OUT_W,
             POOL1_OUT_H * POOL1_OUT_W,
             POOL1_OUT_W,
             1)
        )
        g_x = S.make_tensor(x, S.bf16, x_layout)

        out_layout = S.make_layout(
            (BATCH_SIZE, OUT_CHANNELS, POOL2_OUT_D, POOL2_OUT_H, POOL2_OUT_W),
            (OUT_CHANNELS * POOL2_OUT_D * POOL2_OUT_H * POOL2_OUT_W,
             POOL2_OUT_D * POOL2_OUT_H * POOL2_OUT_W,
             POOL2_OUT_H * POOL2_OUT_W,
             POOL2_OUT_W,
             1)
        )
        g_out = S.make_tensor(out, S.bf16, out_layout)

        m = S.convert(NEG_INF_F32, S.f32)
        for kd in S.range(3):
            for kh in S.range(3):
                for kw in S.range(3):
                    id = od * 3 + kd
                    ih = oh * 3 + kh
                    iw = ow * 3 + kw
                    val = S.convert(g_x[n, c, id, ih, iw], S.f32)
                    m = val if val > m else m
        g_out[n, c, od, oh, ow] = S.convert(m, S.bf16)


# ========== Sum Reduction Kernel ==========

BLOCK_REDUCE = 256


@substrate.jit
def sum_channel_bf16_kernel(
    x: S.Pointer(S.bf16),
    out: S.Pointer(S.bf16),
    total_out: S.u32,
):
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if tid < total_out:
        spatial = POOL2_OUT_D * POOL2_OUT_H * POOL2_OUT_W
        n = tid // spatial
        s = tid - n * spatial
        d = s // (POOL2_OUT_H * POOL2_OUT_W)
        s2 = s - d * (POOL2_OUT_H * POOL2_OUT_W)
        h = s2 // POOL2_OUT_W
        w = s2 - h * POOL2_OUT_W

        x_layout = S.make_layout(
            (BATCH_SIZE, OUT_CHANNELS, POOL2_OUT_D, POOL2_OUT_H, POOL2_OUT_W),
            (OUT_CHANNELS * POOL2_OUT_D * POOL2_OUT_H * POOL2_OUT_W,
             POOL2_OUT_D * POOL2_OUT_H * POOL2_OUT_W,
             POOL2_OUT_H * POOL2_OUT_W,
             POOL2_OUT_W,
             1)
        )
        g_x = S.make_tensor(x, S.bf16, x_layout)

        out_layout = S.make_layout(
            (BATCH_SIZE, 1, POOL2_OUT_D, POOL2_OUT_H, POOL2_OUT_W),
            (POOL2_OUT_D * POOL2_OUT_H * POOL2_OUT_W,
             POOL2_OUT_D * POOL2_OUT_H * POOL2_OUT_W,
             POOL2_OUT_H * POOL2_OUT_W,
             POOL2_OUT_W,
             1)
        )
        g_out = S.make_tensor(out, S.bf16, out_layout)

        acc = S.convert(0.0, S.f32)
        for c in S.range(OUT_CHANNELS):
            acc = acc + S.convert(g_x[n, c, d, h, w], S.f32)
        g_out[n, 0, d, h, w] = S.convert(acc, S.bf16)


@substrate.jit
def sum_channel_f32_kernel(
    x: S.Pointer(S.f32),
    out: S.Pointer(S.f32),
    total_out: S.u32,
):
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if tid < total_out:
        spatial = POOL2_OUT_D * POOL2_OUT_H * POOL2_OUT_W
        n = tid // spatial
        s = tid - n * spatial
        d = s // (POOL2_OUT_H * POOL2_OUT_W)
        s2 = s - d * (POOL2_OUT_H * POOL2_OUT_W)
        h = s2 // POOL2_OUT_W
        w = s2 - h * POOL2_OUT_W

        x_layout = S.make_layout(
            (BATCH_SIZE, OUT_CHANNELS, POOL2_OUT_D, POOL2_OUT_H, POOL2_OUT_W),
            (OUT_CHANNELS * POOL2_OUT_D * POOL2_OUT_H * POOL2_OUT_W,
             POOL2_OUT_D * POOL2_OUT_H * POOL2_OUT_W,
             POOL2_OUT_H * POOL2_OUT_W,
             POOL2_OUT_W,
             1)
        )
        g_x = S.make_tensor(x, S.f32, x_layout)

        out_layout = S.make_layout(
            (BATCH_SIZE, 1, POOL2_OUT_D, POOL2_OUT_H, POOL2_OUT_W),
            (POOL2_OUT_D * POOL2_OUT_H * POOL2_OUT_W,
             POOL2_OUT_D * POOL2_OUT_H * POOL2_OUT_W,
             POOL2_OUT_H * POOL2_OUT_W,
             POOL2_OUT_W,
             1)
        )
        g_out = S.make_tensor(out, S.f32, out_layout)

        acc = S.convert(0.0, S.f32)
        for c in S.range(OUT_CHANNELS):
            acc = acc + g_x[n, c, d, h, w]
        g_out[n, 0, d, h, w] = acc


# ========== Launch Wrappers ==========

def _launch_maxpool3d_k2(x: torch.Tensor) -> torch.Tensor:
    """Launch 3D max pooling with kernel_size=2."""
    out = torch.empty(
        (BATCH_SIZE, OUT_CHANNELS, POOL1_OUT_D, POOL1_OUT_H, POOL1_OUT_W),
        device=x.device, dtype=x.dtype
    )
    total_out = BATCH_SIZE * OUT_CHANNELS * POOL1_OUT_D * POOL1_OUT_H * POOL1_OUT_W
    grid = (total_out + 255) // 256

    if x.dtype == torch.float32:
        maxpool3d_k2_f32_kernel[lambda: ((grid, 1, 1), (256, 1, 1))](x, out, total_out)
    elif x.dtype == torch.bfloat16:
        maxpool3d_k2_bf16_kernel[lambda: ((grid, 1, 1), (256, 1, 1))](x, out, total_out)
    else:
        raise TypeError(f"Unsupported dtype {x.dtype}")
    return out


def _launch_maxpool3d_k3(x: torch.Tensor) -> torch.Tensor:
    """Launch 3D max pooling with kernel_size=3."""
    out = torch.empty(
        (BATCH_SIZE, OUT_CHANNELS, POOL2_OUT_D, POOL2_OUT_H, POOL2_OUT_W),
        device=x.device, dtype=x.dtype
    )
    total_out = BATCH_SIZE * OUT_CHANNELS * POOL2_OUT_D * POOL2_OUT_H * POOL2_OUT_W
    grid = (total_out + 255) // 256

    if x.dtype == torch.float32:
        maxpool3d_k3_f32_kernel[lambda: ((grid, 1, 1), (256, 1, 1))](x, out, total_out)
    elif x.dtype == torch.bfloat16:
        maxpool3d_k3_bf16_kernel[lambda: ((grid, 1, 1), (256, 1, 1))](x, out, total_out)
    else:
        raise TypeError(f"Unsupported dtype {x.dtype}")
    return out


def _launch_sum_channel(x: torch.Tensor) -> torch.Tensor:
    """Launch sum reduction over channel dimension."""
    out = torch.empty(
        (BATCH_SIZE, 1, POOL2_OUT_D, POOL2_OUT_H, POOL2_OUT_W),
        device=x.device, dtype=x.dtype
    )
    total_out = BATCH_SIZE * POOL2_OUT_D * POOL2_OUT_H * POOL2_OUT_W
    grid = (total_out + BLOCK_REDUCE - 1) // BLOCK_REDUCE

    if x.dtype == torch.float32:
        sum_channel_f32_kernel[lambda: ((grid, 1, 1), (BLOCK_REDUCE, 1, 1))](x, out, total_out)
    elif x.dtype == torch.bfloat16:
        sum_channel_bf16_kernel[lambda: ((grid, 1, 1), (BLOCK_REDUCE, 1, 1))](x, out, total_out)
    else:
        raise TypeError(f"Unsupported dtype {x.dtype}")
    return out


class ModelNew(nn.Module):
    """
    Substrate-accelerated model for ConvTranspose3d + MaxPool3d + MaxPool3d + Sum.
    ConvTranspose3d uses PyTorch (no Substrate pattern exists).
    Pooling and reduction use Substrate kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_device = x.device
        if not x.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")
            x = x.cuda()

        # Convert to bfloat16 for optimized computation
        input_dtype = x.dtype
        if x.dtype == torch.float32:
            x = x.to(torch.bfloat16)

        # ConvTranspose3d (PyTorch - no Substrate pattern exists)
        x = self.conv_transpose(x)
        x = x.contiguous()

        # Verify shape after ConvTranspose3d
        assert x.shape == (BATCH_SIZE, OUT_CHANNELS, CONV_OUT_D, CONV_OUT_H, CONV_OUT_W), \
            f"Unexpected shape after ConvTranspose3d: {tuple(x.shape)}"

        # MaxPool3d kernel_size=2 (Substrate kernel)
        x = _launch_maxpool3d_k2(x)

        # MaxPool3d kernel_size=3 (Substrate kernel)
        x = _launch_maxpool3d_k3(x)

        # Sum over channel dimension (Substrate kernel)
        x = _launch_sum_channel(x)

        # Convert back if needed
        if input_dtype == torch.float32:
            x = x.to(torch.float32)

        if orig_device.type != "cuda":
            x = x.to(orig_device)
        return x


batch_size = BATCH_SIZE
in_channels = IN_CHANNELS
out_channels = OUT_CHANNELS
depth, height, width = IN_D, IN_H, IN_W
kernel_size = KERNEL_SIZE
stride = STRIDE
padding = PADDING


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
