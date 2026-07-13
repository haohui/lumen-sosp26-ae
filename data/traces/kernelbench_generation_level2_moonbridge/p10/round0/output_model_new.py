import torch
import torch.nn as nn
import avelang
import avelang.language as al

BATCH_SIZE = 128
IN_CHANNELS = 64
OUT_CHANNELS = 64
HEIGHT = 256
WIDTH = 256
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
MAXPOOL_KERNEL_SIZE = 2
MAXPOOL_STRIDE = 2
HARDTANH_MIN = -1.0
HARDTANH_MAX = 1.0

BLOCK_SIZE: al.constexpr = 256
TILE_H: al.constexpr = 16
TILE_W: al.constexpr = 16
TILE_H_EXT: al.constexpr = 18
TILE_W_EXT: al.constexpr = 18
TILE_ELEMS: al.constexpr = 324
TILE_THREADS: al.constexpr = 256


@avelang.jit
def conv_transpose_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    KH: al.i32,
    KW: al.i32,
    pad: al.i32,
):
    """ConvTranspose2d with bias, 16x16 output tile per block, shared-memory tiling.
    Launch: grid = (H_out // TILE_H, W_out // TILE_W, B * C_out),
            block = (TILE_THREADS, 1, 1)
    """
    tid = al.thread_id(0)
    tile_h = al.block_id(0)
    tile_w = al.block_id(1)
    bid_z = al.block_id(2)

    b = bid_z // C_out
    c_out = bid_z - b * C_out

    h_local = tid // TILE_W
    w_local = tid - h_local * TILE_W
    h_out = tile_h * TILE_H + h_local
    w_out = tile_w * TILE_W + w_local

    if b >= B or c_out >= C_out or h_out >= H_out or w_out >= W_out:
        return

    layout_x = al.make_layout(
        (B, C_in, H_in, W_in),
        (C_in * H_in * W_in, H_in * W_in, W_in, 1),
    )
    x = al.make_tensor(x_ptr, al.bf16, layout_x)

    layout_w = al.make_layout(
        (C_in, C_out, KH, KW),
        (C_out * KH * KW, KH * KW, KW, 1),
    )
    w = al.make_tensor(w_ptr, al.bf16, layout_w)

    layout_bias = al.make_layout((C_out,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, layout_bias)

    layout_out = al.make_layout(
        (B, C_out, H_out, W_out),
        (C_out * H_out * W_out, H_out * W_out, W_out, 1),
    )
    out = al.make_tensor(out_ptr, al.bf16, layout_out)

    smem = al.make_shared((TILE_ELEMS,), al.f32)
    zero_f32 = al.convert(0.0, al.f32)

    acc = zero_f32

    h_start = tile_h * TILE_H - pad
    w_start = tile_w * TILE_W - pad

    for ci in al.range(C_in):
        # Load input tile (18x18) into shared memory, zero-pad out-of-bounds
        for load_id in al.range(2):
            idx = tid + load_id * TILE_THREADS
            if idx < TILE_ELEMS:
                sh = idx // TILE_W_EXT
                sw = idx - sh * TILE_W_EXT
                h_gl = h_start + sh
                w_gl = w_start + sw
                if h_gl >= 0:
                    if h_gl < H_in:
                        if w_gl >= 0:
                            if w_gl < W_in:
                                smem[idx] = al.convert(x[b, ci, h_gl, w_gl], al.f32)
                            else:
                                smem[idx] = zero_f32
                        else:
                            smem[idx] = zero_f32
                    else:
                        smem[idx] = zero_f32
                else:
                    smem[idx] = zero_f32

        al.syncthreads()

        # Compute partial contributions using smem
        for ki in al.range(KH):
            for kj in al.range(KW):
                w_val = al.convert(w[ci, c_out, ki, kj], al.f32)
                sh = h_local + pad + pad - ki
                sw = w_local + pad + pad - kj
                smem_idx = sh * TILE_W_EXT + sw
                x_val = smem[smem_idx]
                acc = acc + w_val * x_val

        al.syncthreads()

    bias_val = al.convert(bias[c_out], al.f32)
    result = acc + bias_val
    out[b, c_out, h_out, w_out] = al.convert(result, al.bf16)


@avelang.jit
def maxpool_hardtanh_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    """Fused MaxPool2d (2x2, stride 2) + Hardtanh [-1, 1].
    Launch: grid = (C * H_out, B, 1), block = (W_out, 1, 1)
    """
    tid = al.thread_id(0)
    bid_x = al.block_id(0)
    b = al.block_id(1)

    c = bid_x // H_out
    h_out = bid_x - c * H_out
    w_out = tid

    if b < B and c < C and h_out < H_out and w_out < W_out:
        layout_x = al.make_layout(
            (B, C, H_in, W_in),
            (C * H_in * W_in, H_in * W_in, W_in, 1),
        )
        x = al.make_tensor(x_ptr, al.bf16, layout_x)

        layout_out = al.make_layout(
            (B, C, H_out, W_out),
            (C * H_out * W_out, H_out * W_out, W_out, 1),
        )
        out = al.make_tensor(out_ptr, al.bf16, layout_out)

        h_base = h_out * 2
        w_base = w_out * 2

        v00 = al.convert(x[b, c, h_base, w_base], al.f32)
        v01 = al.convert(x[b, c, h_base, w_base + 1], al.f32)
        v10 = al.convert(x[b, c, h_base + 1, w_base], al.f32)
        v11 = al.convert(x[b, c, h_base + 1, w_base + 1], al.f32)

        mx = v00
        if v01 > mx:
            mx = v01
        if v10 > mx:
            mx = v10
        if v11 > mx:
            mx = v11

        lo = al.convert(HARDTANH_MIN, al.f32)
        hi = al.convert(HARDTANH_MAX, al.f32)
        if mx < lo:
            mx = lo
        if mx > hi:
            mx = hi

        out[b, c, h_out, w_out] = al.convert(mx, al.bf16)


@avelang.jit
def mean_tanh_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    N: al.i32,
):
    """Mean over last N elements + tanh.
    Launch: grid = (C, B, 1), block = (BLOCK_SIZE, 1, 1)
    """
    tid = al.thread_id(0)
    bid_x = al.block_id(0)
    b = al.block_id(1)

    smem = al.make_shared((BLOCK_SIZE,), al.f32)
    zero_f32 = al.convert(0.0, al.f32)

    if b < B and bid_x < C:
        layout_x = al.make_layout((B * C * N,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_x)

        layout_out = al.make_layout((B, C), (C, 1))
        out = al.make_tensor(out_ptr, al.bf16, layout_out)

        base = (b * C + bid_x) * N

        local_sum = zero_f32
        for i in al.range(tid, N, BLOCK_SIZE):
            idx = base + i
            val = al.convert(x[idx], al.f32)
            local_sum = local_sum + val

        smem[tid] = local_sum
        al.syncthreads()

        if tid < 128:
            smem[tid] = smem[tid] + smem[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem[tid] = smem[tid] + smem[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem[tid] = smem[tid] + smem[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem[tid] = smem[tid] + smem[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem[tid] = smem[tid] + smem[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem[tid] = smem[tid] + smem[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem[tid] = smem[tid] + smem[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem[tid] = smem[tid] + smem[tid + 1]

        if tid == 0:
            n_f32 = al.convert(N, al.f32)
            mean = smem[0] / n_f32
            out[b, bid_x] = al.convert(al.tanh(mean), al.bf16)


def _ensure_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    return t


def avelang_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device required for AveLang kernels.")

    x_bf16 = _ensure_bf16_contiguous(x)
    w_bf16 = _ensure_bf16_contiguous(weight)
    bias_bf16 = _ensure_bf16_contiguous(bias)

    B, C_in, H_in, W_in = x_bf16.shape
    C_out, C_in_w, KH, KW = w_bf16.shape

    H_out_ct = (H_in - 1) * STRIDE - 2 * PADDING + (KERNEL_SIZE - 1) + 1
    W_out_ct = (W_in - 1) * STRIDE - 2 * PADDING + (KERNEL_SIZE - 1) + 1

    # ConvTranspose intermediate: (B, C_out, H_out, W_out)
    ct_out = torch.empty(
        (B, C_out, H_out_ct, W_out_ct),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    grid_ct = (H_out_ct // TILE_H, W_out_ct // TILE_W, B * C_out)
    conv_transpose_bf16_kernel[lambda: (grid_ct, (TILE_THREADS, 1, 1))](
        x_bf16, w_bf16, bias_bf16, ct_out,
        B, C_in, C_out, H_in, W_in, H_out_ct, W_out_ct, KH, KW, PADDING,
    )

    # MaxPool + Hardtanh
    H_pool = H_out_ct // MAXPOOL_STRIDE
    W_pool = W_out_ct // MAXPOOL_STRIDE
    pool_out = torch.empty(
        (B, C_out, H_pool, W_pool),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    grid_pool = (C_out * H_pool, B, 1)
    maxpool_hardtanh_bf16_kernel[lambda: (grid_pool, (W_pool, 1, 1))](
        ct_out, pool_out,
        B, C_out, H_out_ct, W_out_ct, H_pool, W_pool,
    )

    # Mean + Tanh
    N_spatial = H_pool * W_pool
    mean_out = torch.empty(
        (B, C_out),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    grid_mean = (C_out, B, 1)
    mean_tanh_bf16_kernel[lambda: (grid_mean, (BLOCK_SIZE, 1, 1))](
        pool_out, mean_out,
        B, C_out, N_spatial,
    )

    return mean_out.view(B, C_out, 1, 1).to(x.dtype)


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, followed by max pooling,
    hardtanh activation, mean operation, and tanh activation.
    Uses AveLang DSL kernels optimized for BF16 on AMD GPU.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        maxpool_kernel_size,
        maxpool_stride,
        hardtanh_min,
        hardtanh_max,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
        )

    def forward(self, x):
        weight = self.conv_transpose.weight.data
        bias = self.conv_transpose.bias.data
        return avelang_forward(x, weight, bias)


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]


def get_init_inputs():
    return [
        IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING,
        MAXPOOL_KERNEL_SIZE, MAXPOOL_STRIDE,
        HARDTANH_MIN, HARDTANH_MAX,
    ]
