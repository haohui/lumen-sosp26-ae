import torch
import torch.nn as nn
import substrate
import substrate.language as S


# Fixed problem shape from target model/get_inputs.
BATCH_SIZE = 128
IN_CHANNELS = 3
OUT_CHANNELS = 16
IN_D = 16
IN_H = 32
IN_W = 32
K_D = 3
K_H = 3
K_W = 3
POOL_K = 2

# Conv3d output shape (stride=1, padding=0 by default)
OUT_D = IN_D - K_D + 1  # 14
OUT_H = IN_H - K_H + 1  # 30
OUT_W = IN_W - K_W + 1  # 30

# After first MaxPool3d
POOL1_D = OUT_D // POOL_K  # 7
POOL1_H = OUT_H // POOL_K  # 15
POOL1_W = OUT_W // POOL_K  # 15

# After second MaxPool3d
POOL2_D = POOL1_D // POOL_K  # 3
POOL2_H = POOL1_H // POOL_K  # 7
POOL2_W = POOL1_W // POOL_K  # 7

# Conv3d kernel parameters
THREADS_CONV = 256
WEIGHT_ELEMS = IN_CHANNELS * K_D * K_H * K_W  # 3 * 27 = 81
WEIGHT_LOAD_ITERS = (WEIGHT_ELEMS + THREADS_CONV - 1) // THREADS_CONV
SPATIAL_ELEMS = OUT_D * OUT_H * OUT_W  # 14 * 30 * 30 = 12600
SPATIAL_TILES = (SPATIAL_ELEMS + THREADS_CONV - 1) // THREADS_CONV

# Softmax parameters (along channel dimension)
SOFTMAX_CHANNELS = OUT_CHANNELS  # 16
SOFTMAX_SPATIAL = OUT_D * OUT_H * OUT_W  # 12600
SOFTMAX_THREADS = 256

# MaxPool3d parameters
POOL_THREADS = 256
POOL1_ELEMS = POOL1_D * POOL1_H * POOL1_W  # 7 * 15 * 15 = 1575
POOL2_ELEMS = POOL2_D * POOL2_H * POOL2_W  # 3 * 7 * 7 = 147

NEG_INF_F32 = -3.4028234663852886e38
LOG2E = 1.4426950408889634


# ============== Conv3d Kernels ==============

@substrate.jit
def conv3d_3x3x3_bf16_kernel(
    x: S.Tensor((BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W), S.bf16),
    w: S.Tensor((OUT_CHANNELS, IN_CHANNELS, K_D, K_H, K_W), S.bf16),
    b: S.Tensor((OUT_CHANNELS,), S.bf16),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), S.bf16),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    oc = bid % OUT_CHANNELS

    # Shared memory tile for one output channel's kernel weights.
    s_w = S.make_shared((WEIGHT_ELEMS,), S.bf16)

    for it in S.range(WEIGHT_LOAD_ITERS):
        w_flat = it * THREADS_CONV + tid
        if w_flat < WEIGHT_ELEMS:
            ic = w_flat // (K_D * K_H * K_W)
            rem = w_flat % (K_D * K_H * K_W)
            kd = rem // (K_H * K_W)
            rem2 = rem % (K_H * K_W)
            kh = rem2 // K_W
            kw = rem2 % K_W
            s_w[w_flat] = w[oc, ic, kd, kh, kw]

    S.syncthreads()

    for t in S.range(SPATIAL_TILES):
        pos = t * THREADS_CONV + tid
        if pos < SPATIAL_ELEMS:
            od = pos // (OUT_H * OUT_W)
            rem = pos % (OUT_H * OUT_W)
            oh = rem // OUT_W
            ow = rem % OUT_W

            acc = S.convert(b[oc], S.f32)

            for ic in S.range(IN_CHANNELS):
                for kd in S.range(K_D):
                    id_nom = od + kd
                    if id_nom >= 0 and id_nom < IN_D:
                        for kh in S.range(K_H):
                            ih_nom = oh + kh
                            if ih_nom >= 0 and ih_nom < IN_H:
                                for kw in S.range(K_W):
                                    iw_nom = ow + kw
                                    if iw_nom >= 0 and iw_nom < IN_W:
                                        wf = ic * (K_D * K_H * K_W) + kd * (K_H * K_W) + kh * K_W + kw
                                        xv = S.convert(x[n, ic, id_nom, ih_nom, iw_nom], S.f32)
                                        wv = S.convert(s_w[wf], S.f32)
                                        acc = acc + xv * wv

            out[n, oc, od, oh, ow] = S.convert(acc, S.bf16)


@substrate.jit
def conv3d_3x3x3_f32_kernel(
    x: S.Tensor((BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W), S.f32),
    w: S.Tensor((OUT_CHANNELS, IN_CHANNELS, K_D, K_H, K_W), S.f32),
    b: S.Tensor((OUT_CHANNELS,), S.f32),
    out: S.Tensor((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), S.f32),
):
    tid = S.thread_id(0)
    bid = S.block_id(0)

    n = bid // OUT_CHANNELS
    oc = bid % OUT_CHANNELS

    s_w = S.make_shared((WEIGHT_ELEMS,), S.f32)

    for it in S.range(WEIGHT_LOAD_ITERS):
        w_flat = it * THREADS_CONV + tid
        if w_flat < WEIGHT_ELEMS:
            ic = w_flat // (K_D * K_H * K_W)
            rem = w_flat % (K_D * K_H * K_W)
            kd = rem // (K_H * K_W)
            rem2 = rem % (K_H * K_W)
            kh = rem2 // K_W
            kw = rem2 % K_W
            s_w[w_flat] = w[oc, ic, kd, kh, kw]

    S.syncthreads()

    for t in S.range(SPATIAL_TILES):
        pos = t * THREADS_CONV + tid
        if pos < SPATIAL_ELEMS:
            od = pos // (OUT_H * OUT_W)
            rem = pos % (OUT_H * OUT_W)
            oh = rem // OUT_W
            ow = rem % OUT_W

            acc = b[oc]

            for ic in S.range(IN_CHANNELS):
                for kd in S.range(K_D):
                    id_nom = od + kd
                    if id_nom >= 0 and id_nom < IN_D:
                        for kh in S.range(K_H):
                            ih_nom = oh + kh
                            if ih_nom >= 0 and ih_nom < IN_H:
                                for kw in S.range(K_W):
                                    iw_nom = ow + kw
                                    if iw_nom >= 0 and iw_nom < IN_W:
                                        wf = ic * (K_D * K_H * K_W) + kd * (K_H * K_W) + kh * K_W + kw
                                        acc = acc + x[n, ic, id_nom, ih_nom, iw_nom] * s_w[wf]

            out[n, oc, od, oh, ow] = acc


# ============== Softmax Kernels (along channel dimension) ==============

@substrate.jit
def softmax_channel_bf16_kernel(
    x: S.Pointer(S.bf16),
    out: S.Pointer(S.bf16),
    batch: S.u32,
    channels: S.u32,
    spatial: S.u32,
):
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if tid < batch * spatial:
        n = tid // spatial
        s = tid % spatial

        layout = S.make_layout((batch, channels, spatial), (channels * spatial, spatial, 1))
        gx = S.make_tensor(x, S.bf16, layout)
        gout = S.make_tensor(out, S.bf16, layout)

        # Find max for numerical stability
        max_val = S.convert(gx[n, 0, s], S.f32)
        for c in S.range(1, SOFTMAX_CHANNELS):
            v = S.convert(gx[n, c, s], S.f32)
            if v > max_val:
                max_val = v

        # Compute exp(x - max) and sum
        sum_exp = S.convert(0.0, S.f32)
        log2e = S.convert(LOG2E, S.f32)
        neg_max_log2e = -max_val * log2e

        exp_vals = S.make_local((SOFTMAX_CHANNELS,), S.f32)
        for c in S.range(SOFTMAX_CHANNELS):
            xv = S.convert(gx[n, c, s], S.f32)
            exp_vals[c] = S.exp2((xv - max_val) * log2e)
            sum_exp = sum_exp + exp_vals[c]

        # Normalize
        inv_sum = S.convert(1.0, S.f32) / sum_exp
        for c in S.range(SOFTMAX_CHANNELS):
            gout[n, c, s] = S.convert(exp_vals[c] * inv_sum, S.bf16)


@substrate.jit
def softmax_channel_f32_kernel(
    x: S.Pointer(S.f32),
    out: S.Pointer(S.f32),
    batch: S.u32,
    channels: S.u32,
    spatial: S.u32,
):
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)

    if tid < batch * spatial:
        n = tid // spatial
        s = tid % spatial

        layout = S.make_layout((batch, channels, spatial), (channels * spatial, spatial, 1))
        gx = S.make_tensor(x, S.f32, layout)
        gout = S.make_tensor(out, S.f32, layout)

        # Find max for numerical stability
        max_val = gx[n, 0, s]
        for c in S.range(1, SOFTMAX_CHANNELS):
            v = gx[n, c, s]
            if v > max_val:
                max_val = v

        # Compute exp(x - max) and sum
        sum_exp = S.convert(0.0, S.f32)
        log2e = S.convert(LOG2E, S.f32)

        exp_vals = S.make_local((SOFTMAX_CHANNELS,), S.f32)
        for c in S.range(SOFTMAX_CHANNELS):
            xv = gx[n, c, s]
            exp_vals[c] = S.exp2((xv - max_val) * log2e)
            sum_exp = sum_exp + exp_vals[c]

        # Normalize
        inv_sum = S.convert(1.0, S.f32) / sum_exp
        for c in S.range(SOFTMAX_CHANNELS):
            gout[n, c, s] = exp_vals[c] * inv_sum


# ============== MaxPool3d Kernels ==============

@substrate.jit
def maxpool3d_bf16_kernel(
    x: S.Pointer(S.bf16),
    out: S.Pointer(S.bf16),
    in_d: S.u32,
    in_h: S.u32,
    in_w: S.u32,
    out_d: S.u32,
    out_h: S.u32,
    out_w: S.u32,
    pool_k: S.u32,
):
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    bc = S.block_id(1)
    bc_count = S.block_dim(1)

    total_out = out_d * out_h * out_w
    if tid < total_out:
        od = tid // (out_h * out_w)
        rem = tid % (out_h * out_w)
        oh = rem // out_w
        ow = rem % out_w

        layout_in = S.make_layout((bc_count, in_d, in_h, in_w), (in_d * in_h * in_w, in_h * in_w, in_w, 1))
        layout_out = S.make_layout((bc_count, out_d, out_h, out_w), (out_d * out_h * out_w, out_h * out_w, out_w, 1))
        gx = S.make_tensor(x, S.bf16, layout_in)
        gout = S.make_tensor(out, S.bf16, layout_out)

        max_val = S.convert(NEG_INF_F32, S.f32)

        for pd in S.range(2):
            id_pos = od * pool_k + pd
            if id_pos < in_d:
                for ph in S.range(2):
                    ih_pos = oh * pool_k + ph
                    if ih_pos < in_h:
                        for pw in S.range(2):
                            iw_pos = ow * pool_k + pw
                            if iw_pos < in_w:
                                v = S.convert(gx[bc, id_pos, ih_pos, iw_pos], S.f32)
                                if v > max_val:
                                    max_val = v

        gout[bc, od, oh, ow] = S.convert(max_val, S.bf16)


@substrate.jit
def maxpool3d_f32_kernel(
    x: S.Pointer(S.f32),
    out: S.Pointer(S.f32),
    in_d: S.u32,
    in_h: S.u32,
    in_w: S.u32,
    out_d: S.u32,
    out_h: S.u32,
    out_w: S.u32,
    pool_k: S.u32,
):
    tid = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    bc = S.block_id(1)
    bc_count = S.block_dim(1)

    total_out = out_d * out_h * out_w
    if tid < total_out:
        od = tid // (out_h * out_w)
        rem = tid % (out_h * out_w)
        oh = rem // out_w
        ow = rem % out_w

        layout_in = S.make_layout((bc_count, in_d, in_h, in_w), (in_d * in_h * in_w, in_h * in_w, in_w, 1))
        layout_out = S.make_layout((bc_count, out_d, out_h, out_w), (out_d * out_h * out_w, out_h * out_w, out_w, 1))
        gx = S.make_tensor(x, S.f32, layout_in)
        gout = S.make_tensor(out, S.f32, layout_out)

        max_val = S.convert(NEG_INF_F32, S.f32)

        for pd in S.range(2):
            id_pos = od * pool_k + pd
            if id_pos < in_d:
                for ph in S.range(2):
                    ih_pos = oh * pool_k + ph
                    if ih_pos < in_h:
                        for pw in S.range(2):
                            iw_pos = ow * pool_k + pw
                            if iw_pos < in_w:
                                v = gx[bc, id_pos, ih_pos, iw_pos]
                                if v > max_val:
                                    max_val = v

        gout[bc, od, oh, ow] = max_val


# ============== Host wrappers ==============

def _launch_conv3d_bf16(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), device=x.device, dtype=torch.bfloat16)
    conv3d_3x3x3_bf16_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_CONV, 1, 1))
    ](x, w, b, out)
    return out


def _launch_conv3d_f32(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((BATCH_SIZE, OUT_CHANNELS, OUT_D, OUT_H, OUT_W), device=x.device, dtype=torch.float32)
    conv3d_3x3x3_f32_kernel[
        lambda: ((BATCH_SIZE * OUT_CHANNELS, 1, 1), (THREADS_CONV, 1, 1))
    ](x, w, b, out)
    return out


def substrate_conv3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    if x.dtype == torch.bfloat16:
        return _launch_conv3d_bf16(x, weight, bias)
    elif x.dtype == torch.float32:
        return _launch_conv3d_f32(x, weight, bias)
    else:
        raise TypeError(f"Unsupported dtype for conv3d: {x.dtype}")


def _launch_softmax_bf16(x: torch.Tensor) -> torch.Tensor:
    batch = x.shape[0]
    channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3] * x.shape[4]
    out = torch.empty_like(x)
    total = batch * spatial
    grid = (total + SOFTMAX_THREADS - 1) // SOFTMAX_THREADS
    softmax_channel_bf16_kernel[lambda: ((grid, 1, 1), (SOFTMAX_THREADS, 1, 1))](
        x, out, batch, channels, spatial
    )
    return out


def _launch_softmax_f32(x: torch.Tensor) -> torch.Tensor:
    batch = x.shape[0]
    channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3] * x.shape[4]
    out = torch.empty_like(x)
    total = batch * spatial
    grid = (total + SOFTMAX_THREADS - 1) // SOFTMAX_THREADS
    softmax_channel_f32_kernel[lambda: ((grid, 1, 1), (SOFTMAX_THREADS, 1, 1))](
        x, out, batch, channels, spatial
    )
    return out


def substrate_softmax_channel(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    if x.dtype == torch.bfloat16:
        return _launch_softmax_bf16(x)
    elif x.dtype == torch.float32:
        return _launch_softmax_f32(x)
    else:
        raise TypeError(f"Unsupported dtype for softmax: {x.dtype}")


def _launch_maxpool3d_bf16(x: torch.Tensor, pool_k: int) -> torch.Tensor:
    n, c, in_d, in_h, in_w = x.shape
    out_d = in_d // pool_k
    out_h = in_h // pool_k
    out_w = in_w // pool_k

    out = torch.empty((n, c, out_d, out_h, out_w), device=x.device, dtype=torch.bfloat16)

    bc_count = n * c
    total_out = out_d * out_h * out_w
    grid_x = (total_out + POOL_THREADS - 1) // POOL_THREADS
    grid_y = bc_count

    maxpool3d_bf16_kernel[lambda: ((grid_x, grid_y, 1), (POOL_THREADS, 1, 1))](
        x, out, in_d, in_h, in_w, out_d, out_h, out_w, pool_k
    )
    return out


def _launch_maxpool3d_f32(x: torch.Tensor, pool_k: int) -> torch.Tensor:
    n, c, in_d, in_h, in_w = x.shape
    out_d = in_d // pool_k
    out_h = in_h // pool_k
    out_w = in_w // pool_k

    out = torch.empty((n, c, out_d, out_h, out_w), device=x.device, dtype=torch.float32)

    bc_count = n * c
    total_out = out_d * out_h * out_w
    grid_x = (total_out + POOL_THREADS - 1) // POOL_THREADS
    grid_y = bc_count

    maxpool3d_f32_kernel[lambda: ((grid_x, grid_y, 1), (POOL_THREADS, 1, 1))](
        x, out, in_d, in_h, in_w, out_d, out_h, out_w, pool_k
    )
    return out


def substrate_maxpool3d(x: torch.Tensor, pool_k: int) -> torch.Tensor:
    x = x.contiguous()
    if x.dtype == torch.bfloat16:
        return _launch_maxpool3d_bf16(x, pool_k)
    elif x.dtype == torch.float32:
        return _launch_maxpool3d_f32(x, pool_k)
    else:
        raise TypeError(f"Unsupported dtype for maxpool3d: {x.dtype}")


class ModelNew(nn.Module):
    """
    Optimized Substrate implementation of 3D Conv + Softmax + MaxPool3d x2 pipeline.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_kernel_size: int):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check input shape
        expected_shape = (BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W)
        if tuple(x.shape) != expected_shape:
            raise NotImplementedError(
                f"ModelNew currently supports input shape {expected_shape}, got {tuple(x.shape)}"
            )

        orig_device = x.device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

        if not x.is_cuda:
            x = x.cuda()

        # Convert to BF16 for computation
        input_dtype = x.dtype
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        # Conv3d
        w = self.conv.weight
        b = self.conv.bias
        if b is None:
            b = torch.zeros((OUT_CHANNELS,), device=w.device, dtype=w.dtype)

        if w.dtype != torch.bfloat16:
            w = w.to(torch.bfloat16)
        if b.dtype != torch.bfloat16:
            b = b.to(torch.bfloat16)

        w = w.contiguous()
        b = b.contiguous()

        x = substrate_conv3d(x, w, b)

        # Softmax along channel dimension
        x = substrate_softmax_channel(x)

        # MaxPool3d x2
        x = substrate_maxpool3d(x, self.pool_kernel_size)
        x = substrate_maxpool3d(x, self.pool_kernel_size)

        # Convert back to original dtype if needed
        if input_dtype != torch.bfloat16:
            x = x.to(input_dtype)

        if orig_device.type != "cuda":
            x = x.to(orig_device)

        return x


batch_size = BATCH_SIZE
in_channels = IN_CHANNELS
out_channels = OUT_CHANNELS
depth, height, width = IN_D, IN_H, IN_W
kernel_size = 3
pool_kernel_size = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
