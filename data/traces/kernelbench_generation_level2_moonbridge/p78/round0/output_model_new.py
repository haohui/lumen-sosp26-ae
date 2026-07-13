import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Compile-time problem dimensions (fixed by the problem spec)
BATCH: al.constexpr = 16
IN_C: al.constexpr = 32
OUT_C: al.constexpr = 64
IN_D: al.constexpr = 32
IN_H: al.constexpr = 32
IN_W: al.constexpr = 32
OUT_D: al.constexpr = 63
OUT_H: al.constexpr = 63
OUT_W: al.constexpr = 63
K_D: al.constexpr = 5
K_H: al.constexpr = 5
K_W: al.constexpr = 5
STRIDE: al.constexpr = 2
PADDING: al.constexpr = 2
POOL1_OUT: al.constexpr = 31
POOL2_OUT: al.constexpr = 10

BLOCK_SIZE: al.constexpr = 256

# Precomputed stride constants
IN_IC_STRIDE: al.constexpr = IN_D * IN_H * IN_W
IN_D_STRIDE: al.constexpr = IN_H * IN_W
IN_H_STRIDE: al.constexpr = IN_W
W_OC_STRIDE: al.constexpr = K_D * K_H * K_W
W_KD_STRIDE: al.constexpr = K_H * K_W
W_KH_STRIDE: al.constexpr = K_W
OUT_OC_STRIDE: al.constexpr = OUT_D * OUT_H * OUT_W
OUT_D_STRIDE: al.constexpr = OUT_H * OUT_W
OUT_H_STRIDE: al.constexpr = OUT_W

P1_IN_C_STRIDE: al.constexpr = OUT_D * OUT_H * OUT_W
P1_IN_D_STRIDE: al.constexpr = OUT_H * OUT_W
P1_IN_H_STRIDE: al.constexpr = OUT_W
P1_OUT_D_STRIDE: al.constexpr = POOL1_OUT * POOL1_OUT
P1_OUT_H_STRIDE: al.constexpr = POOL1_OUT

P2_IN_C_STRIDE: al.constexpr = POOL1_OUT * POOL1_OUT * POOL1_OUT
P2_IN_D_STRIDE: al.constexpr = POOL1_OUT * POOL1_OUT
P2_IN_H_STRIDE: al.constexpr = POOL1_OUT
P2_OUT_D_STRIDE: al.constexpr = POOL2_OUT * POOL2_OUT
P2_OUT_H_STRIDE: al.constexpr = POOL2_OUT

SR_IN_C_STRIDE: al.constexpr = POOL2_OUT * POOL2_OUT * POOL2_OUT
SR_IN_D_STRIDE: al.constexpr = POOL2_OUT * POOL2_OUT
SR_IN_H_STRIDE: al.constexpr = POOL2_OUT

# ---------------------------------------------------------------------------
# Kernel 1: 3D transposed convolution with shared-memory weight caching
# ---------------------------------------------------------------------------

@avelang.jit
def conv_transpose3d_bf16_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
):
    inp = al.make_tensor(
        input_ptr, al.bf16,
        al.make_layout((BATCH * IN_C * IN_D * IN_H * IN_W,), (1,))
    )
    wgt = al.make_tensor(
        weight_ptr, al.bf16,
        al.make_layout((IN_C * OUT_C * K_D * K_H * K_W,), (1,))
    )
    bias_t = al.make_tensor(bias_ptr, al.bf16, al.make_layout((OUT_C,), (1,)))
    out = al.make_tensor(
        output_ptr, al.bf16,
        al.make_layout((BATCH * OUT_C * OUT_D * OUT_H * OUT_W,), (1,))
    )

    shm_w = al.make_shared((K_D * K_H * K_W,), al.bf16)

    n = al.block_id(0)
    oc = al.block_id(1)
    tile_idx = al.block_id(2)
    tid = al.thread_id(0)

    global_idx = tile_idx * BLOCK_SIZE + tid
    spatial_volume = OUT_D * OUT_H * OUT_W
    kw_total = K_D * K_H * K_W

    ow = global_idx % OUT_W
    tmp = global_idx // OUT_W
    oh = tmp % OUT_H
    od = tmp // OUT_H

    n_in_base = n * IN_C * IN_IC_STRIDE
    n_out_base = n * OUT_C * OUT_OC_STRIDE
    oc_out_off = oc * OUT_OC_STRIDE
    out_off = n_out_base + oc_out_off + od * OUT_D_STRIDE + oh * OUT_H_STRIDE + ow

    acc = al.convert(0.0, al.f32)
    if global_idx < spatial_volume:
        acc = al.convert(bias_t[oc], al.f32)

    od_plus_pad = od + PADDING
    oh_plus_pad = oh + PADDING
    ow_plus_pad = ow + PADDING

    for ic in al.range(IN_C):
        ic_in_off = n_in_base + ic * IN_IC_STRIDE
        ic_w_off = ic * OUT_C * W_OC_STRIDE + oc * W_OC_STRIDE

        for wi in al.range(tid, kw_total, BLOCK_SIZE):
            kd_w = wi // (K_H * K_W)
            tmp_w = wi % (K_H * K_W)
            kh_w = tmp_w // K_W
            kw_w = tmp_w % K_W
            shm_w[wi] = wgt[ic_w_off + kd_w * W_KD_STRIDE + kh_w * W_KH_STRIDE + kw_w]
        al.syncthreads()

        if global_idx < spatial_volume:
            for kd in al.range(K_D):
                val_d = od_plus_pad - kd
                rem_d = val_d - (val_d // STRIDE) * STRIDE
                if rem_d == 0:
                    id_val = val_d // STRIDE
                    if id_val >= 0:
                        if id_val < IN_D:
                            id_off = id_val * IN_D_STRIDE

                            for kh in al.range(K_H):
                                val_h = oh_plus_pad - kh
                                rem_h = val_h - (val_h // STRIDE) * STRIDE
                                if rem_h == 0:
                                    ih_val = val_h // STRIDE
                                    if ih_val >= 0:
                                        if ih_val < IN_H:
                                            ih_off = ih_val * IN_H_STRIDE

                                            for kw in al.range(K_W):
                                                val_w = ow_plus_pad - kw
                                                rem_w = val_w - (val_w // STRIDE) * STRIDE
                                                if rem_w == 0:
                                                    iw_val = val_w // STRIDE
                                                    if iw_val >= 0:
                                                        if iw_val < IN_W:
                                                            in_idx = ic_in_off + id_off + ih_off + iw_val
                                                            shm_w_idx = kd * (K_H * K_W) + kh * K_W + kw
                                                            inp_val = al.convert(inp[in_idx], al.f32)
                                                            wgt_val = al.convert(shm_w[shm_w_idx], al.f32)
                                                            acc = acc + inp_val * wgt_val

        al.syncthreads()

    if global_idx < spatial_volume:
        out[out_off] = al.convert(acc, al.bf16)


# ---------------------------------------------------------------------------
# Kernel 2: 3D max pooling (k=2, unrolled)
# ---------------------------------------------------------------------------

@avelang.jit
def max_pool3d_k2_bf16_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
):
    inp = al.make_tensor(
        input_ptr, al.bf16,
        al.make_layout((BATCH * OUT_C * OUT_D * OUT_H * OUT_W,), (1,))
    )
    out = al.make_tensor(
        output_ptr, al.bf16,
        al.make_layout((BATCH * OUT_C * POOL1_OUT * POOL1_OUT * POOL1_OUT,), (1,))
    )

    block_idx = al.block_id(0)
    n = block_idx // OUT_C
    c = block_idx % OUT_C
    tid = al.thread_id(0)

    nc_in_base = n * OUT_C * P1_IN_C_STRIDE + c * P1_IN_C_STRIDE
    nc_out_base = n * OUT_C * POOL1_OUT * P1_OUT_D_STRIDE + c * POOL1_OUT * P1_OUT_D_STRIDE

    for idx in al.range(tid, POOL1_OUT * POOL1_OUT * POOL1_OUT, BLOCK_SIZE):
        ow = idx % POOL1_OUT
        tmp = idx // POOL1_OUT
        oh = tmp % POOL1_OUT
        od = tmp // POOL1_OUT

        id0 = od * 2 * P1_IN_D_STRIDE
        ih0 = oh * 2 * P1_IN_H_STRIDE
        iw0 = ow * 2
        base = nc_in_base + id0 + ih0 + iw0
        d0 = P1_IN_D_STRIDE
        h0 = P1_IN_H_STRIDE

        cur = inp[base]
        v = inp[base + 1]
        if v > cur:
            cur = v
        v = inp[base + h0]
        if v > cur:
            cur = v
        v = inp[base + h0 + 1]
        if v > cur:
            cur = v
        v = inp[base + d0]
        if v > cur:
            cur = v
        v = inp[base + d0 + 1]
        if v > cur:
            cur = v
        v = inp[base + d0 + h0]
        if v > cur:
            cur = v
        v = inp[base + d0 + h0 + 1]
        if v > cur:
            cur = v

        out_off = nc_out_base + od * P1_OUT_D_STRIDE + oh * P1_OUT_H_STRIDE + ow
        out[out_off] = cur


# ---------------------------------------------------------------------------
# Kernel 3: 3D max pooling (k=3)
# ---------------------------------------------------------------------------

@avelang.jit
def max_pool3d_k3_bf16_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
):
    inp = al.make_tensor(
        input_ptr, al.bf16,
        al.make_layout((BATCH * OUT_C * POOL1_OUT * POOL1_OUT * POOL1_OUT,), (1,))
    )
    out = al.make_tensor(
        output_ptr, al.bf16,
        al.make_layout((BATCH * OUT_C * POOL2_OUT * POOL2_OUT * POOL2_OUT,), (1,))
    )

    block_idx = al.block_id(0)
    n = block_idx // OUT_C
    c = block_idx % OUT_C
    tid = al.thread_id(0)

    nc_in_base = n * OUT_C * P2_IN_C_STRIDE + c * P2_IN_C_STRIDE
    nc_out_base = n * OUT_C * POOL2_OUT * P2_OUT_D_STRIDE + c * POOL2_OUT * P2_OUT_D_STRIDE

    for idx in al.range(tid, POOL2_OUT * POOL2_OUT * POOL2_OUT, BLOCK_SIZE):
        ow = idx % POOL2_OUT
        tmp = idx // POOL2_OUT
        oh = tmp % POOL2_OUT
        od = tmp // POOL2_OUT

        id0 = od * 3 * P2_IN_D_STRIDE
        ih0 = oh * 3 * P2_IN_H_STRIDE
        iw0 = ow * 3
        base = nc_in_base + id0 + ih0 + iw0
        d0 = P2_IN_D_STRIDE
        h0 = P2_IN_H_STRIDE

        cur = inp[base]
        for di in al.range(3):
            for hi in al.range(3):
                for wi in al.range(3):
                    vi = inp[base + di * d0 + hi * h0 + wi]
                    if vi > cur:
                        cur = vi

        out_off = nc_out_base + od * P2_OUT_D_STRIDE + oh * P2_OUT_H_STRIDE + ow
        out[out_off] = cur


# ---------------------------------------------------------------------------
# Kernel 4: Sum reduce over channel dimension
# ---------------------------------------------------------------------------

@avelang.jit
def sum_reduce_channel_bf16_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
):
    inp = al.make_tensor(
        input_ptr, al.bf16,
        al.make_layout((BATCH * OUT_C * POOL2_OUT * POOL2_OUT * POOL2_OUT,), (1,))
    )
    out = al.make_tensor(
        output_ptr, al.bf16,
        al.make_layout((BATCH * POOL2_OUT * POOL2_OUT * POOL2_OUT,), (1,))
    )

    block_idx = al.block_id(0)
    n = block_idx
    tid = al.thread_id(0)

    for idx in al.range(tid, POOL2_OUT * POOL2_OUT * POOL2_OUT, BLOCK_SIZE):
        w = idx % POOL2_OUT
        tmp = idx // POOL2_OUT
        h = tmp % POOL2_OUT
        d = tmp // POOL2_OUT

        acc = al.convert(0.0, al.f32)
        in_base = n * OUT_C * SR_IN_C_STRIDE + d * SR_IN_D_STRIDE + h * SR_IN_H_STRIDE + w
        for c in al.range(OUT_C):
            acc = acc + al.convert(inp[in_base + c * SR_IN_C_STRIDE], al.f32)

        out_off = n * POOL2_OUT * SR_IN_D_STRIDE + d * SR_IN_D_STRIDE + h * SR_IN_H_STRIDE + w
        out[out_off] = al.convert(acc, al.bf16)


# ---------------------------------------------------------------------------
# Host wrappers
# ---------------------------------------------------------------------------

def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: int,
) -> torch.Tensor:
    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = _prepare_bf16_cuda_contiguous(weight)
    b_bf16 = _prepare_bf16_cuda_contiguous(bias)

    N, IC, ID, IH, IW = x_bf16.shape
    weight_IC, OC, KD, KH, KW = w_bf16.shape
    OD = (ID - 1) * stride - 2 * padding + KD
    OH = (IH - 1) * stride - 2 * padding + KH
    OW = (IW - 1) * stride - 2 * padding + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x_bf16.device, dtype=torch.bfloat16)

    num_spatial_tiles = (OD * OH * OW + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (N, OC, num_spatial_tiles)
    block = (BLOCK_SIZE, 1, 1)
    conv_transpose3d_bf16_kernel[lambda: (grid, block)](x_bf16, w_bf16, b_bf16, out)
    return out


def avelang_max_pool3d_k2(x: torch.Tensor) -> torch.Tensor:
    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    N, C, ID, IH, IW = x_bf16.shape
    OD = ID // 2
    OH = IH // 2
    OW = IW // 2

    out = torch.empty((N, C, OD, OH, OW), device=x_bf16.device, dtype=torch.bfloat16)

    grid = (N * C, 1, 1)
    block = (BLOCK_SIZE, 1, 1)
    max_pool3d_k2_bf16_kernel[lambda: (grid, block)](x_bf16, out)
    return out


def avelang_max_pool3d_k3(x: torch.Tensor) -> torch.Tensor:
    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    N, C, ID, IH, IW = x_bf16.shape
    OD = ID // 3
    OH = IH // 3
    OW = IW // 3

    out = torch.empty((N, C, OD, OH, OW), device=x_bf16.device, dtype=torch.bfloat16)

    grid = (N * C, 1, 1)
    block = (BLOCK_SIZE, 1, 1)
    max_pool3d_k3_bf16_kernel[lambda: (grid, block)](x_bf16, out)
    return out


def avelang_sum_reduce_channel(x: torch.Tensor) -> torch.Tensor:
    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    N, C, D, H, W = x_bf16.shape

    out = torch.empty((N, 1, D, H, W), device=x_bf16.device, dtype=torch.bfloat16)

    grid = (N, 1, 1)
    block = (BLOCK_SIZE, 1, 1)
    sum_reduce_channel_bf16_kernel[lambda: (grid, block)](x_bf16, out)
    return out


# ---------------------------------------------------------------------------
# ModelNew
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
        )

    def forward(self, x):
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        stride_val = self.conv_transpose.stride[0]
        padding_val = self.conv_transpose.padding[0]

        x = avelang_conv_transpose3d(x, weight, bias, stride_val, padding_val)
        x = avelang_max_pool3d_k2(x)
        x = avelang_max_pool3d_k3(x)
        x = avelang_sum_reduce_channel(x)
        return x


# ---------------------------------------------------------------------------
# Problem-level constants and harness helpers
# ---------------------------------------------------------------------------

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size = 5
stride = 2
padding = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
