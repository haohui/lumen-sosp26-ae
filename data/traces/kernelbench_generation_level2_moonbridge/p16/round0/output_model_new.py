import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Problem dimensions
BATCH_SIZE = 128
IN_CHANNELS = 64
OUT_CHANNELS = 64
H_IN = 128
W_IN = 128
KH = 3
KW = 3
STRIDE = 2
PADDING = 1
OUTPUT_PADDING = 1
ADD_VALUE = 0.5
SCALE = 2.0

H_OUT = (H_IN - 1) * STRIDE - 2 * PADDING + KH + OUTPUT_PADDING
W_OUT = (W_IN - 1) * STRIDE - 2 * PADDING + KW + OUTPUT_PADDING

# Tile config
TILE_H = 16
TILE_W = 16
THREADS = TILE_H * TILE_W
GRID_X = W_OUT // TILE_W
GRID_Y = H_OUT // TILE_H
GRID_Z = BATCH_SIZE * OUT_CHANNELS

# Shared memory sizes
SHM_W_ELEMS = IN_CHANNELS * KH * KW
TILE_H_IN = TILE_H // STRIDE + 2
TILE_W_IN = TILE_W // STRIDE + 2
SHM_IN_SPATIAL = TILE_H_IN * TILE_W_IN
SHM_IN_ELEMS = IN_CHANNELS * SHM_IN_SPATIAL

# Flat buffer sizes
X_FLAT = BATCH_SIZE * IN_CHANNELS * H_IN * W_IN
W_FLAT = IN_CHANNELS * OUT_CHANNELS * KH * KW
OUT_FLAT = BATCH_SIZE * OUT_CHANNELS * H_OUT * W_OUT

B_STRIDE_IN = IN_CHANNELS * H_IN * W_IN
IC_STRIDE_IN = H_IN * W_IN
OC_STRIDE_W = KH * KW
IC_STRIDE_W = OUT_CHANNELS * OC_STRIDE_W


@avelang.jit
def conv_transpose_mish_fused_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
):
    tid = al.thread_id(0)
    block_x = al.block_id(0)
    block_y = al.block_id(1)
    block_z = al.block_id(2)

    b = block_z // OUT_CHANNELS
    oc = block_z % OUT_CHANNELS

    oh_start = block_y * TILE_H
    ow_start = block_x * TILE_W

    ty = tid // TILE_W
    tx = tid % TILE_W
    oh = oh_start + ty
    ow = ow_start + tx

    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((X_FLAT,), (1,)))
    w = al.make_tensor(w_ptr, al.bf16, al.make_layout((W_FLAT,), (1,)))
    bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((OUT_CHANNELS,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((OUT_FLAT,), (1,)))

    # Input tile boundaries
    ih_start = oh_start // STRIDE
    iw_start = ow_start // STRIDE
    b_off = b * B_STRIDE_IN

    # Load input tile into LDS
    shm_in = al.make_shared((SHM_IN_ELEMS,), al.bf16)
    nload = (SHM_IN_ELEMS + THREADS - 1) // THREADS
    for i in al.range(nload):
        idx = tid + i * THREADS
        if idx < SHM_IN_ELEMS:
            ic = idx // SHM_IN_SPATIAL
            sp = idx % SHM_IN_SPATIAL
            tih = sp // TILE_W_IN
            tiw = sp % TILE_W_IN
            g_ih = ih_start + tih
            g_iw = iw_start + tiw
            if g_ih < H_IN and g_iw < W_IN:
                shm_in[idx] = x[b_off + ic * IC_STRIDE_IN + g_ih * W_IN + g_iw]
            else:
                shm_in[idx] = al.convert(0.0, al.bf16)
    al.syncthreads()

    # Load weight for this oc into LDS
    shm_w = al.make_shared((SHM_W_ELEMS,), al.bf16)
    for i in al.range(3):
        idx = tid + i * THREADS
        if idx < SHM_W_ELEMS:
            ic_w = idx // 9
            rem = idx % 9
            kh_w = rem // 3
            kw_w = rem % 3
            shm_w[idx] = w[ic_w * IC_STRIDE_W + oc * OC_STRIDE_W + kh_w * KW + kw_w]
    al.syncthreads()

    if oh < H_OUT and ow < W_OUT:
        zero_f = al.convert(0.0, al.f32)
        one_f = al.convert(1.0, al.f32)
        neg_one_f = al.convert(-1.0, al.f32)
        add_val_f = al.convert(ADD_VALUE, al.f32)
        scale_f = al.convert(SCALE, al.f32)
        h_in_var = al.convert(H_IN, al.i32)
        w_in_var = al.convert(W_IN, al.i32)

        acc = zero_f

        oh_is_odd = (oh % STRIDE) != 0
        ow_is_odd = (ow % STRIDE) != 0

        if not oh_is_odd:
            ih = oh // STRIDE
            if ih < h_in_var:
                ih_rel = ih - ih_start
                if not ow_is_odd:
                    iw = ow // STRIDE
                    if iw < w_in_var:
                        iw_rel = iw - iw_start
                        for ic in al.range(IN_CHANNELS):
                            wv = al.convert(shm_w[ic * 9 + 4], al.f32)
                            xv = al.convert(shm_in[ic * SHM_IN_SPATIAL + ih_rel * TILE_W_IN + iw_rel], al.f32)
                            acc = acc + xv * wv
                else:
                    iw0 = (ow + PADDING) // STRIDE
                    iw2 = (ow + PADDING - KW + 1) // STRIDE
                    v0 = iw0 < w_in_var
                    v2 = iw2 < w_in_var
                    iw0_rel = iw0 - iw_start
                    iw2_rel = iw2 - iw_start
                    for ic in al.range(IN_CHANNELS):
                        ic_in = ic * SHM_IN_SPATIAL + ih_rel * TILE_W_IN
                        if v0:
                            wv = al.convert(shm_w[ic * 9 + 3], al.f32)
                            xv = al.convert(shm_in[ic_in + iw0_rel], al.f32)
                            acc = acc + xv * wv
                        if v2:
                            wv = al.convert(shm_w[ic * 9 + 5], al.f32)
                            xv = al.convert(shm_in[ic_in + iw2_rel], al.f32)
                            acc = acc + xv * wv
        else:
            ih0 = (oh + PADDING) // STRIDE
            ih2 = (oh + PADDING - KH + 1) // STRIDE
            vh0 = ih0 < h_in_var
            vh2 = ih2 < h_in_var
            ih0_rel = ih0 - ih_start
            ih2_rel = ih2 - ih_start
            if not ow_is_odd:
                iw = ow // STRIDE
                if iw < w_in_var:
                    iw_rel = iw - iw_start
                    for ic in al.range(IN_CHANNELS):
                        ic_in = ic * SHM_IN_SPATIAL + iw_rel
                        if vh0:
                            wv = al.convert(shm_w[ic * 9 + 1], al.f32)
                            xv = al.convert(shm_in[ic_in + ih0_rel * TILE_W_IN], al.f32)
                            acc = acc + xv * wv
                        if vh2:
                            wv = al.convert(shm_w[ic * 9 + 7], al.f32)
                            xv = al.convert(shm_in[ic_in + ih2_rel * TILE_W_IN], al.f32)
                            acc = acc + xv * wv
            else:
                iw0 = (ow + PADDING) // STRIDE
                iw2 = (ow + PADDING - KW + 1) // STRIDE
                vw0 = iw0 < w_in_var
                vw2 = iw2 < w_in_var
                iw0_rel = iw0 - iw_start
                iw2_rel = iw2 - iw_start
                for ic in al.range(IN_CHANNELS):
                    ic_off = ic * SHM_IN_SPATIAL
                    if vh0 and vw0:
                        wv = al.convert(shm_w[ic * 9 + 0], al.f32)
                        xv = al.convert(shm_in[ic_off + ih0_rel * TILE_W_IN + iw0_rel], al.f32)
                        acc = acc + xv * wv
                    if vh0 and vw2:
                        wv = al.convert(shm_w[ic * 9 + 2], al.f32)
                        xv = al.convert(shm_in[ic_off + ih0_rel * TILE_W_IN + iw2_rel], al.f32)
                        acc = acc + xv * wv
                    if vh2 and vw0:
                        wv = al.convert(shm_w[ic * 9 + 6], al.f32)
                        xv = al.convert(shm_in[ic_off + ih2_rel * TILE_W_IN + iw0_rel], al.f32)
                        acc = acc + xv * wv
                    if vh2 and vw2:
                        wv = al.convert(shm_w[ic * 9 + 8], al.f32)
                        xv = al.convert(shm_in[ic_off + ih2_rel * TILE_W_IN + iw2_rel], al.f32)
                        acc = acc + xv * wv

        # Bias + epilogue
        bias_val = al.convert(bias[oc], al.f32)
        acc = acc + bias_val

        sp = al.log(one_f + al.exp(acc))
        mish_val = acc * al.tanh(sp)

        result = mish_val + add_val_f

        if result > one_f:
            result = one_f
        if result < neg_one_f:
            result = neg_one_f

        result = result * scale_f

        out_idx = b * OUT_CHANNELS * H_OUT * W_OUT + oc * H_OUT * W_OUT + oh * W_OUT + ow
        out[out_idx] = al.convert(result, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_transpose_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight)
    bias_bf16 = _prepare_bf16_cuda_contiguous(bias)

    B, IC, H, W = x_bf16.shape
    IC_w, OC_w, KH_w, KW_w = weight_bf16.shape

    if B != BATCH_SIZE or IC != IN_CHANNELS or H != H_IN or W != W_IN:
        raise ValueError(
            f"Expected input shape ({BATCH_SIZE}, {IN_CHANNELS}, {H_IN}, {W_IN}), "
            f"got ({B}, {IC}, {H}, {W})"
        )
    if IC_w != IN_CHANNELS or OC_w != OUT_CHANNELS or KH_w != KH or KW_w != KW:
        raise ValueError(
            f"Expected weight shape ({IN_CHANNELS}, {OUT_CHANNELS}, {KH}, {KW}), "
            f"got ({IC_w}, {OC_w}, {KH_w}, {KW_w})"
        )

    out = torch.empty((B, OUT_CHANNELS, H_OUT, W_OUT), device=x_bf16.device, dtype=torch.bfloat16)

    grid = (GRID_X, GRID_Y, GRID_Z)
    conv_transpose_mish_fused_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, weight_bf16, bias_bf16, out,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.add_value = add_value
        self.scale = scale

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return avelang_conv_transpose_fused(x, self.weight, self.bias)


batch_size = 128
in_channels = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
add_value = 0.5
scale = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]
