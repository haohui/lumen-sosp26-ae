import torch
import torch.nn as nn
import avelang
import avelang.language as al

# =============================================================================
# Compile-time constants
# =============================================================================
BATCH_SIZE = 16
IN_CHANNELS = 16
OUT_CHANNELS = 32
D_IN = 16
H_IN = 32
W_IN = 32
D_OUT = 32
H_OUT = 64
W_OUT = 64
KD_CONST = 3
KH_CONST = 3
KW_CONST = 3

TILE_D = 8
TILE_H = 8
TILE_W = 8
TILE_OC = 8
THREADS = 256

NUM_D_TILES = 4
NUM_H_TILES = 8
NUM_W_TILES = 8
NUM_OC_GROUPS = 4

NEG_SLOPE = 0.2

SHM_WEIGHT_SIZE = 3456

# Wrapped compile-time values
_BATCH_SIZE = al.constexpr(BATCH_SIZE)
_IN_CHANNELS = al.constexpr(IN_CHANNELS)
_OUT_CHANNELS = al.constexpr(OUT_CHANNELS)
_D_IN = al.constexpr(D_IN)
_H_IN = al.constexpr(H_IN)
_W_IN = al.constexpr(W_IN)
_D_OUT = al.constexpr(D_OUT)
_H_OUT = al.constexpr(H_OUT)
_W_OUT = al.constexpr(W_OUT)
_KD_CONST = al.constexpr(KD_CONST)
_KH_CONST = al.constexpr(KH_CONST)
_KW_CONST = al.constexpr(KW_CONST)
_TILE_D = al.constexpr(TILE_D)
_TILE_H = al.constexpr(TILE_H)
_TILE_W = al.constexpr(TILE_W)
_TILE_OC = al.constexpr(TILE_OC)
_THREADS = al.constexpr(THREADS)
_NEG_SLOPE = al.constexpr(NEG_SLOPE)
_SHM_WEIGHT_SIZE = al.constexpr(SHM_WEIGHT_SIZE)
_NUM_D_TILES = al.constexpr(NUM_D_TILES)
_NUM_H_TILES = al.constexpr(NUM_H_TILES)
_NUM_W_TILES = al.constexpr(NUM_W_TILES)
_NUM_OC_GROUPS = al.constexpr(NUM_OC_GROUPS)


# =============================================================================
# ConvTranspose3d kernel
# =============================================================================
@avelang.jit
def conv_transpose_3d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
):
    tid = al.thread_id(0)
    block_id = al.block_id(0)

    d_idx = block_id % _NUM_D_TILES
    rest = block_id // _NUM_D_TILES
    h_idx = rest % _NUM_H_TILES
    rest = rest // _NUM_H_TILES
    w_idx = rest % _NUM_W_TILES
    rest = rest // _NUM_W_TILES
    oc_group = rest % _NUM_OC_GROUPS
    b_idx = rest // _NUM_OC_GROUPS

    d0 = d_idx * _TILE_D
    h0 = h_idx * _TILE_H
    w0 = w_idx * _TILE_W
    oc0 = oc_group * _TILE_OC

    # Global tensor views
    x_stride_c = _D_IN * _H_IN * _W_IN
    x_stride_d = _H_IN * _W_IN
    x_stride_h = _W_IN
    x_layout = al.make_layout(
        (_BATCH_SIZE, _IN_CHANNELS, _D_IN, _H_IN, _W_IN),
        (_IN_CHANNELS * x_stride_c, x_stride_c, x_stride_d, x_stride_h, 1),
    )
    x_global = al.make_tensor(x_ptr, al.bf16, x_layout)

    w_stride_c = _OUT_CHANNELS * _KD_CONST * _KH_CONST * _KW_CONST
    w_stride_oc = _KD_CONST * _KH_CONST * _KW_CONST
    w_stride_kd = _KH_CONST * _KW_CONST
    w_stride_kh = _KW_CONST
    w_layout = al.make_layout(
        (_IN_CHANNELS, _OUT_CHANNELS, _KD_CONST, _KH_CONST, _KW_CONST),
        (w_stride_c, w_stride_oc, w_stride_kd, w_stride_kh, 1),
    )
    w_global = al.make_tensor(w_ptr, al.bf16, w_layout)

    bias_layout = al.make_layout((_OUT_CHANNELS,), (1,))
    bias_global = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    out_stride_oc = _D_OUT * _H_OUT * _W_OUT
    out_stride_d = _H_OUT * _W_OUT
    out_stride_h = _W_OUT
    out_layout = al.make_layout(
        (_BATCH_SIZE, _OUT_CHANNELS, _D_OUT, _H_OUT, _W_OUT),
        (_OUT_CHANNELS * out_stride_oc, out_stride_oc, out_stride_d, out_stride_h, 1),
    )
    out_global = al.make_tensor(out_ptr, al.bf16, out_layout)

    # Preload weights into shared memory
    shm_w = al.make_shared((_SHM_WEIGHT_SIZE,), al.bf16)
    KD_KH_KW = _KD_CONST * _KH_CONST * _KW_CONST
    TOC_KKK = _TILE_OC * KD_KH_KW
    KH_KW = _KH_CONST * _KW_CONST

    for idx in al.range(tid, _SHM_WEIGHT_SIZE, _THREADS):
        ic = idx // TOC_KKK
        r = idx % TOC_KKK
        oc_local = r // KD_KH_KW
        r = r % KD_KH_KW
        kd = r // KH_KW
        r = r % KH_KW
        kh = r // _KW_CONST
        kw = r % _KW_CONST
        shm_w[idx] = w_global[ic, oc0 + oc_local, kd, kh, kw]

    al.syncthreads()

    # Preload bias into registers
    bias_f32 = al.make_local((_TILE_OC,), al.f32)
    for ocl in al.range(_TILE_OC):
        bias_f32[ocl] = al.convert(bias_global[oc0 + ocl], al.f32)

    # Compute
    zero_i32 = al.convert(0, al.i32)
    one_i32 = al.convert(1, al.i32)
    two_i32 = al.convert(2, al.i32)
    d_in_i32 = al.convert(_D_IN, al.i32)
    h_in_i32 = al.convert(_H_IN, al.i32)
    w_in_i32 = al.convert(_W_IN, al.i32)

    num_spatial = _TILE_D * _TILE_H * _TILE_W
    tile_hw = _TILE_H * _TILE_W

    for idx in al.range(tid, num_spatial, _THREADS):
        d_local = idx // tile_hw
        r2 = idx % tile_hw
        h_local = r2 // _TILE_W
        w_local = r2 % _TILE_W

        od = d0 + d_local
        oh = h0 + h_local
        ow = w0 + w_local

        for oc_local in al.range(_TILE_OC):
            acc = bias_f32[oc_local]
            oc_weight_base = oc_local * KD_KH_KW

            for ic in al.range(_IN_CHANNELS):
                ic_weight_base = ic * TOC_KKK + oc_weight_base

                for kd in al.range(_KD_CONST):
                    t_d = od + one_i32 - kd
                    id = t_d // two_i32
                    if id >= zero_i32:
                        if id < d_in_i32:
                            rem_d = t_d - id * two_i32
                            if rem_d == zero_i32:
                                kd_weight_off = kd * KH_KW

                                for kh in al.range(_KH_CONST):
                                    t_h = oh + one_i32 - kh
                                    ih = t_h // two_i32
                                    if ih >= zero_i32:
                                        if ih < h_in_i32:
                                            rem_h = t_h - ih * two_i32
                                            if rem_h == zero_i32:
                                                kh_kw_weight_off = kd_weight_off + kh * _KW_CONST

                                                for kw in al.range(_KW_CONST):
                                                    t_w = ow + one_i32 - kw
                                                    iw = t_w // two_i32
                                                    if iw >= zero_i32:
                                                        if iw < w_in_i32:
                                                            rem_w = t_w - iw * two_i32
                                                            if rem_w == zero_i32:
                                                                w_idx = ic_weight_base + kh_kw_weight_off + kw
                                                                x_val = al.convert(x_global[b_idx, ic, id, ih, iw], al.f32)
                                                                w_val = al.convert(shm_w[w_idx], al.f32)
                                                                acc = acc + x_val * w_val

            out_global[b_idx, oc0 + oc_local, od, oh, ow] = al.convert(acc, al.bf16)


# =============================================================================
# Fused epilogue + MaxPool3d kernel
# =============================================================================
@avelang.jit
def fused_epilogue_pool_kernel(
    conv_out_ptr: al.Pointer(al.bf16),
    mult_ptr: al.Pointer(al.bf16),
    pool_out_ptr: al.Pointer(al.bf16),
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    D_pool: al.i32,
    H_pool: al.i32,
    W_pool: al.i32,
):
    tid = al.thread_id(0)
    block_id = al.block_id(0)

    oc = block_id % _OUT_CHANNELS
    b_idx = block_id // _OUT_CHANNELS

    conv_stride_oc = D_out * H_out * W_out
    conv_stride_d = H_out * W_out
    conv_stride_h = W_out
    conv_layout = al.make_layout(
        (_BATCH_SIZE, _OUT_CHANNELS, D_out, H_out, W_out),
        (_OUT_CHANNELS * conv_stride_oc, conv_stride_oc, conv_stride_d, conv_stride_h, 1),
    )
    conv_global = al.make_tensor(conv_out_ptr, al.bf16, conv_layout)

    mult_layout = al.make_layout((_OUT_CHANNELS,), (1,))
    mult_global = al.make_tensor(mult_ptr, al.bf16, mult_layout)

    pool_stride_oc = D_pool * H_pool * W_pool
    pool_stride_d = H_pool * W_pool
    pool_stride_h = W_pool
    pool_layout = al.make_layout(
        (_BATCH_SIZE, _OUT_CHANNELS, D_pool, H_pool, W_pool),
        (_OUT_CHANNELS * pool_stride_oc, pool_stride_oc, pool_stride_d, pool_stride_h, 1),
    )
    pool_global = al.make_tensor(pool_out_ptr, al.bf16, pool_layout)

    mult_f32 = al.convert(mult_global[oc], al.f32)
    neg_slope_f32 = al.convert(_NEG_SLOPE, al.f32)
    zero_f32 = al.convert(0.0, al.f32)
    neg_inf = al.convert(-3.402823466e+38, al.f32)
    two_i32 = al.convert(2, al.i32)

    num_spatial = D_pool * H_pool * W_pool
    for idx in al.range(tid, num_spatial, _THREADS):
        pd = idx // (H_pool * W_pool)
        r = idx % (H_pool * W_pool)
        ph = r // W_pool
        pw = r % W_pool

        max_val = neg_inf
        cd0 = pd * two_i32
        ch0 = ph * two_i32
        cw0 = pw * two_i32

        for cd in al.range(2):
            od = cd0 + cd
            for ch in al.range(2):
                oh = ch0 + ch
                for cw in al.range(2):
                    ow = cw0 + cw
                    val = al.convert(conv_global[b_idx, oc, od, oh, ow], al.f32)
                    if val < zero_f32:
                        val = val * neg_slope_f32
                    val = val * mult_f32
                    if val < zero_f32:
                        val = val * neg_slope_f32
                    if val > max_val:
                        max_val = val

        pool_global[b_idx, oc, pd, ph, pw] = al.convert(max_val, al.bf16)


# =============================================================================
# Host wrapper
# =============================================================================
def _prepare_bf16_cuda(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_transpose_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    multiplier: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda(x)
    w_bf16 = _prepare_bf16_cuda(weight)
    b_bf16 = _prepare_bf16_cuda(bias)
    m_bf16 = _prepare_bf16_cuda(multiplier.flatten())

    B, IC, D_in_val, H_in_val, W_in_val = x_bf16.shape
    IC_w, OC, KD_val, KH_val, KW_val = w_bf16.shape

    D_out_val = (D_in_val - 1) * 2 - 2 * 1 + 3 + 1
    H_out_val = (H_in_val - 1) * 2 - 2 * 1 + 3 + 1
    W_out_val = (W_in_val - 1) * 2 - 2 * 1 + 3 + 1
    D_pool = D_out_val // 2
    H_pool = H_out_val // 2
    W_pool = W_out_val // 2

    conv_out = torch.empty(
        (B, OC, D_out_val, H_out_val, W_out_val),
        device=x_bf16.device, dtype=torch.bfloat16,
    )

    num_d_tiles_val = D_out_val // TILE_D
    num_h_tiles_val = H_out_val // TILE_H
    num_w_tiles_val = W_out_val // TILE_W
    num_oc_groups_val = OC // TILE_OC
    total_blocks = (
        num_d_tiles_val * num_h_tiles_val * num_w_tiles_val
        * num_oc_groups_val * B
    )

    conv_transpose_3d_kernel[lambda: ((total_blocks, 1, 1), (THREADS, 1, 1))](
        x_bf16, w_bf16, b_bf16, conv_out,
    )

    pool_out = torch.empty(
        (B, OC, D_pool, H_pool, W_pool),
        device=x_bf16.device, dtype=torch.bfloat16,
    )

    epilogue_blocks = B * OC
    fused_epilogue_pool_kernel[lambda: ((epilogue_blocks, 1, 1), (THREADS, 1, 1))](
        conv_out, m_bf16, pool_out,
        D_out_val, H_out_val, W_out_val,
        D_pool, H_pool, W_pool,
    )

    return pool_out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        return avelang_conv_transpose_fused(
            x, self.conv_transpose.weight, self.conv_transpose.bias, self.multiplier,
        )
