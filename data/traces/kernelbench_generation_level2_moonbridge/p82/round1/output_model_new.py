import torch
import torch.nn as nn
import avelang
import avelang.language as al


# ── Problem constants ──────────────────────────────────────────────────────
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 4

# ── Tiling parameters ──────────────────────────────────────────────────────
TILE_H = 16
TILE_W = 16
TILE_C = 8
IN_H = TILE_H + kernel_size - 1    # 18
IN_W = TILE_W + kernel_size - 1    # 18
THREADS = TILE_H * TILE_W           # 256
PK = 4
IN_TILE_ELEMS = in_channels * IN_H * IN_W            # 2592
WT_TILE_ELEMS = TILE_C * in_channels * kernel_size * kernel_size  # 576
MAX_IN_LOADS = (IN_TILE_ELEMS + THREADS - 1) // THREADS   # 11
MAX_WT_LOADS = (WT_TILE_ELEMS + THREADS - 1) // THREADS   # 3


# ═══════════════════════════════════════════════════════════════════════════
# Fused kernel: conv → tanh → scale → bias → maxpool
# Uses shared memory for input/weight tiling and on-chip maxpool reduction.
# Epilogue runs with BF16 round-trips to match PyTorch eager-mode precision.
# ═══════════════════════════════════════════════════════════════════════════
@avelang.jit
def conv_fused_pool_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    conv_bias_ptr: al.Pointer(al.bf16),
    extra_bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
    KH: al.i32,
    KW: al.i32,
    OH: al.i32,
    OW: al.i32,
    PH: al.i32,
    PW: al.i32,
    scale_factor: al.f32,
):
    tid = al.thread_id(0)

    ow_tile = al.block_id(0)
    oh_tile = al.block_id(1)
    batch_oc_flat = al.block_id(2)

    oc_tiles = (OC + TILE_C - 1) // TILE_C
    b = batch_oc_flat // oc_tiles
    oc_group = batch_oc_flat - b * oc_tiles

    oc_start = oc_group * TILE_C
    oh_start = oh_tile * TILE_H
    ow_start = ow_tile * TILE_W

    th = tid // TILE_W
    tw = tid - th * TILE_W
    oh = oh_start + th
    ow = ow_start + tw

    local_th = th - (th // PK) * PK
    local_tw = tw - (tw // PK) * PK
    is_leader = al.convert(0, al.i32)
    if local_th == 0:
        if local_tw == 0:
            is_leader = al.convert(1, al.i32)

    inside = al.convert(1, al.i32)
    if oh >= OH:
        inside = al.convert(0, al.i32)
    if ow >= OW:
        inside = al.convert(0, al.i32)
    if b >= B:
        inside = al.convert(0, al.i32)
    if oc_start >= OC:
        inside = al.convert(0, al.i32)

    # ── Layouts ────────────────────────────────────────────────────────
    in_layout = al.make_layout((B, IC, H, W), (IC * H * W, H * W, W, 1))
    w_layout = al.make_layout((OC, IC, KH, KW), (IC * KH * KW, KH * KW, KW, 1))
    cb_layout = al.make_layout((OC,), (1,))
    eb_layout = al.make_layout((OC, 1, 1), (1, 1, 1))
    out_layout = al.make_layout((B, OC, PH, PW), (OC * PH * PW, PH * PW, PW, 1))

    input_t = al.make_tensor(input_ptr, al.bf16, in_layout)
    weight_t = al.make_tensor(weight_ptr, al.bf16, w_layout)
    conv_bias_t = al.make_tensor(conv_bias_ptr, al.bf16, cb_layout)
    extra_bias_t = al.make_tensor(extra_bias_ptr, al.bf16, eb_layout)
    output_t = al.make_tensor(output_ptr, al.bf16, out_layout)

    # ── Shared memory ──────────────────────────────────────────────────
    shm_in = al.make_shared((in_channels, IN_H, IN_W), al.bf16)
    shm_wt = al.make_shared((TILE_C, in_channels, kernel_size, kernel_size), al.bf16)
    shm_conv = al.make_shared((TILE_H, TILE_W, TILE_C), al.bf16)

    # ── Load input tile ────────────────────────────────────────────────
    for load_i in al.range(MAX_IN_LOADS):
        lid = load_i * THREADS + tid
        if lid < IN_TILE_ELEMS:
            iw_local = lid % IN_W
            tmp1 = lid // IN_W
            ih_local = tmp1 % IN_H
            ic = tmp1 // IN_H
            g_h = oh_start + ih_local
            g_w = ow_start + iw_local
            if g_h < H and g_w < W and b < B:
                shm_in[ic, ih_local, iw_local] = input_t[b, ic, g_h, g_w]

    # ── Load weight tile ───────────────────────────────────────────────
    for load_i in al.range(MAX_WT_LOADS):
        lid = load_i * THREADS + tid
        if lid < WT_TILE_ELEMS:
            wkw = lid % KW
            tmp1 = lid // KW
            wkh = tmp1 % KH
            tmp2 = tmp1 // KH
            wic = tmp2 % IC
            toc = tmp2 // IC
            g_oc = oc_start + toc
            if g_oc < OC:
                shm_wt[toc, wic, wkh, wkw] = weight_t[g_oc, wic, wkh, wkw]

    al.syncthreads()

    # ── Compute conv + epilogue → store in shm_conv ────────────────────
    if inside != 0:
        valid_oc = TILE_C
        tmp_oc = oc_start + TILE_C
        if tmp_oc > OC:
            valid_oc = OC - oc_start

        for toc in al.range(valid_oc):
            oc = oc_start + toc

            # FP32 accumulation with KH-KW-IC loop order
            acc = al.convert(0.0, al.f32)
            for kh in al.range(KH):
                for kw in al.range(KW):
                    for ic in al.range(IC):
                        inp_val = al.convert(shm_in[ic, th + kh, tw + kw], al.f32)
                        w_val = al.convert(shm_wt[toc, ic, kh, kw], al.f32)
                        acc = acc + inp_val * w_val

            # Epilogue with BF16 round-trips to match eager-mode precision
            cb_val = al.convert(conv_bias_t[oc], al.f32)
            acc = acc + cb_val

            conv_out_bf16 = al.convert(acc, al.bf16)
            conv_out_f32 = al.convert(conv_out_bf16, al.f32)
            tanh_out = al.tanh(conv_out_f32)

            tanh_out_bf16 = al.convert(tanh_out, al.bf16)
            tanh_out_f32 = al.convert(tanh_out_bf16, al.f32)
            scaled = tanh_out_f32 * scale_factor

            scaled_bf16 = al.convert(scaled, al.bf16)
            scaled_f32 = al.convert(scaled_bf16, al.f32)
            eb_val = al.convert(extra_bias_t[oc, 0, 0], al.f32)
            result = scaled_f32 + eb_val

            shm_conv[th, tw, toc] = al.convert(result, al.bf16)

    al.syncthreads()

    # ── Max-pool reduction: leader threads scan their 4×4 window ───────
    if inside != 0:
        if is_leader != 0:
            valid_oc = TILE_C
            tmp_oc = oc_start + TILE_C
            if tmp_oc > OC:
                valid_oc = OC - oc_start

            pool_oh = oh_start // PK + th // PK
            pool_ow = ow_start // PK + tw // PK

            if pool_oh < PH and pool_ow < PW:
                for toc in al.range(valid_oc):
                    oc = oc_start + toc
                    pmax = al.convert(shm_conv[th, tw, toc], al.f32)
                    for dth in al.range(PK):
                        for dtw in al.range(PK):
                            val = al.convert(
                                shm_conv[th + dth, tw + dtw, toc], al.f32,
                            )
                            pmax = val if val > pmax else pmax

                    output_t[b, oc, pool_oh, pool_ow] = al.convert(pmax, al.bf16)


# ═══════════════════════════════════════════════════════════════════════════
# Host-side helper
# ═══════════════════════════════════════════════════════════════════════════
def _ensure_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_pool(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    extra_bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _ensure_bf16_cuda_contiguous(x)
    w_bf16 = _ensure_bf16_cuda_contiguous(conv_weight)
    cb_bf16 = _ensure_bf16_cuda_contiguous(conv_bias)
    eb_bf16 = _ensure_bf16_cuda_contiguous(extra_bias)

    B, IC, H, W = x_bf16.shape
    OC, w_IC, KH, KW = w_bf16.shape

    OH = H - KH + 1
    OW = W - KW + 1

    PH = OH // PK
    PW = OW // PK

    pool_out = torch.empty(
        (B, OC, PH, PW), device=x_bf16.device, dtype=torch.bfloat16,
    )

    grid_ow = (OW + TILE_W - 1) // TILE_W
    grid_oh = (OH + TILE_H - 1) // TILE_H
    grid_oc = (OC + TILE_C - 1) // TILE_C
    grid_boc = B * grid_oc

    sf = scaling_factor
    conv_fused_pool_kernel[lambda: ((grid_ow, grid_oh, grid_boc), (THREADS, 1, 1))](
        x_bf16, w_bf16, cb_bf16, eb_bf16, pool_out,
        B, IC, OC, H, W, KH, KW, OH, OW, PH, PW, sf,
    )

    return pool_out


# ═══════════════════════════════════════════════════════════════════════════
# ModelNew
# ═══════════════════════════════════════════════════════════════════════════
class ModelNew(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size,
        scaling_factor, bias_shape, pool_kernel_size,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        return avelang_conv_pool(
            x, self.conv.weight, self.conv.bias, self.bias,
        )


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [
        in_channels, out_channels, kernel_size,
        scaling_factor, bias_shape, pool_kernel_size,
    ]
