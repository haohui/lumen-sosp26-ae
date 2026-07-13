import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ============================================================
# Compile-time constants
# ============================================================
TILE_OH = 8
TILE_OW = 16
TILE_OC = 4
IC_VAL = 8
KH_VAL = 3
KW_VAL = 3
SHM_IN_H = TILE_OH + KH_VAL - 1  # 10
SHM_IN_W = TILE_OW + KW_VAL - 1  # 18
SHM_IN_SIZE = IC_VAL * SHM_IN_H * SHM_IN_W  # 1440
SHM_W_SIZE = TILE_OC * IC_VAL * KH_VAL * KW_VAL  # 288
CONV_THREADS = TILE_OH * TILE_OW  # 128


# ============================================================
# Fused Conv2d + BatchNorm Eval + Scale Kernel
# ============================================================
@avelang.jit
def fused_conv2d_bn_scale_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    rm_ptr: al.Pointer(al.f32),
    rv_ptr: al.Pointer(al.f32),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    N: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
    OH: al.i32,
    OW: al.i32,
    OW_TILES: al.i32,
    eps: al.f32,
    scaling_factor: al.f32,
):
    tid = al.thread_id(0)
    b0 = al.block_id(0)
    oh_block = al.block_id(1)
    oc_block = al.block_id(2)

    n_idx = b0 // OW_TILES
    ow_block = b0 - n_idx * OW_TILES

    oh_start = oh_block * TILE_OH
    ow_start = ow_block * TILE_OW
    oc_start = oc_block * TILE_OC

    tid_h = tid // TILE_OW
    tid_w = tid - tid_h * TILE_OW

    oh = oh_start + tid_h
    ow = ow_start + tid_w

    # Shared memory for input window and weight tile
    shm_in = al.make_shared((SHM_IN_SIZE,), al.bf16)
    shm_w = al.make_shared((SHM_W_SIZE,), al.bf16)

    # Input, weight views
    x_layout = al.make_layout((N, IC_VAL, H, W), (IC_VAL * H * W, H * W, W, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    w_layout = al.make_layout((OC, IC_VAL, KH_VAL, KW_VAL), (IC_VAL * KH_VAL * KW_VAL, KH_VAL * KW_VAL, KW_VAL, 1))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    # BN parameter views (per-channel, size OC)
    ch_layout = al.make_layout((OC,), (1,))
    b_vec = al.make_tensor(b_ptr, al.bf16, ch_layout)
    rm_vec = al.make_tensor(rm_ptr, al.f32, ch_layout)
    rv_vec = al.make_tensor(rv_ptr, al.f32, ch_layout)
    gamma_vec = al.make_tensor(gamma_ptr, al.bf16, ch_layout)
    beta_vec = al.make_tensor(beta_ptr, al.bf16, ch_layout)

    # Preload per-block BN params into registers
    bn_bias = al.make_local((TILE_OC,), al.f32)
    bn_rm = al.make_local((TILE_OC,), al.f32)
    bn_rstd = al.make_local((TILE_OC,), al.f32)
    bn_gamma = al.make_local((TILE_OC,), al.f32)
    bn_beta = al.make_local((TILE_OC,), al.f32)

    for oc_off in al.range(TILE_OC):
        oc = oc_start + oc_off
        if oc < OC:
            bn_bias[oc_off] = al.convert(b_vec[oc], al.f32)
            bn_rm[oc_off] = rm_vec[oc]
            rv_val = rv_vec[oc]
            bn_rstd[oc_off] = al.convert(1.0, al.f32) / al.sqrt(rv_val + eps)
            bn_gamma[oc_off] = al.convert(gamma_vec[oc], al.f32)
            bn_beta[oc_off] = al.convert(beta_vec[oc], al.f32)
        else:
            bn_bias[oc_off] = al.convert(0.0, al.f32)
            bn_rm[oc_off] = al.convert(0.0, al.f32)
            bn_rstd[oc_off] = al.convert(0.0, al.f32)
            bn_gamma[oc_off] = al.convert(0.0, al.f32)
            bn_beta[oc_off] = al.convert(0.0, al.f32)

    # Output view
    out_layout = al.make_layout((N, OC, OH, OW), (OC * OH * OW, OH * OW, OW, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    valid_batch = n_idx < N

    # Cooperative load of input window into shm_in
    for load_idx in al.range(tid, SHM_IN_SIZE, CONV_THREADS):
        ic = load_idx // (SHM_IN_H * SHM_IN_W)
        rem = load_idx - ic * (SHM_IN_H * SHM_IN_W)
        ih_off = rem // SHM_IN_W
        iw_off = rem - ih_off * SHM_IN_W
        ih = oh_start + ih_off
        iw = ow_start + iw_off
        if valid_batch and ih < H and iw < W:
            shm_in[load_idx] = x[n_idx, ic, ih, iw]
        else:
            shm_in[load_idx] = al.convert(0.0, al.bf16)

    # Cooperative load of weight tile into shm_w
    for load_idx in al.range(tid, SHM_W_SIZE, CONV_THREADS):
        oc_off = load_idx // (IC_VAL * KH_VAL * KW_VAL)
        rem = load_idx - oc_off * (IC_VAL * KH_VAL * KW_VAL)
        ic = rem // (KH_VAL * KW_VAL)
        rem2 = rem - ic * (KH_VAL * KW_VAL)
        kh = rem2 // KW_VAL
        kw = rem2 - kh * KW_VAL
        oc = oc_start + oc_off
        if oc < OC:
            shm_w[load_idx] = w[oc, ic, kh, kw]
        else:
            shm_w[load_idx] = al.convert(0.0, al.bf16)

    al.syncthreads()

    # Compute output for this thread, across TILE_OC output channels
    if valid_batch and oh < OH and ow < OW:
        for oc_off in al.range(TILE_OC):
            oc = oc_start + oc_off
            if oc < OC:
                # Conv accumulation: bias + sum over IC, KH, KW
                acc = bn_bias[oc_off]
                for ic in al.range(IC_VAL):
                    for kh in al.range(KH_VAL):
                        for kw in al.range(KW_VAL):
                            in_idx = ic * (SHM_IN_H * SHM_IN_W) + (tid_h + kh) * SHM_IN_W + (tid_w + kw)
                            w_idx = oc_off * (IC_VAL * KH_VAL * KW_VAL) + ic * (KH_VAL * KW_VAL) + kh * KW_VAL + kw
                            x_val = al.convert(shm_in[in_idx], al.f32)
                            w_val = al.convert(shm_w[w_idx], al.f32)
                            acc = acc + x_val * w_val

                # BN eval + scale epilogue
                norm_val = (acc - bn_rm[oc_off]) * bn_rstd[oc_off]
                result = norm_val * bn_gamma[oc_off] + bn_beta[oc_off]
                result = result * scaling_factor
                out[n_idx, oc, oh, ow] = al.convert(result, al.bf16)


# ============================================================
# Host Wrappers
# ============================================================
def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_fused_conv2d_bn_scale(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    bn_gamma: torch.Tensor,
    bn_beta: torch.Tensor,
    scaling_factor: float,
) -> torch.Tensor:
    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(conv_weight)
    b_bf16 = _to_bf16_contiguous(conv_bias)
    gamma_bf16 = _to_bf16_contiguous(bn_gamma)
    beta_bf16 = _to_bf16_contiguous(bn_beta)
    rm = running_mean.contiguous().to(dtype=torch.float32)
    rv = running_var.contiguous().to(dtype=torch.float32)

    N, IC, H, W_in = x_bf16.shape
    OC = w_bf16.shape[0]
    KH = w_bf16.shape[2]
    KW = w_bf16.shape[3]
    OH = H - KH + 1
    OW = W_in - KW + 1

    OH_TILES = (OH + TILE_OH - 1) // TILE_OH
    OW_TILES = (OW + TILE_OW - 1) // TILE_OW
    OC_TILES = (OC + TILE_OC - 1) // TILE_OC

    out = torch.empty((N, OC, OH, OW), dtype=torch.bfloat16, device=x_bf16.device)
    grid = (OW_TILES * N, OH_TILES, OC_TILES)

    fused_conv2d_bn_scale_kernel[lambda: (grid, (CONV_THREADS, 1, 1))](
        x_bf16, w_bf16, b_bf16, out,
        rm, rv, gamma_bf16, beta_bf16,
        N, OC, H, W_in, OH, OW, OW_TILES,
        1e-5,
        scaling_factor,
    )

    return out


# ============================================================
# ModelNew entrypoint
# ============================================================
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x_bf16 = x.to(dtype=torch.bfloat16, device=x.device).contiguous()
        w_bf16 = self.conv.weight.data.to(dtype=torch.bfloat16).contiguous()
        b_bf16 = self.conv.bias.data.to(dtype=torch.bfloat16).contiguous()
        running_mean = self.bn.running_mean.data
        running_var = self.bn.running_var.data
        gamma = self.bn.weight.data
        beta = self.bn.bias.data

        result = avelang_fused_conv2d_bn_scale(
            x_bf16, w_bf16, b_bf16,
            running_mean, running_var, gamma, beta,
            self.scaling_factor,
        )
        return result.to(x.dtype)


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]
