import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── compile-time tile configuration ──
TILE_H = 16
TILE_W = 16
KH = 3
KW = 3
C_IN = 8
C_OUT = 64
C_GROUPS = 1
C_PER_GROUP = C_OUT // C_GROUPS  # 64
INP_H = TILE_H + KH - 1  # 18
INP_W = TILE_W + KW - 1  # 18
INP_TILE_SIZE = INP_H * INP_W * C_IN  # 2592
W_SIZE = C_OUT * C_IN * KH * KW  # 4608
THREADS = 256


@avelang.jit
def conv_divide_leakyrelu_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    divisor: al.f32,
):
    tid = al.thread_id(0)
    block_h = al.block_id(0)
    block_w = al.block_id(1)
    batch_idx = al.block_id(2)

    # flat 1-D views with manual index arithmetic
    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((N * C_IN * H * W,), (1,)))
    w = al.make_tensor(w_ptr, al.bf16, al.make_layout((C_OUT * C_IN * KH * KW,), (1,)))
    b = al.make_tensor(b_ptr, al.bf16, al.make_layout((C_OUT,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((N * C_OUT * H_out * W_out,), (1,)))

    h_base = block_h * TILE_H
    w_base = block_w * TILE_W
    batch_offset = batch_idx * (C_IN * H * W)

    shm_inp = al.make_shared((INP_H, INP_W, C_IN), al.bf16)
    shm_w = al.make_shared((C_OUT, C_IN, KH, KW), al.bf16)

    # ── cooperative load: input tile → shared memory ──
    for idx in al.range(tid, INP_TILE_SIZE, THREADS):
        ic = idx % C_IN
        iw = (idx // C_IN) % INP_W
        ih = idx // (INP_W * C_IN)
        g_h = h_base + ih
        g_w = w_base + iw
        if g_h < H and g_w < W:
            g_off = batch_offset + ic * (H * W) + g_h * W + g_w
            shm_inp[ih, iw, ic] = x[g_off]
        else:
            shm_inp[ih, iw, ic] = al.convert(0.0, al.bf16)

    # ── cooperative load: weights → shared memory ──
    for idx in al.range(tid, W_SIZE, THREADS):
        kw = idx % KW
        kh = (idx // KW) % KH
        ic = (idx // (KH * KW)) % C_IN
        oc = idx // (C_IN * KH * KW)
        w_flat = oc * (C_IN * KH * KW) + ic * (KH * KW) + kh * KW + kw
        shm_w[oc, ic, kh, kw] = w[w_flat]

    al.syncthreads()

    # ── thread → output-element mapping ──
    local_h = tid // (TILE_W * C_GROUPS)
    local_w = (tid // C_GROUPS) % TILE_W
    c_group = tid % C_GROUPS
    c_start = c_group * C_PER_GROUP

    g_h = h_base + local_h
    g_w = w_base + local_w

    if local_h < TILE_H and local_w < TILE_W and g_h < H_out and g_w < W_out:
        out_batch_base = batch_idx * (C_OUT * H_out * W_out)
        zero_f32 = al.convert(0.0, al.f32)
        neg_slope_f32 = al.convert(0.01, al.f32)

        for oc_offset in al.range(C_PER_GROUP):
            oc = c_start + oc_offset
            acc = al.convert(b[oc], al.f32)

            for ic in al.range(C_IN):
                for kh in al.range(KH):
                    for kw in al.range(KW):
                        inp_val = al.convert(shm_inp[local_h + kh, local_w + kw, ic], al.f32)
                        w_val = al.convert(shm_w[oc, ic, kh, kw], al.f32)
                        acc = acc + inp_val * w_val

            acc = acc / divisor
            if acc < zero_f32:
                acc = acc * neg_slope_f32

            out_idx = out_batch_base + oc * (H_out * W_out) + g_h * W_out + g_w
            out[out_idx] = al.convert(acc, al.bf16)


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_divide_leakyrelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    divisor: float,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)
    b_bf16 = _to_bf16_contiguous(bias)

    N, c_in, H, W = x_bf16.shape
    c_out = w_bf16.shape[0]

    H_out = H - KH + 1
    W_out = W - KW + 1

    grid_h = (H_out + TILE_H - 1) // TILE_H
    grid_w = (W_out + TILE_W - 1) // TILE_W

    out = torch.empty((N, c_out, H_out, W_out), device=x_bf16.device, dtype=torch.bfloat16)

    conv_divide_leakyrelu_kernel[lambda: ((grid_h, grid_w, N), (THREADS, 1, 1))](
        x_bf16, w_bf16, b_bf16, out,
        N, H, W, H_out, W_out,
        float(divisor),
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.divisor = divisor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return avelang_conv_divide_leakyrelu(
            x, self.conv.weight, self.conv.bias, float(self.divisor)
        )


# ── contract-preserving module-level exports ──
batch_size = 128
in_channels = 8
out_channels = 64
height = 128
width = 128
kernel_size = 3
divisor = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]
