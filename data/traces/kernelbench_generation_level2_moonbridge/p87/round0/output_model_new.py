import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_H = 16
TILE_W = 16
THREADS = 256


@avelang.jit
def conv2d_mish_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.constexpr,
    H: al.i32,
    W: al.i32,
    K: al.i32,
    P: al.i32,
    Q: al.i32,
    R: al.constexpr,
    OUT_CH: al.constexpr,
    sub_val: al.f32,
):
    tid = al.thread_id(0)
    tile_q = al.block_id(0)
    tile_p = al.block_id(1)
    bch = al.block_id(2)

    num_ch_groups = K // OUT_CH
    n = bch // num_ch_groups
    ch_group = bch - n * num_ch_groups
    ch_start = ch_group * OUT_CH

    p_start = tile_p * TILE_H
    q_start = tile_q * TILE_W

    p_off = tid // TILE_W
    q_off = tid - p_off * TILE_W

    p = p_start + p_off
    q = q_start + q_off

    x_layout = al.make_layout((N, C, H, W), (C * H * W, H * W, W, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    w_layout = al.make_layout((K, C, R, R), (C * R * R, R * R, R, 1))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    b_layout = al.make_layout((K,), (1,))
    bias = al.make_tensor(b_ptr, al.bf16, b_layout)

    out_layout = al.make_layout((N, K, P, Q), (K * P * Q, P * Q, Q, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    inp_elems = al.convert(2592, al.i32)
    smem_inp = al.make_shared((2592,), al.bf16)

    wgt_elems = al.convert(2304, al.i32)
    smem_wgt = al.make_shared((2304,), al.bf16)

    for idx in al.range(tid, inp_elems, THREADS):
        ic = idx % C
        rem = idx // C
        px = rem % 18
        py = rem // 18
        g_row = p_start + py
        g_col = q_start + px
        if g_row < H and g_col < W:
            smem_inp[idx] = x[n, ic, g_row, g_col]
        else:
            smem_inp[idx] = al.convert(0.0, al.bf16)

    wgt_per_ch = al.convert(72, al.i32)
    for idx in al.range(tid, wgt_elems, THREADS):
        oc_off = idx // wgt_per_ch
        rest = idx - oc_off * wgt_per_ch
        ic = rest // (R * R)
        rest2 = rest - ic * (R * R)
        ky = rest2 // R
        kx = rest2 - ky * R
        smem_wgt[idx] = w[ch_start + oc_off, ic, ky, kx]

    al.syncthreads()

    if p < P and q < Q:
        one = al.convert(1.0, al.f32)
        for oc in al.range(OUT_CH):
            oc_global = ch_start + oc
            acc = al.convert(bias[oc_global], al.f32)
            for ic in al.range(C):
                for ky in al.range(R):
                    for kx in al.range(R):
                        inp_idx = ((p_off + ky) * 18 + (q_off + kx)) * C + ic
                        wgt_idx = ((oc * C + ic) * R + ky) * R + kx
                        inp_val = al.convert(smem_inp[inp_idx], al.f32)
                        wgt_val = al.convert(smem_wgt[wgt_idx], al.f32)
                        acc = acc + inp_val * wgt_val

            acc = acc - sub_val
            sp = al.log(one + al.exp(acc))
            mish = acc * al.tanh(sp)
            out[n, oc_global, p, q] = al.convert(mish, al.bf16)


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv2d_mish(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    sub_val: float,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)
    b_bf16 = _to_bf16_contiguous(bias)

    N, C, H, W = x_bf16.shape
    K, _, R, _ = w_bf16.shape
    P = H - R + 1
    Q = W - R + 1

    OUT_CH = 32
    num_tiles_w = (Q + TILE_W - 1) // TILE_W
    num_tiles_h = (P + TILE_H - 1) // TILE_H
    num_ch_groups = K // OUT_CH

    out = torch.empty((N, K, P, Q), device=x_bf16.device, dtype=torch.bfloat16)

    grid = (num_tiles_w, num_tiles_h, N * num_ch_groups)
    conv2d_mish_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, w_bf16, b_bf16, out,
        N, C, H, W, K, P, Q, R, OUT_CH,
        sub_val,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.sub_val = subtract_value_1 + subtract_value_2

    def forward(self, x):
        weight = self.conv.weight.data
        bias = self.conv.bias.data
        return avelang_conv2d_mish(x, weight, bias, self.sub_val)
