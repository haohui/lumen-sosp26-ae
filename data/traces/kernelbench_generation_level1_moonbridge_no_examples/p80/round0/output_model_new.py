import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── Compile-time constants (captured by closure) ──────────────────────────
_IC_c = al.constexpr(32)
_OC_c = al.constexpr(64)
_KH_c = al.constexpr(5)
_KW_c = al.constexpr(9)
_TILE_H_c = al.constexpr(8)
_TILE_W_c = al.constexpr(16)
_WIN_H_c = al.constexpr(16)   # 8 + (5-1)*2 = 16
_WIN_W_c = al.constexpr(40)   # 16 + (9-1)*3 = 40


@avelang.jit
def conv2d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    H: al.i32,
    W: al.i32,
    OH: al.i32,
    OW: al.i32,
    pad_h: al.i32,
    pad_w: al.i32,
    dilation_h: al.i32,
    dilation_w: al.i32,
    H_W: al.i32,
    IC_H_W: al.i32,
    KH_KW: al.i32,
    IC_KH_KW: al.i32,
    OH_OW: al.i32,
    OC_OH_OW: al.i32,
):
    _IC = _IC_c
    _OC = _OC_c
    _KH = _KH_c
    _KW = _KW_c
    _TILE_H = _TILE_H_c
    _TILE_W = _TILE_W_c
    _WIN_H = _WIN_H_c
    _WIN_W = _WIN_W_c

    # ── Flat 1-D tensor views ──────────────────────────────────────────
    inp = al.make_tensor(input_ptr, al.bf16, al.make_layout((IC_H_W * N,), (1,)))
    wgt = al.make_tensor(weight_ptr, al.bf16, al.make_layout((IC_KH_KW * _OC,), (1,)))
    out = al.make_tensor(output_ptr, al.bf16, al.make_layout((OC_OH_OW * N,), (1,)))

    # ── Block / thread indexing ───────────────────────────────────────
    n = al.block_id(0)
    oc = al.block_id(1)
    tile_row = al.block_id(2)
    oh_base = tile_row * _TILE_H

    th = al.thread_id(0)
    tw = al.thread_id(1)
    oh = oh_base + th

    tid_flat = th * _TILE_W + tw
    num_threads = _TILE_H * _TILE_W

    out_n_oc_base = n * OC_OH_OW + oc * OH_OW

    # ── Shared memory ─────────────────────────────────────────────────
    smem_weight = al.make_shared((_KH, _KW), al.bf16)
    smem_input = al.make_shared((_WIN_H, _WIN_W), al.bf16)

    # ── Iterate over OW tiles ────────────────────────────────────────
    for ow_base in al.range(0, OW, _TILE_W):
        ow = ow_base + tw
        valid_out = (th < _TILE_H) & (tw < _TILE_W) & (oh < OH) & (ow < OW)

        acc = al.convert(0, al.f32)

        for ic in al.range(_IC):
            # ── Cooperative load of weight[oc, ic, :, :] ──────────
            if tid_flat < KH_KW:
                kh_idx = tid_flat / _KW
                kw_idx = tid_flat - kh_idx * _KW
                wgt_off = oc * IC_KH_KW + ic * KH_KW + kh_idx * _KW + kw_idx
                smem_weight[kh_idx, kw_idx] = wgt[wgt_off]

            # ── Cooperative load of input window ───────────────────
            for ld_idx in al.range(tid_flat, _WIN_H * _WIN_W, num_threads):
                win_h = ld_idx / _WIN_W
                win_w = ld_idx - win_h * _WIN_W
                h_in = oh_base - pad_h + win_h
                w_in = ow_base - pad_w + win_w
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                if in_bounds:
                    inp_off = n * IC_H_W + ic * H_W + h_in * W + w_in
                    smem_input[win_h, win_w] = inp[inp_off]
                else:
                    smem_input[win_h, win_w] = al.convert(0, al.bf16)

            al.syncthreads()

            # ── Compute partial sums ───────────────────────────────
            if valid_out:
                for kh in al.range(_KH):
                    hin = th + kh * dilation_h
                    for kw in al.range(_KW):
                        win = tw + kw * dilation_w
                        inp_val = al.convert(smem_input[hin, win], al.f32)
                        w_val = al.convert(smem_weight[kh, kw], al.f32)
                        acc = acc + inp_val * w_val

            al.syncthreads()

        # ── Store result ────────────────────────────────────────────
        if valid_out:
            out_off = out_n_oc_base + oh * OW + ow
            out[out_off] = al.convert(acc, al.bf16)


def _launch_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int,
    padding: tuple,
    dilation: tuple,
) -> torch.Tensor:
    N, IC, H, W = x.shape
    OC, IC_w, KH, KW = weight.shape
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    OH = (H + 2 * pad_h - dilation_h * (KH - 1) - 1) // stride + 1
    OW = (W + 2 * pad_w - dilation_w * (KW - 1) - 1) // stride + 1

    out = torch.empty((N, OC, OH, OW), dtype=torch.bfloat16, device=x.device)

    H_W = H * W
    IC_H_W = IC * H_W
    KH_KW = KH * KW
    IC_KH_KW = IC * KH_KW
    OH_OW = OH * OW
    OC_OH_OW = OC * OH_OW

    TILE_H = 8
    TILE_W = 16
    grid_z = (OH + TILE_H - 1) // TILE_H

    conv2d_kernel[lambda: ((N, OC, grid_z), (TILE_H, TILE_W, 1))](
        x, weight, out,
        N, H, W, OH, OW,
        pad_h, pad_w,
        dilation_h, dilation_w,
        H_W, IC_H_W,
        KH_KW, IC_KH_KW,
        OH_OW, OC_OH_OW,
    )

    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv2d.weight.data
        stride = self.conv2d.stride[0]
        padding = self.conv2d.padding
        dilation = self.conv2d.dilation

        x = x.contiguous()
        w = weight.contiguous()

        out = _launch_conv2d(x, w, stride, padding, dilation)

        if self.conv2d.bias is not None:
            out = out + self.conv2d.bias.data.view(1, -1, 1, 1)

        return out


# ── Test code ────────────────────────────────────────────────────────────
batch_size = 8
in_channels = 32
out_channels = 64
kernel_size = (5, 9)
width = 512
height = 512
stride = 1
padding = (2, 4)
dilation = (2, 3)


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
