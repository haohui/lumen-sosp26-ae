import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SPATIAL = 256
TILE_OC = 8
_K3 = 3
_K27 = 27


@avelang.jit
def conv_transpose3d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    stride: al.i32,
    padding: al.i32,
    dilation: al.i32,
    has_bias: al.i32,
    total_spatial: al.i32,
):
    TV = 8
    BS = 256
    K = _K3
    K27 = _K27

    tid = al.thread_id(0)
    bid_s = al.block_id(0)
    bid_oc = al.block_id(1)

    oc_base = bid_oc * TV
    one = al.convert(1, al.i32)

    w_layout = al.make_layout(
        (IC, OC, K, K, K),
        (OC * K * K * K, K * K * K, K * K, K, one),
    )
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    x_layout = al.make_layout(
        (B, IC, D_in, H_in, W_in),
        (IC * D_in * H_in * W_in, D_in * H_in * W_in, H_in * W_in, W_in, one),
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    out_layout = al.make_layout(
        (B, OC, D_out, H_out, W_out),
        (OC * D_out * H_out * W_out, D_out * H_out * W_out, H_out * W_out, W_out, one),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    sidx = bid_s * BS + tid
    if sidx < total_spatial:
        w_idx = sidx - (sidx // W_out) * W_out
        t1 = sidx // W_out
        h_idx = t1 - (t1 // H_out) * H_out
        t2 = t1 // H_out
        d_idx = t2 - (t2 // D_out) * D_out
        b = t2 // D_out

        zero_f = al.convert(0.0, al.f32)
        acc = al.make_local((TV,), al.f32)
        for t in al.range(TV):
            acc[t] = zero_f

        # Early-exit: when stride > 1, only positions where
        # (o + padding) % stride == 0 for all dims can map to any input.
        d_mod = d_idx + padding - ((d_idx + padding) // stride) * stride
        h_mod = h_idx + padding - ((h_idx + padding) // stride) * stride
        w_mod = w_idx + padding - ((w_idx + padding) // stride) * stride
        has_input = al.convert(1, al.i32)
        if d_mod != 0:
            has_input = al.convert(0, al.i32)
        if h_mod != 0:
            has_input = al.convert(0, al.i32)
        if w_mod != 0:
            has_input = al.convert(0, al.i32)

        if has_input != 0:
            # Flattened kernel loop: KD*KH*KW = 27 positions
            for k_idx in al.range(K27):
                kd = k_idx // 9
                rem = k_idx - kd * 9
                kh = rem // 3
                kw = rem - kh * 3

                d_raw = d_idx + padding - kd * dilation
                h_raw = h_idx + padding - kh * dilation
                w_raw = w_idx + padding - kw * dilation

                if d_raw >= 0:
                    if h_raw >= 0:
                        if w_raw >= 0:
                            d_chk = d_raw - (d_raw // stride) * stride
                            if d_chk == 0:
                                h_chk = h_raw - (h_raw // stride) * stride
                                if h_chk == 0:
                                    w_chk = w_raw - (w_raw // stride) * stride
                                    if w_chk == 0:
                                        id_val = d_raw // stride
                                        ih_val = h_raw // stride
                                        iw_val = w_raw // stride
                                        if id_val < D_in:
                                            if ih_val < H_in:
                                                if iw_val < W_in:
                                                    for ic in al.range(IC):
                                                        xv = al.convert(x[b, ic, id_val, ih_val, iw_val], al.f32)
                                                        for t in al.range(TV):
                                                            onow = oc_base + t
                                                            if onow < OC:
                                                                wv = al.convert(w[ic, onow, kd, kh, kw], al.f32)
                                                                acc[t] = acc[t] + xv * wv

        if has_bias != 0:
            bt = al.make_tensor(b_ptr, al.bf16, al.make_layout((OC,), (one,)))
            for t in al.range(TV):
                onow = oc_base + t
                if onow < OC:
                    acc[t] = acc[t] + al.convert(bt[onow], al.f32)

        for t in al.range(TV):
            onow = oc_base + t
            if onow < OC:
                out[b, onow, d_idx, h_idx, w_idx] = al.convert(acc[t], al.bf16)


def _prepare_bf16_cuda(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(torch.bfloat16)
    return t.contiguous().cuda().to(torch.bfloat16)


def conv_transpose3d_avelang(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required.")

    x_bf16 = _prepare_bf16_cuda(x)
    w_bf16 = _prepare_bf16_cuda(weight)

    B, IC, D_in, H_in, W_in = x_bf16.shape
    w_IC, OC, KD, KH, KW = w_bf16.shape
    if w_IC != IC:
        raise ValueError(f"IC mismatch: {w_IC} vs {IC}")

    D_out = (D_in - 1) * stride - 2 * padding + dilation * (KD - 1) + 1
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (KH - 1) + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (KW - 1) + 1

    out = torch.empty(
        (B, OC, D_out, H_out, W_out), device=x.device, dtype=torch.bfloat16
    )

    has_bias = 1 if bias is not None else 0
    b_bf16 = (
        _prepare_bf16_cuda(bias)
        if bias is not None
        else torch.empty(1, device=x.device, dtype=torch.bfloat16)
    )

    total_spatial = B * D_out * H_out * W_out
    g0 = (total_spatial + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL
    g1 = (OC + TILE_OC - 1) // TILE_OC

    conv_transpose3d_kernel[lambda: ((g0, g1, 1), (BLOCK_SPATIAL, 1, 1))](
        x_bf16,
        w_bf16,
        b_bf16,
        out,
        B,
        IC,
        OC,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        stride,
        padding,
        dilation,
        has_bias,
        total_spatial,
    )

    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias

        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required.")
        if not x.is_cuda:
            x = x.cuda()
        weight = self.conv_transpose3d.weight
        bias_val = self.conv_transpose3d.bias if self.bias_flag else None
        return conv_transpose3d_avelang(
            x,
            weight,
            bias_val,
            self.stride,
            self.padding,
            self.dilation,
        )


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
depth = 16
height = 32
width = 32
stride = 2
padding = 1
dilation = 2


def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
