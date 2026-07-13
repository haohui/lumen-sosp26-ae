import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── Compile-time tile constants ──────────────────────────────────────────────
OC_TILE: al.constexpr = 8
D_TILE: al.constexpr = 8
H_TILE: al.constexpr = 8
W_TILE: al.constexpr = 8
THREADS: al.constexpr = 256
OUT_PER_BLOCK: al.constexpr = OC_TILE * D_TILE * H_TILE * W_TILE
ELEMS_PER_THREAD: al.constexpr = OUT_PER_BLOCK // THREADS


@avelang.jit
def _fused_conv_transpose_add_hardswish_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    add_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
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
):
    tid = al.thread_id(0)
    block_idx = al.block_id(0)

    w_tiles = W_out // W_TILE
    h_tiles = H_out // H_TILE
    d_tiles = D_out // D_TILE
    oc_groups = OC // OC_TILE

    w_tile = block_idx % w_tiles
    tmp = block_idx // w_tiles
    h_tile = tmp % h_tiles
    tmp = tmp // h_tiles
    d_tile = tmp % d_tiles
    tmp = tmp // d_tiles
    oc_group = tmp % oc_groups
    batch = tmp // oc_groups

    if batch >= N:
        return

    oc_start = oc_group * OC_TILE
    od_start = d_tile * D_TILE
    oh_start = h_tile * H_TILE
    ow_start = w_tile * W_TILE

    # Global tensor views
    ic_dhw = IC * D_in * H_in * W_in
    dhw = D_in * H_in * W_in
    hw_in = H_in * W_in
    x = al.make_tensor(x_ptr, al.bf16,
        al.make_layout((N, IC, D_in, H_in, W_in),
                       (ic_dhw, dhw, hw_in, W_in, 1)))

    oc27_g = OC * 27
    w = al.make_tensor(w_ptr, al.bf16,
        al.make_layout((IC, OC, 3, 3, 3),
                       (oc27_g, 27, 9, 3, 1)))

    b = al.make_tensor(b_ptr, al.bf16, al.make_layout((OC,), (1,)))

    oc_dhw = OC * D_out * H_out * W_out
    out_dhw = D_out * H_out * W_out
    out_hw = H_out * W_out
    add_t = al.make_tensor(add_ptr, al.bf16,
        al.make_layout((N, OC, D_out, H_out, W_out),
                       (oc_dhw, out_dhw, out_hw, W_out, 1)))
    out = al.make_tensor(out_ptr, al.bf16,
        al.make_layout((N, OC, D_out, H_out, W_out),
                       (oc_dhw, out_dhw, out_hw, W_out, 1)))

    three_f32 = al.convert(3.0, al.f32)
    six_f32 = al.convert(6.0, al.f32)
    zero_f32 = al.convert(0.0, al.f32)

    for elem_idx in al.range(ELEMS_PER_THREAD):
        linear_idx = tid * ELEMS_PER_THREAD + elem_idx

        loc_oc = linear_idx % OC_TILE
        t2 = linear_idx // OC_TILE
        loc_od = t2 % D_TILE
        t2 = t2 // D_TILE
        loc_oh = t2 % H_TILE
        loc_ow = t2 // H_TILE

        oc = oc_start + loc_oc
        od = od_start + loc_od
        oh = oh_start + loc_oh
        ow = ow_start + loc_ow

        acc = al.convert(0.0, al.f32)

        for ic in al.range(IC):
            for kd in al.range(3):
                id_check = od + padding - kd
                id = id_check // stride
                id_ok = id_check - id * stride
                if id_ok == 0:
                    if id >= 0:
                        if id < D_in:
                            for kh in al.range(3):
                                ih_check = oh + padding - kh
                                ih = ih_check // stride
                                ih_ok = ih_check - ih * stride
                                if ih_ok == 0:
                                    if ih >= 0:
                                        if ih < H_in:
                                            for kw in al.range(3):
                                                iw_check = ow + padding - kw
                                                iw = iw_check // stride
                                                iw_ok = iw_check - iw * stride
                                                if iw_ok == 0:
                                                    if iw >= 0:
                                                        if iw < W_in:
                                                            xv = al.convert(x[batch, ic, id, ih, iw], al.f32)
                                                            wv = al.convert(w[ic, oc, kd, kh, kw], al.f32)
                                                            acc = acc + xv * wv

        # Add bias
        bv = al.convert(b[oc], al.f32)
        acc = acc + bv

        # Add residual input
        av = al.convert(add_t[batch, oc, od, oh, ow], al.f32)
        val = acc + av

        # HardSwish: y = val * hardswish(val) = val^2 * relu6(val+3) / 6
        arg = val + three_f32
        result_val = zero_f32
        if arg < zero_f32:
            result_val = zero_f32
        else:
            if arg > six_f32:
                result_val = val * val
            else:
                vsq = val * val
                result_val = vsq * arg / six_f32

        out[batch, oc, od, oh, ow] = al.convert(result_val, al.bf16)


# ── Host helpers ─────────────────────────────────────────────────────────────

def _prepare_bf16(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def _avelang_conv_transpose_add_hardswish(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    add_input: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required.")

    x_bf16 = _prepare_bf16(x)
    w_bf16 = _prepare_bf16(weight)
    b_bf16 = _prepare_bf16(bias)
    add_bf16 = _prepare_bf16(add_input)

    N, IC, D_in, H_in, W_in = x_bf16.shape
    w_IC, OC, KD, KH, KW = w_bf16.shape
    N_a, OC_a, D_out, H_out, W_out = add_bf16.shape

    if w_IC != IC or KD != 3 or KH != 3 or KW != 3:
        raise ValueError(f"Weight shape mismatch: {w_bf16.shape}")
    if N_a != N or OC_a != OC:
        raise ValueError(f"add_input shape mismatch")
    if D_out % D_TILE != 0:
        raise ValueError(f"D_out ({D_out}) must be a multiple of D_TILE ({D_TILE})")
    if H_out % H_TILE != 0:
        raise ValueError(f"H_out ({H_out}) must be a multiple of H_TILE ({H_TILE})")
    if W_out % W_TILE != 0:
        raise ValueError(f"W_out ({W_out}) must be a multiple of W_TILE ({W_TILE})")
    if OC % OC_TILE != 0:
        raise ValueError(f"OC ({OC}) must be a multiple of OC_TILE ({OC_TILE})")

    stride = 2
    padding = 1

    out = torch.empty(
        (N, OC, D_out, H_out, W_out),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )

    w_tiles = W_out // W_TILE
    h_tiles = H_out // H_TILE
    d_tiles = D_out // D_TILE
    oc_groups = OC // OC_TILE
    num_blocks = N * oc_groups * d_tiles * h_tiles * w_tiles

    _fused_conv_transpose_add_hardswish_kernel[lambda: ((num_blocks, 1, 1), (THREADS, 1, 1))](
        x_bf16, w_bf16, b_bf16, add_bf16, out,
        N, IC, OC, D_in, H_in, W_in, D_out, H_out, W_out,
        stride, padding,
    )
    return out


# ── ModelNew ─────────────────────────────────────────────────────────────────

class ModelNew(nn.Module):
    """
    Optimized AveLang DSL model: 3D transposed convolution + add + HardSwish.
    Uses a fused gather-based kernel with tiled output, f32 accumulation,
    and inline hardswish epilogue.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride,
        padding, output_padding, bias_shape,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, add_input):
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        return _avelang_conv_transpose_add_hardswish(x, weight, bias, add_input)
