import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_H = 16
TILE_W = 16


@avelang.jit
def conv_transpose2d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    pad_h: al.i32,
    pad_w: al.i32,
    KH: al.constexpr,
    KW: al.constexpr,
    TILE_H: al.constexpr,
    TILE_W: al.constexpr,
):
    # Flat 1D tensor views
    in_total = B * C_in * H_in * W_in
    w_total = C_in * C_out * KH * KW
    out_total = B * C_out * H_out * W_out

    in_layout = al.make_layout((in_total,), (1,))
    w_layout = al.make_layout((w_total,), (1,))
    out_layout = al.make_layout((out_total,), (1,))

    input_t = al.make_tensor(input_ptr, al.bf16, in_layout)
    weight_t = al.make_tensor(weight_ptr, al.bf16, w_layout)
    output_t = al.make_tensor(output_ptr, al.bf16, out_layout)

    in_stride_b = C_in * H_in * W_in
    in_stride_c = H_in * W_in
    in_stride_h = W_in

    w_stride_ic = C_out * KH * KW
    w_stride_oc = KH * KW
    w_stride_kh = KW

    out_stride_b = C_out * H_out * W_out
    out_stride_c = H_out * W_out
    out_stride_h = W_out

    w_tile = al.block_id(0)
    h_tile = al.block_id(1)
    boc = al.block_id(2)

    b = boc // C_out
    oc = boc % C_out

    tx = al.thread_id(0)
    ty = al.thread_id(1)

    h_out = h_tile * TILE_H + ty
    w_out = w_tile * TILE_W + tx

    IN_H = TILE_H + KH - 1
    IN_W = TILE_W + KW - 1

    in_shared = al.make_shared((IN_H, IN_W), al.bf16)
    w_shared = al.make_shared((KH, KW), al.bf16)

    h_in_start = h_tile * TILE_H + pad_h - (KH - 1)
    w_in_start = w_tile * TILE_W + pad_w - (KW - 1)

    tid = ty * TILE_W + tx
    num_threads = TILE_H * TILE_W
    in_batch_off = b * in_stride_b

    acc = al.convert(0.0, al.f32)

    for ic in al.range(C_in):
        in_tile_elems = IN_H * IN_W

        # Cooperative load of input tile
        if tid < in_tile_elems:
            load_h = tid // IN_W
            load_w = tid % IN_W
            h_in = h_in_start + load_h
            w_in = w_in_start + load_w
            if h_in >= 0 and h_in < H_in and w_in >= 0 and w_in < W_in:
                in_idx = in_batch_off + (ic * in_stride_c) + (h_in * in_stride_h) + w_in
                in_shared[load_h, load_w] = input_t[in_idx]
            else:
                in_shared[load_h, load_w] = al.convert(0.0, al.bf16)

        tid2 = tid + num_threads
        if tid2 < in_tile_elems:
            load_h = tid2 // IN_W
            load_w = tid2 % IN_W
            h_in = h_in_start + load_h
            w_in = w_in_start + load_w
            if h_in >= 0 and h_in < H_in and w_in >= 0 and w_in < W_in:
                in_idx = in_batch_off + (ic * in_stride_c) + (h_in * in_stride_h) + w_in
                in_shared[load_h, load_w] = input_t[in_idx]
            else:
                in_shared[load_h, load_w] = al.convert(0.0, al.bf16)

        # Cooperative load of weight slice
        w_tile_elems = KH * KW
        if tid < w_tile_elems:
            w_kh = tid // KW
            w_kw = tid % KW
            w_idx = (ic * w_stride_ic) + (oc * w_stride_oc) + (w_kh * w_stride_kh) + w_kw
            w_shared[w_kh, w_kw] = weight_t[w_idx]

        al.syncthreads()

        # Accumulate with flipped kernel access into shared memory
        if h_out < H_out and w_out < W_out:
            for kh in al.range(KH):
                in_h = ty + KH - 1 - kh
                for kw in al.range(KW):
                    in_w = tx + KW - 1 - kw
                    in_val = al.convert(in_shared[in_h, in_w], al.f32)
                    w_val = al.convert(w_shared[kh, kw], al.f32)
                    acc = acc + in_val * w_val

        al.syncthreads()

    if h_out < H_out and w_out < W_out:
        out_idx = (b * out_stride_b) + (oc * out_stride_c) + (h_out * out_stride_h) + w_out
        output_t[out_idx] = al.convert(acc, al.bf16)


def avelang_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple,
    padding: tuple,
) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device"
    assert weight.is_cuda, "Weight tensor must be on CUDA/HIP device"
    assert stride == (1, 1), "Only stride=(1,1) is supported in this kernel"

    B, C_in, H_in, W_in = x.shape
    C_in_w, C_out, KH, KW = weight.shape
    assert C_in == C_in_w, (
        f"Input channels mismatch: x has {C_in}, weight expects {C_in_w}"
    )
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    H_out = (H_in - 1) * stride_h - 2 * pad_h + KH
    W_out = (W_in - 1) * stride_w - 2 * pad_w + KW

    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty(B, C_out, H_out, W_out, dtype=x.dtype, device=x.device)

    grid_x = (W_out + TILE_W - 1) // TILE_W
    grid_y = (H_out + TILE_H - 1) // TILE_H
    grid_z = B * C_out

    conv_transpose2d_kernel[
        lambda: ((grid_x, grid_y, grid_z), (TILE_W, TILE_H, 1))
    ](
        x.data_ptr(),
        weight.data_ptr(),
        out.data_ptr(),
        B,
        C_in,
        C_out,
        H_in,
        W_in,
        H_out,
        W_out,
        pad_h,
        pad_w,
        KH,
        KW,
        TILE_H,
        TILE_W,
    )

    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        bias: bool = False,
    ):
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv_transpose2d(
            x,
            self.conv_transpose2d.weight,
            self.conv_transpose2d.stride,
            self.conv_transpose2d.padding,
        )
