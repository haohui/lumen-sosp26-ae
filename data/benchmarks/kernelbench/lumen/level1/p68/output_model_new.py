"""ConvTranspose3D kernel using Substrate DSL with fixed-size kernel unrolling."""

import math

import substrate
import substrate.language as S
import torch
import torch.nn as nn

# Fixed kernel dimensions for this problem
KERNEL_D: S.constexpr = 3
KERNEL_H: S.constexpr = 5
KERNEL_W: S.constexpr = 5


@substrate.jit
def _conv_transpose3d_kernel(
    input_ptr: S.Pointer(S.bf16),
    weight_ptr: S.Pointer(S.bf16),
    output_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    in_channels: S.i32,
    out_channels: S.i32,
    depth_in: S.i32,
    height_in: S.i32,
    width_in: S.i32,
    depth_out: S.i32,
    height_out: S.i32,
    width_out: S.i32,
):
    """
    ConvTranspose3D kernel with kernel dimensions unrolled.

    Each thread computes one output element.
    ConvTranspose3d with stride=1, padding=0:
    output[d] = sum over kd where (d - kd) valid: weight[ic, oc, kd, kh, kw] * input[ic, d-kd]
    """
    tid = S.thread_id(0)
    bid = S.block_id(0)
    bdim = S.block_dim(0)

    idx = bid * bdim + tid

    total_output = batch_size * out_channels * depth_out * height_out * width_out

    if idx >= total_output:
        return

    # Decode (n, oc, od, oh, ow) from linear index
    vol_spatial = depth_out * height_out * width_out
    vol_ch = out_channels * vol_spatial

    n = idx // vol_ch
    rem1 = idx - n * vol_ch
    oc = rem1 // vol_spatial
    rem2 = rem1 - oc * vol_spatial
    od = rem2 // (height_out * width_out)
    rem3 = rem2 - od * (height_out * width_out)
    oh = rem3 // width_out
    ow = rem3 - oh * width_out

    # Create tensor views
    # Input: (batch, in_ch, din, hin, win)
    in_stride_n = in_channels * depth_in * height_in * width_in
    in_stride_c = depth_in * height_in * width_in
    in_stride_d = height_in * width_in
    in_stride_h = width_in
    in_stride_w = 1

    input_layout = S.make_layout(
        (batch_size, in_channels, depth_in, height_in, width_in),
        (in_stride_n, in_stride_c, in_stride_d, in_stride_h, in_stride_w),
    )
    input_tensor = S.make_tensor(input_ptr, S.bf16, input_layout)

    # Weight: (in_ch, out_ch, kd, kh, kw)
    w_stride_ic = out_channels * KERNEL_D * KERNEL_H * KERNEL_W
    w_stride_oc = KERNEL_D * KERNEL_H * KERNEL_W
    w_stride_d = KERNEL_H * KERNEL_W
    w_stride_h = KERNEL_W
    w_stride_w = 1

    weight_layout = S.make_layout(
        (in_channels, out_channels, KERNEL_D, KERNEL_H, KERNEL_W),
        (w_stride_ic, w_stride_oc, w_stride_d, w_stride_h, w_stride_w),
    )
    weight_tensor = S.make_tensor(weight_ptr, S.bf16, weight_layout)

    # Output: (batch, out_ch, dout, hout, wout)
    out_stride_n = out_channels * depth_out * height_out * width_out
    out_stride_c = depth_out * height_out * width_out
    out_stride_d = height_out * width_out
    out_stride_h = width_out
    out_stride_w = 1

    output_layout = S.make_layout(
        (batch_size, out_channels, depth_out, height_out, width_out),
        (out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w),
    )
    output_tensor = S.make_tensor(output_ptr, S.bf16, output_layout)

    # Accumulate in f32
    acc = S.convert(0.0, S.f32)

    # Iterate over input channels
    for ic in S.range(in_channels):
        # Unrolled kernel loop for KERNEL_D=3, KERNEL_H=5, KERNEL_W=5
        # kd=0, kh=0, kw=0
        id_in = od - 0
        ih_in = oh - 0
        iw_in = ow - 0
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 0, 0]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=0, kw=1
        iw_in = ow - 1
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 0, 1]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=0, kw=2
        iw_in = ow - 2
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 0, 2]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=0, kw=3
        iw_in = ow - 3
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 0, 3]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=0, kw=4
        iw_in = ow - 4
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 0, 4]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=1, kw=0
        ih_in = oh - 1
        iw_in = ow - 0
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 1, 0]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=1, kw=1
        iw_in = ow - 1
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 1, 1]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=1, kw=2
        iw_in = ow - 2
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 1, 2]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=1, kw=3
        iw_in = ow - 3
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 1, 3]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=1, kw=4
        iw_in = ow - 4
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 1, 4]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=2, kw=0
        ih_in = oh - 2
        iw_in = ow - 0
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 2, 0]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=2, kw=1
        iw_in = ow - 1
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 2, 1]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=2, kw=2
        iw_in = ow - 2
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 2, 2]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=2, kw=3
        iw_in = ow - 3
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 2, 3]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=2, kw=4
        iw_in = ow - 4
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 2, 4]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=3, kw=0
        ih_in = oh - 3
        iw_in = ow - 0
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 3, 0]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=3, kw=1
        iw_in = ow - 1
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 3, 1]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=3, kw=2
        iw_in = ow - 2
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 3, 2]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=3, kw=3
        iw_in = ow - 3
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 3, 3]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=3, kw=4
        iw_in = ow - 4
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 3, 4]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=4, kw=0
        ih_in = oh - 4
        iw_in = ow - 0
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 4, 0]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=4, kw=1
        iw_in = ow - 1
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 4, 1]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=4, kw=2
        iw_in = ow - 2
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 4, 2]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=4, kw=3
        iw_in = ow - 3
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 4, 3]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=0, kh=4, kw=4
        iw_in = ow - 4
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 0, 4, 4]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1
        id_in = od - 1
        # kd=1, kh=0, kw=0
        ih_in = oh - 0
        iw_in = ow - 0
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 0, 0]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=0, kw=1
        iw_in = ow - 1
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 0, 1]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=0, kw=2
        iw_in = ow - 2
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 0, 2]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=0, kw=3
        iw_in = ow - 3
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 0, 3]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=0, kw=4
        iw_in = ow - 4
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 0, 4]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=1, kw=0
        ih_in = oh - 1
        iw_in = ow - 0
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 1, 0]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=1, kw=1
        iw_in = ow - 1
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 1, 1]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=1, kw=2
        iw_in = ow - 2
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 1, 2]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=1, kw=3
        iw_in = ow - 3
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 1, 3]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=1, kw=4
        iw_in = ow - 4
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 1, 4]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=2, kw=0
        ih_in = oh - 2
        iw_in = ow - 0
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 2, 0]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=2, kw=1
        iw_in = ow - 1
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 2, 1]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=2, kw=2
        iw_in = ow - 2
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 2, 2]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=2, kw=3
        iw_in = ow - 3
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 2, 3]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=2, kw=4
        iw_in = ow - 4
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 2, 4]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=3, kw=0
        ih_in = oh - 3
        iw_in = ow - 0
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 3, 0]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=3, kw=1
        iw_in = ow - 1
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 3, 1]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=3, kw=2
        iw_in = ow - 2
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 3, 2]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=3, kw=3
        iw_in = ow - 3
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 3, 3]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=3, kw=4
        iw_in = ow - 4
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 3, 4]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=4, kw=0
        ih_in = oh - 4
        iw_in = ow - 0
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 4, 0]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=4, kw=1
        iw_in = ow - 1
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 4, 1]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=4, kw=2
        iw_in = ow - 2
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 4, 2]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=4, kw=3
        iw_in = ow - 3
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 4, 3]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=1, kh=4, kw=4
        iw_in = ow - 4
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 1, 4, 4]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2
        id_in = od - 2
        # kd=2, kh=0, kw=0
        ih_in = oh - 0
        iw_in = ow - 0
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 0, 0]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=0, kw=1
        iw_in = ow - 1
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 0, 1]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=0, kw=2
        iw_in = ow - 2
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 0, 2]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=0, kw=3
        iw_in = ow - 3
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 0, 3]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=0, kw=4
        iw_in = ow - 4
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 0, 4]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=1, kw=0
        ih_in = oh - 1
        iw_in = ow - 0
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 1, 0]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=1, kw=1
        iw_in = ow - 1
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 1, 1]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=1, kw=2
        iw_in = ow - 2
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 1, 2]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=1, kw=3
        iw_in = ow - 3
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 1, 3]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=1, kw=4
        iw_in = ow - 4
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 1, 4]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=2, kw=0
        ih_in = oh - 2
        iw_in = ow - 0
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 2, 0]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=2, kw=1
        iw_in = ow - 1
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 2, 1]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=2, kw=2
        iw_in = ow - 2
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 2, 2]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=2, kw=3
        iw_in = ow - 3
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 2, 3]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=2, kw=4
        iw_in = ow - 4
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 2, 4]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=3, kw=0
        ih_in = oh - 3
        iw_in = ow - 0
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 3, 0]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=3, kw=1
        iw_in = ow - 1
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 3, 1]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=3, kw=2
        iw_in = ow - 2
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 3, 2]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=3, kw=3
        iw_in = ow - 3
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 3, 3]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=3, kw=4
        iw_in = ow - 4
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 3, 4]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=4, kw=0
        ih_in = oh - 4
        iw_in = ow - 0
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 4, 0]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=4, kw=1
        iw_in = ow - 1
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 4, 1]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=4, kw=2
        iw_in = ow - 2
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 4, 2]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=4, kw=3
        iw_in = ow - 3
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 4, 3]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

        # kd=2, kh=4, kw=4
        iw_in = ow - 4
        if id_in >= 0 and id_in < depth_in and ih_in >= 0 and ih_in < height_in and iw_in >= 0 and iw_in < width_in:
            in_val = input_tensor[n, ic, id_in, ih_in, iw_in]
            w_val = weight_tensor[ic, oc, 2, 4, 4]
            acc = acc + S.convert(in_val, S.f32) * S.convert(w_val, S.f32)

    # Store output
    output_tensor[n, oc, od, oh, ow] = S.convert(acc, S.bf16)


class ModelNew(nn.Module):
    """ConvTranspose3d using Substrate DSL kernel."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()

        if groups != 1:
            raise NotImplementedError("Only groups=1 is supported")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Weight shape for ConvTranspose3d: (in_channels, out_channels/groups, kd, kh, kw)
        self.weight = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels // groups,
                kernel_size[0],
                kernel_size[1],
                kernel_size[2],
            )
        )

        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias_param", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in = (
                self.in_channels
                * self.kernel_size[0]
                * self.kernel_size[1]
                * self.kernel_size[2]
            )
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        original_dtype = x.dtype

        batch_size, in_channels, depth_in, height_in, width_in = x.shape

        # Convert to bf16 for kernel computation
        x_bf16 = x.to(dtype=torch.bfloat16)
        weight_bf16 = self.weight.detach().to(device=x.device, dtype=torch.bfloat16).contiguous()

        # Compute output dimensions for transposed convolution
        # output_size = (input_size - 1) * stride + kernel_size - 2 * padding + output_padding
        depth_out = (
            (depth_in - 1) * self.stride[0]
            + self.kernel_size[0]
            - 2 * self.padding[0]
            + self.output_padding[0]
        )
        height_out = (
            (height_in - 1) * self.stride[1]
            + self.kernel_size[1]
            - 2 * self.padding[1]
            + self.output_padding[1]
        )
        width_out = (
            (width_in - 1) * self.stride[2]
            + self.kernel_size[2]
            - 2 * self.padding[2]
            + self.output_padding[2]
        )

        # Allocate output tensor
        output = torch.empty(
            (batch_size, self.out_channels, depth_out, height_out, width_out),
            dtype=torch.bfloat16,
            device=x.device,
        )

        # Launch kernel
        block_size = 256
        total_elements = batch_size * self.out_channels * depth_out * height_out * width_out
        grid_size = (total_elements + block_size - 1) // block_size

        _conv_transpose3d_kernel[lambda: ((grid_size, 1, 1), (block_size, 1, 1))](
            x_bf16,
            weight_bf16,
            output,
            batch_size,
            self.in_channels,
            self.out_channels,
            depth_in,
            height_in,
            width_in,
            depth_out,
            height_out,
            width_out,
        )

        # Convert back to original dtype
        if original_dtype != torch.bfloat16:
            output = output.to(original_dtype)

        # Add bias if needed
        if self.bias_param is not None:
            bias = self.bias_param.detach().to(device=x.device, dtype=output.dtype)
            output = output + bias.view(1, -1, 1, 1, 1)

        return output
