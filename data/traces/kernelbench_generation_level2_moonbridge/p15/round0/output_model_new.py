import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def conv_transpose3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    IC: al.i32,
    OC: al.i32,
    ID: al.i32,
    IH: al.i32,
    IW: al.i32,
    OD: al.i32,
    OH: al.i32,
    OW: al.i32,
    KD: al.i32,
    KH: al.i32,
    KW: al.i32,
    stride_v: al.i32,
    pad: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    spatial_size = OD * OH * OW
    total_elements = N * OC * spatial_size
    idx = bid * BLOCK_SIZE + tid

    if idx < total_elements:
        ch_spatial = OC * spatial_size
        n = idx // ch_spatial
        rem = idx - n * ch_spatial
        oc = rem // spatial_size
        rem2 = rem - oc * spatial_size
        od_val = rem2 // (OH * OW)
        rem3 = rem2 - od_val * (OH * OW)
        oh_val = rem3 // OW
        ow_val = rem3 - oh_val * OW

        input_layout = al.make_layout(
            (N, IC, ID, IH, IW),
            (IC * ID * IH * IW, ID * IH * IW, IH * IW, IW, 1),
        )
        inp = al.make_tensor(input_ptr, al.bf16, input_layout)

        weight_layout = al.make_layout(
            (IC, OC, KD, KH, KW),
            (OC * KD * KH * KW, KD * KH * KW, KH * KW, KW, 1),
        )
        wgt = al.make_tensor(weight_ptr, al.bf16, weight_layout)

        bias_layout = al.make_layout((OC,), (1,))
        b = al.make_tensor(bias_ptr, al.bf16, bias_layout)

        output_layout = al.make_layout(
            (N, OC, OD, OH, OW),
            (OC * OD * OH * OW, OD * OH * OW, OH * OW, OW, 1),
        )
        out = al.make_tensor(output_ptr, al.bf16, output_layout)

        acc = al.convert(0.0, al.f32)

        od_even = od_val - (od_val // 2) * 2
        oh_even = oh_val - (oh_val // 2) * 2
        ow_even = ow_val - (ow_val // 2) * 2

        for ic in al.range(0, IC):
            if od_even == 0:
                for kd in al.range(1, KD, 2):
                    id_val = (od_val + pad - kd) // 2
                    if oh_even == 0:
                        for kh in al.range(1, KH, 2):
                            ih_val = (oh_val + pad - kh) // 2
                            if ow_even == 0:
                                for kw in al.range(1, KW, 2):
                                    iw_val = (ow_val + pad - kw) // 2
                                    in_val = al.convert(inp[n, ic, id_val, ih_val, iw_val], al.f32)
                                    w_val = al.convert(wgt[ic, oc, kd, kh, kw], al.f32)
                                    acc = acc + in_val * w_val
                            else:
                                for kw in al.range(0, KW, 2):
                                    iw_val = (ow_val + pad - kw) // 2
                                    in_val = al.convert(inp[n, ic, id_val, ih_val, iw_val], al.f32)
                                    w_val = al.convert(wgt[ic, oc, kd, kh, kw], al.f32)
                                    acc = acc + in_val * w_val
                    else:
                        for kh in al.range(0, KH, 2):
                            ih_val = (oh_val + pad - kh) // 2
                            if ow_even == 0:
                                for kw in al.range(1, KW, 2):
                                    iw_val = (ow_val + pad - kw) // 2
                                    in_val = al.convert(inp[n, ic, id_val, ih_val, iw_val], al.f32)
                                    w_val = al.convert(wgt[ic, oc, kd, kh, kw], al.f32)
                                    acc = acc + in_val * w_val
                            else:
                                for kw in al.range(0, KW, 2):
                                    iw_val = (ow_val + pad - kw) // 2
                                    in_val = al.convert(inp[n, ic, id_val, ih_val, iw_val], al.f32)
                                    w_val = al.convert(wgt[ic, oc, kd, kh, kw], al.f32)
                                    acc = acc + in_val * w_val
            else:
                for kd in al.range(0, KD, 2):
                    id_val = (od_val + pad - kd) // 2
                    if oh_even == 0:
                        for kh in al.range(1, KH, 2):
                            ih_val = (oh_val + pad - kh) // 2
                            if ow_even == 0:
                                for kw in al.range(1, KW, 2):
                                    iw_val = (ow_val + pad - kw) // 2
                                    in_val = al.convert(inp[n, ic, id_val, ih_val, iw_val], al.f32)
                                    w_val = al.convert(wgt[ic, oc, kd, kh, kw], al.f32)
                                    acc = acc + in_val * w_val
                            else:
                                for kw in al.range(0, KW, 2):
                                    iw_val = (ow_val + pad - kw) // 2
                                    in_val = al.convert(inp[n, ic, id_val, ih_val, iw_val], al.f32)
                                    w_val = al.convert(wgt[ic, oc, kd, kh, kw], al.f32)
                                    acc = acc + in_val * w_val
                    else:
                        for kh in al.range(0, KH, 2):
                            ih_val = (oh_val + pad - kh) // 2
                            if ow_even == 0:
                                for kw in al.range(1, KW, 2):
                                    iw_val = (ow_val + pad - kw) // 2
                                    in_val = al.convert(inp[n, ic, id_val, ih_val, iw_val], al.f32)
                                    w_val = al.convert(wgt[ic, oc, kd, kh, kw], al.f32)
                                    acc = acc + in_val * w_val
                            else:
                                for kw in al.range(0, KW, 2):
                                    iw_val = (ow_val + pad - kw) // 2
                                    in_val = al.convert(inp[n, ic, id_val, ih_val, iw_val], al.f32)
                                    w_val = al.convert(wgt[ic, oc, kd, kh, kw], al.f32)
                                    acc = acc + in_val * w_val

        b_val = al.convert(b[oc], al.f32)
        acc = acc + b_val
        out[n, oc, od_val, oh_val, ow_val] = al.convert(acc, al.bf16)


@avelang.jit
def spatial_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    spatial_mean_out_ptr: al.Pointer(al.f32),
    N: al.i32,
    OC_val: al.i32,
    spatial_size: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < N * OC_val:
        n = bid // OC_val
        oc = bid - n * OC_val

        layout_x = al.make_layout(
            (N, OC_val, spatial_size),
            (OC_val * spatial_size, spatial_size, 1),
        )
        x = al.make_tensor(x_ptr, al.bf16, layout_x)

        smem = al.make_shared((BLOCK_SIZE,), al.f32)

        local_sum = al.convert(0.0, al.f32)
        for s in al.range(tid, spatial_size, BLOCK_SIZE):
            local_sum = local_sum + al.convert(x[n, oc, s], al.f32)

        smem[tid] = local_sum
        al.syncthreads()

        if tid < 128:
            smem[tid] = smem[tid] + smem[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem[tid] = smem[tid] + smem[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem[tid] = smem[tid] + smem[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem[tid] = smem[tid] + smem[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem[tid] = smem[tid] + smem[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem[tid] = smem[tid] + smem[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem[tid] = smem[tid] + smem[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem[tid] = smem[tid] + smem[tid + 1]

        if tid == 0:
            sp_mean = smem[0] / al.convert(spatial_size, al.f32)
            layout_sm = al.make_layout((N, OC_val), (OC_val, 1))
            sm_out = al.make_tensor(spatial_mean_out_ptr, al.f32, layout_sm)
            sm_out[n, oc] = sp_mean


@avelang.jit
def apply_kernel(
    x_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    rstd_ptr: al.Pointer(al.f32),
    spatial_mean_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    OC_val: al.i32,
    spatial_size: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    total_elements = N * OC_val * spatial_size
    idx = bid * BLOCK_SIZE + tid

    if idx < total_elements:
        oc_spatial = OC_val * spatial_size
        n = idx // oc_spatial
        rem = idx - n * oc_spatial
        oc = rem // spatial_size
        s = rem - oc * spatial_size

        layout_x = al.make_layout(
            (N, OC_val, spatial_size), (oc_spatial, spatial_size, 1)
        )
        x = al.make_tensor(x_ptr, al.bf16, layout_x)

        layout_gamma = al.make_layout((OC_val,), (1,))
        gamma = al.make_tensor(gamma_ptr, al.bf16, layout_gamma)

        layout_rstd = al.make_layout((OC_val,), (1,))
        rstd = al.make_tensor(rstd_ptr, al.f32, layout_rstd)

        layout_sm = al.make_layout((N, OC_val), (OC_val, 1))
        sm = al.make_tensor(spatial_mean_ptr, al.f32, layout_sm)

        layout_out = al.make_layout(
            (N, OC_val, spatial_size), (oc_spatial, spatial_size, 1)
        )
        out = al.make_tensor(output_ptr, al.bf16, layout_out)

        x_val = al.convert(x[n, oc, s], al.f32)
        g_val = al.convert(gamma[oc], al.f32)
        r_val = rstd[oc]
        s_val = sm[n, oc]

        result = g_val * r_val * (x_val - s_val)
        out[n, oc, s] = al.convert(result, al.bf16)


def avelang_forward(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_running_var: torch.Tensor,
) -> torch.Tensor:
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = conv_weight.shape

    stride_v = 2
    pad = 1
    eps = 1e-5

    OD = (ID - 1) * stride_v - 2 * pad + KD
    OH = (IH - 1) * stride_v - 2 * pad + KH
    OW = (IW - 1) * stride_v - 2 * pad + KW
    spatial_size = OD * OH * OW

    x_contig = x.contiguous().to(torch.bfloat16)
    w_contig = conv_weight.contiguous().to(torch.bfloat16)
    if conv_bias is not None:
        b_contig = conv_bias.contiguous().to(torch.bfloat16)
    else:
        b_contig = torch.zeros(OC, dtype=torch.bfloat16, device=x.device)
    bn_w_contig = bn_weight.contiguous().to(torch.bfloat16)

    conv_out = torch.empty(
        (N, OC, OD, OH, OW), dtype=torch.bfloat16, device=x.device
    )
    total_conv_elements = N * OC * spatial_size
    num_conv_blocks = (total_conv_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    conv_transpose3d_kernel[
        lambda: ((num_conv_blocks, 1, 1), (BLOCK_SIZE, 1, 1))
    ](
        x_contig, w_contig, b_contig, conv_out,
        N, IC, OC, ID, IH, IW, OD, OH, OW, KD, KH, KW, stride_v, pad,
    )

    spatial_mean = torch.empty((N, OC), dtype=torch.float32, device=x.device)
    spatial_reduce_kernel[
        lambda: ((N * OC, 1, 1), (BLOCK_SIZE, 1, 1))
    ](conv_out, spatial_mean, N, OC, spatial_size)

    bn_running_var_contig = bn_running_var.contiguous().to(torch.float32)
    rstd = torch.rsqrt(bn_running_var_contig + eps)

    final_out = torch.empty(
        (N, OC, OD, OH, OW), dtype=torch.bfloat16, device=x.device
    )
    apply_kernel[lambda: ((num_conv_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        conv_out, bn_w_contig, rstd, spatial_mean, final_out, N, OC, spatial_size
    )

    return final_out


class ModelNew(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, bias=True
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return avelang_forward(
            x,
            self.conv_transpose.weight.data,
            self.conv_transpose.bias.data
            if self.conv_transpose.bias is not None
            else None,
            self.batch_norm.weight.data,
            self.batch_norm.running_var.data,
        )
