import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_OC = 16
TILE_H = 16
TILE_W = 16
TILE_IC = 16


@avelang.jit
def conv_transpose2d_tiled_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    stride: al.i32,
    padding: al.i32,
    TILE_OC: al.constexpr,
    TILE_H: al.constexpr,
    TILE_W: al.constexpr,
    TILE_IC: al.constexpr,
    KH: al.constexpr,
    KW: al.constexpr,
):
    batch_oc_chunk = al.block_id(0)
    block_h = al.block_id(1)
    block_w = al.block_id(2)

    tid_h = al.thread_id(0)
    tid_w = al.thread_id(1)

    oc_chunks = (OC + TILE_OC - 1) // TILE_OC
    b = batch_oc_chunk // oc_chunks
    oc_start = (batch_oc_chunk % oc_chunks) * TILE_OC

    oh = block_h * TILE_H + tid_h
    ow = block_w * TILE_W + tid_w

    valid = (oh < H_out) and (ow < W_out)

    in_stride_ic = H * W
    in_layout = al.make_layout((B, IC, H, W), (IC * in_stride_ic, in_stride_ic, W, 1))
    input_t = al.make_tensor(input_ptr, al.bf16, in_layout)

    wt_layout = al.make_layout((IC, OC, KH, KW), (OC * KH * KW, KH * KW, KW, 1))
    weight_t = al.make_tensor(weight_ptr, al.bf16, wt_layout)

    out_stride_oc = H_out * W_out
    out_layout = al.make_layout((B, OC, H_out, W_out), (OC * out_stride_oc, out_stride_oc, W_out, 1))
    output_t = al.make_tensor(output_ptr, al.bf16, out_layout)

    h_in_start = block_h * TILE_H - KH + 1
    w_in_start = block_w * TILE_W - KW + 1

    in_h = TILE_H + KH - 1
    in_w = TILE_W + KW - 1

    in_shared = al.make_shared((TILE_IC, in_h, in_w), al.bf16)
    wt_shared = al.make_shared((TILE_IC, TILE_OC, KH, KW), al.bf16)

    acc = al.make_local((TILE_OC,), al.f32)
    for o in al.range(TILE_OC):
        acc[o] = al.convert(0.0, al.f32)

    thread_linear = tid_h * TILE_W + tid_w
    num_threads = TILE_H * TILE_W

    total_in_elems = TILE_IC * in_h * in_w
    total_wt_elems = TILE_IC * TILE_OC * KH * KW

    elems_per_thread_in = (total_in_elems + num_threads - 1) // num_threads
    elems_per_thread_wt = (total_wt_elems + num_threads - 1) // num_threads

    num_ic_chunks = (IC + TILE_IC - 1) // TILE_IC

    for ic_chunk in al.range(num_ic_chunks):
        ic_start = ic_chunk * TILE_IC

        for i in al.range(elems_per_thread_in):
            elem_idx = thread_linear + i * num_threads
            if elem_idx < total_in_elems:
                tmp = elem_idx
                iw_local = tmp % in_w
                tmp = tmp // in_w
                ih_local = tmp % in_h
                ic_local = tmp // in_h

                ic_global = ic_start + ic_local
                ih_global = h_in_start + ih_local
                iw_global = w_in_start + iw_local

                if ic_global < IC and ih_global >= 0 and ih_global < H and iw_global >= 0 and iw_global < W:
                    in_shared[ic_local, ih_local, iw_local] = input_t[b, ic_global, ih_global, iw_global]
                else:
                    in_shared[ic_local, ih_local, iw_local] = al.convert(0.0, al.bf16)

        for i in al.range(elems_per_thread_wt):
            elem_idx = thread_linear + i * num_threads
            if elem_idx < total_wt_elems:
                tmp = elem_idx
                kw = tmp % KW
                tmp = tmp // KW
                kh = tmp % KH
                tmp = tmp // KH
                oc_local = tmp % TILE_OC
                ic_local = tmp // TILE_OC

                ic_global = ic_start + ic_local
                oc_global = oc_start + oc_local

                if ic_global < IC and oc_global < OC:
                    wt_shared[ic_local, oc_local, kh, kw] = weight_t[ic_global, oc_global, kh, kw]
                else:
                    wt_shared[ic_local, oc_local, kh, kw] = al.convert(0.0, al.bf16)

        al.syncthreads()

        if valid:
            for ic_local in al.range(TILE_IC):
                ic_global = ic_start + ic_local
                if ic_global < IC:
                    for kh in al.range(KH):
                        ih = oh - kh + padding
                        if ih >= 0 and ih < H:
                            ih_local = ih - h_in_start
                            if ih_local >= 0 and ih_local < in_h:
                                for kw in al.range(KW):
                                    iw = ow - kw + padding
                                    if iw >= 0 and iw < W:
                                        iw_local = iw - w_in_start
                                        if iw_local >= 0 and iw_local < in_w:
                                            inp = al.convert(in_shared[ic_local, ih_local, iw_local], al.f32)
                                            for oc_local in al.range(TILE_OC):
                                                oc_global = oc_start + oc_local
                                                if oc_global < OC:
                                                    wt = al.convert(wt_shared[ic_local, oc_local, kh, kw], al.f32)
                                                    acc[oc_local] = acc[oc_local] + inp * wt

        al.syncthreads()

    if valid:
        for oc_local in al.range(TILE_OC):
            oc_global = oc_start + oc_local
            if oc_global < OC:
                output_t[b, oc_global, oh, ow] = al.convert(acc[oc_local], al.bf16)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, IC, H, W = x.shape
        OC = self.conv_transpose2d.out_channels
        KH = self.conv_transpose2d.kernel_size[0]
        KW = self.conv_transpose2d.kernel_size[1]
        stride_val = self.conv_transpose2d.stride[0]
        pad_val = self.conv_transpose2d.padding[0]
        output_padding = self.conv_transpose2d.output_padding[0]

        H_out = (H - 1) * stride_val - 2 * pad_val + (KH - 1) + output_padding + 1
        W_out = (W - 1) * stride_val - 2 * pad_val + (KW - 1) + output_padding + 1

        x_bf16 = x.to(torch.bfloat16).contiguous()
        w = self.conv_transpose2d.weight
        w_bf16 = w.to(torch.bfloat16).contiguous()

        out = torch.zeros(B, OC, H_out, W_out, dtype=torch.bfloat16, device=x.device)

        grid_h = (H_out + TILE_H - 1) // TILE_H
        grid_w = (W_out + TILE_W - 1) // TILE_W
        oc_chunks = (OC + TILE_OC - 1) // TILE_OC

        conv_transpose2d_tiled_kernel[lambda: (
            (B * oc_chunks, grid_h, grid_w),
            (TILE_H, TILE_W, 1),
        )](
            x_bf16, w_bf16, out,
            B, IC, OC, H, W, H_out, W_out, stride_val, pad_val,
            TILE_OC=TILE_OC, TILE_H=TILE_H, TILE_W=TILE_W, TILE_IC=TILE_IC,
            KH=KH, KW=KW,
        )

        if self.conv_transpose2d.bias is not None:
            bias_bf16 = self.conv_transpose2d.bias.to(torch.bfloat16)
            out = out + bias_bf16.view(1, -1, 1, 1)

        return out.to(x.dtype)
