import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv2d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    KH: al.i32,
    KW: al.i32,
    stride: al.i32,
    padding: al.i32,
    dilation: al.i32,
):
    # 16x16 spatial tile handling all C_out=128 output channels per block.
    # Shared memory: input (18,18,16) + weight (128,16,3,3) ≈ 47 KB.

    input_layout = al.make_layout(
        (N, C_in, H, W),
        (C_in * H * W, H * W, W, 1),
    )
    inp = al.make_tensor(input_ptr, al.bf16, input_layout)

    weight_layout = al.make_layout(
        (C_out, C_in, KH, KW),
        (C_in * KH * KW, KH * KW, KW, 1),
    )
    wgt = al.make_tensor(weight_ptr, al.bf16, weight_layout)

    output_layout = al.make_layout(
        (N, C_out, H_out, W_out),
        (C_out * H_out * W_out, H_out * W_out, W_out, 1),
    )
    out = al.make_tensor(output_ptr, al.bf16, output_layout)

    bid_h = al.block_id(0)
    bid_w = al.block_id(1)
    n = al.block_id(2)

    oh_start = bid_h * 16
    ow_start = bid_w * 16

    tid_h = al.thread_id(0)
    tid_w = al.thread_id(1)
    tid = tid_h * 16 + tid_w

    input_smem = al.make_shared((18, 18, 16), al.bf16)
    weight_smem = al.make_shared((128, 16, 3, 3), al.bf16)

    # Cooperative load: input tile
    for idx in al.range(tid, 5184, 256):
        ic = idx % 16
        tmp = idx // 16
        iw_off = tmp % 18
        ih_off = tmp // 18
        ih = oh_start + ih_off
        iw = ow_start + iw_off
        if ih < H and iw < W and ic < C_in:
            input_smem[ih_off, iw_off, ic] = inp[n, ic, ih, iw]
        else:
            input_smem[ih_off, iw_off, ic] = al.convert(0.0, al.bf16)

    # Cooperative load: weight tile
    for idx in al.range(tid, 18432, 256):
        kw = idx % 3
        tmp = idx // 3
        kh = tmp % 3
        tmp = tmp // 3
        ic = tmp % 16
        oc = tmp // 16
        if oc < C_out and ic < C_in:
            weight_smem[oc, ic, kh, kw] = wgt[oc, ic, kh, kw]
        else:
            weight_smem[oc, ic, kh, kw] = al.convert(0.0, al.bf16)

    al.syncthreads()

    oh = oh_start + tid_h
    ow = ow_start + tid_w

    if oh < H_out and ow < W_out:
        for oc in al.range(C_out):
            acc = al.convert(0.0, al.f32)
            for ic in al.range(16):
                for kh in al.range(3):
                    for kw in al.range(3):
                        inp_val = al.convert(
                            input_smem[tid_h + kh, tid_w + kw, ic], al.f32
                        )
                        wgt_val = al.convert(
                            weight_smem[oc, ic, kh, kw], al.f32
                        )
                        acc = acc + inp_val * wgt_val
            out[n, oc, oh, ow] = al.convert(acc, al.bf16)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C_in, H, W = x.shape
        C_out = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        stride = self.stride
        padding = self.padding
        dilation = self.dilation

        H_out = (H + 2 * padding - dilation * (KH - 1) - 1) // stride + 1
        W_out = (W + 2 * padding - dilation * (KW - 1) - 1) // stride + 1

        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = self.conv2d.weight.data.to(torch.bfloat16).contiguous()
        out_bf16 = torch.empty(
            N, C_out, H_out, W_out, dtype=torch.bfloat16, device=x.device
        )

        TILE_H = 16
        TILE_W = 16

        grid_h = (H_out + TILE_H - 1) // TILE_H
        grid_w = (W_out + TILE_W - 1) // TILE_W
        grid_n = N

        conv2d_kernel[lambda: ((grid_h, grid_w, grid_n), (TILE_H, TILE_W, 1))](
            x_bf16,
            w_bf16,
            out_bf16,
            N,
            C_in,
            C_out,
            H,
            W,
            H_out,
            W_out,
            KH,
            KW,
            stride,
            padding,
            dilation,
        )

        return out_bf16.to(x.dtype)
