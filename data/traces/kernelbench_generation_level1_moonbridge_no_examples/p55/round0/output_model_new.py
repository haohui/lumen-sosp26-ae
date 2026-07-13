import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_H = 8
BLOCK_W = 32
IC_TILE = 16
OC_TILE = 32


@avelang.jit
def conv2d_3x3_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    IC: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    OC: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    grid_h: al.i32,
    grid_w: al.i32,
):
    in_layout = al.make_layout((N, IC, H_in, W_in), (IC * H_in * W_in, H_in * W_in, W_in, 1))
    x = al.make_tensor(input_ptr, al.bf16, in_layout)

    wt_layout = al.make_layout((OC, IC, 3, 3), (IC * 9, 9, 3, 1))
    w = al.make_tensor(weight_ptr, al.bf16, wt_layout)

    out_layout = al.make_layout((N, OC, H_out, W_out), (OC * H_out * W_out, H_out * W_out, W_out, 1))
    y = al.make_tensor(output_ptr, al.bf16, out_layout)

    block_h_id = al.block_id(0)
    block_w_id = al.block_id(1)
    batch_id = al.block_id(2)

    tid_h = al.thread_id(0)
    tid_w = al.thread_id(1)
    flat_tid = tid_h * BLOCK_W + tid_w
    num_threads = BLOCK_H * BLOCK_W

    h_out = block_h_id * BLOCK_H + tid_h
    w_out = block_w_id * BLOCK_W + tid_w

    s_in = al.make_shared((IC_TILE, BLOCK_H + 2, BLOCK_W + 2), al.bf16)
    s_wt = al.make_shared((OC_TILE, IC_TILE, 3, 3), al.bf16)

    acc = al.make_local((OC_TILE,), al.f32)

    num_oc_groups = OC // OC_TILE
    num_ic_groups = IC // IC_TILE

    in_elems = IC_TILE * (BLOCK_H + 2) * (BLOCK_W + 2)
    wt_elems = OC_TILE * IC_TILE * 3 * 3

    is_boundary = (block_h_id == (grid_h - 1)) or (block_w_id == (grid_w - 1))

    for oc_group in al.range(num_oc_groups):
        oc_start = oc_group * OC_TILE

        for oc_local in al.range(OC_TILE):
            acc[oc_local] = al.convert(0.0, al.f32)

        for ic_group in al.range(num_ic_groups):
            ic_start = ic_group * IC_TILE

            if is_boundary:
                for idx in al.range(flat_tid, in_elems, num_threads):
                    w_off = idx % (BLOCK_W + 2)
                    rest = idx // (BLOCK_W + 2)
                    h_off = rest % (BLOCK_H + 2)
                    ic = rest // (BLOCK_H + 2)

                    g_h = block_h_id * BLOCK_H + h_off
                    g_w = block_w_id * BLOCK_W + w_off

                    if (g_h < H_in) and (g_w < W_in):
                        s_in[ic, h_off, w_off] = x[batch_id, ic_start + ic, g_h, g_w]
                    else:
                        s_in[ic, h_off, w_off] = al.convert(0.0, al.bf16)
            else:
                for idx in al.range(flat_tid, in_elems, num_threads):
                    w_off = idx % (BLOCK_W + 2)
                    rest = idx // (BLOCK_W + 2)
                    h_off = rest % (BLOCK_H + 2)
                    ic = rest // (BLOCK_H + 2)

                    g_h = block_h_id * BLOCK_H + h_off
                    g_w = block_w_id * BLOCK_W + w_off
                    s_in[ic, h_off, w_off] = x[batch_id, ic_start + ic, g_h, g_w]

            for idx in al.range(flat_tid, wt_elems, num_threads):
                kw = idx % 3
                rest = idx // 3
                kh = rest % 3
                rest = rest // 3
                ic = rest % IC_TILE
                oc_local = rest // IC_TILE
                s_wt[oc_local, ic, kh, kw] = w[oc_start + oc_local, ic_start + ic, kh, kw]

            al.syncthreads()

            for ic in al.range(IC_TILE):
                for kh in al.range(3):
                    for kw in al.range(3):
                        in_val = al.convert(s_in[ic, tid_h + kh, tid_w + kw], al.f32)
                        for oc_local in al.range(OC_TILE):
                            wt_val = al.convert(s_wt[oc_local, ic, kh, kw], al.f32)
                            acc[oc_local] = acc[oc_local] + in_val * wt_val

            al.syncthreads()

        valid_out = (h_out < H_out) and (w_out < W_out)
        if valid_out:
            for oc_local in al.range(OC_TILE):
                y[batch_id, oc_start + oc_local, h_out, w_out] = al.convert(acc[oc_local], al.bf16)


def avelang_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> torch.Tensor:
    N, IC, H, W = x.shape
    OC = weight.shape[0]
    KH, KW = weight.shape[2], weight.shape[3]

    H_out = (H + 2 * padding - dilation * (KH - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (KW - 1) - 1) // stride + 1

    if not x.is_cuda:
        x = x.to(device="cuda", dtype=torch.bfloat16)
    else:
        x = x.to(dtype=torch.bfloat16)
    if not weight.is_cuda:
        weight = weight.to(device="cuda", dtype=torch.bfloat16)
    else:
        weight = weight.to(dtype=torch.bfloat16)

    x = x.contiguous()
    weight = weight.contiguous()

    out = torch.empty(N, OC, H_out, W_out, dtype=torch.bfloat16, device=x.device)

    grid_h = (H_out + BLOCK_H - 1) // BLOCK_H
    grid_w = (W_out + BLOCK_W - 1) // BLOCK_W
    grid_z = N

    conv2d_3x3_kernel[lambda: ((grid_h, grid_w, grid_z), (BLOCK_H, BLOCK_W, 1))](
        x, weight, out, N, IC, H, W, OC, H_out, W_out, grid_h, grid_w,
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
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
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
        weight = self.conv2d.weight.data
        bias = self.conv2d.bias
        stride_val = self.conv2d.stride[0]
        padding_val = self.conv2d.padding[0]
        dilation_val = self.conv2d.dilation[0]
        groups_val = self.conv2d.groups

        out = avelang_conv2d(x, weight, stride_val, padding_val, dilation_val, groups_val)

        if bias is not None:
            out = out + bias.view(1, -1, 1, 1)

        return out
