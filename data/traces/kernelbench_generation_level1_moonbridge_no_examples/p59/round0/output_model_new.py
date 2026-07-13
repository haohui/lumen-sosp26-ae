import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv3d_kernel(
    inp_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_in: al.constexpr,
    C_out: al.constexpr,
    H: al.i32,
    W: al.i32,
    D: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.constexpr,
    TILE_H: al.constexpr,
    TILE_W: al.constexpr,
    BLOCK_OC: al.constexpr,
    in_s0: al.i32,
    in_s1: al.i32,
    in_s2: al.i32,
    in_s3: al.i32,
    in_s4: al.i32,
    w_s0: al.i32,
    w_s1: al.i32,
    w_s2: al.i32,
    w_s3: al.i32,
    out_s0: al.i32,
    out_s1: al.i32,
    out_s2: al.i32,
    out_s3: al.i32,
    out_s4: al.i32,
):
    in_layout = al.make_layout(
        (B, C_in, H, W, D),
        (in_s0, in_s1, in_s2, in_s3, in_s4),
    )
    inp = al.make_tensor(inp_ptr, al.bf16, in_layout)

    w_layout = al.make_layout(
        (C_out, C_in, K, K),
        (w_s0, w_s1, w_s2, w_s3),
    )
    wgt = al.make_tensor(w_ptr, al.bf16, w_layout)

    out_layout = al.make_layout(
        (B, C_out, H_out, W_out, D),
        (out_s0, out_s1, out_s2, out_s3, out_s4),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    tid_x = al.thread_id(0)
    tid_y = al.thread_id(1)
    tid = tid_y * TILE_W + tid_x
    num_threads = TILE_H * TILE_W

    blk_x = al.block_id(0)
    blk_y = al.block_id(1)
    blk_z = al.block_id(2)

    b_idx = blk_z // D
    d_idx = blk_z % D

    h_out = blk_y * TILE_H + tid_y
    w_out = blk_x * TILE_W + tid_x
    valid_out = (h_out < H_out) and (w_out < W_out)

    in_h_ext = TILE_H + K - 1
    in_w_ext = TILE_W + K - 1
    in_shared = al.make_shared((in_h_ext, in_w_ext), al.bf16)
    w_shared = al.make_shared((BLOCK_OC, C_in, K, K), al.bf16)

    in_h_start = blk_y * TILE_H
    in_w_start = blk_x * TILE_W
    total_elems = in_h_ext * in_w_ext

    acc = al.make_local((BLOCK_OC,), al.f32)
    w_tile_sz = C_in * K * K
    w_kk = K * K

    for oc_start in al.range(0, C_out, BLOCK_OC):
        for o in al.range(BLOCK_OC):
            acc[o] = al.convert(0, al.f32)

        # Cooperative load of weight tile
        for idx in al.range(tid, BLOCK_OC * w_tile_sz, num_threads):
            o = idx // w_tile_sz
            r0 = idx % w_tile_sz
            ic = r0 // w_kk
            r1 = r0 % w_kk
            kh = r1 // K
            kw = r1 % K
            oc = oc_start + o
            if oc < C_out:
                w_shared[o, ic, kh, kw] = wgt[oc, ic, kh, kw]
            else:
                w_shared[o, ic, kh, kw] = al.convert(0, al.bf16)

        al.syncthreads()

        for ic in al.range(C_in):
            for idx in al.range(tid, total_elems, num_threads):
                ih = idx // in_w_ext
                iw = idx % in_w_ext
                in_h = in_h_start + ih
                in_w = in_w_start + iw
                valid = (in_h < H) and (in_w < W)
                if valid:
                    in_shared[ih, iw] = inp[b_idx, ic, in_h, in_w, d_idx]
                else:
                    in_shared[ih, iw] = al.convert(0, al.bf16)

            al.syncthreads()

            for o in al.range(BLOCK_OC):
                oc = oc_start + o
                if oc < C_out:
                    for kh in al.range(K):
                        for kw in al.range(K):
                            in_val = al.convert(in_shared[tid_y + kh, tid_x + kw], al.f32)
                            w_val = al.convert(w_shared[o, ic, kh, kw], al.f32)
                            acc[o] = acc[o] + in_val * w_val

            al.syncthreads()

        if valid_out:
            for o in al.range(BLOCK_OC):
                oc = oc_start + o
                if oc < C_out:
                    out[b_idx, oc, h_out, w_out, d_idx] = al.convert(acc[o], al.bf16)


class ModelNew(nn.Module):
    TILE_H = 16
    TILE_W = 16
    BLOCK_OC = 16

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
        self.bias = bias
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, 1),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input must be on CUDA/HIP device"
        B, C_in, H, W, D = x.shape
        C_out = self.out_channels
        K = self.kernel_size

        H_out = (H + 2 * self.padding - self.dilation * (K - 1) - 1) // self.stride + 1
        W_out = (W + 2 * self.padding - self.dilation * (K - 1) - 1) // self.stride + 1
        D_out = (D + 2 * self.padding - self.dilation * (1 - 1) - 1) // self.stride + 1

        x_bf16 = x.contiguous().to(torch.bfloat16)
        w_bf16 = self.conv3d.weight.squeeze(-1).contiguous().to(torch.bfloat16)

        out_bf16 = torch.empty(
            B, C_out, H_out, W_out, D_out,
            dtype=torch.bfloat16,
            device=x.device,
        )

        is0 = C_in * H * W * D
        is1 = H * W * D
        is2 = W * D
        is3 = D
        is4 = 1
        ws0 = C_in * K * K
        ws1 = K * K
        ws2 = K
        ws3 = 1
        os0 = C_out * H_out * W_out * D_out
        os1 = H_out * W_out * D_out
        os2 = W_out * D_out
        os3 = D_out
        os4 = 1

        TH = ModelNew.TILE_H
        TW = ModelNew.TILE_W
        BO = ModelNew.BLOCK_OC
        grid_x = (W_out + TW - 1) // TW
        grid_y = (H_out + TH - 1) // TH
        grid_z = B * D_out

        conv3d_kernel[lambda: ((grid_x, grid_y, grid_z), (TW, TH, 1))](
            x_bf16, w_bf16, out_bf16,
            B, C_in, C_out, H, W, D_out, H_out, W_out, K, TH, TW, BO,
            is0, is1, is2, is3, is4,
            ws0, ws1, ws2, ws3,
            os0, os1, os2, os3, os4,
        )

        return out_bf16


batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256
depth = 10


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width, depth)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
