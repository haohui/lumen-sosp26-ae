import torch
import torch.nn as nn
import avelang
import avelang.language as al

@avelang.jit
def depthwise_conv2d_mfma_kernel(
    X_ptr: al.Pointer(al.bf16),
    W_ptr: al.Pointer(al.bf16),
    Y_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    KH: al.i32,
    KW: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    stride: al.i32,
    padding: al.i32,
    total_outputs: al.i32,
    K_total: al.constexpr,
):
    """Implicit-GEMM MFMA kernel for depthwise Conv2D.

    Each block has 4 active threads (lanes 0-3) with unique (spatial,
    channel) pairs.  Lanes 4-31 are zeroed.  The MFMA diagonal
    c_frag[lane] gives the correct result for each active thread.
    """
    C_H_W = C * H * W
    H_W = H * W
    C_Hout_Wout = C * H_out * W_out
    Hout_Wout = H_out * W_out

    x_layout = al.make_layout((N, C, H, W), (C_H_W, H_W, W, 1))
    X = al.make_tensor(X_ptr, al.bf16, x_layout)

    w_layout = al.make_layout((C, 1, KH, KW), (KH * KW, KH * KW, KW, 1))
    WT = al.make_tensor(W_ptr, al.bf16, w_layout)

    y_layout = al.make_layout((N, C, H_out, W_out), (C_Hout_Wout, Hout_Wout, W_out, 1))
    Y = al.make_tensor(Y_ptr, al.bf16, y_layout)

    block_idx = al.block_id(0)
    lane = al.thread_id(0)

    active = lane < al.convert(4, al.i32)

    lin_idx = block_idx * 4 + lane
    lin_valid = lin_idx < total_outputs

    # Decode linear index -> (n_batch, c, oh, ow)
    c = lin_idx % C
    spatial_idx = lin_idx // C
    n_batch = spatial_idx // Hout_Wout
    spatial_rest = spatial_idx - n_batch * Hout_Wout
    oh = spatial_rest // W_out
    ow = spatial_rest - oh * W_out

    c_frag = al.make_local((16,), al.f32)
    for i in al.range(16):
        c_frag[i] = al.convert(0.0, al.f32)

    a_frag = al.make_local((2,), al.u32)
    b_frag = al.make_local((2,), al.u32)

    for k_call in al.range(3):
        k_base = k_call * 4

        for pair in al.range(2):
            k0 = k_base + pair * 2
            k1 = k0 + 1

            a0_bf16 = al.convert(0.0, al.bf16)
            a1_bf16 = al.convert(0.0, al.bf16)
            b0_bf16 = al.convert(0.0, al.bf16)
            b1_bf16 = al.convert(0.0, al.bf16)

            if active and lin_valid:
                if k0 < K_total:
                    kh0 = k0 // KW
                    kw0 = k0 - kh0 * KW
                    ih0 = oh * stride - padding + kh0
                    iw0 = ow * stride - padding + kw0
                    if (ih0 >= 0) and (ih0 < H) and (iw0 >= 0) and (iw0 < W):
                        a0_bf16 = X[n_batch, c, ih0, iw0]
                    b0_bf16 = WT[c, 0, kh0, kw0]

                if k1 < K_total:
                    kh1 = k1 // KW
                    kw1 = k1 - kh1 * KW
                    ih1 = oh * stride - padding + kh1
                    iw1 = ow * stride - padding + kw1
                    if (ih1 >= 0) and (ih1 < H) and (iw1 >= 0) and (iw1 < W):
                        a1_bf16 = X[n_batch, c, ih1, iw1]
                    b1_bf16 = WT[c, 0, kh1, kw1]

            a_lo = al.bitcast(a0_bf16, al.u16)
            a_hi = al.bitcast(a1_bf16, al.u16)
            b_lo = al.bitcast(b0_bf16, al.u16)
            b_hi = al.bitcast(b1_bf16, al.u16)

            a_packed = al.convert(a_lo, al.u32) | (al.convert(a_hi, al.u32) << al.convert(16, al.u32))
            b_packed = al.convert(b_lo, al.u32) | (al.convert(b_hi, al.u32) << al.convert(16, al.u32))

            a_frag[pair] = a_packed
            b_frag[pair] = b_packed

        c_frag = al.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, c_frag)

    if active and lin_valid:
        result = al.convert(c_frag[lane], al.bf16)
        Y[n_batch, c, oh, ow] = result


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x):
        N, C, H, W_ = x.shape
        KH = self.conv2d.kernel_size[0]
        KW = self.conv2d.kernel_size[1]
        stride = self.conv2d.stride[0]
        padding = self.conv2d.padding[0]
        dilation = self.conv2d.dilation[0]

        H_out = (H + 2 * padding - dilation * (KH - 1) - 1) // stride + 1
        W_out = (W_ + 2 * padding - dilation * (KW - 1) - 1) // stride + 1

        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = self.conv2d.weight.to(device=x.device, dtype=torch.bfloat16).contiguous()

        y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=torch.bfloat16)

        total_outputs = N * C * H_out * W_out
        total_blocks = (total_outputs + 3) // 4

        grid = (total_blocks, 1, 1)
        block = (32, 1, 1)

        depthwise_conv2d_mfma_kernel[lambda: (grid, block)](
            x_bf16,
            w_bf16,
            y,
            N,
            C,
            H,
            W_,
            KH,
            KW,
            H_out,
            W_out,
            stride,
            padding,
            total_outputs,
            9,
        )

        if x.dtype != torch.bfloat16:
            return y.to(x.dtype)
        return y
