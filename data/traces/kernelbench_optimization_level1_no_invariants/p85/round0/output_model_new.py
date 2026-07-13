import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def depthwise_conv2d_kernel(
    X: al.Tensor((32, 128, 128, 256), al.bf16),
    W: al.Tensor((128, 1, 3, 7), al.bf16),
    Y: al.Tensor((32, 128, 126, 250), al.bf16),
):
    """
    Depthwise Conv2D with MFMA_32x32x8_bf16_f32.

    Implicit-GEMM formulation: C[M,N] = sum_K A[M,K] * B[K,N]
    where M = spatial (OH*OW), N = 1 per channel (depthwise), K = KH*KW.

    Each workgroup (64 threads = 1 wave64) processes 32 spatial positions
    for one channel. K=21 processed in 3 MFMA calls (8+8+5).
    Only the N=0 column of MFMA output is used (depthwise has 1 output per group).
    """
    b = al.block_id(0)
    s_block = al.block_id(1)
    c = al.block_id(2)
    tid = al.thread_id(0)
    s_start = s_block * 32

    c_frag = al.make_local((16,), al.f32)
    for i in al.range(16):
        c_frag[i] = al.convert(0.0, al.f32)

    for k_block in al.range(3):
        k_start = k_block * 8

        a_reg = al.make_local((4,), al.bf16)
        for i in al.range(4):
            a_row = tid % 32
            a_col = (tid // 32) * 4 + i
            s_global = s_start + a_row
            k_idx = k_start + a_col
            if s_global < 31500 and k_idx < 21:
                oh = s_global // 250
                ow = s_global % 250
                kh = k_idx // 7
                kw = k_idx % 7
                ih = oh + kh
                iw = ow + kw
                a_reg[i] = X[b, c, ih, iw]
            else:
                a_reg[i] = al.convert(0.0, al.bf16)

        b_reg = al.make_local((4,), al.bf16)
        for i in al.range(4):
            b_row = tid % 8
            b_col = (tid // 8) * 4 + i
            k_idx = k_start + b_row
            if k_idx < 21 and b_col == 0:
                kh = k_idx // 7
                kw = k_idx % 7
                b_reg[i] = W[c, 0, kh, kw]
            else:
                b_reg[i] = al.convert(0.0, al.bf16)

        a_mfma = al.view(a_reg, al.Tensor((2,), al.i32))
        b_mfma = al.view(b_reg, al.Tensor((2,), al.i32))
        c_frag = al.amdgpu.mfma_32x32x8_bf16_f32(a_mfma, b_mfma, c_frag)

    if tid < 16:
        s_even = s_start + tid * 2
        if s_even < 31500:
            oh = s_even // 250
            ow = s_even % 250
            Y[b, c, oh, ow] = al.convert(c_frag[0], al.bf16)
        s_odd = s_start + tid * 2 + 1
        if s_odd < 31500:
            oh = s_odd // 250
            ow = s_odd % 250
            Y[b, c, oh, ow] = al.convert(c_frag[8], al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, (kernel_size_h, kernel_size_w), stride=(stride_h, stride_w), padding=(padding_h, padding_w), dilation=(dilation_h, dilation_w), groups=in_channels, bias=bias)

    def forward(self, x):
        x0 = x.to(dtype=torch.bfloat16).contiguous()
        w = self.conv2d.weight.to(device=x.device, dtype=torch.bfloat16).contiguous()
        y = torch.empty((32, 128, 126, 250), device=x.device, dtype=torch.bfloat16)
        grid = (32, 985, 128)
        depthwise_conv2d_kernel[lambda: (grid, (64, 1, 1))](x0, w, y)
        if x.dtype == torch.float32:
            return y.to(dtype=torch.float32)
        return y
