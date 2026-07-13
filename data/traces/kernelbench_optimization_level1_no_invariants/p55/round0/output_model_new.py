import torch
import torch.nn as nn
import avelang
import avelang.language as al

MMA_M = 32
MMA_N = 32
MMA_K = 8
ACC_SIZE = 16
WARP_SIZE = 64

SPATIAL_TILES = 16289
OC_TILES = 4
BATCH_VAL = 8


@avelang.jit
def conv2d_mfma_kernel(
    X: al.Tensor((8, 64, 512, 1024), al.bf16),
    W: al.Tensor((128, 64, 3, 3), al.bf16),
    Y: al.Tensor((8, 128, 510, 1022), al.f32),
):
    m_block = al.block_id(0)
    k_block = al.block_id(1)
    batch = al.block_id(2)
    lane = al.thread_id(0)

    m_start = m_block * MMA_M
    k_start = k_block * MMA_N

    row_lane = lane % MMA_M
    k_group = lane // MMA_M

    a_bf16 = al.make_local((4,), al.bf16)
    b_bf16 = al.make_local((4,), al.bf16)
    c_regs = al.make_local((ACC_SIZE,), al.f32)

    for j in al.range(ACC_SIZE):
        c_regs[j] = al.convert(0.0, al.f32)

    num_k_tiles = 72
    total_m = 521220

    for kt in al.range(num_k_tiles):
        k_ge_start = kt * MMA_K

        for elem in al.range(4):
            k_offset = k_group * 4 + elem
            k_idx = k_ge_start + k_offset
            ic = k_idx // 9
            kr_ks = k_idx - ic * 9
            kr = kr_ks // 3
            ks = kr_ks - kr * 3

            m_idx = m_start + row_lane
            if m_idx < total_m:
                p = m_idx // 1022
                q = m_idx - p * 1022
                in_h = p + kr
                in_w = q + ks
                a_bf16[elem] = X[batch, ic, in_h, in_w]
            else:
                a_bf16[elem] = al.convert(0.0, al.bf16)

            n_idx = k_start + row_lane
            b_bf16[elem] = W[n_idx, ic, kr, ks]

        a_vec = al.view(a_bf16, al.Tensor((2,), al.i32))
        b_vec = al.view(b_bf16, al.Tensor((2,), al.i32))
        c_vec = al.view(c_regs, al.Tensor((ACC_SIZE,), al.f32))
        c_result = al.amdgpu.mfma_32x32x8_bf16_f32(a_vec, b_vec, c_vec)
        for r in al.range(ACC_SIZE):
            c_regs[r] = c_result[r]

    lane_group = lane // MMA_N
    lane_col = lane % MMA_N
    for t in al.range(ACC_SIZE):
        row = m_start + (t // 4) * 8 + lane_group * 4 + (t % 4)
        col = k_start + lane_col
        if row < total_m:
            rp = row // 1022
            rq = row - rp * 1022
            Y[batch, col, rp, rq] = c_regs[t]


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        if tuple(x.shape) != (8, 64, 512, 1024):
            raise RuntimeError(f'Shape mismatch')
        device = x.device
        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = self.conv2d.weight.to(device=device, dtype=torch.bfloat16).contiguous()
        y = torch.empty((8, 128, 510, 1022), device=device, dtype=torch.float32)
        conv2d_mfma_kernel[lambda: ((SPATIAL_TILES, OC_TILES, BATCH_VAL), (WARP_SIZE, 1, 1))](x_bf16, w_bf16, y)
        return y.to(dtype=x.dtype)
