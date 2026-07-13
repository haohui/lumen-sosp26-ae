import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_M = 32
TILE_N = 32
TILE_K = 16

@avelang.jit
def conv2d_mfma_kernel(
    X_ptr: al.Pointer(al.bf16),
    W_ptr: al.Pointer(al.bf16),
    Y_ptr: al.Pointer(al.f32),
    M_total: al.i32,
    N_total: al.i32,
    IC: al.i32,
    OC: al.i32,
):
    X_flat = al.make_tensor(X_ptr, al.bf16, al.make_layout((M_total, IC), (IC, 1)))
    X_i32 = al.view(X_flat, al.i32, al.make_layout((M_total, IC // 8, 4), (IC // 2, 4, 1)))

    W_flat = al.make_tensor(W_ptr, al.bf16, al.make_layout((OC, IC), (IC, 1)))
    W_i32 = al.view(W_flat, al.i32, al.make_layout((OC, IC // 8, 4), (IC // 2, 4, 1)))

    Y_flat = al.make_tensor(Y_ptr, al.f32, al.make_layout((M_total, OC), (OC, 1)))

    lane = al.thread_id(0)
    lane_col = lane & 31
    lane_group = lane >> 5

    block_m = al.block_id(0) * TILE_M
    block_n = al.block_id(1) * TILE_N

    a_smem = al.make_shared((TILE_M * (TILE_K >> 3), TILE_K >> 2), al.i32)
    b_smem = al.make_shared((TILE_N * (TILE_K >> 3), TILE_K >> 2), al.i32)

    zero_f32 = al.convert(0.0, al.f32)
    acc = al.full((16,), zero_f32, al.f32)

    for kt in al.range(IC // TILE_K):
        k_vec = kt * 2 + lane_group
        a_smem[lane] = X_i32[block_m + lane_col, k_vec]
        b_smem[lane] = W_i32[block_n + lane_col, k_vec]
        al.syncthreads()

        a_words = a_smem[lane]
        b_words = b_smem[lane]
        a_frag = al.view(a_words, al.Tensor((2, 2, 1), al.u32))
        b_frag = al.view(b_words, al.Tensor((2, 2, 1), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)
        al.syncthreads()

    c_smem = al.make_shared((TILE_M, TILE_N), al.f32)
    for r in al.range(16):
        row_offset = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        c_smem[lane_col, row_offset] = acc[r]
    al.syncthreads()

    store_row = lane >> 1
    col_start = (lane & 1) * 16
    for v in al.range(16):
        store_col = col_start + v
        dst_row = block_m + store_row
        dst_col = block_n + store_col
        if dst_row < M_total and dst_col < N_total:
            Y_flat[dst_row, dst_col] = c_smem[store_row, store_col]


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        B_val, IC_val, H_val, W_val = x.shape
        OC_val = self.out_channels
        M_total = B_val * H_val * W_val
        N_total = OC_val

        x_nhwc = x.permute(0, 2, 3, 1).contiguous().to(torch.bfloat16)
        w_bf16 = self.conv1d.weight.to(torch.bfloat16).contiguous()
        y_f32 = torch.zeros(B_val, H_val, W_val, OC_val, device=x.device, dtype=torch.float32)

        grid_m = (M_total + TILE_M - 1) // TILE_M
        grid_n = (N_total + TILE_N - 1) // TILE_N

        conv2d_mfma_kernel[lambda: ((grid_m, grid_n, 1), (64, 1, 1))](
            x_nhwc, w_bf16, y_f32,
            M_total, N_total, IC_val, OC_val,
        )

        y_out = y_f32.permute(0, 3, 1, 2).contiguous()
        if x.dtype == torch.float32:
            return y_out
        else:
            return y_out.to(x.dtype)
