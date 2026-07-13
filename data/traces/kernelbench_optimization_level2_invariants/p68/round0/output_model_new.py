import torch
import torch.nn as nn
import avelang
import avelang.language as al

TM = 64
TN = 64
WM = 32
WN = 32


@avelang.jit
def fused_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    c_tensor: al.Tensor((), al.bf16),
    y_ptr: al.Pointer(al.bf16),
    M: al.i32,
    K: al.i32,
    N: al.i32,
):
    x_layout = al.make_layout((M, K), (K, al.convert(1, al.i32)))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    w_layout = al.make_layout((K, N), (N, al.convert(1, al.i32)))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    y_layout = al.make_layout((M, N), (N, al.convert(1, al.i32)))
    y = al.make_tensor(y_ptr, al.bf16, y_layout)

    bias_layout = al.make_layout((N,), (al.convert(1, al.i32),))
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    c_f32 = al.convert(c_tensor[()], al.f32)

    # Buffer resources for vectorized global loads
    x_rsrc = al.amdgpu.make_rsrc(x, M * K * 2)
    w_rsrc = al.amdgpu.make_rsrc(w, K * N * 2)

    block_m = al.block_id(0)
    block_n = al.block_id(1)
    tid = al.thread_id(0)

    warp_id = tid // 64
    lane_id = tid % 64
    warp_m = warp_id // 2
    warp_n = warp_id % 2

    tile_row_base = block_m * 64
    tile_col_base = block_n * 64
    warp_row_base = tile_row_base + warp_m * 32
    warp_col_base = tile_col_base + warp_n * 32

    acc = al.make_local((16,), al.f32)
    acc[0] = al.convert(0.0, al.f32)
    acc[1] = al.convert(0.0, al.f32)
    acc[2] = al.convert(0.0, al.f32)
    acc[3] = al.convert(0.0, al.f32)
    acc[4] = al.convert(0.0, al.f32)
    acc[5] = al.convert(0.0, al.f32)
    acc[6] = al.convert(0.0, al.f32)
    acc[7] = al.convert(0.0, al.f32)
    acc[8] = al.convert(0.0, al.f32)
    acc[9] = al.convert(0.0, al.f32)
    acc[10] = al.convert(0.0, al.f32)
    acc[11] = al.convert(0.0, al.f32)
    acc[12] = al.convert(0.0, al.f32)
    acc[13] = al.convert(0.0, al.f32)
    acc[14] = al.convert(0.0, al.f32)
    acc[15] = al.convert(0.0, al.f32)

    As = al.make_shared((64, 16), al.bf16)
    Bs = al.make_shared((16, 64), al.bf16)

    zero_off = al.convert(0, al.i32)

    for k_block in al.range(0, K, 16):
        # Cooperative global -> shared: A tile (64 x 16) via raw_buffer_load_x4
        if tid < 128:
            a_row = tid // 2
            a_col_start = (tid % 2) * 8
            g_row = tile_row_base + a_row
            g_col = k_block + a_col_start
            byte_off = (g_row * K + g_col) * 2
            packed = al.amdgpu.raw_buffer_load_x4(x_rsrc, byte_off, zero_off, al.convert(0, al.i32))
            frag = al.view(packed, al.Tensor((8,), al.bf16))
            for c in al.range(8):
                As[a_row, a_col_start + c] = frag[c]

        # Cooperative global -> shared: B tile (16 x 64) via raw_buffer_load_x4
        if tid >= 128:
            t = tid - 128
            b_row = t // 8
            b_col_start = (t % 8) * 8
            g_row = k_block + b_row
            g_col = tile_col_base + b_col_start
            byte_off = (g_row * N + g_col) * 2
            packed = al.amdgpu.raw_buffer_load_x4(w_rsrc, byte_off, zero_off, al.convert(0, al.i32))
            frag = al.view(packed, al.Tensor((8,), al.bf16))
            for c in al.range(8):
                Bs[b_row, b_col_start + c] = frag[c]

        al.syncthreads()

        a_row = warp_m * 32 + (lane_id % 32)
        a0_bf16 = al.make_local((4,), al.bf16)
        a1_bf16 = al.make_local((4,), al.bf16)
        if lane_id < 32:
            a0_bf16[0] = As[a_row, 0]
            a0_bf16[1] = As[a_row, 1]
            a0_bf16[2] = As[a_row, 2]
            a0_bf16[3] = As[a_row, 3]
            a1_bf16[0] = As[a_row, 8]
            a1_bf16[1] = As[a_row, 9]
            a1_bf16[2] = As[a_row, 10]
            a1_bf16[3] = As[a_row, 11]
        else:
            a0_bf16[0] = As[a_row, 4]
            a0_bf16[1] = As[a_row, 5]
            a0_bf16[2] = As[a_row, 6]
            a0_bf16[3] = As[a_row, 7]
            a1_bf16[0] = As[a_row, 12]
            a1_bf16[1] = As[a_row, 13]
            a1_bf16[2] = As[a_row, 14]
            a1_bf16[3] = As[a_row, 15]

        a0_packed = al.view(a0_bf16, al.Tensor((2,), al.u32))
        a1_packed = al.view(a1_bf16, al.Tensor((2,), al.u32))

        b_col = warp_n * 32 + (lane_id % 32)
        b0_bf16 = al.make_local((4,), al.bf16)
        b1_bf16 = al.make_local((4,), al.bf16)
        if lane_id < 32:
            b0_bf16[0] = Bs[0, b_col]
            b0_bf16[1] = Bs[1, b_col]
            b0_bf16[2] = Bs[2, b_col]
            b0_bf16[3] = Bs[3, b_col]
            b1_bf16[0] = Bs[8, b_col]
            b1_bf16[1] = Bs[9, b_col]
            b1_bf16[2] = Bs[10, b_col]
            b1_bf16[3] = Bs[11, b_col]
        else:
            b0_bf16[0] = Bs[4, b_col]
            b0_bf16[1] = Bs[5, b_col]
            b0_bf16[2] = Bs[6, b_col]
            b0_bf16[3] = Bs[7, b_col]
            b1_bf16[0] = Bs[12, b_col]
            b1_bf16[1] = Bs[13, b_col]
            b1_bf16[2] = Bs[14, b_col]
            b1_bf16[3] = Bs[15, b_col]

        b0_packed = al.view(b0_bf16, al.Tensor((2,), al.u32))
        b1_packed = al.view(b1_bf16, al.Tensor((2,), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a0_packed, b0_packed, acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a1_packed, b1_packed, acc)

        al.syncthreads()

    out_col = warp_col_base + (lane_id % 32)
    if out_col < N:
        b = al.convert(bias[out_col], al.f32)

        out_row0 = warp_row_base + 0 + 4 * (lane_id // 32) + 0
        out_row1 = warp_row_base + 0 + 4 * (lane_id // 32) + 1
        out_row2 = warp_row_base + 0 + 4 * (lane_id // 32) + 2
        out_row3 = warp_row_base + 0 + 4 * (lane_id // 32) + 3
        out_row4 = warp_row_base + 8 + 4 * (lane_id // 32) + 0
        out_row5 = warp_row_base + 8 + 4 * (lane_id // 32) + 1
        out_row6 = warp_row_base + 8 + 4 * (lane_id // 32) + 2
        out_row7 = warp_row_base + 8 + 4 * (lane_id // 32) + 3
        out_row8 = warp_row_base + 16 + 4 * (lane_id // 32) + 0
        out_row9 = warp_row_base + 16 + 4 * (lane_id // 32) + 1
        out_row10 = warp_row_base + 16 + 4 * (lane_id // 32) + 2
        out_row11 = warp_row_base + 16 + 4 * (lane_id // 32) + 3
        out_row12 = warp_row_base + 24 + 4 * (lane_id // 32) + 0
        out_row13 = warp_row_base + 24 + 4 * (lane_id // 32) + 1
        out_row14 = warp_row_base + 24 + 4 * (lane_id // 32) + 2
        out_row15 = warp_row_base + 24 + 4 * (lane_id // 32) + 3

        v0 = acc[0] + b
        v1 = acc[1] + b
        v2 = acc[2] + b
        v3 = acc[3] + b
        v4 = acc[4] + b
        v5 = acc[5] + b
        v6 = acc[6] + b
        v7 = acc[7] + b
        v8 = acc[8] + b
        v9 = acc[9] + b
        v10 = acc[10] + b
        v11 = acc[11] + b
        v12 = acc[12] + b
        v13 = acc[13] + b
        v14 = acc[14] + b
        v15 = acc[15] + b

        if v0 > c_f32:
            v0 = c_f32
        if v1 > c_f32:
            v1 = c_f32
        if v2 > c_f32:
            v2 = c_f32
        if v3 > c_f32:
            v3 = c_f32
        if v4 > c_f32:
            v4 = c_f32
        if v5 > c_f32:
            v5 = c_f32
        if v6 > c_f32:
            v6 = c_f32
        if v7 > c_f32:
            v7 = c_f32
        if v8 > c_f32:
            v8 = c_f32
        if v9 > c_f32:
            v9 = c_f32
        if v10 > c_f32:
            v10 = c_f32
        if v11 > c_f32:
            v11 = c_f32
        if v12 > c_f32:
            v12 = c_f32
        if v13 > c_f32:
            v13 = c_f32
        if v14 > c_f32:
            v14 = c_f32
        if v15 > c_f32:
            v15 = c_f32

        v0 = v0 - c_f32
        v1 = v1 - c_f32
        v2 = v2 - c_f32
        v3 = v3 - c_f32
        v4 = v4 - c_f32
        v5 = v5 - c_f32
        v6 = v6 - c_f32
        v7 = v7 - c_f32
        v8 = v8 - c_f32
        v9 = v9 - c_f32
        v10 = v10 - c_f32
        v11 = v11 - c_f32
        v12 = v12 - c_f32
        v13 = v13 - c_f32
        v14 = v14 - c_f32
        v15 = v15 - c_f32

        if out_row0 < M:
            y[out_row0, out_col] = al.convert(v0, al.bf16)
        if out_row1 < M:
            y[out_row1, out_col] = al.convert(v1, al.bf16)
        if out_row2 < M:
            y[out_row2, out_col] = al.convert(v2, al.bf16)
        if out_row3 < M:
            y[out_row3, out_col] = al.convert(v3, al.bf16)
        if out_row4 < M:
            y[out_row4, out_col] = al.convert(v4, al.bf16)
        if out_row5 < M:
            y[out_row5, out_col] = al.convert(v5, al.bf16)
        if out_row6 < M:
            y[out_row6, out_col] = al.convert(v6, al.bf16)
        if out_row7 < M:
            y[out_row7, out_col] = al.convert(v7, al.bf16)
        if out_row8 < M:
            y[out_row8, out_col] = al.convert(v8, al.bf16)
        if out_row9 < M:
            y[out_row9, out_col] = al.convert(v9, al.bf16)
        if out_row10 < M:
            y[out_row10, out_col] = al.convert(v10, al.bf16)
        if out_row11 < M:
            y[out_row11, out_col] = al.convert(v11, al.bf16)
        if out_row12 < M:
            y[out_row12, out_col] = al.convert(v12, al.bf16)
        if out_row13 < M:
            y[out_row13, out_col] = al.convert(v13, al.bf16)
        if out_row14 < M:
            y[out_row14, out_col] = al.convert(v14, al.bf16)
        if out_row15 < M:
            y[out_row15, out_col] = al.convert(v15, al.bf16)


def _launch_grid(M: int, N: int):
    grid_m = (M + TM - 1) // TM
    grid_n = (N + TN - 1) // TN
    return ((grid_m, grid_n, 1), (256, 1, 1))


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))

    def forward(self, x):
        w_t = self.linear.weight.t().contiguous()
        bias = self.linear.bias

        device = x.device
        x = x.to(dtype=torch.bfloat16).contiguous()
        w_dev = w_t.to(device=device, dtype=torch.bfloat16).contiguous()
        bias_dev = bias.to(device=device, dtype=torch.bfloat16).contiguous()
        c_dev = self.constant.to(device=device, dtype=torch.bfloat16).contiguous()

        M_val = x.shape[0]
        K_val = x.shape[1]
        N_val = w_dev.shape[1]

        y = torch.empty((M_val, N_val), device=device, dtype=torch.bfloat16)

        grid_block = _launch_grid(M_val, N_val)
        fused_kernel[lambda: grid_block](
            x, w_dev, bias_dev, c_dev, y,
            M_val, K_val, N_val,
        )

        return y
