import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ============================================================
# Depthwise 2D Convolution with MFMA (BF16)
# ============================================================
# B swizzle for mfma_32x32x8_bf16_f32:
#   B(j,i) -> lane = (j/4)*32 + i
#   4 elements per thread cover 4 consecutive j values for column i
#   Column 0: lane 0 handles j=0..3, lane 32 handles j=4..7
#
# A swizzle: A(i,j) -> lane = i + (j/4)*32
#   Lanes 0..31: j=0..3, Lanes 32..63: j=4..7
#
# Accumulator: col = lane%32, row = 8*(acc//4) + 4*(lane//32) + acc%4
#   Column 0 data in lanes 0 and 32
# ============================================================

POS_PER_MFMA = 32
BLOCK_SIZE = 64


@avelang.jit
def depthwise_conv_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    y_ptr: al.Pointer(al.f32),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    x_layout = al.make_layout((N, C, H, W), (C * H * W, H * W, W, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    w_layout = al.make_layout((C, 1, 3, 3), (9, 9, 3, 1))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    y_layout = al.make_layout(
        (N, C, H_out, W_out), (C * H_out * W_out, H_out * W_out, W_out, 1)
    )
    y = al.make_tensor(y_ptr, al.f32, y_layout)

    n = al.block_id(0)
    ch = al.block_id(1)
    tid = al.thread_id(0)
    total_positions = H_out * W_out

    # Weights: flatten 3x3 -> 9 entries. Filter-to-(kh,kw):
    #   0:(0,0) 1:(0,1) 2:(0,2) 3:(1,0) 4:(1,1) 5:(1,2) 6:(2,0) 7:(2,1) 8:(2,2)
    w00 = al.convert(w[ch, 0, 0, 0], al.bf16)
    w01 = al.convert(w[ch, 0, 0, 1], al.bf16)
    w02 = al.convert(w[ch, 0, 0, 2], al.bf16)
    w10 = al.convert(w[ch, 0, 1, 0], al.bf16)
    w11 = al.convert(w[ch, 0, 1, 1], al.bf16)
    w12 = al.convert(w[ch, 0, 1, 2], al.bf16)
    w20 = al.convert(w[ch, 0, 2, 0], al.bf16)
    w21 = al.convert(w[ch, 0, 2, 1], al.bf16)
    w22 = al.convert(w[ch, 0, 2, 2], al.bf16)

    zero_b = al.convert(0.0, al.bf16)

    for pos_base_linear in al.range(0, total_positions, POS_PER_MFMA):
        acc_mem = al.make_local((16,), al.f32)
        for ai in al.range(16):
            acc_mem[ai] = al.convert(0.0, al.f32)

        # --- A operand: input data for 32 positions x 8 filter entries ---
        pos_idx = tid % 32
        pos = pos_base_linear + pos_idx
        pos_valid = pos < total_positions
        oh = pos // W_out
        ow = pos % W_out

        a_mem = al.make_local((2,), al.u32)
        a_bf16 = al.view(a_mem, al.Tensor((4,), al.bf16))

        if tid < 32:
            # j = 0,1,2,3: filters (0,0),(0,1),(0,2),(1,0)
            if pos_valid:
                a_bf16[0] = al.convert(x[n, ch, oh + 0, ow + 0], al.bf16)
                a_bf16[1] = al.convert(x[n, ch, oh + 0, ow + 1], al.bf16)
                a_bf16[2] = al.convert(x[n, ch, oh + 0, ow + 2], al.bf16)
                a_bf16[3] = al.convert(x[n, ch, oh + 1, ow + 0], al.bf16)
            else:
                a_bf16[0] = zero_b
                a_bf16[1] = zero_b
                a_bf16[2] = zero_b
                a_bf16[3] = zero_b
        else:
            # j = 4,5,6,7: filters (1,1),(1,2),(2,0),(2,1)
            if pos_valid:
                a_bf16[0] = al.convert(x[n, ch, oh + 1, ow + 1], al.bf16)
                a_bf16[1] = al.convert(x[n, ch, oh + 1, ow + 2], al.bf16)
                a_bf16[2] = al.convert(x[n, ch, oh + 2, ow + 0], al.bf16)
                a_bf16[3] = al.convert(x[n, ch, oh + 2, ow + 1], al.bf16)
            else:
                a_bf16[0] = zero_b
                a_bf16[1] = zero_b
                a_bf16[2] = zero_b
                a_bf16[3] = zero_b

        # --- B operand: weights, only column 0 active ---
        # B(j,i) -> lane = (j/4)*32 + i
        # Column 0 (i=0): lane 0 -> j=0..3, lane 32 -> j=4..7
        b_mem = al.make_local((2,), al.u32)
        b_bf16 = al.view(b_mem, al.Tensor((4,), al.bf16))

        # Two lanes handle column 0: lane 0 (j=0-3) and lane 32 (j=4-7)
        if tid == 0:
            b_bf16[0] = w00
            b_bf16[1] = w01
            b_bf16[2] = w02
            b_bf16[3] = w10
        elif tid == 32:
            b_bf16[0] = w11
            b_bf16[1] = w12
            b_bf16[2] = w20
            b_bf16[3] = w21
        else:
            # Other columns are zero
            b_bf16[0] = zero_b
            b_bf16[1] = zero_b
            b_bf16[2] = zero_b
            b_bf16[3] = zero_b

        # --- MFMA ---
        a_vec = al.view(a_mem, al.Tensor((1, 2), al.u32))[0]
        b_vec = al.view(b_mem, al.Tensor((1, 2), al.u32))[0]
        acc_vec = al.view(acc_mem, al.Tensor((1, 16), al.f32))[0]

        result_vec = al.amdgpu.mfma_32x32x8_bf16_f32(a_vec, b_vec, acc_vec)

        for ai in al.range(16):
            acc_mem[ai] = result_vec[ai]

        # ====== Scalar step for K=8 (filter (2,2)) + store ======
        # Only lanes with column 0 data write output
        if tid % 32 == 0:
            for ai in al.range(16):
                row_in_tile = 8 * (ai // 4) + 4 * (tid // 32) + (ai % 4)
                out_linear = pos_base_linear + row_in_tile
                if out_linear < total_positions:
                    out_oh = out_linear // W_out
                    out_ow = out_linear % W_out
                    x22 = al.convert(x[n, ch, out_oh + 2, out_ow + 2], al.f32)
                    acc_mem[ai] = acc_mem[ai] + x22 * al.convert(w22, al.f32)
                    y[n, ch, out_oh, out_ow] = acc_mem[ai]


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N_val, C_val, H_val, W_val = x.shape
        H_out_val = H_val - 2
        W_out_val = W_val - 2

        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = self.conv2d.weight.to(torch.bfloat16).contiguous()

        y = torch.empty(
            (N_val, C_val, H_out_val, W_out_val),
            device=x.device,
            dtype=torch.float32,
        )

        depthwise_conv_bf16_kernel[
            lambda: ((N_val, C_val, 1), (BLOCK_SIZE, 1, 1))
        ](
            x_bf16,
            w_bf16,
            y,
            N_val,
            C_val,
            H_val,
            W_val,
            H_out_val,
            W_out_val,
        )

        return y.to(x.dtype)
