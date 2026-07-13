import torch
import torch.nn as nn
import avelang
import avelang.language as al

BATCH_SIZE = 128
INPUT_SIZE = 32768
HIDDEN_SIZE = 32768


def _launch():
    return ((1, 1, 1), (256, 1, 1))


@avelang.jit
def fused_kernel(
    X_ptr: al.Pointer(al.bf16),
    W_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    Y_ptr: al.Pointer(al.bf16),
):
    M_VAL = al.convert(128, al.i32)
    K_VAL = al.convert(32768, al.i32)
    N_VAL = al.convert(32768, al.i32)

    X = al.make_tensor(
        X_ptr, al.bf16,
        al.make_layout((M_VAL, K_VAL), (K_VAL, al.convert(1, al.i32))),
    )
    W = al.make_tensor(
        W_ptr, al.bf16,
        al.make_layout((K_VAL, N_VAL), (N_VAL, al.convert(1, al.i32))),
    )
    bias = al.make_tensor(
        bias_ptr, al.bf16,
        al.make_layout((N_VAL,), (al.convert(1, al.i32),)),
    )
    Y = al.make_tensor(
        Y_ptr, al.bf16,
        al.make_layout(
            (M_VAL, al.convert(1, al.i32)),
            (al.convert(1, al.i32), al.convert(1, al.i32)),
        ),
    )

    tid = al.thread_id(0)
    warp_id = tid // al.convert(64, al.i32)
    lane_id = tid % al.convert(64, al.i32)

    warp_row = warp_id // al.convert(2, al.i32)
    m_start = warp_row * al.convert(32, al.i32)

    A_shared = al.make_shared((32, 16), al.bf16)
    B_shared = al.make_shared((16, 32), al.bf16)

    X_rsrc = al.amdgpu.make_rsrc(
        X, M_VAL * K_VAL * al.convert(2, al.i32),
    )
    W_rsrc = al.amdgpu.make_rsrc(
        W, K_VAL * N_VAL * al.convert(2, al.i32),
    )

    partial = al.make_local((32,), al.f32)
    for r in al.range(32):
        partial[r] = al.convert(0.0, al.f32)

    TWO = al.convert(2, al.i32)
    FOUR = al.convert(4, al.i32)
    EIGHT = al.convert(8, al.i32)
    SIXTEEN = al.convert(16, al.i32)
    THIRTY_TWO = al.convert(32, al.i32)
    ONE_F = al.convert(1.0, al.f32)
    ZERO_F = al.convert(0.0, al.f32)

    BLK_N = al.convert(32, al.i32)
    BLK_K = al.convert(16, al.i32)

    # LDS vector views for MFMA operand extraction.
    # A: 32x16 bf16 -> u32 layout (32, 4, 2) strides (8, 2, 1)
    A_vec = al.view(
        A_shared, al.u32,
        al.make_layout((32, 4, 2), (8, 2, 1)),
    )
    # B: 16x32 bf16 -> u32 layout (16, 8, 2) strides (16, 2, 1)
    B_vec = al.view(
        B_shared, al.u32,
        al.make_layout((16, 8, 2), (16, 2, 1)),
    )

    for n_tile in al.range(0, N_VAL, BLK_N):
        c_acc = al.make_local((16,), al.f32)
        for i in al.range(16):
            c_acc[i] = ZERO_F

        for k_tile in al.range(0, K_VAL, BLK_K):
            # Vectorised global -> LDS: A tile (32 x 16)
            a_row = lane_id // TWO
            a_col_off = (lane_id % TWO) * EIGHT
            a_byte_off = (
                (m_start + a_row) * K_VAL + k_tile + a_col_off
            ) * TWO
            a_v = al.amdgpu.raw_buffer_load_x4(
                X_rsrc, a_byte_off, 0, 0,
            )
            a_bf16 = al.view(a_v, al.Tensor((8,), al.bf16))
            for i in al.range(8):
                A_shared[a_row, a_col_off + i] = a_bf16[i]

            # Vectorised global -> LDS: B tile (16 x 32)
            b_row = lane_id // FOUR
            b_col_off = (lane_id % FOUR) * EIGHT
            b_byte_off = (
                (k_tile + b_row) * N_VAL + n_tile + b_col_off
            ) * TWO
            b_v = al.amdgpu.raw_buffer_load_x4(
                W_rsrc, b_byte_off, 0, 0,
            )
            b_bf16 = al.view(b_v, al.Tensor((8,), al.bf16))
            for i in al.range(8):
                B_shared[b_row, b_col_off + i] = b_bf16[i]

            al.syncthreads()

            # MFMA 32x32x8: two K=8 slices per 16-wide K tile.
            # A operand: lane%32 row, lane//32 col group.
            # B operand: lane//8 row, lane%8 col group.
            c_acc = al.amdgpu.mfma_32x32x8_bf16_f32(
                A_vec[lane_id % THIRTY_TWO, lane_id // THIRTY_TWO],
                B_vec[lane_id // EIGHT, lane_id % EIGHT],
                c_acc,
            )
            c_acc = al.amdgpu.mfma_32x32x8_bf16_f32(
                A_vec[lane_id % THIRTY_TWO,
                      (lane_id // THIRTY_TWO) + TWO],
                B_vec[(lane_id // EIGHT) + EIGHT, lane_id % EIGHT],
                c_acc,
            )

            al.syncthreads()

        # Post-process: bias + sigmoid + row-wise sum.
        # C output: 8x8 grid of 4x4 blocks per thread.
        tr = lane_id // EIGHT
        tc = lane_id % EIGHT
        bias_local = al.make_local((32,), al.f32)
        for i in al.range(32):
            bias_local[i] = al.convert(bias[n_tile + i], al.f32)

        for idx in al.range(16):
            er = idx // FOUR
            ec = idx % FOUR
            row = tr * FOUR + er
            col = tc * FOUR + ec

            val = c_acc[idx] + bias_local[col]
            sig_val = ONE_F / (ONE_F + al.exp(ZERO_F - val))
            partial[row] = partial[row] + sig_val

    for r in al.range(32):
        Y[m_start + r, al.convert(0, al.i32)] = al.convert(
            partial[r], al.bf16,
        )


class ModelNew(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        if (
            tuple(x.shape) != (BATCH_SIZE, INPUT_SIZE)
            or x.dtype != torch.bfloat16
        ):
            raise RuntimeError(
                "Fused kernel: shape/dtype mismatch",
            )
        w_t = self.linear.weight.t().to(
            device=x.device, dtype=x.dtype,
        ).contiguous()
        bias = self.linear.bias.to(
            device=x.device, dtype=x.dtype,
        ).contiguous()
        y = torch.empty(
            (BATCH_SIZE, 1), device=x.device, dtype=x.dtype,
        )
        fused_kernel[_launch](x.contiguous(), w_t, bias, y)
        return y
