import torch
import torch.nn as nn
import avelang
import avelang.language as al

BATCH_SIZE = 128
INPUT_SIZE = 32768
HIDDEN_SIZE = 32768


def _launch():
    return ((4, 1, 1), (64, 1, 1))


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

    TWO = al.convert(2, al.i32)
    FOUR = al.convert(4, al.i32)
    EIGHT = al.convert(8, al.i32)
    SIXTEEN = al.convert(16, al.i32)
    THIRTY_TWO = al.convert(32, al.i32)
    SIXTY_FOUR = al.convert(64, al.i32)
    ONE = al.convert(1, al.i32)
    ONE_F = al.convert(1.0, al.f32)
    ZERO_F = al.convert(0.0, al.f32)

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

    X_rsrc = al.amdgpu.make_rsrc(
        X, M_VAL * K_VAL * al.convert(2, al.i32),
    )
    W_rsrc = al.amdgpu.make_rsrc(
        W, K_VAL * N_VAL * al.convert(2, al.i32),
    )

    bid = al.block_id(0)
    lane_id = al.thread_id(0)
    m_start = bid * THIRTY_TWO

    # Double-buffered shared memory: one warp per block, no conflicts.
    A0_shared = al.make_shared((32, 16), al.bf16)
    A1_shared = al.make_shared((32, 16), al.bf16)
    B0_shared = al.make_shared((16, 32), al.bf16)
    B1_shared = al.make_shared((16, 32), al.bf16)

    # MFMA operand views.
    A0_vec = al.view(
        A0_shared, al.u32,
        al.make_layout((32, 4, 2), (8, 2, 1)),
    )
    B0_vec = al.view(
        B0_shared, al.u32,
        al.make_layout((16, 8, 2), (16, 2, 1)),
    )
    A1_vec = al.view(
        A1_shared, al.u32,
        al.make_layout((32, 4, 2), (8, 2, 1)),
    )
    B1_vec = al.view(
        B1_shared, al.u32,
        al.make_layout((16, 8, 2), (16, 2, 1)),
    )

    # Per-thread partial sum: accumulates sigmoid values for all columns.
    # Each thread handles 4 rows (tr group) and 4 cols (tc group) per N-tile.
    partial = al.make_local((32,), al.f32)
    for r in al.range(32):
        partial[r] = ZERO_F

    tr = lane_id // EIGHT
    tc = lane_id % EIGHT

    BLK_N = THIRTY_TWO
    BLK_K = SIXTEEN
    TWO_BLK_K = al.convert(32, al.i32)

    a_row = lane_id // TWO
    a_col_off = (lane_id % TWO) * EIGHT
    b_row = lane_id // FOUR
    b_col_off = (lane_id % FOUR) * EIGHT

    # ---- Outer loop over N tiles ----
    for n_tile in al.range(0, N_VAL, BLK_N):
        c_acc = al.make_local((16,), al.f32)
        for i in al.range(16):
            c_acc[i] = ZERO_F

        # ---- Prologue: load tile k=0 into buffer 0 ----
        a_byte_off = ((m_start + a_row) * K_VAL + a_col_off) * TWO
        a_v = al.amdgpu.raw_buffer_load_x4(X_rsrc, a_byte_off, 0, 0)
        a_bf16 = al.view(a_v, al.Tensor((8,), al.bf16))
        for i in al.range(8):
            A0_shared[a_row, a_col_off + i] = a_bf16[i]

        b_byte_off = (b_row * N_VAL + n_tile + b_col_off) * TWO
        b_v = al.amdgpu.raw_buffer_load_x4(W_rsrc, b_byte_off, 0, 0)
        b_bf16 = al.view(b_v, al.Tensor((8,), al.bf16))
        for i in al.range(8):
            B0_shared[b_row, b_col_off + i] = b_bf16[i]

        al.syncthreads()

        # ---- Main loop: software-pipelined, 2x unrolled ----
        for k_start in al.range(BLK_K, K_VAL, TWO_BLK_K):
            # -- Phase 1: load tile k_start → buf1, compute buf0.
            a_byte_off = ((m_start + a_row) * K_VAL + k_start + a_col_off) * TWO
            a_v = al.amdgpu.raw_buffer_load_x4(X_rsrc, a_byte_off, 0, 0)
            a_bf16 = al.view(a_v, al.Tensor((8,), al.bf16))
            for i in al.range(8):
                A1_shared[a_row, a_col_off + i] = a_bf16[i]

            b_byte_off = ((k_start + b_row) * N_VAL + n_tile + b_col_off) * TWO
            b_v = al.amdgpu.raw_buffer_load_x4(W_rsrc, b_byte_off, 0, 0)
            b_bf16 = al.view(b_v, al.Tensor((8,), al.bf16))
            for i in al.range(8):
                B1_shared[b_row, b_col_off + i] = b_bf16[i]

            # MFMA on buffer 0: two ops (K=8 each → K=16 total).
            c_acc = al.amdgpu.mfma_32x32x8_bf16_f32(
                A0_vec[lane_id % THIRTY_TWO, lane_id // THIRTY_TWO],
                B0_vec[lane_id // EIGHT, lane_id % EIGHT],
                c_acc,
            )
            c_acc = al.amdgpu.mfma_32x32x8_bf16_f32(
                A0_vec[lane_id % THIRTY_TWO,
                       (lane_id // THIRTY_TWO) + TWO],
                B0_vec[(lane_id // EIGHT) + EIGHT, lane_id % EIGHT],
                c_acc,
            )

            al.syncthreads()

            # -- Phase 2 (or Epilogue).
            k_next = k_start + BLK_K
            if k_next < K_VAL:
                a_byte_off = ((m_start + a_row) * K_VAL + k_next + a_col_off) * TWO
                a_v = al.amdgpu.raw_buffer_load_x4(X_rsrc, a_byte_off, 0, 0)
                a_bf16 = al.view(a_v, al.Tensor((8,), al.bf16))
                for i in al.range(8):
                    A0_shared[a_row, a_col_off + i] = a_bf16[i]

                b_byte_off = ((k_next + b_row) * N_VAL + n_tile + b_col_off) * TWO
                b_v = al.amdgpu.raw_buffer_load_x4(W_rsrc, b_byte_off, 0, 0)
                b_bf16 = al.view(b_v, al.Tensor((8,), al.bf16))
                for i in al.range(8):
                    B0_shared[b_row, b_col_off + i] = b_bf16[i]

                c_acc = al.amdgpu.mfma_32x32x8_bf16_f32(
                    A1_vec[lane_id % THIRTY_TWO, lane_id // THIRTY_TWO],
                    B1_vec[lane_id // EIGHT, lane_id % EIGHT],
                    c_acc,
                )
                c_acc = al.amdgpu.mfma_32x32x8_bf16_f32(
                    A1_vec[lane_id % THIRTY_TWO,
                           (lane_id // THIRTY_TWO) + TWO],
                    B1_vec[(lane_id // EIGHT) + EIGHT, lane_id % EIGHT],
                    c_acc,
                )

                al.syncthreads()
            else:
                # Epilogue: last tile in buf1, compute it.
                c_acc = al.amdgpu.mfma_32x32x8_bf16_f32(
                    A1_vec[lane_id % THIRTY_TWO, lane_id // THIRTY_TWO],
                    B1_vec[lane_id // EIGHT, lane_id % EIGHT],
                    c_acc,
                )
                c_acc = al.amdgpu.mfma_32x32x8_bf16_f32(
                    A1_vec[lane_id % THIRTY_TWO,
                           (lane_id // THIRTY_TWO) + TWO],
                    B1_vec[(lane_id // EIGHT) + EIGHT, lane_id % EIGHT],
                    c_acc,
                )

                al.syncthreads()

        # ---- Post-process: bias + sigmoid, accumulate into partial ----
        # Each thread handles 4 rows × 4 cols of the 32×32 MFMA output.
        for idx in al.range(16):
            er = idx // FOUR
            ec = idx % FOUR
            row = tr * FOUR + er
            col = tc * FOUR + ec

            bias_val = al.convert(bias[n_tile + col], al.f32)
            val = c_acc[idx] + bias_val
            sig_val = ONE_F / (ONE_F + al.exp(ZERO_F - val))
            partial[row] = partial[row] + sig_val

    # ---- Cross-thread reduction: sum partial across column lanes ----
    # Each thread only accumulated its own 4 columns per row.  We need the
    # sum across all 8 column lanes (tc=0..7) for each row.  Shuffle-xor
    # reduction across lanes sharing the same tr group.
    for er in al.range(4):
        row = tr * FOUR + er
        partial[row] = partial[row] + al.shuffle_xor(
            partial[row], ONE, SIXTY_FOUR,
        )
        partial[row] = partial[row] + al.shuffle_xor(
            partial[row], TWO, SIXTY_FOUR,
        )
        partial[row] = partial[row] + al.shuffle_xor(
            partial[row], FOUR, SIXTY_FOUR,
        )

    # ---- Write output: each thread writes its 4 rows ----
    for er in al.range(4):
        row = tr * FOUR + er
        Y[m_start + row, al.convert(0, al.i32)] = al.convert(
            partial[row], al.bf16,
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
