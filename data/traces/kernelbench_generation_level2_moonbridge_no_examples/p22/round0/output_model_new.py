import torch
import torch.nn as nn
import avelang
import avelang.language as al
import struct


_BM = 64
_BN = 64
_BK = 64
_TM = 16
_TN = 16


@avelang.jit
def gemm_scale_add_clamp_kernel(
    a_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
    scale_bits: al.i32,
    clamp_min_bits: al.i32,
    clamp_max_bits: al.i32,
):
    BM = _BM
    BN = _BN
    BK = _BK

    block_m = al.block_id(1)
    block_n = al.block_id(0)
    thread_m = al.thread_id(1)
    thread_n = al.thread_id(0)

    m_start = block_m * BM
    n_start = block_n * BN

    a_layout = al.make_layout((M, K), (K, 1))
    a = al.make_tensor(a_ptr, al.bf16, a_layout)
    w_layout = al.make_layout((N, K), (K, 1))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)
    bias_layout = al.make_layout((N,), (1,))
    bias = al.make_tensor(bias_ptr, al.f32, bias_layout)
    out_layout = al.make_layout((M, N), (N, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    As = al.make_shared((BM, BK), al.bf16)
    Bs = al.make_shared((BK, BN), al.bf16)

    ROWS_PER_THREAD = BM // _TM
    COLS_PER_THREAD = BN // _TN

    zero = al.convert(0.0, al.f32)
    acc00 = zero; acc01 = zero; acc02 = zero; acc03 = zero
    acc10 = zero; acc11 = zero; acc12 = zero; acc13 = zero
    acc20 = zero; acc21 = zero; acc22 = zero; acc23 = zero
    acc30 = zero; acc31 = zero; acc32 = zero; acc33 = zero

    for k_block in al.range(0, K, BK):
        for i in al.range(ROWS_PER_THREAD):
            a_row = m_start + thread_m * ROWS_PER_THREAD + i
            for j in al.range(COLS_PER_THREAD):
                a_col = k_block + thread_n * COLS_PER_THREAD + j
                if a_row < M and a_col < K:
                    As[thread_m * ROWS_PER_THREAD + i, thread_n * COLS_PER_THREAD + j] = a[a_row, a_col]

        for k_idx in al.range(COLS_PER_THREAD):
            b_row_k = k_block + thread_n * COLS_PER_THREAD + k_idx
            for l_idx in al.range(ROWS_PER_THREAD):
                b_col_n = n_start + thread_m * ROWS_PER_THREAD + l_idx
                if b_row_k < K and b_col_n < N:
                    Bs[thread_n * COLS_PER_THREAD + k_idx, thread_m * ROWS_PER_THREAD + l_idx] = w[b_col_n, b_row_k]

        al.syncthreads()

        for k in al.range(BK):
            a0 = al.convert(As[thread_m * ROWS_PER_THREAD + 0, k], al.f32)
            a1 = al.convert(As[thread_m * ROWS_PER_THREAD + 1, k], al.f32)
            a2 = al.convert(As[thread_m * ROWS_PER_THREAD + 2, k], al.f32)
            a3 = al.convert(As[thread_m * ROWS_PER_THREAD + 3, k], al.f32)
            b0 = al.convert(Bs[k, thread_n * COLS_PER_THREAD + 0], al.f32)
            b1 = al.convert(Bs[k, thread_n * COLS_PER_THREAD + 1], al.f32)
            b2 = al.convert(Bs[k, thread_n * COLS_PER_THREAD + 2], al.f32)
            b3 = al.convert(Bs[k, thread_n * COLS_PER_THREAD + 3], al.f32)

            acc00 = acc00 + a0 * b0; acc01 = acc01 + a0 * b1
            acc02 = acc02 + a0 * b2; acc03 = acc03 + a0 * b3
            acc10 = acc10 + a1 * b0; acc11 = acc11 + a1 * b1
            acc12 = acc12 + a1 * b2; acc13 = acc13 + a1 * b3
            acc20 = acc20 + a2 * b0; acc21 = acc21 + a2 * b1
            acc22 = acc22 + a2 * b2; acc23 = acc23 + a2 * b3
            acc30 = acc30 + a3 * b0; acc31 = acc31 + a3 * b1
            acc32 = acc32 + a3 * b2; acc33 = acc33 + a3 * b3

        al.syncthreads()

    sf = al.bitcast(scale_bits, al.f32)
    cmi = al.bitcast(clamp_min_bits, al.f32)
    cma = al.bitcast(clamp_max_bits, al.f32)

    for i in al.range(ROWS_PER_THREAD):
        row = m_start + thread_m * ROWS_PER_THREAD + i
        for j in al.range(COLS_PER_THREAD):
            col = n_start + thread_n * COLS_PER_THREAD + j
            if row < M and col < N:
                val = zero
                if i == 0:
                    if j == 0: val = acc00
                    elif j == 1: val = acc01
                    elif j == 2: val = acc02
                    else: val = acc03
                elif i == 1:
                    if j == 0: val = acc10
                    elif j == 1: val = acc11
                    elif j == 2: val = acc12
                    else: val = acc13
                elif i == 2:
                    if j == 0: val = acc20
                    elif j == 1: val = acc21
                    elif j == 2: val = acc22
                    else: val = acc23
                else:
                    if j == 0: val = acc30
                    elif j == 1: val = acc31
                    elif j == 2: val = acc32
                    else: val = acc33

                val = val + bias[col]
                val = val * sf
                val = val + val
                if val < cmi: val = cmi
                if val > cma: val = cma
                out[row, col] = al.convert(val, al.bf16)


@avelang.jit
def logsumexp_mish_kernel(
    in_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
):
    row = al.block_id(0)
    tid = al.thread_id(0)
    BD = al.block_dim(0)

    in_layout = al.make_layout((M, N), (N, 1))
    inp = al.make_tensor(in_ptr, al.bf16, in_layout)
    out_layout = al.make_layout((M, 1), (1, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    smax = al.make_shared((256,), al.f32)
    ssum = al.make_shared((256,), al.f32)

    neg_inf = al.convert(-3.402823e38, al.f32)
    local_max = neg_inf
    for j in al.range(tid, N, BD):
        val = al.convert(inp[row, j], al.f32)
        if val > local_max:
            local_max = val

    smax[tid] = local_max
    al.syncthreads()
    if tid < 128:
        if smax[tid + 128] > smax[tid]: smax[tid] = smax[tid + 128]
    al.syncthreads()
    if tid < 64:
        if smax[tid + 64] > smax[tid]: smax[tid] = smax[tid + 64]
    al.syncthreads()
    if tid < 32:
        if smax[tid + 32] > smax[tid]: smax[tid] = smax[tid + 32]
    al.syncthreads()
    if tid < 16:
        if smax[tid + 16] > smax[tid]: smax[tid] = smax[tid + 16]
    al.syncthreads()
    if tid < 8:
        if smax[tid + 8] > smax[tid]: smax[tid] = smax[tid + 8]
    al.syncthreads()
    if tid < 4:
        if smax[tid + 4] > smax[tid]: smax[tid] = smax[tid + 4]
    al.syncthreads()
    if tid < 2:
        if smax[tid + 2] > smax[tid]: smax[tid] = smax[tid + 2]
    al.syncthreads()
    if tid < 1:
        if smax[tid + 1] > smax[tid]: smax[tid] = smax[tid + 1]
    al.syncthreads()
    global_max = smax[0]

    local_sum = al.convert(0.0, al.f32)
    for j in al.range(tid, N, BD):
        val = al.convert(inp[row, j], al.f32)
        diff_f32 = val - global_max
        diff_bf16 = al.convert(diff_f32, al.bf16)
        diff_val = al.convert(diff_bf16, al.f32)
        local_sum = local_sum + al.exp(diff_val)

    ssum[tid] = local_sum
    al.syncthreads()
    if tid < 128: ssum[tid] = ssum[tid] + ssum[tid + 128]
    al.syncthreads()
    if tid < 64: ssum[tid] = ssum[tid] + ssum[tid + 64]
    al.syncthreads()
    if tid < 32: ssum[tid] = ssum[tid] + ssum[tid + 32]
    al.syncthreads()
    if tid < 16: ssum[tid] = ssum[tid] + ssum[tid + 16]
    al.syncthreads()
    if tid < 8: ssum[tid] = ssum[tid] + ssum[tid + 8]
    al.syncthreads()
    if tid < 4: ssum[tid] = ssum[tid] + ssum[tid + 4]
    al.syncthreads()
    if tid < 2: ssum[tid] = ssum[tid] + ssum[tid + 2]
    al.syncthreads()
    if tid < 1: ssum[tid] = ssum[tid] + ssum[tid + 1]
    al.syncthreads()
    global_sum = ssum[0]

    if tid == 0:
        sum_bf16 = al.convert(global_sum, al.bf16)
        sum_f32 = al.convert(sum_bf16, al.f32)
        log_f32 = al.log(sum_f32)
        log_bf16 = al.convert(log_f32, al.bf16)
        log_val = al.convert(log_bf16, al.f32)
        lse_f32 = global_max + log_val
        lse_bf16 = al.convert(lse_f32, al.bf16)
        lse_val = al.convert(lse_bf16, al.f32)
        one = al.convert(1.0, al.f32)
        exp_val = al.exp(lse_val)
        softplus = al.log(one + exp_val)
        mish_val = lse_val * al.tanh(softplus)
        result = lse_val * mish_val
        out[row, 0] = al.convert(result, al.bf16)


def _float_to_bits(f: float) -> int:
    return struct.unpack("i", struct.pack("f", f))[0]


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        batch_size = x.shape[0]
        M_val = batch_size
        N_val = self.hidden_size
        K_val = self.input_size

        w = self.linear.weight.data
        bias = self.linear.bias.data

        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = w.to(torch.bfloat16).contiguous()
        bias_f32 = bias.to(torch.float32).contiguous()

        gemm_out = torch.empty(batch_size, N_val, dtype=torch.bfloat16, device=x.device)

        grid_m = (M_val + _BM - 1) // _BM
        grid_n = (N_val + _BN - 1) // _BN

        gemm_scale_add_clamp_kernel[lambda: ((grid_n, grid_m, 1), (_TN, _TM, 1))](
            x_bf16.data_ptr(), w_bf16.data_ptr(), bias_f32.data_ptr(),
            gemm_out.data_ptr(), M_val, N_val, K_val,
            _float_to_bits(self.scale_factor),
            _float_to_bits(self.clamp_min),
            _float_to_bits(self.clamp_max),
        )

        lse_out = torch.empty(batch_size, 1, dtype=torch.bfloat16, device=x.device)
        logsumexp_mish_kernel[lambda: ((M_val, 1, 1), (256, 1, 1))](
            gemm_out.data_ptr(), lse_out.data_ptr(), M_val, N_val,
        )
        return lse_out
