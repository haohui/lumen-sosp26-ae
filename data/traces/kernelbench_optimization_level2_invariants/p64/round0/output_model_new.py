import torch
import torch.nn as nn
import avelang
import avelang.language as al

SQRT_2 = 1.4142135623730951
NEGATIVE_SLOPE = 0.01

TM = 64
TN = 64
TK = 32


@avelang.jit
def fused_kernel(
    X_ptr: al.Pointer(al.bf16),
    W_ptr: al.Pointer(al.bf16),
    Bias_ptr: al.Pointer(al.bf16),
    Y_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
    stride_X0: al.i32,
    stride_W0: al.i32,
):
    bid = al.block_id(0)
    tid = al.thread_id(0)

    one = al.convert(1, al.i32)
    two = al.convert(2, al.i32)
    four = al.convert(4, al.i32)
    eight = al.convert(8, al.i32)
    sixteen = al.convert(16, al.i32)
    thirty2 = al.convert(32, al.i32)
    sixty4 = al.convert(64, al.i32)
    z128 = al.convert(128, al.i32)
    zero_i = al.convert(0, al.i32)

    X_layout = al.make_layout((M, K), (stride_X0, one))
    X = al.make_tensor(X_ptr, al.bf16, X_layout)
    W_layout = al.make_layout((K, N), (stride_W0, one))
    W = al.make_tensor(W_ptr, al.bf16, W_layout)
    Bias_layout = al.make_layout((N,), (one,))
    Bias = al.make_tensor(Bias_ptr, al.bf16, Bias_layout)
    Y_layout = al.make_layout((M, one), (one, one))
    Y = al.make_tensor(Y_ptr, al.bf16, Y_layout)

    warp_id = tid // sixty4
    warp_m = warp_id // two
    warp_n = warp_id % two
    lane = tid % sixty4
    lane_half = lane // thirty2

    row_base = bid * al.convert(TM, al.i32)

    a_lds = al.make_shared((al.convert(2048, al.i32),), al.bf16)
    b_lds = al.make_shared((al.convert(2048, al.i32),), al.bf16)

    row_max = al.make_local((thirty2,), al.f32)
    row_sum = al.make_local((thirty2,), al.f32)
    for r in al.range(32):
        row_max[r] = al.convert(-1.0e+30, al.f32)
        row_sum[r] = al.convert(0.0, al.f32)

    wave_lds = al.make_shared((z128, al.convert(2, al.i32)), al.f32)

    X_rsrc = al.amdgpu.make_rsrc(X, 0x7FFFFFFF)
    W_rsrc = al.amdgpu.make_rsrc(W, 0x7FFFFFFF)

    for n_tile in al.range(0, N, al.convert(TN, al.i32)):
        acc = al.make_local((sixteen,), al.f32)
        for a in al.range(16):
            acc[a] = al.convert(0.0, al.f32)

        n_base = n_tile + warp_n * thirty2
        n_lds_base = warp_n * thirty2

        for k_tile in al.range(0, K, al.convert(TK, al.i32)):
            a_row = tid % al.convert(TM, al.i32)
            a_col = (tid // al.convert(TM, al.i32)) * eight
            a_byte = ((row_base + a_row) * K + k_tile + a_col) * 2
            a_loaded = al.amdgpu.raw_buffer_load_x4(X_rsrc, a_byte, 0, 0)
            a_bf16 = al.view(a_loaded, al.Tensor((eight,), al.bf16))
            a_lds_base = a_row * thirty2 + a_col
            for v in al.range(8):
                a_lds[a_lds_base + v] = a_bf16[v]

            b_k = tid % al.convert(TK, al.i32)
            b_n = (tid // al.convert(TK, al.i32)) * eight
            b_byte = ((k_tile + b_k) * N + n_tile + b_n) * 2
            b_loaded = al.amdgpu.raw_buffer_load_x4(W_rsrc, b_byte, 0, 0)
            b_bf16 = al.view(b_loaded, al.Tensor((eight,), al.bf16))
            b_lds_base = b_k * sixty4 + b_n
            for v in al.range(8):
                b_lds[b_lds_base + v] = b_bf16[v]

            al.syncthreads()

            wm_row = warp_m * thirty2
            a_lds_row = wm_row + (lane % thirty2)
            j_val = lane % eight
            nq = lane // eight
            b_npos = n_lds_base + nq * four

            if lane < thirty2:
                data_a0 = al.make_local((four,), al.u32)
                data_a1 = al.make_local((four,), al.u32)
                data_b0 = al.make_local((four,), al.u32)
                data_b1 = al.make_local((four,), al.u32)

                av0 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 0], al.u16), al.u32)
                av1 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 1], al.u16), al.u32)
                av2 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 2], al.u16), al.u32)
                av3 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 3], al.u16), al.u32)
                av4 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 8], al.u16), al.u32)
                av5 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 9], al.u16), al.u32)
                av6 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 10], al.u16), al.u32)
                av7 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 11], al.u16), al.u32)
                av10 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 16], al.u16), al.u32)
                av11 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 17], al.u16), al.u32)
                av12 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 18], al.u16), al.u32)
                av13 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 19], al.u16), al.u32)
                av14 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 24], al.u16), al.u32)
                av15 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 25], al.u16), al.u32)
                av16 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 26], al.u16), al.u32)
                av17 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 27], al.u16), al.u32)

                bv0 = al.convert(al.bitcast(b_lds[(j_val) * sixty4 + b_npos + 0], al.u16), al.u32)
                bv1 = al.convert(al.bitcast(b_lds[(j_val) * sixty4 + b_npos + 1], al.u16), al.u32)
                bv2 = al.convert(al.bitcast(b_lds[(j_val) * sixty4 + b_npos + 2], al.u16), al.u32)
                bv3 = al.convert(al.bitcast(b_lds[(j_val) * sixty4 + b_npos + 3], al.u16), al.u32)
                bv4 = al.convert(al.bitcast(b_lds[(j_val + eight) * sixty4 + b_npos + 0], al.u16), al.u32)
                bv5 = al.convert(al.bitcast(b_lds[(j_val + eight) * sixty4 + b_npos + 1], al.u16), al.u32)
                bv6 = al.convert(al.bitcast(b_lds[(j_val + eight) * sixty4 + b_npos + 2], al.u16), al.u32)
                bv7 = al.convert(al.bitcast(b_lds[(j_val + eight) * sixty4 + b_npos + 3], al.u16), al.u32)
                bv10 = al.convert(al.bitcast(b_lds[(j_val + sixteen) * sixty4 + b_npos + 0], al.u16), al.u32)
                bv11 = al.convert(al.bitcast(b_lds[(j_val + sixteen) * sixty4 + b_npos + 1], al.u16), al.u32)
                bv12 = al.convert(al.bitcast(b_lds[(j_val + sixteen) * sixty4 + b_npos + 2], al.u16), al.u32)
                bv13 = al.convert(al.bitcast(b_lds[(j_val + sixteen) * sixty4 + b_npos + 3], al.u16), al.u32)
                bv14 = al.convert(al.bitcast(b_lds[(j_val + al.convert(24, al.i32)) * sixty4 + b_npos + 0], al.u16), al.u32)
                bv15 = al.convert(al.bitcast(b_lds[(j_val + al.convert(24, al.i32)) * sixty4 + b_npos + 1], al.u16), al.u32)
                bv16 = al.convert(al.bitcast(b_lds[(j_val + al.convert(24, al.i32)) * sixty4 + b_npos + 2], al.u16), al.u32)
                bv17 = al.convert(al.bitcast(b_lds[(j_val + al.convert(24, al.i32)) * sixty4 + b_npos + 3], al.u16), al.u32)

                data_a0[0] = av0 | (av1 << sixteen)
                data_a0[1] = av2 | (av3 << sixteen)
                data_a0[2] = av4 | (av5 << sixteen)
                data_a0[3] = av6 | (av7 << sixteen)
                data_a1[0] = av10 | (av11 << sixteen)
                data_a1[1] = av12 | (av13 << sixteen)
                data_a1[2] = av14 | (av15 << sixteen)
                data_a1[3] = av16 | (av17 << sixteen)

                data_b0[0] = bv0 | (bv1 << sixteen)
                data_b0[1] = bv2 | (bv3 << sixteen)
                data_b0[2] = bv4 | (bv5 << sixteen)
                data_b0[3] = bv6 | (bv7 << sixteen)
                data_b1[0] = bv10 | (bv11 << sixteen)
                data_b1[1] = bv12 | (bv13 << sixteen)
                data_b1[2] = bv14 | (bv15 << sixteen)
                data_b1[3] = bv16 | (bv17 << sixteen)

                frag_a0 = al.view(data_a0, al.Tensor((two, two), al.u32))
                frag_b0 = al.view(data_b0, al.Tensor((two, two), al.u32))
                acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a0[0], frag_b0[0], acc)
                acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a0[1], frag_b0[1], acc)

                frag_a1 = al.view(data_a1, al.Tensor((two, two), al.u32))
                frag_b1 = al.view(data_b1, al.Tensor((two, two), al.u32))
                acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a1[0], frag_b1[0], acc)
                acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a1[1], frag_b1[1], acc)
            else:
                data_a0 = al.make_local((four,), al.u32)
                data_a1 = al.make_local((four,), al.u32)
                data_b0 = al.make_local((four,), al.u32)
                data_b1 = al.make_local((four,), al.u32)

                av0 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 4], al.u16), al.u32)
                av1 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 5], al.u16), al.u32)
                av2 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 6], al.u16), al.u32)
                av3 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 7], al.u16), al.u32)
                av4 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 12], al.u16), al.u32)
                av5 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 13], al.u16), al.u32)
                av6 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 14], al.u16), al.u32)
                av7 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 15], al.u16), al.u32)
                av10 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 20], al.u16), al.u32)
                av11 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 21], al.u16), al.u32)
                av12 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 22], al.u16), al.u32)
                av13 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 23], al.u16), al.u32)
                av14 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 28], al.u16), al.u32)
                av15 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 29], al.u16), al.u32)
                av16 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 30], al.u16), al.u32)
                av17 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 31], al.u16), al.u32)

                bv0 = al.convert(al.bitcast(b_lds[(j_val) * sixty4 + b_npos + 0], al.u16), al.u32)
                bv1 = al.convert(al.bitcast(b_lds[(j_val) * sixty4 + b_npos + 1], al.u16), al.u32)
                bv2 = al.convert(al.bitcast(b_lds[(j_val) * sixty4 + b_npos + 2], al.u16), al.u32)
                bv3 = al.convert(al.bitcast(b_lds[(j_val) * sixty4 + b_npos + 3], al.u16), al.u32)
                bv4 = al.convert(al.bitcast(b_lds[(j_val + eight) * sixty4 + b_npos + 0], al.u16), al.u32)
                bv5 = al.convert(al.bitcast(b_lds[(j_val + eight) * sixty4 + b_npos + 1], al.u16), al.u32)
                bv6 = al.convert(al.bitcast(b_lds[(j_val + eight) * sixty4 + b_npos + 2], al.u16), al.u32)
                bv7 = al.convert(al.bitcast(b_lds[(j_val + eight) * sixty4 + b_npos + 3], al.u16), al.u32)
                bv10 = al.convert(al.bitcast(b_lds[(j_val + sixteen) * sixty4 + b_npos + 0], al.u16), al.u32)
                bv11 = al.convert(al.bitcast(b_lds[(j_val + sixteen) * sixty4 + b_npos + 1], al.u16), al.u32)
                bv12 = al.convert(al.bitcast(b_lds[(j_val + sixteen) * sixty4 + b_npos + 2], al.u16), al.u32)
                bv13 = al.convert(al.bitcast(b_lds[(j_val + sixteen) * sixty4 + b_npos + 3], al.u16), al.u32)
                bv14 = al.convert(al.bitcast(b_lds[(j_val + al.convert(24, al.i32)) * sixty4 + b_npos + 0], al.u16), al.u32)
                bv15 = al.convert(al.bitcast(b_lds[(j_val + al.convert(24, al.i32)) * sixty4 + b_npos + 1], al.u16), al.u32)
                bv16 = al.convert(al.bitcast(b_lds[(j_val + al.convert(24, al.i32)) * sixty4 + b_npos + 2], al.u16), al.u32)
                bv17 = al.convert(al.bitcast(b_lds[(j_val + al.convert(24, al.i32)) * sixty4 + b_npos + 3], al.u16), al.u32)

                data_a0[0] = av0 | (av1 << sixteen)
                data_a0[1] = av2 | (av3 << sixteen)
                data_a0[2] = av4 | (av5 << sixteen)
                data_a0[3] = av6 | (av7 << sixteen)
                data_a1[0] = av10 | (av11 << sixteen)
                data_a1[1] = av12 | (av13 << sixteen)
                data_a1[2] = av14 | (av15 << sixteen)
                data_a1[3] = av16 | (av17 << sixteen)

                data_b0[0] = bv0 | (bv1 << sixteen)
                data_b0[1] = bv2 | (bv3 << sixteen)
                data_b0[2] = bv4 | (bv5 << sixteen)
                data_b0[3] = bv6 | (bv7 << sixteen)
                data_b1[0] = bv10 | (bv11 << sixteen)
                data_b1[1] = bv12 | (bv13 << sixteen)
                data_b1[2] = bv14 | (bv15 << sixteen)
                data_b1[3] = bv16 | (bv17 << sixteen)

                frag_a0 = al.view(data_a0, al.Tensor((two, two), al.u32))
                frag_b0 = al.view(data_b0, al.Tensor((two, two), al.u32))
                acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a0[0], frag_b0[0], acc)
                acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a0[1], frag_b0[1], acc)

                frag_a1 = al.view(data_a1, al.Tensor((two, two), al.u32))
                frag_b1 = al.view(data_b1, al.Tensor((two, two), al.u32))
                acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a1[0], frag_b1[0], acc)
                acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a1[1], frag_b1[1], acc)

            al.syncthreads()

        # LogSumExp for this N-tile using shuffle reduction
        col_global = n_base + (lane % thirty2)
        bias_val = al.convert(Bias[col_global], al.f32)

        # Compute per-lane values and shuffle-reduce within wave
        for a in al.range(16):
            r = eight * (a // four) + four * lane_half + (a % four)
            my_val = acc[a] + bias_val

            # Tree reduction across 32 lanes in the same lane_half group
            # Shuffle down within 32-lane group
            max_val = my_val
            v1 = al.shuffle_down(max_val, al.convert(16, al.i32), thirty2)
            if v1 > max_val:
                max_val = v1
            v2 = al.shuffle_down(max_val, eight, thirty2)
            if v2 > max_val:
                max_val = v2
            v3 = al.shuffle_down(max_val, four, thirty2)
            if v3 > max_val:
                max_val = v3
            v4 = al.shuffle_down(max_val, two, thirty2)
            if v4 > max_val:
                max_val = v4
            v5 = al.shuffle_down(max_val, one, thirty2)
            if v5 > max_val:
                max_val = v5
            # Broadcast max back to all lanes
            tmax = al.shuffle_xor(max_val, al.convert(1, al.i32), thirty2)
            tmax = al.shuffle_xor(tmax, two, thirty2)
            tmax = al.shuffle_xor(tmax, four, thirty2)
            tmax = al.shuffle_xor(tmax, eight, thirty2)
            tmax = al.shuffle_xor(tmax, al.convert(16, al.i32), thirty2)

            # Compute sum of exps
            exp_val = al.exp(my_val - tmax)
            sum_val = exp_val
            s1 = al.shuffle_down(sum_val, al.convert(16, al.i32), thirty2)
            sum_val = sum_val + s1
            s2 = al.shuffle_down(sum_val, eight, thirty2)
            sum_val = sum_val + s2
            s3 = al.shuffle_down(sum_val, four, thirty2)
            sum_val = sum_val + s3
            s4 = al.shuffle_down(sum_val, two, thirty2)
            sum_val = sum_val + s4
            s5 = al.shuffle_down(sum_val, one, thirty2)
            sum_val = sum_val + s5
            # Broadcast sum to all lanes
            tsum = al.shuffle_xor(sum_val, al.convert(1, al.i32), thirty2)
            tsum = al.shuffle_xor(tsum, two, thirty2)
            tsum = al.shuffle_xor(tsum, four, thirty2)
            tsum = al.shuffle_xor(tsum, eight, thirty2)
            tsum = al.shuffle_xor(tsum, al.convert(16, al.i32), thirty2)

            if tmax > row_max[r]:
                row_sum[r] = row_sum[r] * al.exp(row_max[r] - tmax) + tsum
                row_max[r] = tmax
            else:
                row_sum[r] = row_sum[r] + tsum * al.exp(tmax - row_max[r])

        al.syncthreads()

    # Cross-wave combine
    base_off = warp_m * thirty2 + warp_n * sixty4
    for r in al.range(32):
        wave_lds[base_off + r, 0] = row_max[r]
        wave_lds[base_off + r, 1] = row_sum[r]

    al.syncthreads()

    if warp_n == zero_i:
        for r in al.range(32):
            gr = warp_m * thirty2 + r
            m0 = wave_lds[gr, 0]
            s0 = wave_lds[gr, 1]
            m1 = wave_lds[gr + sixty4, 0]
            s1 = wave_lds[gr + sixty4, 1]

            final_max = m0
            if m1 > final_max:
                final_max = m1

            sum0_term = s0
            sum1_term = s1
            if m0 < final_max:
                sum0_term = s0 * al.exp(m0 - final_max)
            if m1 < final_max:
                sum1_term = s1 * al.exp(m1 - final_max)
            final_sum = sum0_term + sum1_term

            x = final_max + al.log(final_sum)

            zero_f = al.convert(0.0, al.f32)
            slope = al.convert(NEGATIVE_SLOPE, al.f32)
            if x < zero_f:
                x = x * slope
            if x < zero_f:
                x = x * slope

            half = al.convert(0.5, al.f32)
            one_f = al.convert(1.0, al.f32)
            sqrt2_f = al.convert(SQRT_2, al.f32)
            x = half * x * (one_f + al.erf(x / sqrt2_f))
            x = half * x * (one_f + al.erf(x / sqrt2_f))

            Y[row_base + gr, 0] = al.convert(x, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        if x.dtype != torch.bfloat16:
            raise RuntimeError('This fused kernel only supports bfloat16 input.')
        M_val = x.shape[0]
        K_val = x.shape[1]
        w_t = self.linear.weight.t().to(device=x.device, dtype=torch.bfloat16).contiguous()
        bias = self.linear.bias.to(device=x.device, dtype=torch.bfloat16).contiguous()
        N_val = w_t.shape[1]

        y = torch.empty((M_val, 1), device=x.device, dtype=torch.bfloat16)

        grid = (M_val // TM, 1, 1)
        block = (256, 1, 1)

        fused_kernel[lambda: (grid, block)](
            x.contiguous().data_ptr(),
            w_t.data_ptr(),
            bias.data_ptr(),
            y.data_ptr(),
            M_val,
            N_val,
            K_val,
            x.stride(0),
            w_t.stride(0),
            num_warps=4,
        )
        return y
