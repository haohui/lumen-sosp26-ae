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

    a_lds_0 = al.make_shared((al.convert(2048, al.i32),), al.bf16)
    a_lds_1 = al.make_shared((al.convert(2048, al.i32),), al.bf16)
    b_lds_0 = al.make_shared((al.convert(2048, al.i32),), al.bf16)
    b_lds_1 = al.make_shared((al.convert(2048, al.i32),), al.bf16)

    row_max = al.make_local((thirty2,), al.f32)
    row_sum = al.make_local((thirty2,), al.f32)
    for r in al.range(32):
        row_max[r] = al.convert(-1.0e+30, al.f32)
        row_sum[r] = al.convert(0.0, al.f32)

    wave_lds = al.make_shared((z128, al.convert(2, al.i32)), al.f32)

    bf16_sz = al.convert(2, al.i32)
    X_bytes = M * K * bf16_sz
    W_bytes = K * N * bf16_sz

    X_rsrc = al.amdgpu.make_rsrc(X, X_bytes)
    W_rsrc = al.amdgpu.make_rsrc(W, W_bytes)

    tk2 = al.convert(64, al.i32)
    tk_val = al.convert(TK, al.i32)
    tn_val = al.convert(TN, al.i32)
    tm_val = al.convert(TM, al.i32)

    for n_tile in al.range(0, N, tn_val):
        acc = al.make_local((sixteen,), al.f32)
        for a in al.range(16):
            acc[a] = al.convert(0.0, al.f32)

        n_base = n_tile + warp_n * thirty2
        n_lds_base = warp_n * thirty2

        # Prologue: load first tile K=[0, TK) into buf[0]
        a_row = tid % tm_val
        a_col = (tid // tm_val) * eight
        a_byte_p0 = ((row_base + a_row) * K + zero_i + a_col) * two
        a_loaded_p0 = al.amdgpu.raw_buffer_load_x4(X_rsrc, a_byte_p0, 0, 0)
        a_bf16_p0 = al.view(a_loaded_p0, al.Tensor((eight,), al.bf16))
        a_lds_base_p0 = a_row * thirty2 + a_col
        for v in al.range(8):
            a_lds_0[a_lds_base_p0 + v] = a_bf16_p0[v]

        b_k = tid % tk_val
        b_n = (tid // tk_val) * eight
        b_byte_p0 = ((zero_i + b_k) * N + n_tile + b_n) * two
        b_loaded_p0 = al.amdgpu.raw_buffer_load_x4(W_rsrc, b_byte_p0, 0, 0)
        b_bf16_p0 = al.view(b_loaded_p0, al.Tensor((eight,), al.bf16))
        b_lds_base_p0 = b_k * sixty4 + b_n
        for v in al.range(8):
            b_lds_0[b_lds_base_p0 + v] = b_bf16_p0[v]

        al.syncthreads()

        # Main K-loop: unrolled by 2, double-buffered software pipeline
        for k_tile in al.range(0, K, tk2):
            k_p1 = k_tile + tk_val
            k_p2 = k_tile + tk2

            # --- Phase 1: load buf[1] while computing from buf[0] ---
            a_row_p1 = tid % tm_val
            a_col_p1 = (tid // tm_val) * eight
            a_byte_p1 = ((row_base + a_row_p1) * K + k_p1 + a_col_p1) * two
            a_loaded_p1 = al.amdgpu.raw_buffer_load_x4(X_rsrc, a_byte_p1, 0, 0)
            a_bf16_p1 = al.view(a_loaded_p1, al.Tensor((eight,), al.bf16))
            a_lds_base_p1 = a_row_p1 * thirty2 + a_col_p1
            for v in al.range(8):
                a_lds_1[a_lds_base_p1 + v] = a_bf16_p1[v]

            b_k_p1 = tid % tk_val
            b_n_p1 = (tid // tk_val) * eight
            b_byte_p1 = ((k_p1 + b_k_p1) * N + n_tile + b_n_p1) * two
            b_loaded_p1 = al.amdgpu.raw_buffer_load_x4(W_rsrc, b_byte_p1, 0, 0)
            b_bf16_p1 = al.view(b_loaded_p1, al.Tensor((eight,), al.bf16))
            b_lds_base_p1 = b_k_p1 * sixty4 + b_n_p1
            for v in al.range(8):
                b_lds_1[b_lds_base_p1 + v] = b_bf16_p1[v]

            # MFMA from buf[0]
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

                av0 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 0], al.u16), al.u32)
                av1 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 1], al.u16), al.u32)
                av2 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 2], al.u16), al.u32)
                av3 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 3], al.u16), al.u32)
                av4 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 8], al.u16), al.u32)
                av5 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 9], al.u16), al.u32)
                av6 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 10], al.u16), al.u32)
                av7 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 11], al.u16), al.u32)
                av10 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 16], al.u16), al.u32)
                av11 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 17], al.u16), al.u32)
                av12 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 18], al.u16), al.u32)
                av13 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 19], al.u16), al.u32)
                av14 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 24], al.u16), al.u32)
                av15 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 25], al.u16), al.u32)
                av16 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 26], al.u16), al.u32)
                av17 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 27], al.u16), al.u32)

                bv0 = al.convert(al.bitcast(b_lds_0[(j_val) * sixty4 + b_npos + 0], al.u16), al.u32)
                bv1 = al.convert(al.bitcast(b_lds_0[(j_val) * sixty4 + b_npos + 1], al.u16), al.u32)
                bv2 = al.convert(al.bitcast(b_lds_0[(j_val) * sixty4 + b_npos + 2], al.u16), al.u32)
                bv3 = al.convert(al.bitcast(b_lds_0[(j_val) * sixty4 + b_npos + 3], al.u16), al.u32)
                bv4 = al.convert(al.bitcast(b_lds_0[(j_val + eight) * sixty4 + b_npos + 0], al.u16), al.u32)
                bv5 = al.convert(al.bitcast(b_lds_0[(j_val + eight) * sixty4 + b_npos + 1], al.u16), al.u32)
                bv6 = al.convert(al.bitcast(b_lds_0[(j_val + eight) * sixty4 + b_npos + 2], al.u16), al.u32)
                bv7 = al.convert(al.bitcast(b_lds_0[(j_val + eight) * sixty4 + b_npos + 3], al.u16), al.u32)
                bv10 = al.convert(al.bitcast(b_lds_0[(j_val + sixteen) * sixty4 + b_npos + 0], al.u16), al.u32)
                bv11 = al.convert(al.bitcast(b_lds_0[(j_val + sixteen) * sixty4 + b_npos + 1], al.u16), al.u32)
                bv12 = al.convert(al.bitcast(b_lds_0[(j_val + sixteen) * sixty4 + b_npos + 2], al.u16), al.u32)
                bv13 = al.convert(al.bitcast(b_lds_0[(j_val + sixteen) * sixty4 + b_npos + 3], al.u16), al.u32)
                bv14 = al.convert(al.bitcast(b_lds_0[(j_val + al.convert(24, al.i32)) * sixty4 + b_npos + 0], al.u16), al.u32)
                bv15 = al.convert(al.bitcast(b_lds_0[(j_val + al.convert(24, al.i32)) * sixty4 + b_npos + 1], al.u16), al.u32)
                bv16 = al.convert(al.bitcast(b_lds_0[(j_val + al.convert(24, al.i32)) * sixty4 + b_npos + 2], al.u16), al.u32)
                bv17 = al.convert(al.bitcast(b_lds_0[(j_val + al.convert(24, al.i32)) * sixty4 + b_npos + 3], al.u16), al.u32)

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

                av0 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 4], al.u16), al.u32)
                av1 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 5], al.u16), al.u32)
                av2 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 6], al.u16), al.u32)
                av3 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 7], al.u16), al.u32)
                av4 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 12], al.u16), al.u32)
                av5 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 13], al.u16), al.u32)
                av6 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 14], al.u16), al.u32)
                av7 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 15], al.u16), al.u32)
                av10 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 20], al.u16), al.u32)
                av11 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 21], al.u16), al.u32)
                av12 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 22], al.u16), al.u32)
                av13 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 23], al.u16), al.u32)
                av14 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 28], al.u16), al.u32)
                av15 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 29], al.u16), al.u32)
                av16 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 30], al.u16), al.u32)
                av17 = al.convert(al.bitcast(a_lds_0[a_lds_row * thirty2 + 31], al.u16), al.u32)

                bv0 = al.convert(al.bitcast(b_lds_0[(j_val) * sixty4 + b_npos + 0], al.u16), al.u32)
                bv1 = al.convert(al.bitcast(b_lds_0[(j_val) * sixty4 + b_npos + 1], al.u16), al.u32)
                bv2 = al.convert(al.bitcast(b_lds_0[(j_val) * sixty4 + b_npos + 2], al.u16), al.u32)
                bv3 = al.convert(al.bitcast(b_lds_0[(j_val) * sixty4 + b_npos + 3], al.u16), al.u32)
                bv4 = al.convert(al.bitcast(b_lds_0[(j_val + eight) * sixty4 + b_npos + 0], al.u16), al.u32)
                bv5 = al.convert(al.bitcast(b_lds_0[(j_val + eight) * sixty4 + b_npos + 1], al.u16), al.u32)
                bv6 = al.convert(al.bitcast(b_lds_0[(j_val + eight) * sixty4 + b_npos + 2], al.u16), al.u32)
                bv7 = al.convert(al.bitcast(b_lds_0[(j_val + eight) * sixty4 + b_npos + 3], al.u16), al.u32)
                bv10 = al.convert(al.bitcast(b_lds_0[(j_val + sixteen) * sixty4 + b_npos + 0], al.u16), al.u32)
                bv11 = al.convert(al.bitcast(b_lds_0[(j_val + sixteen) * sixty4 + b_npos + 1], al.u16), al.u32)
                bv12 = al.convert(al.bitcast(b_lds_0[(j_val + sixteen) * sixty4 + b_npos + 2], al.u16), al.u32)
                bv13 = al.convert(al.bitcast(b_lds_0[(j_val + sixteen) * sixty4 + b_npos + 3], al.u16), al.u32)
                bv14 = al.convert(al.bitcast(b_lds_0[(j_val + al.convert(24, al.i32)) * sixty4 + b_npos + 0], al.u16), al.u32)
                bv15 = al.convert(al.bitcast(b_lds_0[(j_val + al.convert(24, al.i32)) * sixty4 + b_npos + 1], al.u16), al.u32)
                bv16 = al.convert(al.bitcast(b_lds_0[(j_val + al.convert(24, al.i32)) * sixty4 + b_npos + 2], al.u16), al.u32)
                bv17 = al.convert(al.bitcast(b_lds_0[(j_val + al.convert(24, al.i32)) * sixty4 + b_npos + 3], al.u16), al.u32)

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

            # --- Phase 2: load buf[0] (OOB guarded by rsrc range), compute from buf[1] ---
            a_row_p2 = tid % tm_val
            a_col_p2 = (tid // tm_val) * eight
            a_byte_p2 = ((row_base + a_row_p2) * K + k_p2 + a_col_p2) * two
            a_loaded_p2 = al.amdgpu.raw_buffer_load_x4(X_rsrc, a_byte_p2, 0, 0)
            a_bf16_p2 = al.view(a_loaded_p2, al.Tensor((eight,), al.bf16))
            a_lds_base_p2 = a_row_p2 * thirty2 + a_col_p2
            for v in al.range(8):
                a_lds_0[a_lds_base_p2 + v] = a_bf16_p2[v]

                b_k_p2 = tid % tk_val
                b_n_p2 = (tid // tk_val) * eight
                b_byte_p2 = ((k_p2 + b_k_p2) * N + n_tile + b_n_p2) * two
                b_loaded_p2 = al.amdgpu.raw_buffer_load_x4(W_rsrc, b_byte_p2, 0, 0)
                b_bf16_p2 = al.view(b_loaded_p2, al.Tensor((eight,), al.bf16))
                b_lds_base_p2 = b_k_p2 * sixty4 + b_n_p2
                for v in al.range(8):
                    b_lds_0[b_lds_base_p2 + v] = b_bf16_p2[v]

            # MFMA from buf[1]
            wm_row_2 = warp_m * thirty2
            a_lds_row_2 = wm_row_2 + (lane % thirty2)
            j_val_2 = lane % eight
            nq_2 = lane // eight
            b_npos_2 = n_lds_base + nq_2 * four

            if lane < thirty2:
                data_a0_2 = al.make_local((four,), al.u32)
                data_a1_2 = al.make_local((four,), al.u32)
                data_b0_2 = al.make_local((four,), al.u32)
                data_b1_2 = al.make_local((four,), al.u32)

                av0_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 0], al.u16), al.u32)
                av1_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 1], al.u16), al.u32)
                av2_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 2], al.u16), al.u32)
                av3_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 3], al.u16), al.u32)
                av4_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 8], al.u16), al.u32)
                av5_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 9], al.u16), al.u32)
                av6_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 10], al.u16), al.u32)
                av7_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 11], al.u16), al.u32)
                av10_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 16], al.u16), al.u32)
                av11_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 17], al.u16), al.u32)
                av12_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 18], al.u16), al.u32)
                av13_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 19], al.u16), al.u32)
                av14_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 24], al.u16), al.u32)
                av15_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 25], al.u16), al.u32)
                av16_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 26], al.u16), al.u32)
                av17_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 27], al.u16), al.u32)

                bv0_2 = al.convert(al.bitcast(b_lds_1[(j_val_2) * sixty4 + b_npos_2 + 0], al.u16), al.u32)
                bv1_2 = al.convert(al.bitcast(b_lds_1[(j_val_2) * sixty4 + b_npos_2 + 1], al.u16), al.u32)
                bv2_2 = al.convert(al.bitcast(b_lds_1[(j_val_2) * sixty4 + b_npos_2 + 2], al.u16), al.u32)
                bv3_2 = al.convert(al.bitcast(b_lds_1[(j_val_2) * sixty4 + b_npos_2 + 3], al.u16), al.u32)
                bv4_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + eight) * sixty4 + b_npos_2 + 0], al.u16), al.u32)
                bv5_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + eight) * sixty4 + b_npos_2 + 1], al.u16), al.u32)
                bv6_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + eight) * sixty4 + b_npos_2 + 2], al.u16), al.u32)
                bv7_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + eight) * sixty4 + b_npos_2 + 3], al.u16), al.u32)
                bv10_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + sixteen) * sixty4 + b_npos_2 + 0], al.u16), al.u32)
                bv11_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + sixteen) * sixty4 + b_npos_2 + 1], al.u16), al.u32)
                bv12_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + sixteen) * sixty4 + b_npos_2 + 2], al.u16), al.u32)
                bv13_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + sixteen) * sixty4 + b_npos_2 + 3], al.u16), al.u32)
                bv14_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + al.convert(24, al.i32)) * sixty4 + b_npos_2 + 0], al.u16), al.u32)
                bv15_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + al.convert(24, al.i32)) * sixty4 + b_npos_2 + 1], al.u16), al.u32)
                bv16_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + al.convert(24, al.i32)) * sixty4 + b_npos_2 + 2], al.u16), al.u32)
                bv17_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + al.convert(24, al.i32)) * sixty4 + b_npos_2 + 3], al.u16), al.u32)

                data_a0_2[0] = av0_2 | (av1_2 << sixteen)
                data_a0_2[1] = av2_2 | (av3_2 << sixteen)
                data_a0_2[2] = av4_2 | (av5_2 << sixteen)
                data_a0_2[3] = av6_2 | (av7_2 << sixteen)
                data_a1_2[0] = av10_2 | (av11_2 << sixteen)
                data_a1_2[1] = av12_2 | (av13_2 << sixteen)
                data_a1_2[2] = av14_2 | (av15_2 << sixteen)
                data_a1_2[3] = av16_2 | (av17_2 << sixteen)

                data_b0_2[0] = bv0_2 | (bv1_2 << sixteen)
                data_b0_2[1] = bv2_2 | (bv3_2 << sixteen)
                data_b0_2[2] = bv4_2 | (bv5_2 << sixteen)
                data_b0_2[3] = bv6_2 | (bv7_2 << sixteen)
                data_b1_2[0] = bv10_2 | (bv11_2 << sixteen)
                data_b1_2[1] = bv12_2 | (bv13_2 << sixteen)
                data_b1_2[2] = bv14_2 | (bv15_2 << sixteen)
                data_b1_2[3] = bv16_2 | (bv17_2 << sixteen)

                frag_a0_2 = al.view(data_a0_2, al.Tensor((two, two), al.u32))
                frag_b0_2 = al.view(data_b0_2, al.Tensor((two, two), al.u32))
                acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a0_2[0], frag_b0_2[0], acc)
                acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a0_2[1], frag_b0_2[1], acc)

                frag_a1_2 = al.view(data_a1_2, al.Tensor((two, two), al.u32))
                frag_b1_2 = al.view(data_b1_2, al.Tensor((two, two), al.u32))
                acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a1_2[0], frag_b1_2[0], acc)
                acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a1_2[1], frag_b1_2[1], acc)
            else:
                data_a0_2 = al.make_local((four,), al.u32)
                data_a1_2 = al.make_local((four,), al.u32)
                data_b0_2 = al.make_local((four,), al.u32)
                data_b1_2 = al.make_local((four,), al.u32)

                av0_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 4], al.u16), al.u32)
                av1_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 5], al.u16), al.u32)
                av2_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 6], al.u16), al.u32)
                av3_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 7], al.u16), al.u32)
                av4_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 12], al.u16), al.u32)
                av5_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 13], al.u16), al.u32)
                av6_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 14], al.u16), al.u32)
                av7_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 15], al.u16), al.u32)
                av10_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 20], al.u16), al.u32)
                av11_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 21], al.u16), al.u32)
                av12_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 22], al.u16), al.u32)
                av13_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 23], al.u16), al.u32)
                av14_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 28], al.u16), al.u32)
                av15_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 29], al.u16), al.u32)
                av16_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 30], al.u16), al.u32)
                av17_2 = al.convert(al.bitcast(a_lds_1[a_lds_row_2 * thirty2 + 31], al.u16), al.u32)

                bv0_2 = al.convert(al.bitcast(b_lds_1[(j_val_2) * sixty4 + b_npos_2 + 0], al.u16), al.u32)
                bv1_2 = al.convert(al.bitcast(b_lds_1[(j_val_2) * sixty4 + b_npos_2 + 1], al.u16), al.u32)
                bv2_2 = al.convert(al.bitcast(b_lds_1[(j_val_2) * sixty4 + b_npos_2 + 2], al.u16), al.u32)
                bv3_2 = al.convert(al.bitcast(b_lds_1[(j_val_2) * sixty4 + b_npos_2 + 3], al.u16), al.u32)
                bv4_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + eight) * sixty4 + b_npos_2 + 0], al.u16), al.u32)
                bv5_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + eight) * sixty4 + b_npos_2 + 1], al.u16), al.u32)
                bv6_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + eight) * sixty4 + b_npos_2 + 2], al.u16), al.u32)
                bv7_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + eight) * sixty4 + b_npos_2 + 3], al.u16), al.u32)
                bv10_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + sixteen) * sixty4 + b_npos_2 + 0], al.u16), al.u32)
                bv11_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + sixteen) * sixty4 + b_npos_2 + 1], al.u16), al.u32)
                bv12_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + sixteen) * sixty4 + b_npos_2 + 2], al.u16), al.u32)
                bv13_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + sixteen) * sixty4 + b_npos_2 + 3], al.u16), al.u32)
                bv14_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + al.convert(24, al.i32)) * sixty4 + b_npos_2 + 0], al.u16), al.u32)
                bv15_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + al.convert(24, al.i32)) * sixty4 + b_npos_2 + 1], al.u16), al.u32)
                bv16_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + al.convert(24, al.i32)) * sixty4 + b_npos_2 + 2], al.u16), al.u32)
                bv17_2 = al.convert(al.bitcast(b_lds_1[(j_val_2 + al.convert(24, al.i32)) * sixty4 + b_npos_2 + 3], al.u16), al.u32)

                data_a0_2[0] = av0_2 | (av1_2 << sixteen)
                data_a0_2[1] = av2_2 | (av3_2 << sixteen)
                data_a0_2[2] = av4_2 | (av5_2 << sixteen)
                data_a0_2[3] = av6_2 | (av7_2 << sixteen)
                data_a1_2[0] = av10_2 | (av11_2 << sixteen)
                data_a1_2[1] = av12_2 | (av13_2 << sixteen)
                data_a1_2[2] = av14_2 | (av15_2 << sixteen)
                data_a1_2[3] = av16_2 | (av17_2 << sixteen)

                data_b0_2[0] = bv0_2 | (bv1_2 << sixteen)
                data_b0_2[1] = bv2_2 | (bv3_2 << sixteen)
                data_b0_2[2] = bv4_2 | (bv5_2 << sixteen)
                data_b0_2[3] = bv6_2 | (bv7_2 << sixteen)
                data_b1_2[0] = bv10_2 | (bv11_2 << sixteen)
                data_b1_2[1] = bv12_2 | (bv13_2 << sixteen)
                data_b1_2[2] = bv14_2 | (bv15_2 << sixteen)
                data_b1_2[3] = bv16_2 | (bv17_2 << sixteen)

                frag_a0_2 = al.view(data_a0_2, al.Tensor((two, two), al.u32))
                frag_b0_2 = al.view(data_b0_2, al.Tensor((two, two), al.u32))
                acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a0_2[0], frag_b0_2[0], acc)
                acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a0_2[1], frag_b0_2[1], acc)

                frag_a1_2 = al.view(data_a1_2, al.Tensor((two, two), al.u32))
                frag_b1_2 = al.view(data_b1_2, al.Tensor((two, two), al.u32))
                acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a1_2[0], frag_b1_2[0], acc)
                acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a1_2[1], frag_b1_2[1], acc)

            al.syncthreads()

        # LogSumExp for this N-tile using shuffle reduction
        col_global = n_base + (lane % thirty2)
        bias_val = al.convert(Bias[col_global], al.f32)

        for a in al.range(16):
            r = eight * (a // four) + four * lane_half + (a % four)
            my_val = acc[a] + bias_val

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
            tmax = al.shuffle_xor(max_val, al.convert(1, al.i32), thirty2)
            tmax = al.shuffle_xor(tmax, two, thirty2)
            tmax = al.shuffle_xor(tmax, four, thirty2)
            tmax = al.shuffle_xor(tmax, eight, thirty2)
            tmax = al.shuffle_xor(tmax, al.convert(16, al.i32), thirty2)

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

    # Cross-wave combine — each lane writes only rows it computed (fixes race)
    base_off = warp_m * thirty2 + warp_n * sixty4
    for a in al.range(16):
        r = eight * (a // four) + four * lane_half + (a % four)
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
