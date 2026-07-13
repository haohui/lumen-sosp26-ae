import torch
import torch.nn as nn
import avelang
import avelang.language as al

BATCH_SIZE = 128
IN_FEATURES = 16384
OUT_FEATURES = 16384
DROPOUT_P = 0.2

BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 16
WARP_SIZE = 64
THREADS = WARP_SIZE


@avelang.jit
def matmul_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    y_ptr: al.Pointer(al.bf16),
    M: al.i32,
    K: al.i32,
    N: al.i32,
):
    block_m = al.block_id(1)
    block_n = al.block_id(0)
    tid = al.thread_id(0)
    lane_id = tid % 64

    m_base = block_m * 32
    n_base = block_n * 32

    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    w = al.make_tensor(w_ptr, al.bf16, al.make_layout((K, N), (N, 1)))
    y = al.make_tensor(y_ptr, al.bf16, al.make_layout((M, N), (N, 1)))
    bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((N,), (1,)))

    # Double-buffered LDS
    a0_lds = al.make_shared((256,), al.u32)
    a1_lds = al.make_shared((256,), al.u32)
    b0_lds = al.make_shared((256,), al.u32)
    b1_lds = al.make_shared((256,), al.u32)

    acc = al.make_local((4, 4), al.f32)
    for si in al.range(4):
        for i in al.range(4):
            acc[si, i] = al.convert(0.0, al.f32)

    thread_u32_base = tid * 4

    # ---- PROLOGUE: load tile 0 into buffer 0 ----
    for ui in al.range(4):
        u32_pos = thread_u32_base + ui
        if u32_pos < 256:
            row = u32_pos // 8
            rem = u32_pos % 8
            vnni_lane = rem // 2
            vnni_pair = rem % 2
            col0 = vnni_lane + vnni_pair * 8
            col1 = col0 + 4
            gr = m_base + row
            bf_lo = al.convert(0.0, al.bf16)
            bf_hi = al.convert(0.0, al.bf16)
            if gr < M:
                bf_lo = x[gr, col0]
                bf_hi = x[gr, col1]
            lo_u = al.bitcast(bf_lo, al.u16)
            hi_u = al.bitcast(bf_hi, al.u16)
            a0_lds[u32_pos] = al.convert(lo_u, al.u32) | (al.convert(hi_u, al.u32) << 16)

            n_idx = u32_pos // 8
            k0 = vnni_lane + vnni_pair * 8
            k1 = k0 + 4
            gn = n_base + n_idx
            bf_lo_b = al.convert(0.0, al.bf16)
            bf_hi_b = al.convert(0.0, al.bf16)
            if k0 < K and gn < N:
                bf_lo_b = w[k0, gn]
                bf_hi_b = w[k1, gn]
            lo_u_b = al.bitcast(bf_lo_b, al.u16)
            hi_u_b = al.bitcast(bf_hi_b, al.u16)
            b0_lds[u32_pos] = al.convert(lo_u_b, al.u32) | (al.convert(hi_u_b, al.u32) << 16)

    al.syncthreads()

    # ---- MAIN LOOP: double-buffered, unrolled by 2 ----
    for k_block in al.range(16, K - 16, 32):
        # Phase A: load k_block into buffer 1
        for ui in al.range(4):
            u32_pos = thread_u32_base + ui
            if u32_pos < 256:
                row = u32_pos // 8
                rem = u32_pos % 8
                vnni_lane = rem // 2
                vnni_pair = rem % 2
                col0 = vnni_lane + vnni_pair * 8
                col1 = col0 + 4
                gr = m_base + row
                gc0 = k_block + col0
                gc1 = k_block + col1
                bf_lo = al.convert(0.0, al.bf16)
                bf_hi = al.convert(0.0, al.bf16)
                if gr < M:
                    bf_lo = x[gr, gc0]
                    bf_hi = x[gr, gc1]
                lo_u = al.bitcast(bf_lo, al.u16)
                hi_u = al.bitcast(bf_hi, al.u16)
                a1_lds[u32_pos] = al.convert(lo_u, al.u32) | (al.convert(hi_u, al.u32) << 16)

                n_idx = u32_pos // 8
                k0 = vnni_lane + vnni_pair * 8
                k1 = k0 + 4
                gk0 = k_block + k0
                gk1 = k_block + k1
                gn = n_base + n_idx
                bf_lo_b = al.convert(0.0, al.bf16)
                bf_hi_b = al.convert(0.0, al.bf16)
                if gk0 < K and gn < N:
                    bf_lo_b = w[gk0, gn]
                    bf_hi_b = w[gk1, gn]
                lo_u_b = al.bitcast(bf_lo_b, al.u16)
                hi_u_b = al.bitcast(bf_hi_b, al.u16)
                b1_lds[u32_pos] = al.convert(lo_u_b, al.u32) | (al.convert(hi_u_b, al.u32) << 16)

        al.syncthreads()

        # Compute tile (k_block-16) from buffer 0
        for si in al.range(4):
            sub_m = si // 2
            sub_n = si % 2
            a_row = sub_m * 16 + lane_id // 4
            a_off = a_row * 8 + (lane_id % 4) * 2
            a_packed = al.make_local((1, 2), al.u32)
            a_packed[0, 0] = a0_lds[a_off + 0]
            a_packed[0, 1] = a0_lds[a_off + 1]
            b_col = sub_n * 16 + lane_id // 4
            b_off = b_col * 8 + (lane_id % 4) * 2
            b_packed = al.make_local((1, 2), al.u32)
            b_packed[0, 0] = b0_lds[b_off + 0]
            b_packed[0, 1] = b0_lds[b_off + 1]
            acc[si] = al.amdgpu.mfma_16x16x16_bf16_f32(a_packed[0], b_packed[0], acc[si])

        # Phase B: load (k_block+16) into buffer 0
        k_next = k_block + 16
        for ui in al.range(4):
            u32_pos = thread_u32_base + ui
            if u32_pos < 256:
                row = u32_pos // 8
                rem = u32_pos % 8
                vnni_lane = rem // 2
                vnni_pair = rem % 2
                col0 = vnni_lane + vnni_pair * 8
                col1 = col0 + 4
                gr = m_base + row
                gc0 = k_next + col0
                gc1 = k_next + col1
                bf_lo = al.convert(0.0, al.bf16)
                bf_hi = al.convert(0.0, al.bf16)
                if gr < M:
                    bf_lo = x[gr, gc0]
                    bf_hi = x[gr, gc1]
                lo_u = al.bitcast(bf_lo, al.u16)
                hi_u = al.bitcast(bf_hi, al.u16)
                a0_lds[u32_pos] = al.convert(lo_u, al.u32) | (al.convert(hi_u, al.u32) << 16)

                n_idx = u32_pos // 8
                k0 = vnni_lane + vnni_pair * 8
                k1 = k0 + 4
                gk0 = k_next + k0
                gk1 = k_next + k1
                gn = n_base + n_idx
                bf_lo_b = al.convert(0.0, al.bf16)
                bf_hi_b = al.convert(0.0, al.bf16)
                if gk0 < K and gn < N:
                    bf_lo_b = w[gk0, gn]
                    bf_hi_b = w[gk1, gn]
                lo_u_b = al.bitcast(bf_lo_b, al.u16)
                hi_u_b = al.bitcast(bf_hi_b, al.u16)
                b0_lds[u32_pos] = al.convert(lo_u_b, al.u32) | (al.convert(hi_u_b, al.u32) << 16)

        al.syncthreads()

        # Compute tile k_block from buffer 1
        for si in al.range(4):
            sub_m = si // 2
            sub_n = si % 2
            a_row = sub_m * 16 + lane_id // 4
            a_off = a_row * 8 + (lane_id % 4) * 2
            a_packed = al.make_local((1, 2), al.u32)
            a_packed[0, 0] = a1_lds[a_off + 0]
            a_packed[0, 1] = a1_lds[a_off + 1]
            b_col = sub_n * 16 + lane_id // 4
            b_off = b_col * 8 + (lane_id % 4) * 2
            b_packed = al.make_local((1, 2), al.u32)
            b_packed[0, 0] = b1_lds[b_off + 0]
            b_packed[0, 1] = b1_lds[b_off + 1]
            acc[si] = al.amdgpu.mfma_16x16x16_bf16_f32(a_packed[0], b_packed[0], acc[si])

    # ---- EPILOGUE: last two tiles ----
    # 1) Compute penultimate tile from buffer 0
    al.syncthreads()
    for si in al.range(4):
        sub_m = si // 2
        sub_n = si % 2
        a_row = sub_m * 16 + lane_id // 4
        a_off = a_row * 8 + (lane_id % 4) * 2
        a_packed = al.make_local((1, 2), al.u32)
        a_packed[0, 0] = a0_lds[a_off + 0]
        a_packed[0, 1] = a0_lds[a_off + 1]
        b_col = sub_n * 16 + lane_id // 4
        b_off = b_col * 8 + (lane_id % 4) * 2
        b_packed = al.make_local((1, 2), al.u32)
        b_packed[0, 0] = b0_lds[b_off + 0]
        b_packed[0, 1] = b0_lds[b_off + 1]
        acc[si] = al.amdgpu.mfma_16x16x16_bf16_f32(a_packed[0], b_packed[0], acc[si])

    # 2) Load and compute final tile into buffer 1
    k_epi = K - 16
    for ui in al.range(4):
        u32_pos = thread_u32_base + ui
        if u32_pos < 256:
            row = u32_pos // 8
            rem = u32_pos % 8
            vnni_lane = rem // 2
            vnni_pair = rem % 2
            col0 = vnni_lane + vnni_pair * 8
            col1 = col0 + 4
            gr = m_base + row
            gc0 = k_epi + col0
            gc1 = k_epi + col1
            bf_lo = al.convert(0.0, al.bf16)
            bf_hi = al.convert(0.0, al.bf16)
            if gr < M:
                bf_lo = x[gr, gc0]
                bf_hi = x[gr, gc1]
            lo_u = al.bitcast(bf_lo, al.u16)
            hi_u = al.bitcast(bf_hi, al.u16)
            a1_lds[u32_pos] = al.convert(lo_u, al.u32) | (al.convert(hi_u, al.u32) << 16)

            n_idx = u32_pos // 8
            k0 = vnni_lane + vnni_pair * 8
            k1 = k0 + 4
            gk0 = k_epi + k0
            gk1 = k_epi + k1
            gn = n_base + n_idx
            bf_lo_b = al.convert(0.0, al.bf16)
            bf_hi_b = al.convert(0.0, al.bf16)
            if gk0 < K and gn < N:
                bf_lo_b = w[gk0, gn]
                bf_hi_b = w[gk1, gn]
            lo_u_b = al.bitcast(bf_lo_b, al.u16)
            hi_u_b = al.bitcast(bf_hi_b, al.u16)
            b1_lds[u32_pos] = al.convert(lo_u_b, al.u32) | (al.convert(hi_u_b, al.u32) << 16)

    al.syncthreads()

    for si in al.range(4):
        sub_m = si // 2
        sub_n = si % 2
        a_row = sub_m * 16 + lane_id // 4
        a_off = a_row * 8 + (lane_id % 4) * 2
        a_packed = al.make_local((1, 2), al.u32)
        a_packed[0, 0] = a1_lds[a_off + 0]
        a_packed[0, 1] = a1_lds[a_off + 1]
        b_col = sub_n * 16 + lane_id // 4
        b_off = b_col * 8 + (lane_id % 4) * 2
        b_packed = al.make_local((1, 2), al.u32)
        b_packed[0, 0] = b1_lds[b_off + 0]
        b_packed[0, 1] = b1_lds[b_off + 1]
        acc[si] = al.amdgpu.mfma_16x16x16_bf16_f32(a_packed[0], b_packed[0], acc[si])

    # ---- STORE ----
    for si in al.range(4):
        sub_m = si // 2
        sub_n = si % 2
        for i in al.range(4):
            out_row = sub_m * 16 + (lane_id // 4)
            out_col = sub_n * 16 + (lane_id % 4) * 4 + i
            global_m = m_base + out_row
            global_n = n_base + out_col
            if global_m < M and global_n < N:
                val = acc[si, i] + al.convert(bias[global_n], al.f32)
                y[global_m, global_n] = al.convert(val, al.bf16)


@avelang.jit
def softmax_kernel(
    inp_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
):
    row = al.block_id(0)
    tid = al.thread_id(0)

    inp_layout = al.make_layout((N,), (1,))
    inp = al.make_tensor(inp_ptr, al.bf16, inp_layout)
    out_layout = al.make_layout((N,), (1,))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    shared = al.make_shared((256,), al.f32)

    thread_max = al.convert(-1e+30, al.f32)
    for i in al.range(tid, N, 256):
        v = al.convert(inp[i], al.f32)
        if v > thread_max:
            thread_max = v

    shared[tid] = thread_max
    al.syncthreads()

    if tid < 128:
        other = shared[tid + 128]
        if other > shared[tid]:
            shared[tid] = other
    al.syncthreads()
    if tid < 64:
        other = shared[tid + 64]
        if other > shared[tid]:
            shared[tid] = other
    al.syncthreads()
    if tid < 32:
        other = shared[tid + 32]
        if other > shared[tid]:
            shared[tid] = other
    al.syncthreads()
    if tid < 16:
        other = shared[tid + 16]
        if other > shared[tid]:
            shared[tid] = other
    al.syncthreads()
    if tid < 8:
        other = shared[tid + 8]
        if other > shared[tid]:
            shared[tid] = other
    al.syncthreads()
    if tid < 4:
        other = shared[tid + 4]
        if other > shared[tid]:
            shared[tid] = other
    al.syncthreads()
    if tid < 2:
        other = shared[tid + 2]
        if other > shared[tid]:
            shared[tid] = other
    al.syncthreads()
    if tid < 1:
        other = shared[tid + 1]
        if other > shared[tid]:
            shared[tid] = other
    al.syncthreads()
    row_max = shared[0]

    thread_sum = al.convert(0.0, al.f32)
    for i in al.range(tid, N, 256):
        thread_sum = thread_sum + al.exp(al.convert(inp[i], al.f32) - row_max)

    shared[tid] = thread_sum
    al.syncthreads()

    if tid < 128:
        shared[tid] = shared[tid] + shared[tid + 128]
    al.syncthreads()
    if tid < 64:
        shared[tid] = shared[tid] + shared[tid + 64]
    al.syncthreads()
    if tid < 32:
        shared[tid] = shared[tid] + shared[tid + 32]
    al.syncthreads()
    if tid < 16:
        shared[tid] = shared[tid] + shared[tid + 16]
    al.syncthreads()
    if tid < 8:
        shared[tid] = shared[tid] + shared[tid + 8]
    al.syncthreads()
    if tid < 4:
        shared[tid] = shared[tid] + shared[tid + 4]
    al.syncthreads()
    if tid < 2:
        shared[tid] = shared[tid] + shared[tid + 2]
    al.syncthreads()
    if tid < 1:
        shared[tid] = shared[tid] + shared[tid + 1]
    al.syncthreads()
    row_sum = shared[0]

    for i in al.range(tid, N, 256):
        val = al.exp(al.convert(inp[i], al.f32) - row_max) / row_sum
        out[i] = al.convert(val, al.bf16)


class ModelNew(nn.Module):

    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)
        self._cached_w_ptr = None
        self._cached_bias_ptr = None
        self._w_t = None
        self._bias = None

    def forward(self, x):
        device = torch.device('cuda')
        M_val = x.shape[0]
        K_val = x.shape[1]
        N_val = self.matmul.out_features

        w = self.matmul.weight
        bias = self.matmul.bias

        w_data_ptr = w.data_ptr()
        bias_data_ptr = bias.data_ptr()
        if self._cached_w_ptr != w_data_ptr:
            self._w_t = w.t().to(device=device, dtype=torch.bfloat16).contiguous()
            self._cached_w_ptr = w_data_ptr
        if self._cached_bias_ptr != bias_data_ptr:
            self._bias = bias.to(device=device, dtype=torch.bfloat16).contiguous()
            self._cached_bias_ptr = bias_data_ptr

        x_dev = x.to(device=device, dtype=torch.bfloat16).contiguous()
        y_tmp = torch.empty((M_val, N_val), device=device, dtype=torch.bfloat16)
        y_out = torch.empty((M_val, N_val), device=device, dtype=torch.bfloat16)

        grid_m = (M_val + BLOCK_M - 1) // BLOCK_M
        grid_n = (N_val + BLOCK_N - 1) // BLOCK_N

        matmul_kernel[lambda: ((grid_n, grid_m, 1), (THREADS, 1, 1))](
            x_dev, self._w_t, self._bias, y_tmp, M_val, K_val, N_val,
        )
        y_tmp = self.dropout(y_tmp)

        softmax_kernel[lambda: ((M_val, 1, 1), (256, 1, 1))](
            y_tmp, y_out, N_val,
        )

        return y_out
