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

    # Build resource descriptors with byte ranges so OOB loads return 0
    # and OOB stores are discarded -- no explicit guards needed.
    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    x_rsrc = al.amdgpu.make_rsrc(x, M * K * 2)

    w = al.make_tensor(w_ptr, al.bf16, al.make_layout((K, N), (N, 1)))
    w_rsrc = al.amdgpu.make_rsrc(w, K * N * 2)

    y = al.make_tensor(y_ptr, al.bf16, al.make_layout((M, N), (N, 1)))
    y_rsrc = al.amdgpu.make_rsrc(y, M * N * 2)

    bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((N,), (1,)))
    bias_rsrc = al.amdgpu.make_rsrc(bias, N * 2)

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

    # ---- PROLOGUE: load tile k=0 into buffer 0 ----
    # No OOB branches: resource-descriptor range handles out-of-bounds.
    for ui in al.range(4):
        u32_pos = thread_u32_base + ui
        row = u32_pos // 8
        rem = u32_pos % 8
        vnni_lane = rem // 2
        vnni_pair = rem % 2
        col0 = vnni_lane + vnni_pair * 8
        col1 = col0 + 4
        gr = m_base + row

        # A load -- byte offsets into x
        boff_a_lo = (gr * K + col0) * 2
        boff_a_hi = (gr * K + col1) * 2
        lo_a = al.amdgpu.raw_buffer_load_x1(x_rsrc, boff_a_lo, 0, 0)
        hi_a = al.amdgpu.raw_buffer_load_x1(x_rsrc, boff_a_hi, 0, 0)
        a0_lds[u32_pos] = al.convert(al.convert(lo_a, al.u16), al.u32) | (al.convert(al.convert(hi_a, al.u16), al.u32) << 16)

        # B load -- byte offsets into w^T (K x N layout)
        n_idx = u32_pos // 8
        k0 = vnni_lane + vnni_pair * 8
        k1 = k0 + 4
        gn = n_base + n_idx
        boff_b_lo = (k0 * N + gn) * 2
        boff_b_hi = (k1 * N + gn) * 2
        lo_b = al.amdgpu.raw_buffer_load_x1(w_rsrc, boff_b_lo, 0, 0)
        hi_b = al.amdgpu.raw_buffer_load_x1(w_rsrc, boff_b_hi, 0, 0)
        b0_lds[u32_pos] = al.convert(al.convert(lo_b, al.u16), al.u32) | (al.convert(al.convert(hi_b, al.u16), al.u32) << 16)

    al.syncthreads()

    # ---- MAIN LOOP: double-buffered, unrolled by 2 ----
    for k_block in al.range(16, K - 16, 32):
        # Phase A: load tile k_block into buffer 1
        for ui in al.range(4):
            u32_pos = thread_u32_base + ui
            row = u32_pos // 8
            rem = u32_pos % 8
            vnni_lane = rem // 2
            vnni_pair = rem % 2
            col0 = vnni_lane + vnni_pair * 8
            col1 = col0 + 4
            gr = m_base + row
            gc0 = k_block + col0
            gc1 = k_block + col1

            boff_a_lo = (gr * K + gc0) * 2
            boff_a_hi = (gr * K + gc1) * 2
            lo_a = al.amdgpu.raw_buffer_load_x1(x_rsrc, boff_a_lo, 0, 0)
            hi_a = al.amdgpu.raw_buffer_load_x1(x_rsrc, boff_a_hi, 0, 0)
            a1_lds[u32_pos] = al.convert(al.convert(lo_a, al.u16), al.u32) | (al.convert(al.convert(hi_a, al.u16), al.u32) << 16)

            n_idx = u32_pos // 8
            k0 = vnni_lane + vnni_pair * 8
            k1 = k0 + 4
            gk0 = k_block + k0
            gk1 = k_block + k1
            gn = n_base + n_idx
            boff_b_lo = (gk0 * N + gn) * 2
            boff_b_hi = (gk1 * N + gn) * 2
            lo_b = al.amdgpu.raw_buffer_load_x1(w_rsrc, boff_b_lo, 0, 0)
            hi_b = al.amdgpu.raw_buffer_load_x1(w_rsrc, boff_b_hi, 0, 0)
            b1_lds[u32_pos] = al.convert(al.convert(lo_b, al.u16), al.u32) | (al.convert(al.convert(hi_b, al.u16), al.u32) << 16)

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

        # Phase B: load tile (k_block+16) into buffer 0
        k_next = k_block + 16
        for ui in al.range(4):
            u32_pos = thread_u32_base + ui
            row = u32_pos // 8
            rem = u32_pos % 8
            vnni_lane = rem // 2
            vnni_pair = rem % 2
            col0 = vnni_lane + vnni_pair * 8
            col1 = col0 + 4
            gr = m_base + row
            gc0 = k_next + col0
            gc1 = k_next + col1

            boff_a_lo = (gr * K + gc0) * 2
            boff_a_hi = (gr * K + gc1) * 2
            lo_a = al.amdgpu.raw_buffer_load_x1(x_rsrc, boff_a_lo, 0, 0)
            hi_a = al.amdgpu.raw_buffer_load_x1(x_rsrc, boff_a_hi, 0, 0)
            a0_lds[u32_pos] = al.convert(al.convert(lo_a, al.u16), al.u32) | (al.convert(al.convert(hi_a, al.u16), al.u32) << 16)

            n_idx = u32_pos // 8
            k0 = vnni_lane + vnni_pair * 8
            k1 = k0 + 4
            gk0 = k_next + k0
            gk1 = k_next + k1
            gn = n_base + n_idx
            boff_b_lo = (gk0 * N + gn) * 2
            boff_b_hi = (gk1 * N + gn) * 2
            lo_b = al.amdgpu.raw_buffer_load_x1(w_rsrc, boff_b_lo, 0, 0)
            hi_b = al.amdgpu.raw_buffer_load_x1(w_rsrc, boff_b_hi, 0, 0)
            b0_lds[u32_pos] = al.convert(al.convert(lo_b, al.u16), al.u32) | (al.convert(al.convert(hi_b, al.u16), al.u32) << 16)

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

    # ---- EPILOGUE: compute penultimate tile from buffer 0 ----
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

    # ---- Load final tile (K-16) into buffer 1 ----
    k_epi = K - 16
    for ui in al.range(4):
        u32_pos = thread_u32_base + ui
        row = u32_pos // 8
        rem = u32_pos % 8
        vnni_lane = rem // 2
        vnni_pair = rem % 2
        col0 = vnni_lane + vnni_pair * 8
        col1 = col0 + 4
        gr = m_base + row
        gc0 = k_epi + col0
        gc1 = k_epi + col1

        boff_a_lo = (gr * K + gc0) * 2
        boff_a_hi = (gr * K + gc1) * 2
        lo_a = al.amdgpu.raw_buffer_load_x1(x_rsrc, boff_a_lo, 0, 0)
        hi_a = al.amdgpu.raw_buffer_load_x1(x_rsrc, boff_a_hi, 0, 0)
        a1_lds[u32_pos] = al.convert(al.convert(lo_a, al.u16), al.u32) | (al.convert(al.convert(hi_a, al.u16), al.u32) << 16)

        n_idx = u32_pos // 8
        k0 = vnni_lane + vnni_pair * 8
        k1 = k0 + 4
        gk0 = k_epi + k0
        gk1 = k_epi + k1
        gn = n_base + n_idx
        boff_b_lo = (gk0 * N + gn) * 2
        boff_b_hi = (gk1 * N + gn) * 2
        lo_b = al.amdgpu.raw_buffer_load_x1(w_rsrc, boff_b_lo, 0, 0)
        hi_b = al.amdgpu.raw_buffer_load_x1(w_rsrc, boff_b_hi, 0, 0)
        b1_lds[u32_pos] = al.convert(al.convert(lo_b, al.u16), al.u32) | (al.convert(al.convert(hi_b, al.u16), al.u32) << 16)

    al.syncthreads()

    # ---- Compute final tile from buffer 1 ----
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

    # ---- STORE: use raw_buffer_store_x2, OOB stores are discarded ----
    for si in al.range(4):
        sub_m = si // 2
        sub_n = si % 2
        out_row = sub_m * 16 + (lane_id // 4)
        out_col_base = sub_n * 16 + (lane_id % 4) * 4
        global_m = m_base + out_row
        global_n = n_base + out_col_base

        # Load 4 bias values via raw buffer (2 u32 loads = 4 bf16)
        boff_bias = global_n * 2
        bias_u = al.amdgpu.raw_buffer_load_x1(bias_rsrc, boff_bias, 0, 0)
        bias_v = al.amdgpu.raw_buffer_load_x1(bias_rsrc, boff_bias + 4, 0, 0)

        # Extract bf16: truncate u32->u16 keeps lower bits, then bitcast to bf16
        b0 = al.convert(al.bitcast(al.convert(bias_u, al.u16), al.bf16), al.f32)
        b1 = al.convert(al.bitcast(al.convert(bias_u >> 16, al.u16), al.bf16), al.f32)
        b2 = al.convert(al.bitcast(al.convert(bias_v, al.u16), al.bf16), al.f32)
        b3 = al.convert(al.bitcast(al.convert(bias_v >> 16, al.u16), al.bf16), al.f32)

        # Accumulate + bias -> bf16 -> pack into 2 u32s
        v0 = al.convert(acc[si, 0] + b0, al.bf16)
        v1 = al.convert(acc[si, 1] + b1, al.bf16)
        v2 = al.convert(acc[si, 2] + b2, al.bf16)
        v3 = al.convert(acc[si, 3] + b3, al.bf16)

        u0 = al.bitcast(v0, al.u16)
        u1 = al.bitcast(v1, al.u16)
        u2 = al.bitcast(v2, al.u16)
        u3 = al.bitcast(v3, al.u16)

        w0 = al.convert(u0, al.u32) | (al.convert(u1, al.u32) << 16)
        w1 = al.convert(u2, al.u32) | (al.convert(u3, al.u32) << 16)

        # Store 4 bf16 values as one x2 store
        store_vec = al.make_local((2,), al.u32)
        store_vec[0] = w0
        store_vec[1] = w1

        boff_y = (global_m * N + global_n) * 2
        al.amdgpu.raw_buffer_store_x2(store_vec, y_rsrc, boff_y, 0, 0)


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
