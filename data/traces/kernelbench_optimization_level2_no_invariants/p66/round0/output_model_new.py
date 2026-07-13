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

    x_layout = al.make_layout((M, K), (K, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    w_layout = al.make_layout((K, N), (N, 1))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)
    y_layout = al.make_layout((M, N), (N, 1))
    y = al.make_tensor(y_ptr, al.bf16, y_layout)
    bias_layout = al.make_layout((N,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    # LDS stored as u32 for explicit packing
    a_lds = al.make_shared((256,), al.u32)
    b_lds = al.make_shared((256,), al.u32)

    acc = al.make_local((4, 4), al.f32)
    for si in al.range(4):
        for i in al.range(4):
            acc[si, i] = al.convert(0.0, al.f32)

    for k_block in al.range(0, K, 16):
        # Load A: pack bf16 pairs into u32 with VNNI interleaving
        # Each thread loads 512/64 = 8 bf16 = 4 u32 values from global into LDS
        # Row r (0-31), bf16 col c (0-15): bf16 pair at (c, c+4) packed into u32 at VNNI position
        # VNNI layout: u32[row*8 + lane*2 + pair] contains bf16(col=lane+pair*4, col=lane+pair*4+4)
        # Simpler: each thread handles 4 u32 positions in LDS
        thread_u32_base = tid * 4
        for ui in al.range(4):
            u32_pos = thread_u32_base + ui
            if u32_pos < 256:
                # Decode u32_pos to (row, lane, pair)
                row = u32_pos // 8
                remainder = u32_pos % 8
                vnni_lane = remainder // 2  # 0-3
                vnni_pair = remainder % 2  # 0-1
                # This u32 contains bf16 at K positions: vnni_lane + vnni_pair*4 and vnni_lane+4 + vnni_pair*4
                col0 = vnni_lane + vnni_pair * 4
                col1 = col0 + 4
                gr = m_base + row
                gc0 = k_block + col0
                gc1 = k_block + col1
                bf16_lo = al.convert(0.0, al.bf16)
                bf16_hi = al.convert(0.0, al.bf16)
                if gr < M:
                    bf16_lo = x[gr, gc0]
                    bf16_hi = x[gr, gc1]
                # Pack into u32: lo in lower 16 bits, hi in upper 16 bits
                lo_u32 = al.bitcast(bf16_lo, al.u16)
                hi_u32 = al.bitcast(bf16_hi, al.u16)
                packed = al.convert(lo_u32, al.u32) | (al.convert(hi_u32, al.u32) << 16)
                a_lds[u32_pos] = packed

        # Load B: pack bf16 pairs into u32, column-major with VNNI on K
        # B LDS: col n (0-31), VNNI lane 0-3, pair 0-1
        for ui in al.range(4):
            u32_pos = thread_u32_base + ui
            if u32_pos < 256:
                n = u32_pos // 8
                remainder = u32_pos % 8
                vnni_lane = remainder // 2
                vnni_pair = remainder % 2
                k0 = vnni_lane + vnni_pair * 4
                k1 = k0 + 4
                gk0 = k_block + k0
                gk1 = k_block + k1
                gn = n_base + n
                bf16_lo = al.convert(0.0, al.bf16)
                bf16_hi = al.convert(0.0, al.bf16)
                if gk0 < K and gn < N:
                    bf16_lo = w[gk0, gn]
                    bf16_hi = w[gk1, gn]
                lo_u32_b = al.bitcast(bf16_lo, al.u16)
                hi_u32_b = al.bitcast(bf16_hi, al.u16)
                packed_b = al.convert(lo_u32_b, al.u32) | (al.convert(hi_u32_b, al.u32) << 16)
                b_lds[u32_pos] = packed_b

        al.syncthreads()

        # 4 MFMA calls: 2×2 sub-tiles
        for si in al.range(4):
            sub_m = si // 2
            sub_n = si % 2

            a_row = sub_m * 16 + lane_id // 4
            a_u32_off = a_row * 8 + (lane_id % 4) * 2
            a_packed = al.make_local((1, 2), al.u32)
            a_packed[0, 0] = a_lds[a_u32_off + 0]
            a_packed[0, 1] = a_lds[a_u32_off + 1]

            b_col = sub_n * 16 + lane_id // 4
            b_u32_off = b_col * 8 + (lane_id % 4) * 2
            b_packed = al.make_local((1, 2), al.u32)
            b_packed[0, 0] = b_lds[b_u32_off + 0]
            b_packed[0, 1] = b_lds[b_u32_off + 1]

            acc[si] = al.amdgpu.mfma_16x16x16_bf16_f32(a_packed[0], b_packed[0], acc[si])

        al.syncthreads()

    for si in al.range(4):
        sub_m = si // 2
        sub_n = si % 2
        for i in al.range(4):
            out_row = sub_m * 16 + (lane_id // 16) * 4 + i
            out_col = sub_n * 16 + (lane_id % 16)
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
        M_val = x.shape[0]
        K_val = x.shape[1]
        N_val = self.matmul.out_features

        w = self.matmul.weight
        bias = self.matmul.bias

        w_data_ptr = w.data_ptr()
        bias_data_ptr = bias.data_ptr()
        if self._cached_w_ptr != w_data_ptr:
            self._w_t = w.t().to(device=x.device, dtype=torch.bfloat16).contiguous()
            self._cached_w_ptr = w_data_ptr
        if self._cached_bias_ptr != bias_data_ptr:
            self._bias = bias.to(device=x.device, dtype=torch.bfloat16).contiguous()
            self._cached_bias_ptr = bias_data_ptr

        x_dev = x.contiguous()
        y_tmp = torch.empty((M_val, N_val), device=x.device, dtype=torch.bfloat16)
        y_out = torch.empty((M_val, N_val), device=x.device, dtype=torch.bfloat16)

        grid_m = (M_val + BLOCK_M - 1) // BLOCK_M
        grid_n = (N_val + BLOCK_N - 1) // BLOCK_N

        matmul_kernel[lambda: ((grid_n, grid_m, 1), (THREADS, 1, 1))](
            x_dev, self._w_t, self._bias, y_tmp, M_val, K_val, N_val,
        )

        softmax_kernel[lambda: ((M_val, 1, 1), (256, 1, 1))](
            y_tmp, y_out, N_val,
        )

        return y_out
