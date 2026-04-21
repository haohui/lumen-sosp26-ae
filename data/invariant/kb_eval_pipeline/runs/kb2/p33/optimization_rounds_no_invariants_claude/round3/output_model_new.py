import torch
import torch.nn as nn
import substrate
import substrate.language as S

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
EPS = 1e-5

TILE_M = 32
TILE_N = 32
TILE_K = 8
BLOCK_K = 16
WARP_SIZE = 64
BLOCK_SIZE = 64

# Range in bytes for OOB handling
X_RANGE = BATCH_SIZE * IN_FEATURES * 2
W_RANGE = IN_FEATURES * OUT_FEATURES * 2


@substrate.jit
def gemm_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((IN_FEATURES, OUT_FEATURES), S.bf16),
    BIAS0: S.Tensor((OUT_FEATURES,), S.bf16),
    SCALE: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    bx = S.block_id(0)
    by = S.block_id(1)
    tid = S.thread_id(0)

    lane = tid

    block_row_base = by * TILE_M
    block_col_base = bx * TILE_N

    # Create resource descriptors with range for OOB handling
    rsrc_X = S.amdgpu.make_rsrc(X, X_RANGE)
    rsrc_W = S.amdgpu.make_rsrc(W, W_RANGE)

    # Double-buffered shared memory for software pipelining
    A_shared_0 = S.make_shared((TILE_M, BLOCK_K), S.bf16)
    A_shared_1 = S.make_shared((TILE_M, BLOCK_K), S.bf16)
    B_shared_0 = S.make_shared((BLOCK_K, TILE_N), S.bf16)
    B_shared_1 = S.make_shared((BLOCK_K, TILE_N), S.bf16)

    # Fragment staging area for MFMA input
    A_frag = S.make_shared((64, 2), S.u32)
    B_frag = S.make_shared((64, 2), S.u32)

    acc = S.full((16,), 0.0, S.f32)

    num_k_tiles = IN_FEATURES // BLOCK_K  # 512

    # Preload first tile into buffer 0 using raw_buffer_load_x4
    k_base = 0
    elem_idx = tid * 8
    row = elem_idx // BLOCK_K
    col = elem_idx % BLOCK_K
    global_row = block_row_base + row
    global_col = k_base + col
    byte_offset = global_row * IN_FEATURES * 2 + global_col * 2
    data = S.amdgpu.raw_buffer_load_x4(rsrc_X, byte_offset, 0, 0)
    data_bf16 = S.view(data, S.Tensor((8,), S.bf16))
    for j in S.range(8):
        A_shared_0[row, col + j] = data_bf16[j]

    elem_idx = tid * 8
    row = elem_idx // TILE_N
    col = elem_idx % TILE_N
    global_row = k_base + row
    global_col = block_col_base + col
    byte_offset = global_row * OUT_FEATURES * 2 + global_col * 2
    data = S.amdgpu.raw_buffer_load_x4(rsrc_W, byte_offset, 0, 0)
    data_bf16 = S.view(data, S.Tensor((8,), S.bf16))
    for j in S.range(8):
        B_shared_0[row, col + j] = data_bf16[j]

    S.syncthreads()

    # Software pipelined main loop with K-loop unrolled by 2
    # Process pairs of K-tiles: load next pair while computing current pair
    num_k_pairs = num_k_tiles // 2  # 256

    for k_pair in S.range(num_k_pairs):
        k_odd = k_pair * 2 + 1
        k_next_even = (k_pair + 1) * 2

        k_base_odd = k_odd * BLOCK_K
        k_base_next_even = k_next_even * BLOCK_K

        # Load odd tile into buffer 1 (overlapping with compute)
        elem_idx = tid * 8
        row = elem_idx // BLOCK_K
        col = elem_idx % BLOCK_K
        global_row = block_row_base + row
        global_col = k_base_odd + col
        byte_offset = global_row * IN_FEATURES * 2 + global_col * 2
        data = S.amdgpu.raw_buffer_load_x4(rsrc_X, byte_offset, 0, 0)
        data_bf16 = S.view(data, S.Tensor((8,), S.bf16))
        for j in S.range(8):
            A_shared_1[row, col + j] = data_bf16[j]

        elem_idx = tid * 8
        row = elem_idx // TILE_N
        col = elem_idx % TILE_N
        global_row = k_base_odd + row
        global_col = block_col_base + col
        byte_offset = global_row * OUT_FEATURES * 2 + global_col * 2
        data = S.amdgpu.raw_buffer_load_x4(rsrc_W, byte_offset, 0, 0)
        data_bf16 = S.view(data, S.Tensor((8,), S.bf16))
        for j in S.range(8):
            B_shared_1[row, col + j] = data_bf16[j]

        # Compute on buffer 0 (even tile)
        for k_sub in S.range(2):
            k_offset = k_sub * TILE_K

            a_row_lds = lane % 32
            a_col_lds = k_offset + (lane // 32) * 4

            a_bf16_0 = A_shared_0[a_row_lds, a_col_lds + 0]
            a_bf16_1 = A_shared_0[a_row_lds, a_col_lds + 1]
            a_bf16_2 = A_shared_0[a_row_lds, a_col_lds + 2]
            a_bf16_3 = A_shared_0[a_row_lds, a_col_lds + 3]

            a_u16_0 = S.bitcast(a_bf16_0, S.u16)
            a_u16_1 = S.bitcast(a_bf16_1, S.u16)
            a_u16_2 = S.bitcast(a_bf16_2, S.u16)
            a_u16_3 = S.bitcast(a_bf16_3, S.u16)

            a_u32_0 = a_u16_0 | (a_u16_1 << 16)
            a_u32_1 = a_u16_2 | (a_u16_3 << 16)

            A_frag[lane, 0] = a_u32_0
            A_frag[lane, 1] = a_u32_1

            b_row_lds = k_offset + (lane // 32) * 4
            b_col_lds = lane % 32

            b_bf16_0 = B_shared_0[b_row_lds + 0, b_col_lds]
            b_bf16_1 = B_shared_0[b_row_lds + 1, b_col_lds]
            b_bf16_2 = B_shared_0[b_row_lds + 2, b_col_lds]
            b_bf16_3 = B_shared_0[b_row_lds + 3, b_col_lds]

            b_u16_0 = S.bitcast(b_bf16_0, S.u16)
            b_u16_1 = S.bitcast(b_bf16_1, S.u16)
            b_u16_2 = S.bitcast(b_bf16_2, S.u16)
            b_u16_3 = S.bitcast(b_bf16_3, S.u16)

            b_u32_0 = b_u16_0 | (b_u16_1 << 16)
            b_u32_1 = b_u16_2 | (b_u16_3 << 16)

            B_frag[lane, 0] = b_u32_0
            B_frag[lane, 1] = b_u32_1

            S.syncthreads()

            a_view = S.view(A_frag[lane], S.Tensor((1, 4, 1), S.bf16))
            b_view = S.view(B_frag[lane], S.Tensor((1, 4, 1), S.bf16))

            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_view[0], b_view[0], acc)

        S.syncthreads()

        # Load next even tile into buffer 0 (for next iteration)
        # No branch needed: range in make_rsrc handles OOB - returns 0 for OOB elements
        elem_idx = tid * 8
        row = elem_idx // BLOCK_K
        col = elem_idx % BLOCK_K
        global_row = block_row_base + row
        global_col = k_base_next_even + col
        byte_offset = global_row * IN_FEATURES * 2 + global_col * 2
        data = S.amdgpu.raw_buffer_load_x4(rsrc_X, byte_offset, 0, 0)
        data_bf16 = S.view(data, S.Tensor((8,), S.bf16))
        for j in S.range(8):
            A_shared_0[row, col + j] = data_bf16[j]

        elem_idx = tid * 8
        row = elem_idx // TILE_N
        col = elem_idx % TILE_N
        global_row = k_base_next_even + row
        global_col = block_col_base + col
        byte_offset = global_row * OUT_FEATURES * 2 + global_col * 2
        data = S.amdgpu.raw_buffer_load_x4(rsrc_W, byte_offset, 0, 0)
        data_bf16 = S.view(data, S.Tensor((8,), S.bf16))
        for j in S.range(8):
            B_shared_0[row, col + j] = data_bf16[j]

        # Compute on buffer 1 (odd tile)
        for k_sub in S.range(2):
            k_offset = k_sub * TILE_K

            a_row_lds = lane % 32
            a_col_lds = k_offset + (lane // 32) * 4

            a_bf16_0 = A_shared_1[a_row_lds, a_col_lds + 0]
            a_bf16_1 = A_shared_1[a_row_lds, a_col_lds + 1]
            a_bf16_2 = A_shared_1[a_row_lds, a_col_lds + 2]
            a_bf16_3 = A_shared_1[a_row_lds, a_col_lds + 3]

            a_u16_0 = S.bitcast(a_bf16_0, S.u16)
            a_u16_1 = S.bitcast(a_bf16_1, S.u16)
            a_u16_2 = S.bitcast(a_bf16_2, S.u16)
            a_u16_3 = S.bitcast(a_bf16_3, S.u16)

            a_u32_0 = a_u16_0 | (a_u16_1 << 16)
            a_u32_1 = a_u16_2 | (a_u16_3 << 16)

            A_frag[lane, 0] = a_u32_0
            A_frag[lane, 1] = a_u32_1

            b_row_lds = k_offset + (lane // 32) * 4
            b_col_lds = lane % 32

            b_bf16_0 = B_shared_1[b_row_lds + 0, b_col_lds]
            b_bf16_1 = B_shared_1[b_row_lds + 1, b_col_lds]
            b_bf16_2 = B_shared_1[b_row_lds + 2, b_col_lds]
            b_bf16_3 = B_shared_1[b_row_lds + 3, b_col_lds]

            b_u16_0 = S.bitcast(b_bf16_0, S.u16)
            b_u16_1 = S.bitcast(b_bf16_1, S.u16)
            b_u16_2 = S.bitcast(b_bf16_2, S.u16)
            b_u16_3 = S.bitcast(b_bf16_3, S.u16)

            b_u32_0 = b_u16_0 | (b_u16_1 << 16)
            b_u32_1 = b_u16_2 | (b_u16_3 << 16)

            B_frag[lane, 0] = b_u32_0
            B_frag[lane, 1] = b_u32_1

            S.syncthreads()

            a_view = S.view(A_frag[lane], S.Tensor((1, 4, 1), S.bf16))
            b_view = S.view(B_frag[lane], S.Tensor((1, 4, 1), S.bf16))

            acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_view[0], b_view[0], acc)

        S.syncthreads()

    # Store output
    for acc_idx in S.range(16):
        row_offset = 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        col_offset = lane % 32

        global_row = block_row_base + row_offset
        global_col = block_col_base + col_offset

        bias_val = S.convert(BIAS0[global_col], S.f32)
        scale_val = S.convert(SCALE[global_col], S.f32)

        result = (acc[acc_idx] + bias_val) * scale_val
        Y[global_row, global_col] = S.convert(result, S.bf16)


@substrate.jit
def batchnorm_kernel(
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
    BN_WEIGHT: S.Tensor((OUT_FEATURES,), S.bf16),
    BN_BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
):
    for j in S.range(OUT_FEATURES):
        mean = S.convert(0.0, S.f32)
        for i in S.range(BATCH_SIZE):
            mean += S.convert(Y[i, j], S.f32)
        mean = mean / S.convert(BATCH_SIZE, S.f32)

        var = S.convert(0.0, S.f32)
        for i in S.range(BATCH_SIZE):
            d = S.convert(Y[i, j], S.f32) - mean
            var += d * d
        var = var / S.convert(BATCH_SIZE, S.f32)

        denom = S.sqrt(var + S.convert(EPS, S.f32))
        for i in S.range(BATCH_SIZE):
            v = (S.convert(Y[i, j], S.f32) - mean) / denom
            v = v * S.convert(BN_WEIGHT[j], S.f32) + S.convert(BN_BIAS[j], S.f32)
            Y[i, j] = S.convert(v, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES) or x.dtype != torch.bfloat16:
            raise RuntimeError('This fused kernel only supports the benchmark input shape and dtype.')

        w_t = self.gemm.weight.t().to(device=x.device, dtype=x.dtype).contiguous()
        bias = self.gemm.bias.to(device=x.device, dtype=x.dtype).contiguous()
        scale = self.scale.to(device=x.device, dtype=x.dtype).contiguous()
        bn_w = self.bn.weight.to(device=x.device, dtype=x.dtype).contiguous()
        bn_b = self.bn.bias.to(device=x.device, dtype=x.dtype).contiguous()

        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=x.dtype)

        grid_x = OUT_FEATURES // TILE_N
        grid_y = BATCH_SIZE // TILE_M

        def launch_gemm():
            return ((grid_x, grid_y, 1), (BLOCK_SIZE, 1, 1))

        gemm_kernel[launch_gemm](x, w_t, bias, scale, y)

        def launch_bn():
            return ((1, 1, 1), (1, 1, 1))

        batchnorm_kernel[launch_bn](y, bn_w, bn_b)

        return y
