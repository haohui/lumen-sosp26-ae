import torch
import torch.nn as nn
import avelang
import avelang.language as al

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
BN_EPS = 1e-05


# ── GEMM + Bias kernel using MFMA ──────────────────────────────
# Output is f32 to match MFMA accumulator precision.

@avelang.jit
def gemm_bias_kernel(
    X_ptr: al.Pointer(al.bf16),
    W_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    Y_ptr: al.Pointer(al.f32),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    TILE_M = 32
    TILE_N = 32
    TILE_K = 16

    lane = al.thread_id(0)
    warp_id = lane // 64
    lane_in_warp = lane % 64
    lane_col = lane_in_warp & 31
    lane_group = lane_in_warp >> 5
    warp_m = warp_id >> 1
    warp_n = warp_id & 1

    BLOCK_M = TILE_M * 2
    BLOCK_N = TILE_N * 2

    block_m = al.block_id(1) * BLOCK_M
    block_n = al.block_id(0) * BLOCK_N

    # Tensor views for bf16 inputs and f32 output
    X_bf16 = al.make_tensor(X_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    W_bf16 = al.make_tensor(W_ptr, al.bf16, al.make_layout((N, K), (K, 1)))
    Y_f32 = al.make_tensor(Y_ptr, al.f32, al.make_layout((M, N), (N, 1)))

    # Packed i32 views for bf16 inputs
    # K bf16 per row = K/2 i32 per row. Groups of 4 i32: (K/2)/4 = K>>3 groups per row.
    X_vec = al.view(X_bf16, al.i32, al.make_layout((M, K >> 3, 4), (K >> 1, 4, 1)))
    W_vec = al.view(W_bf16, al.i32, al.make_layout((N, K >> 3, 4), (K >> 1, 4, 1)))

    # Packed i32 view for f32 output
    # N f32 per row = N i32 per row. Groups of 4: N>>2 groups per row.
    Y_vec = al.view(Y_f32, al.i32, al.make_layout((M, N >> 2, 4), (N, 4, 1)))

    # LDS buffers
    a_smem = al.make_shared((BLOCK_M * (TILE_K >> 3), TILE_K >> 2), al.i32)
    b_smem = al.make_shared((BLOCK_N * (TILE_K >> 3), TILE_K >> 2), al.i32)

    # Bias in LDS (f32)
    bias_lds = al.make_shared((BLOCK_N,), al.f32)
    bias_flat = al.make_tensor(bias_ptr, al.bf16, al.make_layout((N,), (1,)))
    if lane < BLOCK_N:
        bias_lds[lane] = al.convert(bias_flat[block_n + lane], al.f32)

    # Intermediate C in LDS and packed view for writeback
    c_smem = al.make_shared((BLOCK_M, BLOCK_N), al.f32)
    c_smem_vec = al.view(c_smem, al.i32, al.make_layout((BLOCK_M, BLOCK_N >> 2, 4), (BLOCK_N, 4, 1)))

    # Accumulators: 16 f32 per lane
    acc = al.full((16,), 0.0, al.f32)

    K_TILES = K // TILE_K
    for kt in al.range(K_TILES):
        k_vec = kt * 2 + lane_group

        # Load A tile from X
        a_row = warp_m * TILE_M + lane_col
        a_lds = a_row * (TILE_K >> 3) + lane_group
        a_smem[a_lds] = X_vec[block_m + a_row, k_vec]

        # Load B tile from W (W is (N,K) stored row-major)
        b_row = warp_n * TILE_N + lane_col
        b_lds = b_row * (TILE_K >> 3) + lane_group
        b_smem[b_lds] = W_vec[block_n + b_row, k_vec]

        al.syncthreads()

        # Read back from LDS and feed to MFMA
        a_words = a_smem[a_lds]
        b_words = b_smem[b_lds]
        a_frag = al.view(a_words, al.Tensor((2, 2, 1), al.u32))
        b_frag = al.view(b_words, al.Tensor((2, 2, 1), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)

        al.syncthreads()

    # Write accumulators to shared memory, adding bias
    for r in al.range(16):
        col_offset = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        col = warp_n * TILE_N + col_offset
        val = acc[r] + bias_lds[col]
        c_smem[warp_m * TILE_M + lane_col, col] = val

    al.syncthreads()

    # Writeback to global memory as f32 (packed i32 view)
    store_row = warp_m * TILE_M + (lane_in_warp >> 1)
    store_col_base = warp_n * (TILE_N >> 2) + (lane_in_warp & 1) * (TILE_N >> 3)

    for v in al.range(TILE_N >> 3):
        c_vals = c_smem_vec[store_row, store_col_base + v]
        Y_vec[block_m + store_row, (block_n >> 2) + store_col_base + v] = c_vals


# ── BN apply + Scale + Softmax kernel ───────────────────────────

@avelang.jit
def bn_apply_scale_softmax_kernel(
    Y_ptr: al.Pointer(al.f32),
    mean_ptr: al.Pointer(al.f32),
    var_ptr: al.Pointer(al.f32),
    bn_weight_ptr: al.Pointer(al.bf16),
    bn_bias_ptr: al.Pointer(al.bf16),
    scale_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
):
    SOFTMAX_BLOCK = 256
    NUM_ELTS = 32
    row = al.block_id(0)

    Y_f32 = al.make_tensor(Y_ptr, al.f32, al.make_layout((M, N), (N, 1)))
    mean_f32 = al.make_tensor(mean_ptr, al.f32, al.make_layout((N,), (1,)))
    var_f32 = al.make_tensor(var_ptr, al.f32, al.make_layout((N,), (1,)))
    bn_w = al.make_tensor(bn_weight_ptr, al.bf16, al.make_layout((N,), (1,)))
    bn_b = al.make_tensor(bn_bias_ptr, al.bf16, al.make_layout((N,), (1,)))
    scale_val = al.convert(
        al.make_tensor(scale_ptr, al.bf16, al.make_layout((1,), (1,)))[0],
        al.f32,
    )
    out_bf16 = al.make_tensor(out_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    tid = al.thread_id(0)

    # Local storage for BN+Scale values (32 elements per thread for N=8192, BLOCK=256)
    local_vals = al.make_local((NUM_ELTS,), al.f32)
    local_indices = al.make_local((NUM_ELTS,), al.i32)

    # Shared memory for reductions
    smem = al.make_shared((SOFTMAX_BLOCK,), al.f32)

    # Pass 1: Apply BN + Scale, store locally, find per-thread max
    max_val = al.convert(-1.0e30, al.f32)
    local_count = 0
    for j in al.range(tid, N, SOFTMAX_BLOCK):
        val = Y_f32[row, j]
        norm = (val - mean_f32[j]) / al.sqrt(var_f32[j])
        norm = norm * al.convert(bn_w[j], al.f32) + al.convert(bn_b[j], al.f32)
        norm = norm * scale_val
        local_vals[local_count] = norm
        local_indices[local_count] = j
        local_count = local_count + 1
        if norm > max_val:
            max_val = norm
    smem[tid] = max_val
    al.syncthreads()

    # Reduce max (unrolled for SOFTMAX_BLOCK=256)
    if tid < 128:
        if smem[tid + 128] > smem[tid]:
            smem[tid] = smem[tid + 128]
    al.syncthreads()
    if tid < 64:
        if smem[tid + 64] > smem[tid]:
            smem[tid] = smem[tid + 64]
    al.syncthreads()
    if tid < 32:
        if smem[tid + 32] > smem[tid]:
            smem[tid] = smem[tid + 32]
    al.syncthreads()
    if tid < 16:
        if smem[tid + 16] > smem[tid]:
            smem[tid] = smem[tid + 16]
    al.syncthreads()
    if tid < 8:
        if smem[tid + 8] > smem[tid]:
            smem[tid] = smem[tid + 8]
    al.syncthreads()
    if tid < 4:
        if smem[tid + 4] > smem[tid]:
            smem[tid] = smem[tid + 4]
    al.syncthreads()
    if tid < 2:
        if smem[tid + 2] > smem[tid]:
            smem[tid] = smem[tid + 2]
    al.syncthreads()
    if tid < 1:
        if smem[tid + 1] > smem[tid]:
            smem[tid] = smem[tid + 1]
    al.syncthreads()

    global_max = smem[0]

    # Pass 2: Compute exp sum using local values
    sum_exp = al.convert(0.0, al.f32)
    for e in al.range(local_count):
        sum_exp = sum_exp + al.exp(local_vals[e] - global_max)
    smem[tid] = sum_exp
    al.syncthreads()

    # Reduce sum (unrolled)
    if tid < 128: smem[tid] = smem[tid] + smem[tid + 128]
    al.syncthreads()
    if tid < 64: smem[tid] = smem[tid] + smem[tid + 64]
    al.syncthreads()
    if tid < 32: smem[tid] = smem[tid] + smem[tid + 32]
    al.syncthreads()
    if tid < 16: smem[tid] = smem[tid] + smem[tid + 16]
    al.syncthreads()
    if tid < 8: smem[tid] = smem[tid] + smem[tid + 8]
    al.syncthreads()
    if tid < 4: smem[tid] = smem[tid] + smem[tid + 4]
    al.syncthreads()
    if tid < 2: smem[tid] = smem[tid] + smem[tid + 2]
    al.syncthreads()
    if tid < 1: smem[tid] = smem[tid] + smem[tid + 1]
    al.syncthreads()

    global_sum_exp = smem[0]

    # Pass 3: Normalize and write output using local values
    for e in al.range(local_count):
        softmax_val = al.exp(local_vals[e] - global_max) / global_sum_exp
        out_bf16[row, local_indices[e]] = al.convert(softmax_val, al.bf16)

# ── ModelNew ────────────────────────────────────────────────────


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-05, bn_momentum=0.1, scale_shape=(1,)):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M = x.shape[0]
        N = self.gemm.out_features
        K = self.gemm.in_features

        x_bf16 = x.to(dtype=torch.bfloat16, device=x.device).contiguous()

        W = self.gemm.weight.to(device=x.device, dtype=torch.bfloat16).contiguous()
        bias = self.gemm.bias.to(device=x.device, dtype=torch.bfloat16).contiguous()

        bn_w = self.bn.weight.to(device=x.device, dtype=torch.bfloat16).contiguous()
        bn_b = self.bn.bias.to(device=x.device, dtype=torch.bfloat16).contiguous()
        scale = self.scale.to(device=x.device, dtype=torch.bfloat16).contiguous()

        # Use running stats (eval mode BN)
        running_mean = self.bn.running_mean.to(device=x.device, dtype=torch.float32).contiguous()
        running_var = self.bn.running_var.to(device=x.device, dtype=torch.float32).contiguous()
        running_var = running_var + BN_EPS  # BN formula uses var + eps in denominator

        # GEMM output is f32 to match MFMA accumulator precision
        Y = torch.empty((M, N), device=x.device, dtype=torch.float32)
        out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

        M_i32 = M
        N_i32 = N
        K_i32 = K

        TILE_M = 64
        TILE_N = 64

        grid_m = M_i32 // TILE_M
        grid_n = N_i32 // TILE_N

        gemm_bias_kernel[lambda: (
            (grid_n, grid_m, 1),
            (256, 1, 1),
        )](x_bf16, W, bias, Y, M_i32, N_i32, K_i32)

        # BN + Scale + Softmax using running stats (eval mode)
        bn_apply_scale_softmax_kernel[lambda: ((M_i32, 1, 1), (256, 1, 1))](
            Y, running_mean, running_var, bn_w, bn_b, scale, out, M_i32, N_i32
        )

        return out
