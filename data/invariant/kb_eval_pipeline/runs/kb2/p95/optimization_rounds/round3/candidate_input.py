import torch
import torch.nn as nn
import substrate
import substrate.language as S

SQRT_2 = 1.4142135623730951

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WARP_M = 32
WARP_N = 32
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = 64 * WAVES_PER_BLOCK
NUM_K_TILES = IN_FEATURES // BLOCK_K
NUM_K_PAIRS = NUM_K_TILES // 2
X_RANGE_BYTES = BATCH_SIZE * IN_FEATURES * 2
W_RANGE_BYTES = IN_FEATURES * OUT_FEATURES * 2


def _launch():
    return ((OUT_FEATURES // BLOCK_N, BATCH_SIZE // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))


@substrate.jit
def fused_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((IN_FEATURES, OUT_FEATURES), S.bf16),
    BIAS0: S.Tensor((OUT_FEATURES,), S.bf16),
    ADDV: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % 64
    wave = tid // 64
    warp_row = wave // 2
    warp_col = wave % 2
    lane_row = lane % 32
    lane_group = lane // 32

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N
    warp_row_base = block_row + warp_row * WARP_M
    warp_col_base = block_col + warp_col * WARP_N

    zero_i32 = S.convert(0, S.i32)
    x_rsrc = S.amdgpu.make_rsrc(X, S.convert(X_RANGE_BYTES, S.i32))
    w_rsrc = S.amdgpu.make_rsrc(W, S.convert(W_RANGE_BYTES, S.i32))

    shared_words = S.make_shared((2048,), S.u32)
    pack_bf16 = S.make_shared((THREADS_PER_BLOCK, 16), S.bf16)

    a_words0 = S.subview(shared_words, (0,), (512,), (1,))
    b_words0 = S.subview(shared_words, (512,), (512,), (1,))
    a_words1 = S.subview(shared_words, (1024,), (512,), (1,))
    b_words1 = S.subview(shared_words, (1536,), (512,), (1,))

    a_tile0 = S.view(a_words0, S.bf16, S.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K, 1)))
    b_tile0 = S.view(b_words0, S.bf16, S.make_layout((BLOCK_K, BLOCK_N), (BLOCK_N, 1)))
    a_tile1 = S.view(a_words1, S.bf16, S.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K, 1)))
    b_tile1 = S.view(b_words1, S.bf16, S.make_layout((BLOCK_K, BLOCK_N), (BLOCK_N, 1)))

    c_lane = S.full((16,), 0.0, S.f32)

    a_chunk = tid % (BLOCK_M * (BLOCK_K // 8))
    a_row = a_chunk // (BLOCK_K // 8)
    a_vec = a_chunk % (BLOCK_K // 8)
    b_chunk = tid % (BLOCK_K * (BLOCK_N // 8))
    b_row = b_chunk // (BLOCK_N // 8)
    b_vec = b_chunk % (BLOCK_N // 8)

    a_offset_elems = (block_row + a_row) * IN_FEATURES + a_vec * 8
    a_packed = S.amdgpu.raw_buffer_load_x4(x_rsrc, zero_i32, S.convert(a_offset_elems * 2, S.i32), 0)
    for u in S.range(4):
        a_words0[a_chunk * 4 + u] = a_packed[u]

    b_offset_elems = b_row * OUT_FEATURES + block_col + b_vec * 8
    b_packed = S.amdgpu.raw_buffer_load_x4(w_rsrc, zero_i32, S.convert(b_offset_elems * 2, S.i32), 0)
    for u in S.range(4):
        b_words0[b_chunk * 4 + u] = b_packed[u]

    S.syncthreads()

    for pair_idx in S.range(NUM_K_PAIRS):
        k_base1 = (pair_idx * 2 + 1) * BLOCK_K

        a_offset_elems = (block_row + a_row) * IN_FEATURES + k_base1 + a_vec * 8
        a_packed = S.amdgpu.raw_buffer_load_x4(x_rsrc, zero_i32, S.convert(a_offset_elems * 2, S.i32), 0)
        for u in S.range(4):
            a_words1[a_chunk * 4 + u] = a_packed[u]

        b_offset_elems = (k_base1 + b_row) * OUT_FEATURES + block_col + b_vec * 8
        b_packed = S.amdgpu.raw_buffer_load_x4(w_rsrc, zero_i32, S.convert(b_offset_elems * 2, S.i32), 0)
        for u in S.range(4):
            b_words1[b_chunk * 4 + u] = b_packed[u]

        for e in S.range(4):
            pack_bf16[tid, e] = a_tile0[warp_row * WARP_M + lane_row, lane_group * 4 + e]
            pack_bf16[tid, 8 + e] = b_tile0[lane_group * 4 + e, warp_col * WARP_N + lane_row]
        thread_pack = S.view(pack_bf16[tid], S.Tensor((4, 4, 1), S.bf16))
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(thread_pack[0], thread_pack[2], c_lane)

        for e in S.range(4):
            pack_bf16[tid, 4 + e] = a_tile0[warp_row * WARP_M + lane_row, 8 + lane_group * 4 + e]
            pack_bf16[tid, 12 + e] = b_tile0[8 + lane_group * 4 + e, warp_col * WARP_N + lane_row]
        thread_pack = S.view(pack_bf16[tid], S.Tensor((4, 4, 1), S.bf16))
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(thread_pack[1], thread_pack[3], c_lane)

        S.syncthreads()

        if pair_idx + 1 < NUM_K_PAIRS:
            k_base2 = (pair_idx * 2 + 2) * BLOCK_K

            a_offset_elems = (block_row + a_row) * IN_FEATURES + k_base2 + a_vec * 8
            a_packed = S.amdgpu.raw_buffer_load_x4(x_rsrc, zero_i32, S.convert(a_offset_elems * 2, S.i32), 0)
            for u in S.range(4):
                a_words0[a_chunk * 4 + u] = a_packed[u]

            b_offset_elems = (k_base2 + b_row) * OUT_FEATURES + block_col + b_vec * 8
            b_packed = S.amdgpu.raw_buffer_load_x4(w_rsrc, zero_i32, S.convert(b_offset_elems * 2, S.i32), 0)
            for u in S.range(4):
                b_words0[b_chunk * 4 + u] = b_packed[u]

        for e in S.range(4):
            pack_bf16[tid, e] = a_tile1[warp_row * WARP_M + lane_row, lane_group * 4 + e]
            pack_bf16[tid, 8 + e] = b_tile1[lane_group * 4 + e, warp_col * WARP_N + lane_row]
        thread_pack = S.view(pack_bf16[tid], S.Tensor((4, 4, 1), S.bf16))
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(thread_pack[0], thread_pack[2], c_lane)

        for e in S.range(4):
            pack_bf16[tid, 4 + e] = a_tile1[warp_row * WARP_M + lane_row, 8 + lane_group * 4 + e]
            pack_bf16[tid, 12 + e] = b_tile1[8 + lane_group * 4 + e, warp_col * WARP_N + lane_row]
        thread_pack = S.view(pack_bf16[tid], S.Tensor((4, 4, 1), S.bf16))
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(thread_pack[1], thread_pack[3], c_lane)

        if pair_idx + 1 < NUM_K_PAIRS:
            S.syncthreads()

    col = warp_col_base + lane_row
    bias = S.convert(BIAS0[col], S.f32) + S.convert(ADDV[col], S.f32)
    one = S.convert(1.0, S.f32)
    half = S.convert(0.5, S.f32)
    neg_one = S.convert(-1.0, S.f32)

    for acc_idx in S.range(16):
        row = warp_row_base + 8 * (acc_idx // 4) + 4 * lane_group + (acc_idx % 4)
        x = c_lane[acc_idx] + bias
        x = x * (one / (one + S.exp(-x)))
        x = S.tanh(x)
        x = half * x * (one + S.erf(x / S.convert(SQRT_2, S.f32)))
        if x < neg_one:
            x = neg_one
        if x > one:
            x = one
        Y[row, col] = S.convert(x, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        self._cached_weight_t = None
        self._cached_bias = None
        self._cached_addv = None
        self._cache_device = None
        self._cache_dtype = None

    def _refresh_cache(self, device, dtype):
        if (
            self._cached_weight_t is None
            or self._cache_device != device
            or self._cache_dtype != dtype
        ):
            self._cached_weight_t = torch.empty(
                (IN_FEATURES, OUT_FEATURES), device=device, dtype=dtype
            )
            self._cached_bias = torch.empty((OUT_FEATURES,), device=device, dtype=dtype)
            self._cached_addv = torch.empty((OUT_FEATURES,), device=device, dtype=dtype)
            self._cache_device = device
            self._cache_dtype = dtype

        self._cached_weight_t.copy_(self.matmul.weight.detach().t(), non_blocking=True)
        self._cached_bias.copy_(self.matmul.bias.detach(), non_blocking=True)
        self._cached_addv.copy_(self.add_value.detach(), non_blocking=True)

    def forward(self, x):
        if (
            tuple(x.shape) != (BATCH_SIZE, IN_FEATURES)
            or x.dtype != torch.bfloat16
            or tuple(self.add_value.shape) != (OUT_FEATURES,)
        ):
            raise RuntimeError("This fused kernel only supports the benchmark input shape and dtype.")

        if not x.is_contiguous():
            x = x.contiguous()

        self._refresh_cache(x.device, x.dtype)
        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=x.dtype)
        fused_kernel[_launch](x, self._cached_weight_t, self._cached_bias, self._cached_addv, y)
        return y
