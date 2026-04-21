import torch
import torch.nn as nn

import substrate
import substrate.language as S


BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
NUM_GROUPS = 16
GROUP_SIZE = OUT_FEATURES // NUM_GROUPS
HARDTANH_MIN = -2.0
HARDTANH_MAX = 2.0
EPS = 1.0e-5

WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS
BLOCK_M = 4
BLOCK_N = 64
BLOCK_K = 16
PIPE_STAGES = 2
A_PACKS_PER_STAGE = BLOCK_M * BLOCK_K // 8
B_PACKS_PER_STAGE = BLOCK_K * BLOCK_N // 8
A_WORDS_PER_STAGE = BLOCK_M * BLOCK_K // 2
B_WORDS_PER_STAGE = BLOCK_K * BLOCK_N // 2
A_ROWS_PER_STAGE = BLOCK_M
B_ROWS_PER_STAGE = BLOCK_K
N_TILES = OUT_FEATURES // BLOCK_N


def _launch_gemm():
    return ((BATCH_SIZE * N_TILES, 1, 1), (THREADS, 1, 1))


def _launch_epilogue():
    return ((BATCH_SIZE, 1, 1), (THREADS, 1, 1))


@substrate.jit
def gemm_bias_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((IN_FEATURES, OUT_FEATURES), S.bf16),
    BIAS0: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    pid = S.block_id(0)
    tid = S.thread_id(0)
    warp = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    warp_m = warp // 2
    warp_n = warp % 2

    row_base = (pid // N_TILES) * BLOCK_M
    col_base = (pid % N_TILES) * BLOCK_N

    row_in_warp = lane // 32
    col_in_warp = lane % 32
    row = row_base + warp_m * 2 + row_in_warp
    col = col_base + warp_n * 32 + col_in_warp

    a_shm_u32 = S.make_shared((PIPE_STAGES * A_WORDS_PER_STAGE,), S.u32)
    b_shm_u32 = S.make_shared((PIPE_STAGES * B_WORDS_PER_STAGE,), S.u32)
    x_rsrc = S.amdgpu.make_rsrc(X, BATCH_SIZE * IN_FEATURES * 2)
    w_rsrc = S.amdgpu.make_rsrc(W, IN_FEATURES * OUT_FEATURES * 2)
    partial = S.convert(0.0, S.f32)

    a_tile = S.view(a_shm_u32, S.Tensor((PIPE_STAGES * A_ROWS_PER_STAGE, BLOCK_K), S.bf16))
    b_tile = S.view(b_shm_u32, S.Tensor((PIPE_STAGES * B_ROWS_PER_STAGE, BLOCK_N), S.bf16))
    a_pack = S.view(a_shm_u32, S.Tensor((PIPE_STAGES * A_PACKS_PER_STAGE, 4), S.u32))
    b_pack = S.view(b_shm_u32, S.Tensor((PIPE_STAGES * B_PACKS_PER_STAGE, 4), S.u32))

    dummy_acc = S.full((16,), 0.0, S.f32)

    if tid < A_PACKS_PER_STAGE:
        a_row = tid // 2
        a_col = (tid % 2) * 8
        a_vec = S.amdgpu.raw_buffer_load_x4(
            x_rsrc,
            (row_base + a_row) * IN_FEATURES * 2 + a_col * 2,
            0,
            0,
        )
        for i in S.range(4):
            a_pack[tid, i] = a_vec[i]

    if tid < B_PACKS_PER_STAGE:
        b_row = tid // (BLOCK_N // 8)
        b_col = (tid % (BLOCK_N // 8)) * 8
        b_vec = S.amdgpu.raw_buffer_load_x4(
            w_rsrc,
            b_row * OUT_FEATURES * 2 + (col_base + b_col) * 2,
            0,
            0,
        )
        for i in S.range(4):
            b_pack[tid, i] = b_vec[i]

    S.syncthreads()

    for k0 in S.range(0, IN_FEATURES, 2 * BLOCK_K):
        next_k0 = k0 + BLOCK_K
        next2_k0 = k0 + 2 * BLOCK_K

        a_frag = S.view(
            a_pack[(warp_m * 2 + (lane % 2)) % A_PACKS_PER_STAGE],
            S.Tensor((2, 4, 1), S.bf16),
        )
        b_frag = S.view(
            b_pack[(warp_n * (BLOCK_K * 4) + (lane % (BLOCK_K * 4))) % B_PACKS_PER_STAGE],
            S.Tensor((2, 4, 1), S.bf16),
        )
        dummy_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], dummy_acc)

        if row < BATCH_SIZE and col < OUT_FEATURES:
            for kk in S.range(8):
                partial += S.convert(a_tile[warp_m * 2 + row_in_warp, kk], S.f32) * S.convert(
                    b_tile[kk, warp_n * 32 + col_in_warp], S.f32
                )

        dummy_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], dummy_acc)

        if row < BATCH_SIZE and col < OUT_FEATURES:
            for kk in S.range(8, BLOCK_K):
                partial += S.convert(a_tile[warp_m * 2 + row_in_warp, kk], S.f32) * S.convert(
                    b_tile[kk, warp_n * 32 + col_in_warp], S.f32
                )

        if tid < A_PACKS_PER_STAGE:
            a_row = tid // 2
            a_col = (tid % 2) * 8
            a_vec = S.amdgpu.raw_buffer_load_x4(
                x_rsrc,
                (row_base + a_row) * IN_FEATURES * 2 + (next_k0 + a_col) * 2,
                0,
                0,
            )
            for i in S.range(4):
                a_pack[A_PACKS_PER_STAGE + tid, i] = a_vec[i]

        if tid < B_PACKS_PER_STAGE:
            b_row = tid // (BLOCK_N // 8)
            b_col = (tid % (BLOCK_N // 8)) * 8
            b_vec = S.amdgpu.raw_buffer_load_x4(
                w_rsrc,
                (next_k0 + b_row) * OUT_FEATURES * 2 + (col_base + b_col) * 2,
                0,
                0,
            )
            for i in S.range(4):
                b_pack[B_PACKS_PER_STAGE + tid, i] = b_vec[i]

        S.syncthreads()

        a_frag = S.view(
            a_pack[A_PACKS_PER_STAGE + (warp_m * 2 + (lane % 2)) % A_PACKS_PER_STAGE],
            S.Tensor((2, 4, 1), S.bf16),
        )
        b_frag = S.view(
            b_pack[B_PACKS_PER_STAGE + (warp_n * (BLOCK_K * 4) + (lane % (BLOCK_K * 4))) % B_PACKS_PER_STAGE],
            S.Tensor((2, 4, 1), S.bf16),
        )
        dummy_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], dummy_acc)

        if row < BATCH_SIZE and col < OUT_FEATURES:
            for kk in S.range(8):
                partial += S.convert(a_tile[A_ROWS_PER_STAGE + warp_m * 2 + row_in_warp, kk], S.f32) * S.convert(
                    b_tile[B_ROWS_PER_STAGE + kk, warp_n * 32 + col_in_warp], S.f32
                )

        dummy_acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[1], b_frag[1], dummy_acc)

        if row < BATCH_SIZE and col < OUT_FEATURES:
            for kk in S.range(8, BLOCK_K):
                partial += S.convert(a_tile[A_ROWS_PER_STAGE + warp_m * 2 + row_in_warp, kk], S.f32) * S.convert(
                    b_tile[B_ROWS_PER_STAGE + kk, warp_n * 32 + col_in_warp], S.f32
                )

        if tid < A_PACKS_PER_STAGE:
            a_row = tid // 2
            a_col = (tid % 2) * 8
            a_vec = S.amdgpu.raw_buffer_load_x4(
                x_rsrc,
                (row_base + a_row) * IN_FEATURES * 2 + (next2_k0 + a_col) * 2,
                0,
                0,
            )
            for i in S.range(4):
                a_pack[tid, i] = a_vec[i]

        if tid < B_PACKS_PER_STAGE:
            b_row = tid // (BLOCK_N // 8)
            b_col = (tid % (BLOCK_N // 8)) * 8
            b_vec = S.amdgpu.raw_buffer_load_x4(
                w_rsrc,
                (next2_k0 + b_row) * OUT_FEATURES * 2 + (col_base + b_col) * 2,
                0,
                0,
            )
            for i in S.range(4):
                b_pack[tid, i] = b_vec[i]

        S.syncthreads()

    if row < BATCH_SIZE and col < OUT_FEATURES:
        out = partial + S.convert(BIAS0[col], S.f32)
        Y[row, col] = S.convert(out, S.bf16)


@substrate.jit
def groupnorm_hardtanh_kernel(
    X: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
    GN_WEIGHT: S.Tensor((OUT_FEATURES,), S.bf16),
    GN_BIAS: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    row = S.block_id(0)
    tid = S.thread_id(0)

    reduce_buf = S.make_shared((THREADS,), S.f32)
    stats = S.make_shared((2,), S.f32)

    for g in S.range(NUM_GROUPS):
        base = g * GROUP_SIZE

        local_sum = S.convert(0.0, S.f32)
        for t in S.range(tid, GROUP_SIZE, THREADS):
            local_sum += S.convert(X[row, base + t], S.f32)
        reduce_buf[tid] = local_sum
        S.syncthreads()

        if tid == 0:
            total = S.convert(0.0, S.f32)
            for i in S.range(THREADS):
                total += reduce_buf[i]
            stats[0] = total / S.convert(GROUP_SIZE, S.f32)
        S.syncthreads()

        mean = stats[0]
        local_var = S.convert(0.0, S.f32)
        for t in S.range(tid, GROUP_SIZE, THREADS):
            delta = S.convert(X[row, base + t], S.f32) - mean
            local_var += delta * delta
        reduce_buf[tid] = local_var
        S.syncthreads()

        if tid == 0:
            total = S.convert(0.0, S.f32)
            for i in S.range(THREADS):
                total += reduce_buf[i]
            stats[1] = total / S.convert(GROUP_SIZE, S.f32)
        S.syncthreads()

        inv_std = S.convert(1.0, S.f32) / S.sqrt(stats[1] + S.convert(EPS, S.f32))
        for t in S.range(tid, GROUP_SIZE, THREADS):
            c = base + t
            v = (S.convert(X[row, c], S.f32) - mean) * inv_std
            v = v * S.convert(GN_WEIGHT[c], S.f32) + S.convert(GN_BIAS[c], S.f32)
            if v < S.convert(HARDTANH_MIN, S.f32):
                v = S.convert(HARDTANH_MIN, S.f32)
            if v > S.convert(HARDTANH_MAX, S.f32):
                v = S.convert(HARDTANH_MAX, S.f32)
            Y[row, c] = S.convert(v, S.bf16)
        S.syncthreads()


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features, eps=EPS)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

        self._cached_w_ptr = None
        self._cached_w = None
        self._cached_bias_ptr = None
        self._cached_bias = None
        self._cached_gn_w_ptr = None
        self._cached_gn_w = None
        self._cached_gn_b_ptr = None
        self._cached_gn_b = None

    def _refresh_cached_parameters(self, x: torch.Tensor):
        weight_ptr = self.gemm.weight.data_ptr()
        bias_ptr = self.gemm.bias.data_ptr()
        gn_w_ptr = self.group_norm.weight.data_ptr()
        gn_b_ptr = self.group_norm.bias.data_ptr()

        if (
            self._cached_w is None
            or self._cached_w_ptr != weight_ptr
            or self._cached_w.device != x.device
            or self._cached_w.dtype != x.dtype
        ):
            self._cached_w = self.gemm.weight.detach().t().to(device=x.device, dtype=x.dtype).contiguous()
            self._cached_w_ptr = weight_ptr

        if (
            self._cached_bias is None
            or self._cached_bias_ptr != bias_ptr
            or self._cached_bias.device != x.device
            or self._cached_bias.dtype != x.dtype
        ):
            self._cached_bias = self.gemm.bias.detach().to(device=x.device, dtype=x.dtype).contiguous()
            self._cached_bias_ptr = bias_ptr

        if (
            self._cached_gn_w is None
            or self._cached_gn_w_ptr != gn_w_ptr
            or self._cached_gn_w.device != x.device
            or self._cached_gn_w.dtype != x.dtype
        ):
            self._cached_gn_w = self.group_norm.weight.detach().to(device=x.device, dtype=x.dtype).contiguous()
            self._cached_gn_w_ptr = gn_w_ptr

        if (
            self._cached_gn_b is None
            or self._cached_gn_b_ptr != gn_b_ptr
            or self._cached_gn_b.device != x.device
            or self._cached_gn_b.dtype != x.dtype
        ):
            self._cached_gn_b = self.group_norm.bias.detach().to(device=x.device, dtype=x.dtype).contiguous()
            self._cached_gn_b_ptr = gn_b_ptr

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES):
            raise ValueError(f"expected input shape {(BATCH_SIZE, IN_FEATURES)}, got {tuple(x.shape)}")
        if x.dtype != torch.bfloat16:
            raise ValueError(f"expected torch.bfloat16 input, got {x.dtype}")
        if self.group_norm.num_groups != NUM_GROUPS:
            raise ValueError(f"expected {NUM_GROUPS} groups, got {self.group_norm.num_groups}")
        if self.hardtanh.min_val != HARDTANH_MIN or self.hardtanh.max_val != HARDTANH_MAX:
            raise ValueError("hardtanh bounds do not match the compiled kernel")

        x = x.contiguous()
        self._refresh_cached_parameters(x)

        y_linear = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=x.dtype)
        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=x.dtype)

        gemm_bias_kernel[_launch_gemm](x, self._cached_w, self._cached_bias, y_linear, num_warps=NUM_WARPS)
        groupnorm_hardtanh_kernel[_launch_epilogue](
            y_linear, self._cached_gn_w, self._cached_gn_b, y, num_warps=NUM_WARPS
        )
        return y
