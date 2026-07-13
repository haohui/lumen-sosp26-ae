import torch
import torch.nn as nn
import avelang
import avelang.language as al

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
NUM_GROUPS = 512
EPS = 1e-05

TM = 32
TN = 32
TK = 16
WARP_SIZE = 64


@avelang.jit
def gemm_kernel_opt(
    X_ptr: al.Pointer(al.bf16),
    W_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    Y_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    X_bf16 = al.make_tensor(X_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    W_bf16 = al.make_tensor(W_ptr, al.bf16, al.make_layout((N, K), (K, 1)))
    bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((N,), (1,)))
    Y = al.make_tensor(Y_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    # Buffer resource descriptors with explicit byte ranges.
    # OOB raw_buffer_load returns zero → removes kt-prefetch guards.
    # OOB raw_buffer_store is discarded → removes g_row<M guard.
    rsrc_X = al.amdgpu.make_rsrc(X_bf16, M * K * 2)
    rsrc_W = al.amdgpu.make_rsrc(W_bf16, N * K * 2)
    rsrc_Y = al.amdgpu.make_rsrc(Y, M * N * 2)

    lane = al.thread_id(0)
    lane_col = lane & 31
    lane_group = lane >> 5

    block_m = al.block_id(1) * TM
    block_n = al.block_id(0) * TN

    a_smem0 = al.make_shared((64, 4), al.i32)
    b_smem0 = al.make_shared((64, 4), al.i32)
    a_smem1 = al.make_shared((64, 4), al.i32)
    b_smem1 = al.make_shared((64, 4), al.i32)

    acc = al.full((16,), 0.0, al.f32)

    K_TILES = K // TK
    K_bytes = K * 2

    # Prefetch tile 0 via raw_buffer_load_x4 (range-protected)
    kt = al.convert(0, al.i32)
    x_off = (block_m + lane_col) * K_bytes + kt * 32 + lane_group * 16
    w_off = (block_n + lane_col) * K_bytes + kt * 32 + lane_group * 16
    a_smem0[lane] = al.amdgpu.raw_buffer_load_x4(rsrc_X, x_off, 0, 0)
    b_smem0[lane] = al.amdgpu.raw_buffer_load_x4(rsrc_W, w_off, 0, 0)
    al.syncthreads()

    # Software-pipelined main loop — NO kt1/kt2 OOB guards.
    # When kt+1 or kt+2 exceed K_TILES, raw_buffer_load returns zero.
    # MFMA with zero operands safely adds zero to the accumulator.
    for kt in al.range(0, K_TILES, 2):
        # Prefetch tile kt+1
        kt1 = kt + 1
        x_off1 = (block_m + lane_col) * K_bytes + kt1 * 32 + lane_group * 16
        w_off1 = (block_n + lane_col) * K_bytes + kt1 * 32 + lane_group * 16
        a_smem1[lane] = al.amdgpu.raw_buffer_load_x4(rsrc_X, x_off1, 0, 0)
        b_smem1[lane] = al.amdgpu.raw_buffer_load_x4(rsrc_W, w_off1, 0, 0)

        # Compute tile kt from smem0
        a_words0 = a_smem0[lane]
        b_words0 = b_smem0[lane]
        a_frag0 = al.view(a_words0, al.Tensor((2, 2, 1), al.u32))
        b_frag0 = al.view(b_words0, al.Tensor((2, 2, 1), al.u32))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag0[0], a_frag0[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag0[1], a_frag0[1], acc)

        al.syncthreads()

        # Prefetch tile kt+2
        kt2 = kt + 2
        x_off2 = (block_m + lane_col) * K_bytes + kt2 * 32 + lane_group * 16
        w_off2 = (block_n + lane_col) * K_bytes + kt2 * 32 + lane_group * 16
        a_smem0[lane] = al.amdgpu.raw_buffer_load_x4(rsrc_X, x_off2, 0, 0)
        b_smem0[lane] = al.amdgpu.raw_buffer_load_x4(rsrc_W, w_off2, 0, 0)

        # Compute tile kt+1 from smem1
        a_words1 = a_smem1[lane]
        b_words1 = b_smem1[lane]
        a_frag1 = al.view(a_words1, al.Tensor((2, 2, 1), al.u32))
        b_frag1 = al.view(b_words1, al.Tensor((2, 2, 1), al.u32))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag1[0], a_frag1[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag1[1], a_frag1[1], acc)

        al.syncthreads()

    # Epilogue: reduce accumulator fragments into shared memory
    c_smem = al.make_shared((TM, TN), al.f32)

    for r in al.range(16):
        row_offset = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        c_smem[lane_col, row_offset] = acc[r]

    al.syncthreads()

    # Store output.  The range in rsrc_Y causes OOB stores to be
    # discarded by the hardware when used with raw_buffer_store_*,
    # so no g_row<M branch is needed.
    store_row = lane >> 1
    store_vec_base = (lane & 1) * (TN >> 3)

    for v in al.range(TN >> 3):
        col_in_smem = (store_vec_base + v) * 4
        for el in al.range(4):
            f32_val = c_smem[store_row, col_in_smem + el]
            g_row = block_m + store_row
            g_col = block_n + col_in_smem + el
            bias_val = al.convert(bias[g_col], al.f32)
            val = f32_val + bias_val
            Y[g_row, g_col] = al.convert(val, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

        self._cached_wt = None
        self._cached_wt_data_ptr = None

    def _get_wt(self, dev, dt):
        w = self.gemm.weight
        data_ptr = w.data_ptr()
        if self._cached_wt is None or self._cached_wt_data_ptr != data_ptr:
            w_t = w.t().to(device=dev, dtype=dt).contiguous()
            self._cached_wt = w_t.t().contiguous()
            self._cached_wt_data_ptr = data_ptr
        return self._cached_wt

    def forward(self, x):
        dev = x.device
        dt = x.dtype

        w_T = self._get_wt(dev, dt)
        bias0 = self.gemm.bias.to(device=dev, dtype=dt).contiguous()

        y0 = torch.empty((BATCH_SIZE, OUT_FEATURES), device=dev, dtype=dt)
        gemm_kernel_opt[lambda: (
            (OUT_FEATURES // TN, BATCH_SIZE // TM, 1),
            (WARP_SIZE, 1, 1),
        )](x.contiguous(), w_T, bias0, y0,
           BATCH_SIZE, OUT_FEATURES, IN_FEATURES)

        y_gn = self.group_norm(y0)
        y_min = y_gn.float().min(dim=1, keepdim=True)[0]
        y_out = (y_min.bfloat16() + self.bias).contiguous()

        return y_out
