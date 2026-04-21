import torch
import torch.nn as nn
import substrate
import substrate.language as S

M = 128
K = 16384
N = 16384
MFMA_K = 8
NUM_MFMA = K // MFMA_K  # 2048


@substrate.jit
def gemm_bias_kernel(
    X_u32  : S.Tensor((128, 8192), S.u32),
    W_u32  : S.Tensor((16384, 8192), S.u32),
    bias   : S.Tensor((16384,), S.bf16),
    Out    : S.Tensor((128, 16384), S.f32),
):
    tid   = S.thread_id(0)
    lane  = tid % 64
    wid   = tid // 64
    wy    = wid // 2
    wx    = wid % 2
    bid_x = S.block_id(0)
    bid_y = S.block_id(1)

    m0 = bid_y * 64 + wy * 32
    n0 = bid_x * 64 + wx * 32

    c = S.full((16,), 0.0, S.f32)

    # Create resource descriptors with range (in bytes) for OOB handling
    rsrc_X = S.amdgpu.make_rsrc(X_u32, 128 * 8192 * 4)
    rsrc_W = S.amdgpu.make_rsrc(W_u32, 16384 * 8192 * 4)

    # Per-thread LDS buffers for double buffering
    A0 = S.make_shared((256, 2), S.u32)
    A1 = S.make_shared((256, 2), S.u32)
    B0 = S.make_shared((256, 2), S.u32)
    B1 = S.make_shared((256, 2), S.u32)

    my_row  = m0 + lane % 32
    my_col  = n0 + lane % 32
    k_group = (lane // 32) * 4

    # ── prologue: load first MFMA tile into buf0 ──
    u32_col_0 = k_group // 2
    a_vals = S.amdgpu.raw_buffer_load_x4(rsrc_X, (my_row * 8192 + u32_col_0) * 4, 0, 0)
    b_vals = S.amdgpu.raw_buffer_load_x4(rsrc_W, (my_col * 8192 + u32_col_0) * 4, 0, 0)
    A0[tid, 0] = a_vals[0]
    A0[tid, 1] = a_vals[1]
    B0[tid, 0] = b_vals[0]
    B0[tid, 1] = b_vals[1]

    S.syncthreads()

    # ── main loop: double-buffered, unrolled by 2 ──
    for ki in S.range(1023):
        k_odd  = (ki * 2 + 1) * 8
        k_next = (ki * 2 + 2) * 8

        # even sub-step: MFMA from buf0, load into buf1
        ma = S.view(A0[tid], S.Tensor((1, 4, 1), S.bf16))
        mb = S.view(B0[tid], S.Tensor((1, 4, 1), S.bf16))
        c  = S.amdgpu.mfma_32x32x8_bf16_f32(ma[0], mb[0], c)

        u32_col_odd = (k_odd + k_group) // 2
        a_vals1 = S.amdgpu.raw_buffer_load_x4(rsrc_X, (my_row * 8192 + u32_col_odd) * 4, 0, 0)
        b_vals1 = S.amdgpu.raw_buffer_load_x4(rsrc_W, (my_col * 8192 + u32_col_odd) * 4, 0, 0)
        A1[tid, 0] = a_vals1[0]
        A1[tid, 1] = a_vals1[1]
        B1[tid, 0] = b_vals1[0]
        B1[tid, 1] = b_vals1[1]

        S.syncthreads()

        # odd sub-step: MFMA from buf1, load into buf0
        ma = S.view(A1[tid], S.Tensor((1, 4, 1), S.bf16))
        mb = S.view(B1[tid], S.Tensor((1, 4, 1), S.bf16))
        c  = S.amdgpu.mfma_32x32x8_bf16_f32(ma[0], mb[0], c)

        u32_col_next = (k_next + k_group) // 2
        a_vals0 = S.amdgpu.raw_buffer_load_x4(rsrc_X, (my_row * 8192 + u32_col_next) * 4, 0, 0)
        b_vals0 = S.amdgpu.raw_buffer_load_x4(rsrc_W, (my_col * 8192 + u32_col_next) * 4, 0, 0)
        A0[tid, 0] = a_vals0[0]
        A0[tid, 1] = a_vals0[1]
        B0[tid, 0] = b_vals0[0]
        B0[tid, 1] = b_vals0[1]

        S.syncthreads()

    # ── epilogue: last 2 MFMA tiles ──
    ma = S.view(A0[tid], S.Tensor((1, 4, 1), S.bf16))
    mb = S.view(B0[tid], S.Tensor((1, 4, 1), S.bf16))
    c  = S.amdgpu.mfma_32x32x8_bf16_f32(ma[0], mb[0], c)

    k_last = 16376
    u32_col_last = (k_last + k_group) // 2
    a_vals_l = S.amdgpu.raw_buffer_load_x4(rsrc_X, (my_row * 8192 + u32_col_last) * 4, 0, 0)
    b_vals_l = S.amdgpu.raw_buffer_load_x4(rsrc_W, (my_col * 8192 + u32_col_last) * 4, 0, 0)
    A1[tid, 0] = a_vals_l[0]
    A1[tid, 1] = a_vals_l[1]
    B1[tid, 0] = b_vals_l[0]
    B1[tid, 1] = b_vals_l[1]

    S.syncthreads()

    ma = S.view(A1[tid], S.Tensor((1, 4, 1), S.bf16))
    mb = S.view(B1[tid], S.Tensor((1, 4, 1), S.bf16))
    c  = S.amdgpu.mfma_32x32x8_bf16_f32(ma[0], mb[0], c)

    # ── write-back: f32 output + bias ──
    row_off = (lane // 32) * 4
    for j in S.range(16):
        row_in_warp = (j // 4) * 8 + (j % 4) + row_off
        Out[m0 + row_in_warp, my_col] = c[j] + bias[my_col]


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))

    def forward(self, x):
        batch = x.shape[0]
        out_f32 = torch.empty(batch, self.out_features, dtype=torch.float32, device=x.device)

        x_u32 = x.contiguous().view(torch.int32).reshape(batch, self.in_features // 2)
        w_u32 = self.linear.weight.contiguous().view(torch.int32).reshape(self.out_features, self.in_features // 2)

        gx = self.out_features // 64
        gy = batch // 64
        gemm_bias_kernel[lambda: ((gx, gy, 1), (256, 1, 1))](
            x_u32, w_u32, self.linear.bias, out_f32
        )
        result = torch.minimum(out_f32, self.constant.float())
        result = result - self.constant.float()
        return result.to(torch.bfloat16)
