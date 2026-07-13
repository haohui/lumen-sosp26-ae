import torch
import torch.nn as nn
import avelang
import avelang.language as al
import math

# ---------------------------------------------------------------------------
# Tiled GEMM + bias kernel  (64x64 tiles, 256 threads, BF16/FP32)
# ---------------------------------------------------------------------------

_BM = 64
_BN = 64
_BK = 32
_THREADS = 256
_TM = 16
_TN = 16


@avelang.jit
def gemm_add_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    bx = al.block_id(0)
    by = al.block_id(1)
    tid = al.thread_id(0)

    bm_start = by * 64
    bn_start = bx * 64

    # --- tensor views -------------------------------------------------------
    x_layout = al.make_layout((M, K), (K, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    w_layout = al.make_layout((N, K), (K, 1))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    out_layout = al.make_layout((M, N), (N, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    bias_layout = al.make_layout((N,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    # --- shared memory ------------------------------------------------------
    x_smem = al.make_shared((64, 32), al.bf16)
    w_smem = al.make_shared((64, 32), al.bf16)

    # --- thread mapping -----------------------------------------------------
    tid_m = tid // 16
    tid_n = tid % 16

    # --- FP32 accumulators (4x4 = 16) --------------------------------------
    acc = al.make_local((4, 4), al.f32)
    ri = al.convert(0, al.i32)
    for _ in al.range(4):
        ci = al.convert(0, al.i32)
        for _ in al.range(4):
            acc[ri, ci] = al.convert(0.0, al.f32)
            ci = ci + 1
        ri = ri + 1

    # --- main K loop --------------------------------------------------------
    _bk = al.convert(0, al.i32)
    for _ in al.range(0, K, 32):
        # ---- load X tile ---------------------------------------------------
        _off = tid
        for _ in al.range(8):
            sr = _off // 32
            sc = _off % 32
            gr = bm_start + sr
            gc = _bk + sc
            if (gr < M) and (gc < K):
                x_smem[sr, sc] = x[gr, gc]
            else:
                x_smem[sr, sc] = al.convert(0.0, al.bf16)
            _off = _off + 256

        # ---- load W tile ---------------------------------------------------
        _off = tid
        for _ in al.range(8):
            sr = _off // 32
            sc = _off % 32
            gr = bn_start + sr
            gc = _bk + sc
            if (gr < N) and (gc < K):
                w_smem[sr, sc] = w[gr, gc]
            else:
                w_smem[sr, sc] = al.convert(0.0, al.bf16)
            _off = _off + 256

        al.syncthreads()

        # ---- dot products ---------------------------------------------------
        kk = al.convert(0, al.i32)
        for _ in al.range(32):
            ri = al.convert(0, al.i32)
            for _ in al.range(4):
                x_val = al.convert(x_smem[tid_m * 4 + ri, kk], al.f32)
                ci = al.convert(0, al.i32)
                for _ in al.range(4):
                    w_val = al.convert(w_smem[tid_n * 4 + ci, kk], al.f32)
                    acc[ri, ci] = acc[ri, ci] + x_val * w_val
                    ci = ci + 1
                ri = ri + 1
            kk = kk + 1

        al.syncthreads()

        _bk = _bk + 32

    # --- store with bias ----------------------------------------------------
    ri = al.convert(0, al.i32)
    for _ in al.range(4):
        out_row = bm_start + tid_m * 4 + ri
        ci = al.convert(0, al.i32)
        for _ in al.range(4):
            out_col = bn_start + tid_n * 4 + ci
            if (out_row < M) and (out_col < N):
                bv = al.convert(bias[out_col], al.f32)
                val = acc[ri, ci] + bv
                out[out_row, out_col] = al.convert(val, al.bf16)
            ci = ci + 1
        ri = ri + 1


# ---------------------------------------------------------------------------
# Activation kernel
# ---------------------------------------------------------------------------

@avelang.jit
def activation_kernel(
    inout_ptr: al.Pointer(al.bf16),
    numel: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    base = bid * (256 * 8) + tid
    buf_layout = al.make_layout((numel,), (1,))
    buf = al.make_tensor(inout_ptr, al.bf16, buf_layout)

    idx = base
    for _ in al.range(8):
        if idx < numel:
            val = al.convert(buf[idx], al.f32)

            # --- Swish ---
            neg_val = -val
            exp_neg = al.exp(neg_val)
            one = al.convert(1.0, al.f32)
            sigmoid_val = one / (one + exp_neg)
            val = val * sigmoid_val

            # --- Tanh ---
            val = al.tanh(val)

            # --- GELU ---
            sqrt2 = al.convert(1.4142135623730951, al.f32)
            half = al.convert(0.5, al.f32)
            erf_arg = val / sqrt2
            erf_val = al.erf(erf_arg)
            val = half * val * (one + erf_val)

            # --- Hardtanh [-1, 1] ---
            neg_one = al.convert(-1.0, al.f32)
            pos_one = al.convert(1.0, al.f32)
            if val > pos_one:
                val = pos_one
            if val < neg_one:
                val = neg_one

            buf[idx] = al.convert(val, al.bf16)

        idx = idx + 256


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))

    def forward(self, x):
        orig_dtype = x.dtype

        weight = self.matmul.weight
        bias = self.matmul.bias
        addv = self.add_value
        combined_bias = (bias if bias is not None else torch.zeros_like(addv)) + addv

        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = weight.to(torch.bfloat16).contiguous()
        b_bf16 = combined_bias.to(torch.bfloat16).contiguous()

        M = x_bf16.shape[0]
        K_int = x_bf16.shape[1]
        N = w_bf16.shape[0]

        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)

        grid_m = (M + 63) // 64
        grid_n = (N + 63) // 64
        gemm_add_kernel[lambda: ((grid_n, grid_m, 1), (256, 1, 1))](
            x_bf16, w_bf16, b_bf16, out,
            M, N, K_int,
        )

        numel = M * N
        act_elems = 256 * 8
        grid_act = (numel + act_elems - 1) // act_elems
        activation_kernel[lambda: ((grid_act, 1, 1), (256, 1, 1))](
            out, numel,
        )

        return out.to(orig_dtype)
