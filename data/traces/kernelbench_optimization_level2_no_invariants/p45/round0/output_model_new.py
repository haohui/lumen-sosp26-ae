import torch
import torch.nn as nn
import avelang
import avelang.language as al

_BM = 128
_BN = 128
_BK = 32
_TM = 8
_TN = 8
_THREADS_M = 16
_THREADS_N = 16
_THREADS = 256


@avelang.jit
def gemm1_sigmoid_kernel(
    X_ptr: al.Pointer(al.bf16),
    W1_ptr: al.Pointer(al.bf16),
    B1_ptr: al.Pointer(al.bf16),
    H_ptr: al.Pointer(al.bf16),
    B: al.i32,
    I_size: al.i32,
    H_size: al.i32,
):
    batch_tile = al.block_id(0)
    n_tile = al.block_id(1)
    tid = al.thread_id(0)
    thread_m = tid // _THREADS_N
    thread_n = tid % _THREADS_N
    row_base = batch_tile * _BM
    col_base = n_tile * _BN

    layout_X = al.make_layout((B, I_size), (I_size, al.convert(1, al.i32)))
    X = al.make_tensor(X_ptr, al.bf16, layout_X)
    layout_W1 = al.make_layout((I_size, H_size), (H_size, al.convert(1, al.i32)))
    W1 = al.make_tensor(W1_ptr, al.bf16, layout_W1)
    layout_B1 = al.make_layout((H_size,), (al.convert(1, al.i32),))
    B1 = al.make_tensor(B1_ptr, al.bf16, layout_B1)
    layout_H = al.make_layout((B, H_size), (H_size, al.convert(1, al.i32)))
    H = al.make_tensor(H_ptr, al.bf16, layout_H)

    lds_a = al.make_shared((_BM * _BK,), al.bf16)
    lds_b = al.make_shared((_BN * _BK,), al.bf16)
    zero_f32_val = al.convert(0.0, al.f32)
    one = al.convert(1.0, al.f32)

    acc = al.make_local((_TM * _TN,), al.f32)
    for e in al.range(_TM * _TN):
        acc[e] = zero_f32_val

    num_a_elems = _BM * _BK
    num_b_elems = _BN * _BK
    total_threads = _THREADS
    bk_val = _BK
    bn_val = _BN

    for k_block in al.range(al.convert(0, al.i32), I_size, al.convert(bk_val, al.i32)):
        for a_idx in al.range(tid, al.convert(num_a_elems, al.i32), al.convert(total_threads, al.i32)):
            a_r = a_idx // al.convert(bk_val, al.i32)
            a_c = a_idx % al.convert(bk_val, al.i32)
            lds_a[a_idx] = X[row_base + a_r, k_block + a_c]
        for b_idx in al.range(tid, al.convert(num_b_elems, al.i32), al.convert(total_threads, al.i32)):
            b_k = b_idx // al.convert(bn_val, al.i32)
            b_n = b_idx % al.convert(bn_val, al.i32)
            lds_b[b_idx] = W1[k_block + b_k, col_base + b_n]

        al.syncthreads()

        for tm in al.range(_TM):
            a_l = thread_m * _TM + tm
            for tn in al.range(_TN):
                b_l = thread_n * _TN + tn
                ei = tm * _TN + tn
                for kk in al.range(bk_val):
                    av = al.convert(lds_a[a_l * bk_val + kk], al.f32)
                    bv = al.convert(lds_b[kk * bn_val + b_l], al.f32)
                    acc[ei] = acc[ei] + av * bv

        al.syncthreads()

    for tm in al.range(_TM):
        gr = row_base + thread_m * _TM + tm
        for tn in al.range(_TN):
            gc = col_base + thread_n * _TN + tn
            ei = tm * _TN + tn
            val = acc[ei] + al.convert(B1[gc], al.f32)
            val = one / (one + al.exp(-val))
            H[gr, gc] = al.convert(val, al.bf16)


@avelang.jit
def gemm2_lse_kernel(
    H_ptr: al.Pointer(al.bf16),
    W2_ptr: al.Pointer(al.bf16),
    B2_ptr: al.Pointer(al.bf16),
    Y_ptr: al.Pointer(al.bf16),
    B: al.i32,
    H_size: al.i32,
    O_size: al.i32,
):
    batch_tile = al.block_id(0)
    tid = al.thread_id(0)
    thread_m = tid // _THREADS_N
    thread_n = tid % _THREADS_N
    row_base = batch_tile * _BM

    layout_H = al.make_layout((B, H_size), (H_size, al.convert(1, al.i32)))
    H = al.make_tensor(H_ptr, al.bf16, layout_H)
    layout_W2 = al.make_layout((H_size, O_size), (O_size, al.convert(1, al.i32)))
    W2 = al.make_tensor(W2_ptr, al.bf16, layout_W2)
    layout_B2 = al.make_layout((O_size,), (al.convert(1, al.i32),))
    B2 = al.make_tensor(B2_ptr, al.bf16, layout_B2)
    layout_Y = al.make_layout((B,), (al.convert(1, al.i32),))
    Y = al.make_tensor(Y_ptr, al.bf16, layout_Y)

    lds_a = al.make_shared((_BM * _BK,), al.bf16)
    lds_b = al.make_shared((_BN * _BK,), al.bf16)
    sm_red = al.make_shared((_THREADS_N * _BM,), al.f32)
    zero_f32_val = al.convert(0.0, al.f32)
    neg_inf = al.convert(-1e30, al.f32)

    num_a_elems = _BM * _BK
    num_b_elems = _BN * _BK
    total_threads = _THREADS
    bk_val = _BK
    bn_val = _BN

    lse_max = al.make_local((_BM,), al.f32)
    lse_sum = al.make_local((_BM,), al.f32)
    for r in al.range(_BM):
        lse_max[r] = neg_inf
        lse_sum[r] = zero_f32_val

    for o_tile in al.range(al.convert(0, al.i32), O_size, al.convert(bn_val, al.i32)):
        acc = al.make_local((_TM * _TN,), al.f32)
        for e in al.range(_TM * _TN):
            acc[e] = zero_f32_val

        for k_block in al.range(al.convert(0, al.i32), H_size, al.convert(bk_val, al.i32)):
            for a_idx in al.range(tid, al.convert(num_a_elems, al.i32), al.convert(total_threads, al.i32)):
                a_r = a_idx // al.convert(bk_val, al.i32)
                a_c = a_idx % al.convert(bk_val, al.i32)
                lds_a[a_idx] = H[row_base + a_r, k_block + a_c]
            for b_idx in al.range(tid, al.convert(num_b_elems, al.i32), al.convert(total_threads, al.i32)):
                b_k = b_idx // al.convert(bn_val, al.i32)
                b_n = b_idx % al.convert(bn_val, al.i32)
                lds_b[b_idx] = W2[k_block + b_k, o_tile + b_n]

            al.syncthreads()

            for tm in al.range(_TM):
                a_l = thread_m * _TM + tm
                for tn in al.range(_TN):
                    b_l = thread_n * _TN + tn
                    ei = tm * _TN + tn
                    for kk in al.range(bk_val):
                        av = al.convert(lds_a[a_l * bk_val + kk], al.f32)
                        bv = al.convert(lds_b[kk * bn_val + b_l], al.f32)
                        acc[ei] = acc[ei] + av * bv

            al.syncthreads()

        for tm in al.range(_TM):
            lr = thread_m * _TM + tm
            gr_base = row_base + lr
            for tn in al.range(_TN):
                gc = o_tile + thread_n * _TN + tn
                ei = tm * _TN + tn
                val = acc[ei] + al.convert(B2[gc], al.f32)
                if val > lse_max[lr]:
                    lse_max[lr] = val

    # ── Reduce max across thread_n ─────────────────────────────────────
    for tm in al.range(_TM):
        lr = thread_m * _TM + tm
        sm_red[thread_n * al.convert(_BM, al.i32) + lr] = lse_max[lr]
    al.syncthreads()
    if al.convert(thread_n, al.i32) == al.convert(0, al.i32):
        for tm in al.range(_TM):
            lr = thread_m * _TM + tm
            best = neg_inf
            for tn_r in al.range(_THREADS_N):
                v = sm_red[tn_r * al.convert(_BM, al.i32) + lr]
                if v > best:
                    best = v
            sm_red[lr] = best
    al.syncthreads()
    for tm in al.range(_TM):
        lr = thread_m * _TM + tm
        lse_max[lr] = sm_red[lr]

    # ── Second pass: sum exp(x - max) ───────────────────────────────────
    for r in al.range(_BM):
        lse_sum[r] = zero_f32_val

    for o_tile in al.range(al.convert(0, al.i32), O_size, al.convert(bn_val, al.i32)):
        acc = al.make_local((_TM * _TN,), al.f32)
        for e in al.range(_TM * _TN):
            acc[e] = zero_f32_val

        for k_block in al.range(al.convert(0, al.i32), H_size, al.convert(bk_val, al.i32)):
            for a_idx in al.range(tid, al.convert(num_a_elems, al.i32), al.convert(total_threads, al.i32)):
                a_r = a_idx // al.convert(bk_val, al.i32)
                a_c = a_idx % al.convert(bk_val, al.i32)
                lds_a[a_idx] = H[row_base + a_r, k_block + a_c]
            for b_idx in al.range(tid, al.convert(num_b_elems, al.i32), al.convert(total_threads, al.i32)):
                b_k = b_idx // al.convert(bn_val, al.i32)
                b_n = b_idx % al.convert(bn_val, al.i32)
                lds_b[b_idx] = W2[k_block + b_k, o_tile + b_n]

            al.syncthreads()

            for tm in al.range(_TM):
                a_l = thread_m * _TM + tm
                for tn in al.range(_TN):
                    b_l = thread_n * _TN + tn
                    ei = tm * _TN + tn
                    for kk in al.range(bk_val):
                        av = al.convert(lds_a[a_l * bk_val + kk], al.f32)
                        bv = al.convert(lds_b[kk * bn_val + b_l], al.f32)
                        acc[ei] = acc[ei] + av * bv

            al.syncthreads()

        for tm in al.range(_TM):
            lr = thread_m * _TM + tm
            row_max = lse_max[lr]
            for tn in al.range(_TN):
                gc = o_tile + thread_n * _TN + tn
                ei = tm * _TN + tn
                val = acc[ei] + al.convert(B2[gc], al.f32)
                lse_sum[lr] = lse_sum[lr] + al.exp(val - row_max)

    # ── Reduce sum across thread_n ──────────────────────────────────────
    for tm in al.range(_TM):
        lr = thread_m * _TM + tm
        sm_red[thread_n * al.convert(_BM, al.i32) + lr] = lse_sum[lr]
    al.syncthreads()
    if al.convert(thread_n, al.i32) == al.convert(0, al.i32):
        for tm in al.range(_TM):
            lr = thread_m * _TM + tm
            total = zero_f32_val
            for tn_r in al.range(_THREADS_N):
                total = total + sm_red[tn_r * al.convert(_BM, al.i32) + lr]
            sm_red[lr] = total
    al.syncthreads()
    for tm in al.range(_TM):
        lr = thread_m * _TM + tm
        lse_sum[lr] = sm_red[lr]

    # ── Final output ────────────────────────────────────────────────────
    for tm in al.range(_TM):
        lr = thread_m * _TM + tm
        gr = row_base + lr
        result = lse_max[lr] + al.log(lse_sum[lr])
        Y[gr] = al.convert(result, al.bf16)


class ModelNew(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        dev = x.device
        B_val = int(x.shape[0])
        grid_x = (B_val + _BM - 1) // _BM
        grid_y = (self.hidden_size + _BN - 1) // _BN

        w1 = self.linear1.weight.t().to(device=dev, dtype=torch.bfloat16).contiguous()
        b1 = self.linear1.bias.to(device=dev, dtype=torch.bfloat16).contiguous()
        w2 = self.linear2.weight.t().to(device=dev, dtype=torch.bfloat16).contiguous()
        b2 = self.linear2.bias.to(device=dev, dtype=torch.bfloat16).contiguous()

        h = torch.empty((B_val, self.hidden_size), device=dev, dtype=torch.bfloat16)
        y = torch.empty((B_val,), device=dev, dtype=torch.bfloat16)

        gemm1_sigmoid_kernel[lambda: ((grid_x, grid_y, 1), (_THREADS, 1, 1))](
            x.contiguous(), w1, b1, h,
            B_val, self.input_size, self.hidden_size,
        )
        gemm2_lse_kernel[lambda: ((grid_x, 1, 1), (_THREADS, 1, 1))](
            h, w2, b2, y,
            B_val, self.hidden_size, self.output_size,
        )
        return y
