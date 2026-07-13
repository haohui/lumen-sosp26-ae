import avelang
import avelang.language as al

@avelang.jit
def gemm_kernel(
    a_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
    cols_per_thread: al.i32,
    bk_inner: al.i32,
    bk_outer: al.i32,
):
    a = al.make_tensor(a_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    w = al.make_tensor(w_ptr, al.bf16, al.make_layout((N, K), (K, 1)))
    bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((N,), (1,)))
    c = al.make_tensor(c_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    row = al.block_id(1)
    tid = al.thread_id(0)
    col_start = al.block_id(0) * cols_per_thread * al.block_dim(0) + tid * cols_per_thread

    if row < M and col_start < N:
        for col in al.range(col_start, col_start + cols_per_thread):
            if col >= N:
                break
            acc = al.convert(0.0, al.f32)
            for k_block in al.range(0, K, bk_outer):
                for kk in al.range(bk_inner):
                    k_idx = k_block + kk
                    if k_idx < K:
                        a_val = al.convert(a[row, k_idx], al.f32)
                        w_val = al.convert(w[col, k_idx], al.f32)
                        acc = acc + a_val * w_val
            val = acc + al.convert(bias[col], al.f32)
            c[row, col] = al.convert(val, al.bf16)
