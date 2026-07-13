import avelang
import avelang.language as al

W_NORM: al.constexpr = 64

@avelang.jit
def ln_stats(
    x_ptr: al.Pointer(al.bf16),
    out_mean_ptr: al.Pointer(al.f32),
    out_var_ptr: al.Pointer(al.f32),
    W_val: al.i32,
    eps: al.f32,
):
    tid = al.thread_id(0)
    flat_layout = al.make_layout((W_val,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, flat_layout)

    smem = al.make_shared((W_NORM,), al.f32)
    smem_sq = al.make_shared((W_NORM,), al.f32)

    x_val = al.convert(x[tid], al.f32)
    smem[tid] = x_val
    smem_sq[tid] = x_val * x_val
    al.syncthreads()

    if tid < 32:
        smem[tid] = smem[tid] + smem[tid + 32]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem[tid] = smem[tid] + smem[tid + 16]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem[tid] = smem[tid] + smem[tid + 8]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem[tid] = smem[tid] + smem[tid + 4]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem[tid] = smem[tid] + smem[tid + 2]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem[tid] = smem[tid] + smem[tid + 1]
        smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 1]
    al.syncthreads()

    if tid == 0:
        out_mean = al.make_tensor(out_mean_ptr, al.f32, al.make_layout((1,), (1,)))
        out_var = al.make_tensor(out_var_ptr, al.f32, al.make_layout((1,), (1,)))
        slice_sum = smem[0]
        slice_sq = smem_sq[0]
        W_f32 = al.convert(W_val, al.f32)
        mean_val = slice_sum / W_f32
        var_val = slice_sq / W_f32 - mean_val * mean_val
        if var_val < al.convert(0.0, al.f32):
            var_val = al.convert(0.0, al.f32)
        out_mean[0] = mean_val
        out_var[0] = var_val
