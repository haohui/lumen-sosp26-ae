import avelang
import avelang.language as al

W_NORM: al.constexpr = 64

@avelang.jit
def ln_gelu_only(
    x_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    W_val: al.i32,
    eps: al.f32,
    scaling_factor: al.f32,
):
    tid = al.thread_id(0)
    w_idx = tid

    flat_layout = al.make_layout((W_val,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, flat_layout)
    out = al.make_tensor(out_ptr, al.bf16, flat_layout)
    gamma = al.make_tensor(gamma_ptr, al.bf16, flat_layout)
    beta_t = al.make_tensor(beta_ptr, al.bf16, flat_layout)

    smem = al.make_shared((W_NORM,), al.f32)
    smem_sq = al.make_shared((W_NORM,), al.f32)

    x_val = al.convert(x[w_idx], al.f32)
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

    slice_sum = smem[0]
    slice_sq = smem_sq[0]
    W_f32 = al.convert(W_val, al.f32)
    mean = slice_sum / W_f32
    var = slice_sq / W_f32 - mean * mean
    zero_f32 = al.convert(0.0, al.f32)
    if var < zero_f32:
        var = zero_f32
    rstd = al.convert(1.0, al.f32) / al.sqrt(var + eps)

    normalized = (x_val - mean) * rstd
    g_val = al.convert(gamma[w_idx], al.f32)
    b_val = al.convert(beta_t[w_idx], al.f32)
    ln_out = normalized * g_val + b_val

    c1 = al.convert(0.7978845608028654, al.f32)
    c2 = al.convert(0.044715, al.f32)
    half = al.convert(0.5, al.f32)
    one = al.convert(1.0, al.f32)
    x3 = ln_out * ln_out * ln_out
    inner = c1 * (ln_out + c2 * x3)
    tanh_val = al.tanh(inner)
    gelu_val = half * ln_out * (one + tanh_val)

    result = gelu_val * scaling_factor
    out[w_idx] = al.convert(result, al.bf16)
