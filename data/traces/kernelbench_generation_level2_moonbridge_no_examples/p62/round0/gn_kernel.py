import avelang
import avelang.language as al

@avelang.jit
def groupnorm_leakyrelu_double_kernel(
    x_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    G: al.i32,
    eps_bits: al.i32,
    neg_slope_bits: al.i32,
):
    eps = al.bitcast(eps_bits, al.f32)
    neg_slope = al.bitcast(neg_slope_bits, al.f32)

    total_pairs = B * G

    tid = al.thread_id(0)
    bid = al.block_id(0)
    bdim = al.block_dim(0)

    pair_idx = bid * bdim + tid
    if pair_idx >= total_pairs:
        return

    batch_idx = pair_idx // G
    group_idx = pair_idx % G
    channel_start = group_idx * 16

    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((B, C), (C, 1)))
    gamma = al.make_tensor(gamma_ptr, al.bf16, al.make_layout((C,), (1,)))
    beta = al.make_tensor(beta_ptr, al.bf16, al.make_layout((C,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((B, C), (C, 1)))

    vals = al.make_local((16,), al.f32)
    for d in al.range(16):
        vals[d] = al.convert(x[batch_idx, channel_start + d], al.f32)

    mean = al.convert(0.0, al.f32)
    for d in al.range(16):
        mean = mean + vals[d]
    mean = mean / al.convert(16, al.f32)

    var = al.convert(0.0, al.f32)
    for d in al.range(16):
        diff = vals[d] - mean
        var = var + diff * diff
    var = var / al.convert(16, al.f32)

    for d in al.range(16):
        norm_val = (vals[d] - mean) / al.sqrt(var + eps)
        ch = channel_start + d
        g = al.convert(gamma[ch], al.f32)
        b = al.convert(beta[ch], al.f32)
        affine_val = norm_val * g + b
        zero = al.convert(0.0, al.f32)
        relu_val = al.convert(0.0, al.f32)
        if affine_val > zero:
            relu_val = affine_val
        else:
            relu_val = neg_slope * affine_val
        result = relu_val + relu_val
        out[batch_idx, channel_start + d] = al.convert(result, al.bf16)
