import avelang
import avelang.language as al

@avelang.jit
def single_conv(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32, Cin: al.i32, Cout: al.i32,
    D: al.i32, H: al.i32, W_in: al.i32,
    D_out: al.i32, H_out: al.i32, W_out: al.i32,
    KD_v: al.i32, KH_v: al.i32, KW_v: al.i32,
    stride: al.i32, padding: al.i32,
    n: al.i32, oc: al.i32, od: al.i32, oh: al.i32, ow: al.i32,
):
    x_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((N * Cin * D * H * W_in,), (1,)))
    w_flat = al.make_tensor(w_ptr, al.bf16, al.make_layout((Cin * Cout * KD_v * KH_v * KW_v,), (1,)))
    b_flat = al.make_tensor(b_ptr, al.bf16, al.make_layout((Cout,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((1,), (1,)))

    acc = al.convert(b_flat[oc], al.f32)
    for ic in al.range(Cin):
        for kd in al.range(KD_v):
            idx_d = od + padding - kd
            if idx_d % stride == 0:
                id_val = idx_d // stride
                if id_val >= 0:
                    if id_val < D:
                        for kh in al.range(KH_v):
                            idx_h = oh + padding - kh
                            if idx_h % stride == 0:
                                ih_val = idx_h // stride
                                if ih_val >= 0:
                                    if ih_val < H:
                                        for kw in al.range(KW_v):
                                            idx_w = ow + padding - kw
                                            if idx_w % stride == 0:
                                                iw_val = idx_w // stride
                                                if iw_val >= 0:
                                                    if iw_val < W_in:
                                                        x_off = n * (Cin * D * H * W_in) + ic * (D * H * W_in) + id_val * (H * W_in) + ih_val * W_in + iw_val
                                                        w_off = ic * (Cout * KD_v * KH_v * KW_v) + oc * (KD_v * KH_v * KW_v) + kd * (KH_v * KW_v) + kh * KW_v + kw
                                                        xv = al.convert(x_flat[x_off], al.f32)
                                                        wv = al.convert(w_flat[w_off], al.f32)
                                                        acc = acc + xv * wv
    out[0] = al.convert(acc, al.bf16)
