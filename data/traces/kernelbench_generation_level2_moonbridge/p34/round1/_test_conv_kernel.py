import avelang
import avelang.language as al

@avelang.jit
def test_conv_flat(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    result_ptr: al.Pointer(al.f32),
    N: al.i32, Cin: al.i32, Cout: al.i32,
    D: al.i32, H: al.i32, W_in: al.i32,
    KD_v: al.i32, KH_v: al.i32, KW_v: al.i32,
    stride: al.i32, padding: al.i32,
    n: al.i32, oc: al.i32, od: al.i32, oh: al.i32, ow: al.i32,
    x_stride_n: al.i32, x_stride_c: al.i32, x_stride_d: al.i32, x_stride_h: al.i32,
    w_stride_cin: al.i32, w_stride_cout: al.i32, w_stride_kd: al.i32, w_stride_kh: al.i32,
):
    x_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((N * Cin * D * H * W_in,), (1,)))
    w_flat = al.make_tensor(w_ptr, al.bf16, al.make_layout((Cin * Cout * KD_v * KH_v * KW_v,), (1,)))
    b_flat = al.make_tensor(b_ptr, al.bf16, al.make_layout((Cout,), (1,)))
    out = al.make_tensor(result_ptr, al.f32, al.make_layout((1,), (1,)))

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
                                                        x_off = n * x_stride_n + ic * x_stride_c + id_val * x_stride_d + ih_val * x_stride_h + iw_val
                                                        w_off = ic * w_stride_cin + oc * w_stride_cout + kd * w_stride_kd + kh * w_stride_kh + kw
                                                        xv = al.convert(x_flat[x_off], al.f32)
                                                        wv = al.convert(w_flat[w_off], al.f32)
                                                        acc = acc + xv * wv
    out[0] = acc
