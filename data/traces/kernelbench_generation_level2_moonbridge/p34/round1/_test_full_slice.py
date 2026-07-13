import avelang
import avelang.language as al

@avelang.jit
def conv_one_slice(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32, Cin: al.i32, Cout: al.i32,
    D: al.i32, H: al.i32, W_in: al.i32,
    D_out: al.i32, H_out: al.i32, W_out: al.i32,
    KD_v: al.i32, KH_v: al.i32, KW_v: al.i32,
    stride: al.i32, padding: al.i32,
    n: al.i32, oc: al.i32, od: al.i32, oh: al.i32,
):
    tid = al.thread_id(0)
    ow = tid

    x_layout = al.make_layout(
        (N, Cin, D, H, W_in),
        (Cin * D * H * W_in, D * H * W_in, H * W_in, W_in, 1),
    )
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    w_layout = al.make_layout(
        (Cin, Cout, KD_v, KH_v, KW_v),
        (Cout * KD_v * KH_v * KW_v, KD_v * KH_v * KW_v, KH_v * KW_v, KW_v, 1),
    )
    w = al.make_tensor(w_ptr, al.bf16, w_layout)
    b_layout = al.make_layout((Cout,), (1,))
    b = al.make_tensor(b_ptr, al.bf16, b_layout)
    out_layout = al.make_layout(
        (N, Cout, D_out, H_out, W_out),
        (Cout * D_out * H_out * W_out, D_out * H_out * W_out, H_out * W_out, W_out, 1),
    )
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    acc = al.convert(b[oc], al.f32)
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
                                                        xv = al.convert(x[n, ic, id_val, ih_val, iw_val], al.f32)
                                                        wv = al.convert(w[ic, oc, kd, kh, kw], al.f32)
                                                        acc = acc + xv * wv
    out[n, oc, od, oh, ow] = al.convert(acc, al.bf16)
