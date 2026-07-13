import avelang
import avelang.language as al

@avelang.jit
def conv_transpose2d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
    OH: al.i32,
    OW: al.i32,
    K: al.i32,
    S: al.i32,
    P: al.i32,
    SPATIAL_SIZE: al.i32,
    TILE_SIZE: al.i32,
):
    input_layout = al.make_layout((B, IC, H, W), (IC * H * W, H * W, W, 1))
    input_t = al.make_tensor(input_ptr, al.bf16, input_layout)
    weight_layout = al.make_layout((IC, OC, K, K), (OC * K * K, K * K, K, 1))
    weight_t = al.make_tensor(weight_ptr, al.bf16, weight_layout)
    bias_layout = al.make_layout((OC,), (1,))
    bias_t = al.make_tensor(bias_ptr, al.bf16, bias_layout)
    output_layout = al.make_layout((B, OC, OH, OW), (OC * OH * OW, OH * OW, OW, 1))
    output_t = al.make_tensor(output_ptr, al.bf16, output_layout)

    block_spatial = al.block_id(0)
    block_bc = al.block_id(1)
    tid = al.thread_id(0)
    spatial_idx = block_spatial * TILE_SIZE + tid

    if spatial_idx < SPATIAL_SIZE:
        b = block_bc // OC
        oc = block_bc % OC
        if b < B:
            oh = spatial_idx // OW
            ow = spatial_idx % OW
            accum = al.convert(0.0, al.f32)
            for ic in al.range(IC):
                for ky in al.range(K):
                    ih_base = oh + P - ky
                    if (ih_base % S) == 0:
                        ih = ih_base // S
                        if ih >= 0:
                            if ih < H:
                                for kx in al.range(K):
                                    iw_base = ow + P - kx
                                    if (iw_base % S) == 0:
                                        iw = iw_base // S
                                        if iw >= 0:
                                            if iw < W:
                                                inp_val = al.convert(input_t[b, ic, ih, iw], al.f32)
                                                w_val = al.convert(weight_t[ic, oc, ky, kx], al.f32)
                                                accum = accum + inp_val * w_val
            bias_val = al.convert(bias_t[oc], al.f32)
            accum = accum + bias_val
            output_t[b, oc, oh, ow] = al.convert(accum, al.bf16)
