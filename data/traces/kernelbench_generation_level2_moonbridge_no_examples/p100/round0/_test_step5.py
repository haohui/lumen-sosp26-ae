import torch
import avelang
import avelang.language as al

@avelang.jit
def test_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.bf16),
    total_elems: al.i32,
    B: al.i32, IC: al.i32, OC: al.i32,
    ID: al.i32, IH: al.i32, IW: al.i32,
    OD: al.i32, OH: al.i32, OW: al.i32,
    KD: al.i32, KH: al.i32, KW: al.i32,
    stride: al.i32, padding: al.i32,
    has_bias: al.i32,
    min_value_bits: al.i32, divisor_bits: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_dim0 = al.block_dim(0)

    gid = bid * block_dim0 + tid
    if gid >= total_elems:
        return

    is_IC = ID * IH * IW
    is_ID = IH * IW
    is_IH = IW

    ws_OC = KD * KH * KW
    ws_KD = KH * KW
    ws_KH = KW

    os_OC = OD * OH * OW
    os_OD = OH * OW
    os_OH = OW

    input_t = al.make_tensor(input_ptr, al.bf16,
        al.make_layout((B, IC, ID, IH, IW), (IC * is_IC, is_IC, is_ID, is_IH, 1)))
    weight_t = al.make_tensor(weight_ptr, al.bf16,
        al.make_layout((IC, OC, KD, KH, KW), (OC * ws_OC, ws_OC, ws_KD, ws_KH, 1)))
    output_t = al.make_tensor(output_ptr, al.bf16,
        al.make_layout((B, OC, OD, OH, OW), (OC * os_OC, os_OC, os_OD, os_OH, 1)))

    min_value = al.bitcast(min_value_bits, al.f32)
    divisor_val = al.bitcast(divisor_bits, al.f32)

    ow = gid % OW
    tmp1 = gid // OW
    oh = tmp1 % OH
    tmp2 = tmp1 // OH
    od = tmp2 % OD
    tmp3 = tmp2 // OD
    oc = tmp3 % OC
    b = tmp3 // OC

    acc = al.convert(0.0, al.f32)

    for kd in al.range(KD):
        id_val = od + padding - kd
        if id_val >= 0:
            id_rem = id_val % stride
            if id_rem == 0:
                id = id_val // stride
                if id < ID:
                    for kh in al.range(KH):
                        ih_val = oh + padding - kh
                        if ih_val >= 0:
                            ih_rem = ih_val % stride
                            if ih_rem == 0:
                                ih = ih_val // stride
                                if ih < IH:
                                    for kw in al.range(KW):
                                        iw_val = ow + padding - kw
                                        if iw_val >= 0:
                                            iw_rem = iw_val % stride
                                            if iw_rem == 0:
                                                iw = iw_val // stride
                                                if iw < IW:
                                                    for ic in al.range(IC):
                                                        w_val = weight_t[ic, oc, kd, kh, kw]
                                                        in_val = input_t[b, ic, id, ih, iw]
                                                        acc = acc + al.convert(w_val, al.f32) * al.convert(in_val, al.f32)

    if has_bias != 0:
        bias_t = al.make_tensor(bias_ptr, al.f32, al.make_layout((OC,), (1,)))
        acc = acc + bias_t[oc]

    if acc < min_value:
        acc = min_value

    acc = acc / divisor_val

    output_t[b, oc, od, oh, ow] = al.convert(acc, al.bf16)

import struct
def f32b(f): return struct.unpack('<i', struct.pack('<f', float(f)))[0]

if __name__ == "__main__":
    # Medium size
    B, IC, OC = 2, 64, 4
    ID, IH, IW = 12, 24, 24
    KD, KH, KW = 3, 3, 3
    stride, padding = 2, 1
    OD = (ID - 1) * stride - 2 * padding + KD
    OH = (IH - 1) * stride - 2 * padding + KH
    OW = (IW - 1) * stride - 2 * padding + KW
    total = B * OC * OD * OH * OW
    
    print(f'Output shape: ({B}, {OC}, {OD}, {OH}, {OW}), total={total}')
    print(f'Input shape: ({B}, {IC}, {ID}, {IH}, {IW})')
    
    inp = torch.randn(B, IC, ID, IH, IW, device='cuda', dtype=torch.bfloat16)
    wgt = torch.randn(IC, OC, KD, KH, KW, device='cuda', dtype=torch.bfloat16)
    bias = torch.randn(OC, device='cuda', dtype=torch.float32)
    out = torch.zeros(B, OC, OD, OH, OW, device='cuda', dtype=torch.bfloat16)
    
    num_blocks = (total + 255) // 256
    print(f'Blocks: {num_blocks}')
    test_kernel[lambda: ((num_blocks, 1, 1), (256, 1, 1))](
        inp, wgt, bias, out, total,
        B, IC, OC, ID, IH, IW, OD, OH, OW, KD, KH, KW, stride, padding,
        1, f32b(-1.0), f32b(2.0))
    torch.cuda.synchronize()
    print('Step 5 (medium) test passed')
