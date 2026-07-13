import torch
import avelang
import avelang.language as al

B = 128
IC = 8
OC = 64
H = 128
W = 128
KH = 3
KW = 3
OH = H - KH + 1
OW = W - KW + 1
OC_TILE = 4
OC_TILES = OC // OC_TILE
OH_TILE = 8
OW_TILE = 8
THREADS = OC_TILE * OH_TILE * OW_TILE  # 256

@avelang.jit
def conv2d_test(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
):
    tid = al.thread_id(0)
    bid_x = al.block_id(0)
    bid_y = al.block_id(1)
    bid_z = al.block_id(2)

    batch = bid_x // OC_TILES
    oc_tile_idx = bid_x - batch * OC_TILES

    spat_per_oc = OH_TILE * OW_TILE
    oc_local = tid // spat_per_oc
    spat = tid - oc_local * spat_per_oc
    oh_local = spat // OW_TILE
    ow_local = spat - oh_local * OW_TILE

    oc = oc_tile_idx * OC_TILE + oc_local
    oh = bid_y * OH_TILE + oh_local
    ow = bid_z * OW_TILE + ow_local

    if oc < OC and oh < OH and ow < OW:
        layout_in = al.make_layout(
            (B, IC, H, W),
            (IC * H * W, H * W, W, 1),
        )
        inp = al.make_tensor(input_ptr, al.bf16, layout_in)

        layout_w = al.make_layout(
            (OC, IC, KH, KW),
            (IC * KH * KW, KH * KW, KW, 1),
        )
        wgt = al.make_tensor(weight_ptr, al.bf16, layout_w)

        layout_b = al.make_layout((OC,), (1,))
        bias = al.make_tensor(bias_ptr, al.bf16, layout_b)

        layout_out = al.make_layout(
            (B, OC, OH, OW),
            (OC * OH * OW, OH * OW, OW, 1),
        )
        out = al.make_tensor(output_ptr, al.bf16, layout_out)

        acc = al.convert(bias[oc], al.f32)

        for ic in al.range(IC):
            for kh in al.range(KH):
                for kw in al.range(KW):
                    ih = oh + kh
                    iw = ow + kw
                    ival = al.convert(inp[batch, ic, ih, iw], al.f32)
                    wval = al.convert(wgt[oc, ic, kh, kw], al.f32)
                    acc = acc + ival * wval

        out[batch, oc, oh, ow] = al.convert(acc, al.bf16)


torch.manual_seed(42)
inp = torch.randn(B, IC, H, W, dtype=torch.bfloat16, device='cuda')
wgt = torch.randn(OC, IC, KH, KW, dtype=torch.bfloat16, device='cuda')
bias = torch.randn(OC, dtype=torch.bfloat16, device='cuda')
out = torch.empty(B, OC, OH, OW, dtype=torch.bfloat16, device='cuda')

oh_tiles = (OH + OH_TILE - 1) // OH_TILE
ow_tiles = (OW + OW_TILE - 1) // OW_TILE
grid = (B * OC_TILES, oh_tiles, ow_tiles)

conv2d_test[lambda: (grid, (THREADS, 1, 1))](inp, wgt, bias, out)

ref = torch.nn.functional.conv2d(inp.float(), wgt.float(), bias.float())
max_diff = (out.float() - ref.float()).abs().max().item()
print(f"Max diff vs PyTorch: {max_diff}")
print(f"Out sum: {out.sum().item()}, Ref sum: {ref.to(torch.bfloat16).sum().item()}")
