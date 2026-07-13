import torch
import avelang
import avelang.language as al

@avelang.jit
def test_kernel(
    output_ptr: al.Pointer(al.bf16),
    total_elems: al.i32,
    B: al.i32, OC: al.i32,
    OD: al.i32, OH: al.i32, OW: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_dim0 = al.block_dim(0)

    gid = bid * block_dim0 + tid
    if gid >= total_elems:
        return

    os_OC = OD * OH * OW
    os_OD = OH * OW
    os_OH = OW

    output_t = al.make_tensor(output_ptr, al.bf16,
        al.make_layout((B, OC, OD, OH, OW), (OC * os_OC, os_OC, os_OD, os_OH, 1)))

    ow = gid % OW
    tmp1 = gid // OW
    oh = tmp1 % OH
    tmp2 = tmp1 // OH
    od = tmp2 % OD
    tmp3 = tmp2 // OD
    oc = tmp3 % OC
    b = tmp3 // OC

    output_t[b, oc, od, oh, ow] = al.convert(0.0, al.bf16)

if __name__ == "__main__":
    B, OC, OD, OH, OW = 1, 1, 7, 17, 17
    total = B * OC * OD * OH * OW
    out = torch.zeros(B, OC, OD, OH, OW, device='cuda', dtype=torch.bfloat16)
    num_blocks = (total + 255) // 256
    test_kernel[lambda: ((num_blocks, 1, 1), (256, 1, 1))](out, total, B, OC, OD, OH, OW)
    torch.cuda.synchronize()
    print('Minimal test passed')
    print('Output sample:', out[0,0,0,0,0].item())
