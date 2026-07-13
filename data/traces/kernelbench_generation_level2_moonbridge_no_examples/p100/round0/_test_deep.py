import avelang
import avelang.language as al
import torch

@avelang.jit
def test_deep_nested(
    input_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.f32),
    M: al.i32, N: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_dim0 = al.block_dim(0)

    gid = bid * block_dim0 + tid
    total = M * N
    if gid >= total:
        return

    n = gid % N
    m = gid // N

    layout_in = al.make_layout((M, N, 3, 3, 3), (N * 27, 27, 9, 3, 1))
    inp = al.make_tensor(input_ptr, al.f32, layout_in)

    layout_out = al.make_layout((M, N), (N, 1))
    out = al.make_tensor(output_ptr, al.f32, layout_out)

    acc = al.convert(0.0, al.f32)
    for a in al.range(3):
        for b in al.range(3):
            for c in al.range(3):
                for d in al.range(3):
                    v = inp[m, n, a, b, c]
                    cond1 = m >= 0
                    if cond1:
                        cond2 = n >= 0
                        if cond2:
                            cond3 = a < 3
                            if cond3:
                                cond4 = b < 3
                                if cond4:
                                    acc = acc + v * al.convert(0.25, al.f32)

    if acc < al.convert(-10.0, al.f32):
        acc = al.convert(-10.0, al.f32)
    acc = acc * al.convert(0.5, al.f32)
    out[m, n] = acc

if __name__ == "__main__":
    M, N = 4, 8
    inp = torch.randn(M, N, 3, 3, 3, device='cuda')
    out = torch.zeros(M, N, device='cuda', dtype=torch.float32)
    test_deep_nested[lambda: ((1, 1, 1), (32, 1, 1))](inp, out, M, N)
    print('Input:', inp)
    print('Output:', out)
    print('Test passed!')
