import avelang
import avelang.language as al
import torch

@avelang.jit
def test_nested_range(
    input_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.f32),
    M: al.i32, N: al.i32, K: al.i32,
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

    layout_in = al.make_layout((M, N, K), (N * K, K, 1))
    inp = al.make_tensor(input_ptr, al.f32, layout_in)

    layout_out = al.make_layout((M, N), (N, 1))
    out = al.make_tensor(output_ptr, al.f32, layout_out)

    acc = al.convert(0.0, al.f32)
    for k in al.range(K):
        for i in al.range(3):
            v = inp[m, n, k]
            acc = acc + v * al.convert(0.5, al.f32)
    out[m, n] = acc

if __name__ == "__main__":
    M, N, K = 4, 8, 5
    inp = torch.randn(M, N, K, device='cuda')
    out = torch.zeros(M, N, device='cuda')
    test_nested_range[lambda: ((1, 1, 1), (32, 1, 1))](inp, out, M, N, K)
    print('Input:', inp)
    print('Output:', out)
    print('Test passed!')
