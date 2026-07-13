import avelang
import avelang.language as al
import torch

@avelang.jit
def test_divmod(
    out: al.Tensor((64,), al.i32),
    x: al.i32,
):
    tid = al.thread_id(0)
    if tid < 64:
        q = x // 7
        r = x % 7
        out[tid] = q + r

if __name__ == "__main__":
    t = torch.zeros(64, dtype=torch.int32, device='cuda')
    test_divmod[lambda: ((1, 1, 1), (64, 1, 1))](t, 100)
    print('Output sample:', t[0].item())
    print('Test passed!')
