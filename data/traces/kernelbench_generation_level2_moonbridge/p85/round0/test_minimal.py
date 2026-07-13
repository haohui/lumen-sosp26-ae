import torch
import avelang
import avelang.language as al

@avelang.jit
def test_kernel(
    inp: al.Pointer(al.bf16),
    out: al.Pointer(al.bf16),
):
    tid = al.thread_id(0)
    layout = al.make_layout((128,), (1,))
    i = al.make_tensor(inp, al.bf16, layout)
    o = al.make_tensor(out, al.bf16, layout)
    if tid < 128:
        o[tid] = i[tid]

x = torch.randn(128, dtype=torch.bfloat16, device='cuda')
y = torch.empty_like(x)
test_kernel[lambda: ((1, 1, 1), (128, 1, 1))](x, y)
print("Minimal kernel OK, sum:", y.sum().item())
