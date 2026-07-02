import torch
import substrate
import substrate.language as S

@substrate.jit
def probe(A: S.Tensor((64,4), S.u32), B: S.Tensor((64,4), S.u32), O: S.Tensor((64,16), S.f32)):
    lane = S.thread_id(0)
    acc = S.full((16,), 0.0, S.f32)
    a = S.view(A[lane], S.Tensor((2,4,1), S.bf16))
    b = S.view(B[lane], S.Tensor((2,4,1), S.bf16))
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a[0], b[0], acc)
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a[1], b[1], acc)
    O[lane] = acc

A_dense = torch.zeros((32,16), dtype=torch.bfloat16, device='cuda')
B_dense = torch.zeros((16,32), dtype=torch.bfloat16, device='cuda')
for r in range(32):
    k = r % 16
    A_dense[r, k] = float(r + 1)
for k in range(16):
    for c in range(32):
        B_dense[k, c] = float((k + 1) * 100 + c)

A_frag = torch.empty((64,8), dtype=torch.bfloat16, device='cuda')
B_frag = torch.empty((64,8), dtype=torch.bfloat16, device='cuda')
for lane in range(64):
    row = lane >> 1
    a_k8 = (lane & 1) * 8
    A_frag[lane] = A_dense[row, a_k8:a_k8+8]
    b_k = lane >> 3
    b_col8 = (lane & 7) * 8
    B_frag[lane] = B_dense[b_k, b_col8:b_col8+8]

A_u32 = A_frag.view(torch.uint32)
B_u32 = B_frag.view(torch.uint32)
O = torch.empty((64,16), dtype=torch.float32, device='cuda')
probe[lambda: ((1,1,1),(64,1,1))](A_u32, B_u32, O)
C = (A_dense.float() @ B_dense.float()).cpu()
O_cpu = O.cpu()

matches = []
used = set()
for lane in range(64):
    for i in range(16):
        v = float(O_cpu[lane,i].item())
        found = None
        for r in range(32):
            for c in range(32):
                if (r,c) in used:
                    continue
                if abs(float(C[r,c].item()) - v) < 1e-3:
                    found = (r,c)
                    used.add(found)
                    break
            if found:
                break
        matches.append((lane, i, v, found))

for m in matches[:64]:
    print(m)
print('matched', sum(1 for m in matches if m[3] is not None), 'of', len(matches))
