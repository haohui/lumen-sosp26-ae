import torch
import substrate
import substrate.language as S


@substrate.jit
def probe(A: S.Tensor((32, 16), S.bf16), B: S.Tensor((16, 32), S.bf16), C: S.Tensor((64, 16), S.f32)):
    tid = S.thread_id(0)
    lane = tid & 63
    a_rsrc = S.amdgpu.make_rsrc(A, S.convert(32 * 16 * 2, S.i32))
    b_rsrc = S.amdgpu.make_rsrc(B, S.convert(16 * 32 * 2, S.i32))
    smem = S.make_shared((512,), S.u32)
    a_words = S.subview(smem, (0,), (256,), (1,))
    b_words = S.subview(smem, (256,), (256,), (1,))
    chunk_layout = S.make_layout((64, 4), (4, 1))
    a_chunks = S.view(a_words, S.u32, chunk_layout)
    b_chunks = S.view(b_words, S.u32, chunk_layout)
    zero = S.convert(0, S.i32)

    if tid < 64:
        row = tid >> 1
        col = (tid & 1) * 8
        offset = S.convert((row * 16 + col) * 2, S.i32)
        packed = S.amdgpu.raw_buffer_load_x4(a_rsrc, zero, offset, 0)
        for i in S.range(4):
            a_words[tid * 4 + i] = packed[i]
    else:
        row = (tid - 64) >> 2
        col = ((tid - 64) & 3) * 8
        offset = S.convert((row * 32 + col) * 2, S.i32)
        packed = S.amdgpu.raw_buffer_load_x4(b_rsrc, zero, offset, 0)
        for i in S.range(4):
            b_words[(tid - 64) * 4 + i] = packed[i]

    S.syncthreads()

    c = S.full((16,), 0.0, S.f32)
    a = S.view(a_chunks[lane], S.Tensor((2, 4, 1), S.bf16))
    b = S.view(b_chunks[(lane >> 2) * 4 + (lane & 3)], S.Tensor((2, 4, 1), S.bf16))
    c = S.amdgpu.mfma_32x32x8_bf16_f32(a[0], b[0], c)
    c = S.amdgpu.mfma_32x32x8_bf16_f32(a[1], b[1], c)
    C[lane] = c


def main():
    a_tile = torch.zeros((32, 16), dtype=torch.bfloat16, device="cuda")
    b_tile = torch.zeros((16, 32), dtype=torch.bfloat16, device="cuda")
    for m in range(32):
        a_tile[m, 0] = 1
        a_tile[m, 1] = m
    for n in range(32):
        b_tile[0, n] = n
        b_tile[1, n] = 64

    c = torch.empty((64, 16), dtype=torch.float32, device="cuda")
    probe[lambda: ((1, 1, 1), (128, 1, 1))](a_tile, b_tile, c)

    raw = c.cpu()
    ref = (a_tile.float() @ b_tile.float()).cpu()
    value_to_pos = {}
    for i in range(32):
        for j in range(32):
            value_to_pos[float(ref[i, j].item())] = (i, j)

    for lane in range(64):
        for elem in range(16):
            v = float(raw[lane, elem].item())
            print(lane, elem, *value_to_pos[v])


if __name__ == "__main__":
    main()
