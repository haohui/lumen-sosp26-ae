import torch, sys
import avelang, avelang.language as al

@avelang.jit
def test_mfma_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    y_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    one_i = al.convert(1, al.i32)
    eight_i = al.convert(8, al.i32)
    sixteen_i = al.convert(16, al.i32)
    thirty2 = al.convert(32, al.i32)
    sixty4 = al.convert(64, al.i32)

    x_layout = al.make_layout((M, K), (K, one_i))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    w_layout = al.make_layout((K, N), (N, one_i))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)
    y_layout = al.make_layout((M, N), (N, one_i))
    y = al.make_tensor(y_ptr, al.bf16, y_layout)

    x_rsrc = al.amdgpu.make_rsrc(x, M * K * 2)
    w_rsrc = al.amdgpu.make_rsrc(w, K * N * 2)

    block_m = al.block_id(0) * sixty4
    block_n = al.block_id(0) * sixty4  # using block_id(0) for both since 1D grid
    tid = al.thread_id(0)
    warp_id = tid // sixty4
    lane = tid % sixty4
    warp_m = warp_id // 2
    warp_n = warp_id % 2

    m_base = block_m + warp_m * thirty2
    n_base = block_n + warp_n * thirty2

    # Flat LDS like test_fix
    a_lds = al.make_shared((1024,), al.bf16)
    b_lds = al.make_shared((1024,), al.bf16)

    acc = al.make_local((16,), al.f32)
    z = al.convert(0.0, al.f32)
    for i in al.range(16):
        acc[i] = z

    # Load A: 64x16
    a_row = tid % sixty4
    a_col = (tid // sixty4) * eight_i
    a_byte = ((block_m + a_row) * K + a_col) * 2
    a_load = al.amdgpu.raw_buffer_load_x4(x_rsrc, a_byte, 0, 0)
    a_bf16 = al.view(a_load, al.Tensor((8,), al.bf16))
    a_base = a_row * sixteen_i + a_col
    for v in al.range(8):
        a_lds[a_base + v] = a_bf16[v]

    # Load B: 16x64
    b_k = tid % sixteen_i
    b_n = (tid // sixteen_i) * eight_i
    b_byte = (b_k * N + block_n + b_n) * 2
    b_load = al.amdgpu.raw_buffer_load_x4(w_rsrc, b_byte, 0, 0)
    b_bf16 = al.view(b_load, al.Tensor((8,), al.bf16))
    b_base = b_k * sixty4 + b_n
    for v in al.range(8):
        b_lds[b_base + v] = b_bf16[v]

    al.syncthreads()

    # MFMA: exactly like test_fix pattern
    wm_row = warp_m * thirty2
    a_lds_row = wm_row + (lane % thirty2)
    j_val = lane % eight_i
    nq = lane // eight_i
    b_npos = warp_n * thirty2 + nq * 4

    if lane < thirty2:
        av0 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 0], al.i32), al.i32)
        av1 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 1], al.i32), al.i32)
        av2 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 2], al.i32), al.i32)
        av3 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 3], al.i32), al.i32)
        av4 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 8], al.i32), al.i32)
        av5 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 9], al.i32), al.i32)
        av6 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 10], al.i32), al.i32)
        av7 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 11], al.i32), al.i32)
        av10 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 16], al.i32), al.i32)
        av11 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 17], al.i32), al.i32)
        av12 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 18], al.i32), al.i32)
        av13 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 19], al.i32), al.i32)
        av14 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 24], al.i32), al.i32)
        av15 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 25], al.i32), al.i32)
        av16 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 26], al.i32), al.i32)
        av17 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 27], al.i32), al.i32)
    else:
        av0 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 4], al.i32), al.i32)
        av1 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 5], al.i32), al.i32)
        av2 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 6], al.i32), al.i32)
        av3 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 7], al.i32), al.i32)
        av4 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 12], al.i32), al.i32)
        av5 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 13], al.i32), al.i32)
        av6 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 14], al.i32), al.i32)
        av7 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 15], al.i32), al.i32)
        av10 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 20], al.i32), al.i32)
        av11 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 21], al.i32), al.i32)
        av12 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 22], al.i32), al.i32)
        av13 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 23], al.i32), al.i32)
        av14 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 28], al.i32), al.i32)
        av15 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 29], al.i32), al.i32)
        av16 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 30], al.i32), al.i32)
        av17 = al.convert(al.bitcast(a_lds[a_lds_row * thirty2 + 31], al.i32), al.i32)

    data_a0 = al.make_local((4,), al.i32)
    data_a1 = al.make_local((4,), al.i32)
    data_a0[0] = av0 | (av1 << sixteen_i)
    data_a0[1] = av2 | (av3 << sixteen_i)
    data_a0[2] = av4 | (av5 << sixteen_i)
    data_a0[3] = av6 | (av7 << sixteen_i)
    data_a1[0] = av10 | (av11 << sixteen_i)
    data_a1[1] = av12 | (av13 << sixteen_i)
    data_a1[2] = av14 | (av15 << sixteen_i)
    data_a1[3] = av16 | (av17 << sixteen_i)

    bv0 = al.convert(al.bitcast(b_lds[(j_val) * sixty4 + b_npos + 0], al.i32), al.i32)
    bv1 = al.convert(al.bitcast(b_lds[(j_val) * sixty4 + b_npos + 1], al.i32), al.i32)
    bv2 = al.convert(al.bitcast(b_lds[(j_val) * sixty4 + b_npos + 2], al.i32), al.i32)
    bv3 = al.convert(al.bitcast(b_lds[(j_val) * sixty4 + b_npos + 3], al.i32), al.i32)
    bv4 = al.convert(al.bitcast(b_lds[(j_val + eight_i) * sixty4 + b_npos + 0], al.i32), al.i32)
    bv5 = al.convert(al.bitcast(b_lds[(j_val + eight_i) * sixty4 + b_npos + 1], al.i32), al.i32)
    bv6 = al.convert(al.bitcast(b_lds[(j_val + eight_i) * sixty4 + b_npos + 2], al.i32), al.i32)
    bv7 = al.convert(al.bitcast(b_lds[(j_val + eight_i) * sixty4 + b_npos + 3], al.i32), al.i32)
    bv10 = al.convert(al.bitcast(b_lds[(j_val + sixteen_i) * sixty4 + b_npos + 0], al.i32), al.i32)
    bv11 = al.convert(al.bitcast(b_lds[(j_val + sixteen_i) * sixty4 + b_npos + 1], al.i32), al.i32)
    bv12 = al.convert(al.bitcast(b_lds[(j_val + sixteen_i) * sixty4 + b_npos + 2], al.i32), al.i32)
    bv13 = al.convert(al.bitcast(b_lds[(j_val + sixteen_i) * sixty4 + b_npos + 3], al.i32), al.i32)
    bv14 = al.convert(al.bitcast(b_lds[(j_val + al.convert(24, al.i32)) * sixty4 + b_npos + 0], al.i32), al.i32)
    bv15 = al.convert(al.bitcast(b_lds[(j_val + al.convert(24, al.i32)) * sixty4 + b_npos + 1], al.i32), al.i32)
    bv16 = al.convert(al.bitcast(b_lds[(j_val + al.convert(24, al.i32)) * sixty4 + b_npos + 2], al.i32), al.i32)
    bv17 = al.convert(al.bitcast(b_lds[(j_val + al.convert(24, al.i32)) * sixty4 + b_npos + 3], al.i32), al.i32)

    data_b0 = al.make_local((4,), al.i32)
    data_b1 = al.make_local((4,), al.i32)
    data_b0[0] = bv0 | (bv1 << sixteen_i)
    data_b0[1] = bv2 | (bv3 << sixteen_i)
    data_b0[2] = bv4 | (bv5 << sixteen_i)
    data_b0[3] = bv6 | (bv7 << sixteen_i)
    data_b1[0] = bv10 | (bv11 << sixteen_i)
    data_b1[1] = bv12 | (bv13 << sixteen_i)
    data_b1[2] = bv14 | (bv15 << sixteen_i)
    data_b1[3] = bv16 | (bv17 << sixteen_i)

    frag_a0 = al.view(data_a0, al.Tensor((2, 2), al.i32))
    frag_b0 = al.view(data_b0, al.Tensor((2, 2), al.i32))
    acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a0[0], frag_b0[0], acc)
    acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a0[1], frag_b0[1], acc)

    frag_a1 = al.view(data_a1, al.Tensor((2, 2), al.i32))
    frag_b1 = al.view(data_b1, al.Tensor((2, 2), al.i32))
    acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a1[0], frag_b1[0], acc)
    acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag_a1[1], frag_b1[1], acc)

    # Writeback (same as candidate)
    lane_half = lane // thirty2
    for acc_idx in al.range(16):
        row = m_base + eight_i * (acc_idx // 4) + 4 * lane_half + (acc_idx % 4)
        col = n_base + (lane % thirty2)
        if row < M:
            if col < N:
                y[row, col] = al.convert(acc[acc_idx], al.bf16)


# Test
M, N, K = 64, 64, 32
x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
w_t = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
y = torch.zeros((M, N), device='cuda', dtype=torch.bfloat16)

test_mfma_kernel[lambda: ((1, 1, 1), (256, 1, 1))](x.contiguous(), w_t, y, M, N, K)

ref = x @ w_t
d = (ref.float() - y.float()).abs()
print(f'max={d.max():.6f}, mean={d.mean():.6f}')
print(f'ref[0,:4]={ref.float()[0,:4]}')
print(f'y[0,:4]={y.float()[0,:4]}')

# All-ones
x1 = torch.ones(M,K,device='cuda',dtype=torch.bfloat16)
w1 = torch.ones(K,N,device='cuda',dtype=torch.bfloat16)
y1 = torch.zeros((M,N),device='cuda',dtype=torch.bfloat16)
test_mfma_kernel[lambda:((1,1,1),(256,1,1))](x1.contiguous(),w1,y1,M,N,K)
print(f'Ones: unique={torch.unique(y1.float()).tolist()}')
