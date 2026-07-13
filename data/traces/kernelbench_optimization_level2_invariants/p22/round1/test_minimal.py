import torch
import avelang
import avelang.language as al

@avelang.jit
def tiny_gemm(
    X: al.Tensor((4, 8), al.bf16),
    W: al.Tensor((8, 4), al.bf16),
    Y: al.Tensor((4, 4), al.bf16),
):
    tid = al.thread_id(0)
    block_m = al.block_id(0) * 4
    block_n = al.block_id(0) * 4

    A_LDS = al.make_shared((4, 4), al.i32)
    B_LDS = al.make_shared((8, 2), al.i32)
    acc = al.make_local((1,), al.f32)
    acc[0] = al.convert(0.0, al.f32)

    rsrc_X = al.amdgpu.make_rsrc(X, al.convert(4 * 8 * 2, al.i32))
    rsrc_W = al.amdgpu.make_rsrc(W, al.convert(8 * 4 * 2, al.i32))

    a_r = tid // 4
    a_c = tid % 4
    b_r = tid // 1
    b_c = 0

    a_vindex = ((block_m + a_r) * 8 + a_c * 2) * 2
    a_data = al.amdgpu.raw_buffer_load_x4(rsrc_X, a_vindex, 0, 0)
    a_i32 = al.view(a_data, al.Tensor((4,), al.i32))
    for i in al.range(4):
        A_LDS[a_r, a_c + i * 1] = a_i32[i]

    b_vindex = ((b_r) * 4 + block_n + b_c * 8) * 2
    b_data = al.amdgpu.raw_buffer_load_x4(rsrc_W, b_vindex, 0, 0)
    b_i32 = al.view(b_data, al.Tensor((4,), al.i32))
    for i in al.range(4):
        B_LDS[b_r, b_c + i * 1] = b_i32[i]

    al.syncthreads()

    mfma_a_row = tid % 4
    mfma_a_col = (tid // 4) * 2
    mfma_b_row = tid % 8
    mfma_b_col = (tid // 8) * 2
    
    a_op = al.make_local((2,), al.i32)
    a_op[0] = A_LDS[mfma_a_row, mfma_a_col]
    a_op[1] = A_LDS[mfma_a_row, mfma_a_col + 1]
    b_op = al.make_local((2,), al.i32)
    b_op[0] = B_LDS[mfma_b_row, mfma_b_col]
    b_op[1] = B_LDS[mfma_b_row, mfma_b_col + 1]
    a_vec = al.view(a_op, al.Tensor((2,), al.i32))
    b_vec = al.view(b_op, al.Tensor((2,), al.i32))
    acc_vec = al.view(acc, al.Tensor((1,), al.f32))
    result = al.amdgpu.mfma_32x32x8_bf16_f32(a_vec, b_vec, acc_vec)
    acc[0] = result[0]

    for i in al.range(1):
        row = block_m + (tid // 4)
        col = block_n + (tid % 4)
        Y[row, col] = al.convert(acc[i], al.bf16)

print('Testing tiny GEMM...')
X = torch.ones(4, 8, device='cuda', dtype=torch.bfloat16)
W = torch.ones(8, 4, device='cuda', dtype=torch.bfloat16)
Y = torch.zeros(4, 4, device='cuda', dtype=torch.bfloat16)

try:
    tiny_gemm[lambda: ((1, 1, 1), (8, 1, 1))](X, W, Y)
    print('Y:', Y)
    print('Expected: all 8s (1*1 summed 8 times)')
except Exception as e:
    print(f'Error: {e}')
