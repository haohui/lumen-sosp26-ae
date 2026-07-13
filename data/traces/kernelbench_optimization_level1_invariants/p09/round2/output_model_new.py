import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def _load_a_shm(
    shm_a: al.Tensor((256, 4), al.u32),
    a_rsrc: al.Tensor((4,), al.u32),
    block_m: al.u32,
    k_base: al.u32,
    tid: al.u32,
):
    idx = tid
    zero = al.convert(0, al.u32)
    for _ in al.range(1):
        row = idx // 2
        col_vec = idx % 2
        off = ((block_m * 128 + row) * 32 + k_base + col_vec * 8) * 2
        shm_a[idx] = al.amdgpu.raw_buffer_load_x4(a_rsrc, zero, off, 0)
        idx += 256


@avelang.jit
def _load_b_shm(
    shm_b: al.Tensor((256, 4), al.u32),
    b_rsrc: al.Tensor((4,), al.u32),
    block_n: al.u32,
    k_base: al.u32,
    tid: al.u32,
):
    idx = tid
    zero = al.convert(0, al.u32)
    for _ in al.range(1):
        row = idx // 2
        col_vec = idx % 2
        off = ((block_n * 128 + row) * 32 + k_base + col_vec * 8) * 2
        shm_b[idx] = al.amdgpu.raw_buffer_load_x4(b_rsrc, zero, off, 0)
        idx += 256


@avelang.jit
def _fetch_operand(
    ret: al.Tensor((2, 4), al.bf16),
    shm: al.Tensor((256, 4), al.u32),
    tile_idx: al.u32,
    lane: al.u32,
):
    ret_u32 = al.view(ret, al.Tensor((4,), al.u32))
    shm_u32 = al.view(shm, al.Tensor((1024,), al.u32))
    row = tile_idx * 32 + (lane % 32)
    k_u32 = (lane // 32) * 2
    row_base = row * 8
    ret_u32[0] = shm_u32[row_base + k_u32]
    ret_u32[1] = shm_u32[row_base + k_u32 + 1]
    ret_u32[2] = shm_u32[row_base + 4 + k_u32]
    ret_u32[3] = shm_u32[row_base + 5 + k_u32]


@avelang.jit
def gemm_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
):
    tid = al.thread_id(0)
    block_m = al.block_id(1)
    block_n = al.block_id(0)
    wid = tid // 64
    lane = tid % 64
    warp_row = wid // 2
    warp_col = wid % 2

    a_mem = al.make_tensor(a_ptr, al.bf16, al.make_layout((32768 * 32,), (1,)))
    b_mem = al.make_tensor(b_ptr, al.bf16, al.make_layout((32768 * 32,), (1,)))
    c_mem = al.make_tensor(c_ptr, al.bf16, al.make_layout((32768, 32768), (32768, 1)))
    a_rsrc = al.amdgpu.make_rsrc(a_mem, 32768 * 32 * 2)
    b_rsrc = al.amdgpu.make_rsrc(b_mem, 32768 * 32 * 2)

    shm_a = al.make_shared((256, 4), al.u32)
    shm_b = al.make_shared((256, 4), al.u32)
    a_reg = al.make_local((2, 2, 4), al.bf16)
    b_reg = al.make_local((2, 2, 4), al.bf16)
    acc = al.make_local((4, 16), al.f32)

    for i in al.range(4):
        for j in al.range(16):
            acc[i, j] = al.convert(0.0, al.f32)

    for kt in al.range(2):
        k_base = kt * 16
        _load_a_shm(shm_a, a_rsrc, block_m, k_base, tid)
        _load_b_shm(shm_b, b_rsrc, block_n, k_base, tid)
        al.syncthreads()

        for i in al.range(2):
            _fetch_operand(a_reg[i], shm_a, warp_row * 2 + i, lane)
        for j in al.range(2):
            _fetch_operand(b_reg[j], shm_b, warp_col * 2 + j, lane)

        for i in al.range(2):
            for j in al.range(2):
                acc_idx = i * 2 + j
                a_vec0 = al.view(a_reg[i, 0], al.Tensor((2,), al.u32))
                b_vec0 = al.view(b_reg[j, 0], al.Tensor((2,), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_vec0, b_vec0, acc[acc_idx])
                a_vec1 = al.view(a_reg[i, 1], al.Tensor((2,), al.u32))
                b_vec1 = al.view(b_reg[j, 1], al.Tensor((2,), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_vec1, b_vec1, acc[acc_idx])

        al.syncthreads()

    lane_group = lane // 32
    lane_col = lane % 32

    for j in al.range(2):
        col = block_n * 128 + (warp_col * 2 + j) * 32 + lane_col
        for i in al.range(2):
            acc_idx = i * 2 + j
            row_base = block_m * 128 + (warp_row * 2 + i) * 32
            for t in al.range(16):
                row = row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
                c_mem[row, col] = al.convert(acc[acc_idx, t], al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise RuntimeError('Kernel requires bfloat16 inputs.')
        if A.device != B.device:
            raise RuntimeError('A and B must be on the same device.')

        A = A.contiguous()
        B_T = B.T.contiguous()
        C = torch.empty((32768, 32768), device=A.device, dtype=torch.bfloat16)

        grid_n = 32768 // 128
        grid_m = 32768 // 128

        gemm_kernel[lambda: ((grid_n, grid_m, 1), (256, 1, 1))](A, B_T, C)
        return C
