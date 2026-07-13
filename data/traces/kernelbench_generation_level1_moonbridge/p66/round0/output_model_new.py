import torch
import torch.nn as nn
import avelang
import avelang.language as al


BLOCK_OC = 16
BLOCK_SPATIAL = 16
THREADS = BLOCK_OC * BLOCK_SPATIAL
BF16_BYTES = 2
VEC_ELEMS = 8
SHM_ROWS = THREADS


@avelang.jit
def conv3d_partial_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    acc_ptr: al.Pointer(al.f32),
    B: al.u32,
    IC: al.u32,
    OC: al.u32,
    D: al.u32,
    H: al.u32,
    W: al.u32,
    KD: al.u32,
    D_out: al.u32,
    H_out: al.u32,
    W_out: al.u32,
    SPATIAL_OUT: al.u32,
    X_TOTAL: al.u32,
    W_TOTAL: al.u32,
    O_TOTAL: al.u32,
    IC_IDX: al.u32,
    KD_IDX: al.u32,
    IC_KD_KH_KW: al.u32,
    KD_KH_KW: al.u32,
):
    tid = al.thread_id(0)
    spatial_block = al.block_id(0)
    oc_block = al.block_id(1)
    batch = al.block_id(2)

    oc_thread = tid // BLOCK_SPATIAL
    spatial_thread = tid % BLOCK_SPATIAL

    oc = oc_block * BLOCK_OC + oc_thread
    spatial_idx = spatial_block * BLOCK_SPATIAL + spatial_thread

    if oc >= OC:
        return
    if spatial_idx >= SPATIAL_OUT:
        return

    ow = spatial_idx % W_out
    temp = spatial_idx // W_out
    oh = temp % H_out
    od = temp // H_out

    x_1d = al.make_tensor(x_ptr, al.bf16, al.make_layout((X_TOTAL,), (1,)))
    w_1d = al.make_tensor(w_ptr, al.bf16, al.make_layout((W_TOTAL,), (1,)))
    acc_1d = al.make_tensor(acc_ptr, al.f32, al.make_layout((O_TOTAL,), (1,)))

    x_rsrc = al.amdgpu.make_rsrc(x_1d, X_TOTAL * BF16_BYTES)
    w_rsrc = al.amdgpu.make_rsrc(w_1d, W_TOTAL * BF16_BYTES)

    zero = al.convert(0, al.u32)

    shm_x = al.make_shared((SHM_ROWS, 4), al.u32)
    shm_w = al.make_shared((SHM_ROWS, 4), al.u32)
    x_bf16 = al.view(shm_x, al.Tensor((SHM_ROWS * VEC_ELEMS,), al.bf16))
    w_bf16 = al.view(shm_w, al.Tensor((SHM_ROWS * VEC_ELEMS,), al.bf16))

    x_batch_off = batch * IC * D * H * W
    x_ic_off = x_batch_off + IC_IDX * D * H * W
    w_ic_off = oc * IC_KD_KH_KW + IC_IDX * KD_KH_KW
    x_d_off = x_ic_off + (od + KD_IDX) * H * W
    w_kd_off = w_ic_off + KD_IDX * 35

    out_idx = batch * OC * D_out * H_out * W_out + oc * D_out * H_out * W_out + od * H_out * W_out + oh * W_out + ow
    prev_acc = acc_1d[out_idx]

    # KH=0
    x_h = x_d_off + (oh + 0) * W
    w_kh = w_kd_off + 0 * 7
    _x0 = x_h + ow + 0; _w0 = w_kh + 0; xb0 = (_x0 // 8) * 8; xr0 = _x0 - xb0; wb0 = (_w0 // 8) * 8; wr0 = _w0 - wb0
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb0 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb0 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr0], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr0], al.f32)

    _x1 = x_h + ow + 1; _w1 = w_kh + 1; xb1 = (_x1 // 8) * 8; xr1 = _x1 - xb1; wb1 = (_w1 // 8) * 8; wr1 = _w1 - wb1
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb1 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb1 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr1], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr1], al.f32)

    _x2 = x_h + ow + 2; _w2 = w_kh + 2; xb2 = (_x2 // 8) * 8; xr2 = _x2 - xb2; wb2 = (_w2 // 8) * 8; wr2 = _w2 - wb2
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb2 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb2 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr2], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr2], al.f32)

    _x3 = x_h + ow + 3; _w3 = w_kh + 3; xb3 = (_x3 // 8) * 8; xr3 = _x3 - xb3; wb3 = (_w3 // 8) * 8; wr3 = _w3 - wb3
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb3 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb3 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr3], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr3], al.f32)

    _x4 = x_h + ow + 4; _w4 = w_kh + 4; xb4 = (_x4 // 8) * 8; xr4 = _x4 - xb4; wb4 = (_w4 // 8) * 8; wr4 = _w4 - wb4
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb4 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb4 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr4], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr4], al.f32)

    _x5 = x_h + ow + 5; _w5 = w_kh + 5; xb5 = (_x5 // 8) * 8; xr5 = _x5 - xb5; wb5 = (_w5 // 8) * 8; wr5 = _w5 - wb5
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb5 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb5 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr5], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr5], al.f32)

    _x6 = x_h + ow + 6; _w6 = w_kh + 6; xb6 = (_x6 // 8) * 8; xr6 = _x6 - xb6; wb6 = (_w6 // 8) * 8; wr6 = _w6 - wb6
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb6 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb6 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr6], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr6], al.f32)

    # KH=1
    x_h1 = x_d_off + (oh + 1) * W
    w_kh1 = w_kd_off + 1 * 7
    _x0 = x_h1 + ow + 0; _w0 = w_kh1 + 0; xb0 = (_x0 // 8) * 8; xr0 = _x0 - xb0; wb0 = (_w0 // 8) * 8; wr0 = _w0 - wb0
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb0 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb0 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr0], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr0], al.f32)

    _x1 = x_h1 + ow + 1; _w1 = w_kh1 + 1; xb1 = (_x1 // 8) * 8; xr1 = _x1 - xb1; wb1 = (_w1 // 8) * 8; wr1 = _w1 - wb1
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb1 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb1 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr1], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr1], al.f32)

    _x2 = x_h1 + ow + 2; _w2 = w_kh1 + 2; xb2 = (_x2 // 8) * 8; xr2 = _x2 - xb2; wb2 = (_w2 // 8) * 8; wr2 = _w2 - wb2
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb2 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb2 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr2], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr2], al.f32)

    _x3 = x_h1 + ow + 3; _w3 = w_kh1 + 3; xb3 = (_x3 // 8) * 8; xr3 = _x3 - xb3; wb3 = (_w3 // 8) * 8; wr3 = _w3 - wb3
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb3 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb3 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr3], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr3], al.f32)

    _x4 = x_h1 + ow + 4; _w4 = w_kh1 + 4; xb4 = (_x4 // 8) * 8; xr4 = _x4 - xb4; wb4 = (_w4 // 8) * 8; wr4 = _w4 - wb4
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb4 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb4 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr4], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr4], al.f32)

    _x5 = x_h1 + ow + 5; _w5 = w_kh1 + 5; xb5 = (_x5 // 8) * 8; xr5 = _x5 - xb5; wb5 = (_w5 // 8) * 8; wr5 = _w5 - wb5
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb5 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb5 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr5], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr5], al.f32)

    _x6 = x_h1 + ow + 6; _w6 = w_kh1 + 6; xb6 = (_x6 // 8) * 8; xr6 = _x6 - xb6; wb6 = (_w6 // 8) * 8; wr6 = _w6 - wb6
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb6 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb6 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr6], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr6], al.f32)

    # KH=2
    x_h2 = x_d_off + (oh + 2) * W
    w_kh2 = w_kd_off + 2 * 7
    _x0 = x_h2 + ow + 0; _w0 = w_kh2 + 0; xb0 = (_x0 // 8) * 8; xr0 = _x0 - xb0; wb0 = (_w0 // 8) * 8; wr0 = _w0 - wb0
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb0 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb0 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr0], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr0], al.f32)

    _x1 = x_h2 + ow + 1; _w1 = w_kh2 + 1; xb1 = (_x1 // 8) * 8; xr1 = _x1 - xb1; wb1 = (_w1 // 8) * 8; wr1 = _w1 - wb1
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb1 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb1 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr1], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr1], al.f32)

    _x2 = x_h2 + ow + 2; _w2 = w_kh2 + 2; xb2 = (_x2 // 8) * 8; xr2 = _x2 - xb2; wb2 = (_w2 // 8) * 8; wr2 = _w2 - wb2
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb2 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb2 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr2], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr2], al.f32)

    _x3 = x_h2 + ow + 3; _w3 = w_kh2 + 3; xb3 = (_x3 // 8) * 8; xr3 = _x3 - xb3; wb3 = (_w3 // 8) * 8; wr3 = _w3 - wb3
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb3 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb3 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr3], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr3], al.f32)

    _x4 = x_h2 + ow + 4; _w4 = w_kh2 + 4; xb4 = (_x4 // 8) * 8; xr4 = _x4 - xb4; wb4 = (_w4 // 8) * 8; wr4 = _w4 - wb4
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb4 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb4 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr4], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr4], al.f32)

    _x5 = x_h2 + ow + 5; _w5 = w_kh2 + 5; xb5 = (_x5 // 8) * 8; xr5 = _x5 - xb5; wb5 = (_w5 // 8) * 8; wr5 = _w5 - wb5
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb5 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb5 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr5], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr5], al.f32)

    _x6 = x_h2 + ow + 6; _w6 = w_kh2 + 6; xb6 = (_x6 // 8) * 8; xr6 = _x6 - xb6; wb6 = (_w6 // 8) * 8; wr6 = _w6 - wb6
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb6 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb6 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr6], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr6], al.f32)

    # KH=3
    x_h3 = x_d_off + (oh + 3) * W
    w_kh3 = w_kd_off + 3 * 7
    _x0 = x_h3 + ow + 0; _w0 = w_kh3 + 0; xb0 = (_x0 // 8) * 8; xr0 = _x0 - xb0; wb0 = (_w0 // 8) * 8; wr0 = _w0 - wb0
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb0 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb0 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr0], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr0], al.f32)

    _x1 = x_h3 + ow + 1; _w1 = w_kh3 + 1; xb1 = (_x1 // 8) * 8; xr1 = _x1 - xb1; wb1 = (_w1 // 8) * 8; wr1 = _w1 - wb1
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb1 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb1 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr1], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr1], al.f32)

    _x2 = x_h3 + ow + 2; _w2 = w_kh3 + 2; xb2 = (_x2 // 8) * 8; xr2 = _x2 - xb2; wb2 = (_w2 // 8) * 8; wr2 = _w2 - wb2
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb2 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb2 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr2], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr2], al.f32)

    _x3 = x_h3 + ow + 3; _w3 = w_kh3 + 3; xb3 = (_x3 // 8) * 8; xr3 = _x3 - xb3; wb3 = (_w3 // 8) * 8; wr3 = _w3 - wb3
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb3 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb3 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr3], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr3], al.f32)

    _x4 = x_h3 + ow + 4; _w4 = w_kh3 + 4; xb4 = (_x4 // 8) * 8; xr4 = _x4 - xb4; wb4 = (_w4 // 8) * 8; wr4 = _w4 - wb4
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb4 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb4 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr4], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr4], al.f32)

    _x5 = x_h3 + ow + 5; _w5 = w_kh3 + 5; xb5 = (_x5 // 8) * 8; xr5 = _x5 - xb5; wb5 = (_w5 // 8) * 8; wr5 = _w5 - wb5
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb5 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb5 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr5], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr5], al.f32)

    _x6 = x_h3 + ow + 6; _w6 = w_kh3 + 6; xb6 = (_x6 // 8) * 8; xr6 = _x6 - xb6; wb6 = (_w6 // 8) * 8; wr6 = _w6 - wb6
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb6 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb6 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr6], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr6], al.f32)

    # KH=4
    x_h4 = x_d_off + (oh + 4) * W
    w_kh4 = w_kd_off + 4 * 7
    _x0 = x_h4 + ow + 0; _w0 = w_kh4 + 0; xb0 = (_x0 // 8) * 8; xr0 = _x0 - xb0; wb0 = (_w0 // 8) * 8; wr0 = _w0 - wb0
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb0 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb0 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr0], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr0], al.f32)

    _x1 = x_h4 + ow + 1; _w1 = w_kh4 + 1; xb1 = (_x1 // 8) * 8; xr1 = _x1 - xb1; wb1 = (_w1 // 8) * 8; wr1 = _w1 - wb1
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb1 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb1 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr1], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr1], al.f32)

    _x2 = x_h4 + ow + 2; _w2 = w_kh4 + 2; xb2 = (_x2 // 8) * 8; xr2 = _x2 - xb2; wb2 = (_w2 // 8) * 8; wr2 = _w2 - wb2
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb2 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb2 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr2], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr2], al.f32)

    _x3 = x_h4 + ow + 3; _w3 = w_kh4 + 3; xb3 = (_x3 // 8) * 8; xr3 = _x3 - xb3; wb3 = (_w3 // 8) * 8; wr3 = _w3 - wb3
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb3 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb3 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr3], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr3], al.f32)

    _x4 = x_h4 + ow + 4; _w4 = w_kh4 + 4; xb4 = (_x4 // 8) * 8; xr4 = _x4 - xb4; wb4 = (_w4 // 8) * 8; wr4 = _w4 - wb4
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb4 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb4 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr4], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr4], al.f32)

    _x5 = x_h4 + ow + 5; _w5 = w_kh4 + 5; xb5 = (_x5 // 8) * 8; xr5 = _x5 - xb5; wb5 = (_w5 // 8) * 8; wr5 = _w5 - wb5
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb5 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb5 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr5], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr5], al.f32)

    _x6 = x_h4 + ow + 6; _w6 = w_kh4 + 6; xb6 = (_x6 // 8) * 8; xr6 = _x6 - xb6; wb6 = (_w6 // 8) * 8; wr6 = _w6 - wb6
    shm_x[tid] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, xb6 * BF16_BYTES, 0)
    shm_w[tid] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, wb6 * BF16_BYTES, 0)
    al.syncthreads()
    prev_acc = prev_acc + al.convert(x_bf16[tid * VEC_ELEMS + xr6], al.f32) * al.convert(w_bf16[tid * VEC_ELEMS + wr6], al.f32)

    acc_1d[out_idx] = prev_acc


def avelang_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = x.contiguous().cuda().to(torch.bfloat16)
    w_bf16 = weight.contiguous().cuda().to(torch.bfloat16)

    B, IC, D, H_in, W_in = x_bf16.shape
    OC, w_IC, wKD, wKH, wKW = w_bf16.shape
    SD, SH, SW = stride
    PD, PH, PW = padding
    DD, DH, DW = dilation

    D_out = (D + 2 * PD - DD * (wKD - 1) - 1) // SD + 1
    H_out = (H_in + 2 * PH - DH * (wKH - 1) - 1) // SH + 1
    W_out = (W_in + 2 * PW - DW * (wKW - 1) - 1) // SW + 1

    spatial_out = D_out * H_out * W_out

    x_total = B * IC * D * H_in * W_in
    w_total = OC * IC * wKD * wKH * wKW
    o_total = B * OC * D_out * H_out * W_out
    ic_kd_kh_kw = IC * wKD * wKH * wKW
    kd_kh_kw = wKD * wKH * wKW

    out = torch.empty((B, OC, D_out, H_out, W_out), device=x_bf16.device, dtype=torch.bfloat16)

    spatial_blocks = (spatial_out + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL
    oc_blocks = (OC + BLOCK_OC - 1) // BLOCK_OC

    grid = (spatial_blocks, oc_blocks, B)

    # Accumulator buffer in f32
    acc_buf = torch.zeros((o_total,), device=x_bf16.device, dtype=torch.float32)

    for ic_idx in range(IC):
        for kd_idx in range(wKD):
            conv3d_partial_kernel[lambda: (grid, (THREADS, 1, 1))](
                x_bf16, w_bf16, acc_buf,
                B, IC, OC, D, H_in, W_in, wKD,
                D_out, H_out, W_out,
                spatial_out,
                x_total, w_total, o_total,
                ic_idx, kd_idx,
                ic_kd_kh_kw, kd_kh_kw,
            )

    # Convert f32 accumulator to bf16 output
    out = acc_buf.view(B, OC, D_out, H_out, W_out).to(torch.bfloat16)

    if bias is not None:
        bias_bf16 = bias.contiguous().cuda().to(torch.bfloat16)
        out = out + bias_bf16.view(1, OC, 1, 1, 1)

    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        dilation: tuple = (1, 1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv3d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation
        )


# Test code
batch_size = 8
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)
depth = 16
height = 128
width = 128


def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
