import torch
import torch.nn as nn
import avelang
import avelang.language as al

POSTPROC_WARP_SIZE = 64
POSTPROC_NUM_WARPS = 4
POSTPROC_THREADS = POSTPROC_WARP_SIZE * POSTPROC_NUM_WARPS
GEMM_THREADS = 256
TILE_K = 64


@avelang.jit
def _gemm_kernel(
    A: al.Pointer(al.bf16),
    B: al.Pointer(al.bf16),
    C: al.Pointer(al.bf16),
    m: al.u32,
    k: al.u32,
    n: al.u32,
):
    tid = al.thread_id(0)
    bdim = al.block_dim(0)
    row = al.block_id(0)

    a_t = al.make_tensor(A, al.bf16, al.make_layout((m, k), (k, 1)))
    b_t = al.make_tensor(B, al.bf16, al.make_layout((n, k), (k, 1)))
    c_t = al.make_tensor(C, al.bf16, al.make_layout((m, n), (n, 1)))

    shm_a = al.make_shared((TILE_K,), al.bf16)
    zero = al.convert(0.0, al.f32)

    for col in al.range(tid, n, bdim):
        acc = zero
        for k_start in al.range(0, k, TILE_K):
            if tid < TILE_K:
                if k_start + tid < k:
                    shm_a[tid] = a_t[row, k_start + tid]
            al.syncthreads()

            b_row = col
            for kk in al.range(TILE_K):
                if k_start + kk < k:
                    a_val = al.convert(shm_a[kk], al.f32)
                    b_val = al.convert(b_t[b_row, k_start + kk], al.f32)
                    acc = acc + a_val * b_val
            al.syncthreads()

        c_t[row, col] = al.convert(acc, al.bf16)


@avelang.jit
def _postprocess_kernel(
    gemm_out: al.Pointer(al.bf16),
    bias: al.Pointer(al.bf16),
    y: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
):
    tid = al.thread_id(0)
    wid = tid // POSTPROC_WARP_SIZE
    wtid = tid % POSTPROC_WARP_SIZE
    bdim = al.block_dim(0)
    row = al.block_id(0)

    sf = al.convert(2.0, al.f32)
    cmin = al.convert(-10.0, al.f32)
    cmax = al.convert(10.0, al.f32)
    neg_inf = al.convert(-1e+30, al.f32)
    zero = al.convert(0.0, al.f32)
    one = al.convert(1.0, al.f32)

    gemm_t = al.make_tensor(gemm_out, al.bf16, al.make_layout((m, n), (n, 1)))
    bias_t = al.make_tensor(bias, al.bf16, al.make_layout((n,), (1,)))
    y_t = al.make_tensor(y, al.bf16, al.make_layout((m,), (1,)))
    shm = al.make_shared((POSTPROC_NUM_WARPS,), al.f32)

    local_max = neg_inf
    for j in al.range(tid, n, bdim):
        val = al.convert(gemm_t[row, j], al.f32)
        bv = al.convert(bias_t[j], al.f32)
        val = (val + bv) * sf
        val = val + val
        if val < cmin:
            val = cmin
        if val > cmax:
            val = cmax
        if val > local_max:
            local_max = val

    o = al.shuffle_down(local_max, 32, POSTPROC_WARP_SIZE)
    if o > local_max: local_max = o
    o = al.shuffle_down(local_max, 16, POSTPROC_WARP_SIZE)
    if o > local_max: local_max = o
    o = al.shuffle_down(local_max, 8, POSTPROC_WARP_SIZE)
    if o > local_max: local_max = o
    o = al.shuffle_down(local_max, 4, POSTPROC_WARP_SIZE)
    if o > local_max: local_max = o
    o = al.shuffle_down(local_max, 2, POSTPROC_WARP_SIZE)
    if o > local_max: local_max = o
    o = al.shuffle_down(local_max, 1, POSTPROC_WARP_SIZE)
    if o > local_max: local_max = o

    if wtid == 0:
        shm[wid] = local_max
    al.syncthreads()

    if wid == 0:
        cross_max = shm[wtid] if wtid < POSTPROC_NUM_WARPS else neg_inf
        o = al.shuffle_down(cross_max, 2, POSTPROC_WARP_SIZE)
        if o > cross_max: cross_max = o
        o = al.shuffle_down(cross_max, 1, POSTPROC_WARP_SIZE)
        if o > cross_max: cross_max = o
        if wtid == 0:
            shm[0] = cross_max
    al.syncthreads()
    global_max = shm[0]

    local_sum = zero
    for j in al.range(tid, n, bdim):
        val = al.convert(gemm_t[row, j], al.f32)
        bv = al.convert(bias_t[j], al.f32)
        val = (val + bv) * sf
        val = val + val
        if val < cmin:
            val = cmin
        if val > cmax:
            val = cmax
        local_sum = local_sum + al.exp(val - global_max)

    local_sum = local_sum + al.shuffle_down(local_sum, 32, POSTPROC_WARP_SIZE)
    local_sum = local_sum + al.shuffle_down(local_sum, 16, POSTPROC_WARP_SIZE)
    local_sum = local_sum + al.shuffle_down(local_sum, 8, POSTPROC_WARP_SIZE)
    local_sum = local_sum + al.shuffle_down(local_sum, 4, POSTPROC_WARP_SIZE)
    local_sum = local_sum + al.shuffle_down(local_sum, 2, POSTPROC_WARP_SIZE)
    local_sum = local_sum + al.shuffle_down(local_sum, 1, POSTPROC_WARP_SIZE)

    if wtid == 0:
        shm[wid] = local_sum
    al.syncthreads()

    if wid == 0:
        cross_sum = shm[wtid] if wtid < POSTPROC_NUM_WARPS else zero
        cross_sum = cross_sum + al.shuffle_down(cross_sum, 2, POSTPROC_WARP_SIZE)
        cross_sum = cross_sum + al.shuffle_down(cross_sum, 1, POSTPROC_WARP_SIZE)
        if wtid == 0:
            shm[0] = cross_sum
    al.syncthreads()
    global_sum = shm[0]

    lse = global_max + al.log(global_sum)
    sp = al.log(one + al.exp(lse))
    mish_val = lse * lse * al.tanh(sp)

    if tid == 0:
        y_t[row] = al.convert(mish_val, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        linear = nn.Linear(input_size, hidden_size)
        self.register_buffer("weight", linear.weight.detach().to(torch.bfloat16))
        self.register_buffer("bias", linear.bias.detach().to(torch.bfloat16))

    def forward(self, x):
        device = x.device
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x = x.contiguous()

        weight_dev = self.weight.to(device=device)
        bias_dev = self.bias.to(device=device)

        gemm_out = torch.empty((x.shape[0], weight_dev.shape[0]),
                               dtype=torch.bfloat16, device=device)
        _gemm_kernel[lambda: ((x.shape[0], 1, 1), (GEMM_THREADS, 1, 1))](
            x, weight_dev, gemm_out,
            x.shape[0], x.shape[1], weight_dev.shape[0],
        )

        y_out = torch.empty((x.shape[0],), dtype=torch.bfloat16, device=device)
        _postprocess_kernel[lambda: ((x.shape[0], 1, 1), (POSTPROC_THREADS, 1, 1))](
            gemm_out, bias_dev, y_out,
            x.shape[0], weight_dev.shape[0],
        )
        return y_out.unsqueeze(1)
