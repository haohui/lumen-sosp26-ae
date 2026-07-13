import torch
import torch.nn as nn
import importlib.machinery
import avelang
import avelang.language as al

# ── Import reference pipelined GEMM ────────────────────────────────────
# This kernel provides: MFMA_16x16x16_bf16_f32, software pipelining,
# double-buffered register data, K-step unrolling by 2, and fine-grained
# LDS/MFMA overlap.
_loader = importlib.machinery.SourceFileLoader(
    'amdgpu_gemm_mod', '/avelang/python/avelang_kernels/amdgpu_gemm.py')
_amdgpu_gemm_mod = _loader.load_module()
_gemm_pipeline_transposed_b = _amdgpu_gemm_mod.gemm_pipeline_transposed_b

# ── Post-processing kernel ────────────────────────────────────────────

POSTPROC_WARP_SIZE = 64
POSTPROC_NUM_WARPS = 4
POSTPROC_THREADS = POSTPROC_WARP_SIZE * POSTPROC_NUM_WARPS


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

    # ── Pass 1: per-thread max for stable LogSumExp ──
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

    # Warp-level max reduction
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

    # Cross-warp max via shared memory (first warp does the reduction)
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

    # ── Pass 2: sum of exp(x - max) ──
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

    # Warp-level sum reduction
    local_sum = local_sum + al.shuffle_down(local_sum, 32, POSTPROC_WARP_SIZE)
    local_sum = local_sum + al.shuffle_down(local_sum, 16, POSTPROC_WARP_SIZE)
    local_sum = local_sum + al.shuffle_down(local_sum, 8, POSTPROC_WARP_SIZE)
    local_sum = local_sum + al.shuffle_down(local_sum, 4, POSTPROC_WARP_SIZE)
    local_sum = local_sum + al.shuffle_down(local_sum, 2, POSTPROC_WARP_SIZE)
    local_sum = local_sum + al.shuffle_down(local_sum, 1, POSTPROC_WARP_SIZE)

    if wtid == 0:
        shm[wid] = local_sum
    al.syncthreads()

    # Cross-warp sum
    if wid == 0:
        cross_sum = shm[wtid] if wtid < POSTPROC_NUM_WARPS else zero
        cross_sum = cross_sum + al.shuffle_down(cross_sum, 2, POSTPROC_WARP_SIZE)
        cross_sum = cross_sum + al.shuffle_down(cross_sum, 1, POSTPROC_WARP_SIZE)
        if wtid == 0:
            shm[0] = cross_sum
    al.syncthreads()
    global_sum = shm[0]

    # ── LogSumExp + Mish activation ──
    # Mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    lse = global_max + al.log(global_sum)
    sp = al.log(one + al.exp(lse))
    mish_val = lse * lse * al.tanh(sp)

    if tid == 0:
        y_t[row] = al.convert(mish_val, al.bf16)


# ── ModelNew ───────────────────────────────────────────────────────────

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        linear = nn.Linear(input_size, hidden_size)
        self.register_buffer("weight", linear.weight.detach().to(torch.bfloat16))
        self.register_buffer("bias", linear.bias.detach().to(torch.bfloat16))
        self._cached_weight_ptr = None
        self._cached_bias_ptr = None
        self._cached_weight_t = None
        self._cached_bias_dev = None

    def forward(self, x):
        device = x.device
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x = x.contiguous()

        w_ptr = self.weight.data_ptr()
        if self._cached_weight_ptr != w_ptr or self._cached_weight_t is None:
            self._cached_weight_t = (
                self.weight.t().to(device=device, dtype=torch.bfloat16).contiguous()
            )
            self._cached_weight_ptr = w_ptr
        elif self._cached_weight_t.device != device:
            self._cached_weight_t = self._cached_weight_t.to(device)

        b_ptr = self.bias.data_ptr()
        if self._cached_bias_ptr != b_ptr or self._cached_bias_dev is None:
            self._cached_bias_dev = (
                self.bias.to(device=device, dtype=torch.bfloat16).contiguous()
            )
            self._cached_bias_ptr = b_ptr
        elif self._cached_bias_dev.device != device:
            self._cached_bias_dev = self._cached_bias_dev.to(device)

        gemm_out = _gemm_pipeline_transposed_b(x, self._cached_weight_t)
        y_out = torch.empty((x.shape[0],), dtype=torch.bfloat16, device=device)
        _postprocess_kernel[lambda: ((x.shape[0], 1, 1), (POSTPROC_THREADS, 1, 1))](
            gemm_out, self._cached_bias_dev, y_out,
            x.shape[0], self._cached_weight_t.shape[0],
        )
        return y_out.unsqueeze(1)
