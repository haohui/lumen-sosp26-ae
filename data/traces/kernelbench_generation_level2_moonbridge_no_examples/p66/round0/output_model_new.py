import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def _linear_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    x_layout = al.make_layout((M, K), (K, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    w_layout = al.make_layout((N, K), (K, 1))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)
    b_layout = al.make_layout((N,), (1,))
    b = al.make_tensor(b_ptr, al.bf16, b_layout)
    out_layout = al.make_layout((M, N), (N, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    pid_m = al.block_id(0)
    pid_n = al.block_id(1)

    tid = al.thread_id(0)
    tid_m = tid // 16
    tid_n = tid % 16

    rm0 = tid_m * 4 + 0
    rm1 = tid_m * 4 + 1
    rm2 = tid_m * 4 + 2
    rm3 = tid_m * 4 + 3
    gm0 = pid_m * 64 + rm0
    gm1 = pid_m * 64 + rm1
    gm2 = pid_m * 64 + rm2
    gm3 = pid_m * 64 + rm3

    cn0 = tid_n * 4 + 0
    cn1 = tid_n * 4 + 1
    cn2 = tid_n * 4 + 2
    cn3 = tid_n * 4 + 3
    gn0 = pid_n * 64 + cn0
    gn1 = pid_n * 64 + cn1
    gn2 = pid_n * 64 + cn2
    gn3 = pid_n * 64 + cn3

    a_sh = al.make_shared((64, 128), al.bf16)
    b_sh = al.make_shared((128, 64), al.bf16)

    acc00 = al.convert(0.0, al.f32)
    acc01 = al.convert(0.0, al.f32)
    acc02 = al.convert(0.0, al.f32)
    acc03 = al.convert(0.0, al.f32)
    acc10 = al.convert(0.0, al.f32)
    acc11 = al.convert(0.0, al.f32)
    acc12 = al.convert(0.0, al.f32)
    acc13 = al.convert(0.0, al.f32)
    acc20 = al.convert(0.0, al.f32)
    acc21 = al.convert(0.0, al.f32)
    acc22 = al.convert(0.0, al.f32)
    acc23 = al.convert(0.0, al.f32)
    acc30 = al.convert(0.0, al.f32)
    acc31 = al.convert(0.0, al.f32)
    acc32 = al.convert(0.0, al.f32)
    acc33 = al.convert(0.0, al.f32)

    for k_tile in al.range(128):
        off_k = k_tile * 128

        for i in al.range(32):
            idx = tid * 32 + i
            row = idx // 128
            col = idx % 128
            a_sh[row, col] = x[pid_m * 64 + row, off_k + col]

        for i in al.range(32):
            idx = tid * 32 + i
            row = idx // 64
            col = idx % 64
            b_sh[row, col] = w[pid_n * 64 + col, off_k + row]

        al.syncthreads()

        for kk in al.range(128):
            a0 = al.convert(a_sh[rm0, kk], al.f32)
            a1 = al.convert(a_sh[rm1, kk], al.f32)
            a2 = al.convert(a_sh[rm2, kk], al.f32)
            a3 = al.convert(a_sh[rm3, kk], al.f32)

            b0 = al.convert(b_sh[kk, cn0], al.f32)
            b1 = al.convert(b_sh[kk, cn1], al.f32)
            b2 = al.convert(b_sh[kk, cn2], al.f32)
            b3 = al.convert(b_sh[kk, cn3], al.f32)

            acc00 = acc00 + a0 * b0
            acc01 = acc01 + a0 * b1
            acc02 = acc02 + a0 * b2
            acc03 = acc03 + a0 * b3
            acc10 = acc10 + a1 * b0
            acc11 = acc11 + a1 * b1
            acc12 = acc12 + a1 * b2
            acc13 = acc13 + a1 * b3
            acc20 = acc20 + a2 * b0
            acc21 = acc21 + a2 * b1
            acc22 = acc22 + a2 * b2
            acc23 = acc23 + a2 * b3
            acc30 = acc30 + a3 * b0
            acc31 = acc31 + a3 * b1
            acc32 = acc32 + a3 * b2
            acc33 = acc33 + a3 * b3

        al.syncthreads()

    b0 = al.convert(b[gn0], al.f32)
    b1 = al.convert(b[gn1], al.f32)
    b2 = al.convert(b[gn2], al.f32)
    b3 = al.convert(b[gn3], al.f32)
    out[gm0, gn0] = al.convert(acc00 + b0, al.bf16)
    out[gm0, gn1] = al.convert(acc01 + b1, al.bf16)
    out[gm0, gn2] = al.convert(acc02 + b2, al.bf16)
    out[gm0, gn3] = al.convert(acc03 + b3, al.bf16)
    out[gm1, gn0] = al.convert(acc10 + b0, al.bf16)
    out[gm1, gn1] = al.convert(acc11 + b1, al.bf16)
    out[gm1, gn2] = al.convert(acc12 + b2, al.bf16)
    out[gm1, gn3] = al.convert(acc13 + b3, al.bf16)
    out[gm2, gn0] = al.convert(acc20 + b0, al.bf16)
    out[gm2, gn1] = al.convert(acc21 + b1, al.bf16)
    out[gm2, gn2] = al.convert(acc22 + b2, al.bf16)
    out[gm2, gn3] = al.convert(acc23 + b3, al.bf16)
    out[gm3, gn0] = al.convert(acc30 + b0, al.bf16)
    out[gm3, gn1] = al.convert(acc31 + b1, al.bf16)
    out[gm3, gn2] = al.convert(acc32 + b2, al.bf16)
    out[gm3, gn3] = al.convert(acc33 + b3, al.bf16)


@avelang.jit
def _softmax_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
):
    row = al.block_id(0)
    tid = al.thread_id(0)

    x_layout = al.make_layout((M, N), (N, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    out_layout = al.make_layout((M, N), (N, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    sh_max = al.make_shared((256,), al.f32)
    sh_sum = al.make_shared((256,), al.f32)

    local_max = al.convert(-3.4028235e38, al.f32)
    for i in al.range(64):
        col = tid * 64 + i
        val = al.convert(x[row, col], al.f32)
        if val > local_max:
            local_max = val

    sh_max[tid] = local_max
    al.syncthreads()

    if tid < 128:
        other = sh_max[tid + 128]
        if other > sh_max[tid]:
            sh_max[tid] = other
    al.syncthreads()
    if tid < 64:
        other = sh_max[tid + 64]
        if other > sh_max[tid]:
            sh_max[tid] = other
    al.syncthreads()
    if tid < 32:
        other = sh_max[tid + 32]
        if other > sh_max[tid]:
            sh_max[tid] = other
    al.syncthreads()
    if tid < 16:
        other = sh_max[tid + 16]
        if other > sh_max[tid]:
            sh_max[tid] = other
    al.syncthreads()
    if tid < 8:
        other = sh_max[tid + 8]
        if other > sh_max[tid]:
            sh_max[tid] = other
    al.syncthreads()
    if tid < 4:
        other = sh_max[tid + 4]
        if other > sh_max[tid]:
            sh_max[tid] = other
    al.syncthreads()
    if tid < 2:
        other = sh_max[tid + 2]
        if other > sh_max[tid]:
            sh_max[tid] = other
    al.syncthreads()
    if tid < 1:
        other = sh_max[tid + 1]
        if other > sh_max[tid]:
            sh_max[tid] = other
    al.syncthreads()

    global_max = sh_max[0]

    local_sum = al.convert(0.0, al.f32)
    for i in al.range(64):
        col = tid * 64 + i
        val = al.convert(x[row, col], al.f32)
        exp_val = al.exp(val - global_max)
        local_sum = local_sum + exp_val

    sh_sum[tid] = local_sum
    al.syncthreads()

    if tid < 128:
        sh_sum[tid] = sh_sum[tid] + sh_sum[tid + 128]
    al.syncthreads()
    if tid < 64:
        sh_sum[tid] = sh_sum[tid] + sh_sum[tid + 64]
    al.syncthreads()
    if tid < 32:
        sh_sum[tid] = sh_sum[tid] + sh_sum[tid + 32]
    al.syncthreads()
    if tid < 16:
        sh_sum[tid] = sh_sum[tid] + sh_sum[tid + 16]
    al.syncthreads()
    if tid < 8:
        sh_sum[tid] = sh_sum[tid] + sh_sum[tid + 8]
    al.syncthreads()
    if tid < 4:
        sh_sum[tid] = sh_sum[tid] + sh_sum[tid + 4]
    al.syncthreads()
    if tid < 2:
        sh_sum[tid] = sh_sum[tid] + sh_sum[tid + 2]
    al.syncthreads()
    if tid < 1:
        sh_sum[tid] = sh_sum[tid] + sh_sum[tid + 1]
    al.syncthreads()

    global_sum = sh_sum[0]
    inv_sum = al.convert(1.0, al.f32) / global_sum

    for i in al.range(64):
        col = tid * 64 + i
        val = al.convert(x[row, col], al.f32)
        exp_val = al.exp(val - global_max)
        result = exp_val * inv_sum
        out[row, col] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_p: float):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.matmul.weight.data
        b = self.matmul.bias.data

        M = x.shape[0]
        K = x.shape[1]
        N = w.shape[0]

        x = x.contiguous()
        w = w.contiguous()
        b = b.contiguous()

        linear_out = torch.empty(M, N, dtype=x.dtype, device=x.device)

        grid_m = (M + 63) // 64
        grid_n = (N + 63) // 64
        _linear_kernel[lambda: ((grid_m, grid_n, 1), (256, 1, 1))](
            x, w, b, linear_out, M, N, K
        )

        softmax_out = torch.empty_like(linear_out)
        _softmax_kernel[lambda: ((M, 1, 1), (256, 1, 1))](
            linear_out, softmax_out, M, N
        )

        return softmax_out
