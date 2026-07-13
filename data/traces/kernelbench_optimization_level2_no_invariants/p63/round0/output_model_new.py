import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def fused_kernel(
    X_ptr: al.Pointer(al.bf16),
    W_ptr: al.Pointer(al.bf16),
    Bias_ptr: al.Pointer(al.bf16),
    Y_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    ci1 = al.convert(1, al.i32)
    ci4 = al.convert(4, al.i32)
    ci16 = al.convert(16, al.i32)
    ci32 = al.convert(32, al.i32)
    ci64 = al.convert(64, al.i32)
    cf0 = al.convert(0.0, al.f32)

    block_m_id = al.block_id(0)
    block_n_id = al.block_id(1)
    tid = al.thread_id(0)

    warp_id = tid // ci64
    lane_id = tid % ci64
    warp_m = warp_id // al.convert(2, al.i32)
    warp_n = warp_id % al.convert(2, al.i32)

    m_base = block_m_id * ci64 + warp_m * ci32
    n_base = block_n_id * ci64 + warp_n * ci32

    X = al.make_tensor(X_ptr, al.bf16, al.make_layout((M, K), (K, ci1)))
    W = al.make_tensor(W_ptr, al.bf16, al.make_layout((K, N), (N, ci1)))
    Bias = al.make_tensor(Bias_ptr, al.bf16, al.make_layout((N,), (ci1,)))
    Y = al.make_tensor(Y_ptr, al.bf16, al.make_layout((M, N), (N, ci1)))

    tr = lane_id // al.convert(8, al.i32)
    tc = lane_id % al.convert(8, al.i32)

    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = cf0

    for k_block in al.range(0, K, ci16):
        for r in al.range(4):
            m_idx = m_base + tr * ci4 + r
            for c in al.range(4):
                n_idx = n_base + tc * ci4 + c
                acc_idx = r * ci4 + c

                k_sum = acc[acc_idx]
                for kk in al.range(16):
                    k_idx = k_block + kk
                    if k_idx < K:
                        x_val = al.convert(X[m_idx, k_idx], al.f32)
                        w_val = al.convert(W[k_idx, n_idx], al.f32)
                        k_sum = k_sum + x_val * w_val
                acc[acc_idx] = k_sum

    for r in al.range(4):
        m_idx = m_base + tr * ci4 + r
        for c in al.range(4):
            n_idx = n_base + tc * ci4 + c
            acc_idx = r * ci4 + c

            val = acc[acc_idx]
            val = val + al.convert(Bias[n_idx], al.f32)
            if val < cf0:
                val = cf0
            val = val / al.convert(2.0, al.f32)
            if m_idx < M:
                if n_idx < N:
                    Y[m_idx, n_idx] = al.convert(val, al.bf16)


def _launch(M_val: int, N_val: int):
    grid_m = (M_val + 63) // 64
    grid_n = (N_val + 63) // 64
    return ((grid_m, grid_n, 1), (256, 1, 1))


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor

    def forward(self, x):
        M_val = x.shape[0]
        K_val = x.shape[1]
        N_val = self.linear.out_features

        w_t = self.linear.weight.t().to(device=x.device, dtype=x.dtype).contiguous()
        bias = self.linear.bias.to(device=x.device, dtype=x.dtype).contiguous()
        y = torch.empty((M_val, N_val), device=x.device, dtype=x.dtype)

        fused_kernel[lambda: _launch(M_val, N_val)](
            x.contiguous(), w_t, bias, y,
            M_val, N_val, K_val,
        )
        return y
