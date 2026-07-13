import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_M = 16
TILE_N = 16
TILE_K = 32


@avelang.jit
def linear_relu_div_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    K: al.i32,
    N: al.i32,
    DIVISOR: al.constexpr,
    TILE_M: al.constexpr,
    TILE_N: al.constexpr,
    TILE_K: al.constexpr,
):
    pid_m = al.block_id(0)
    pid_n = al.block_id(1)
    tx = al.thread_id(0)
    ty = al.thread_id(1)

    off_m = pid_m * TILE_M
    off_n = pid_n * TILE_N
    row = off_m + tx
    col = off_n + ty

    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    w = al.make_tensor(w_ptr, al.bf16, al.make_layout((N, K), (K, 1)))
    b = al.make_tensor(b_ptr, al.f32, al.make_layout((N,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    a_sh = al.make_shared((TILE_M, TILE_K), al.bf16)
    b_sh = al.make_shared((TILE_N, TILE_K), al.bf16)

    acc = al.convert(0.0, al.f32)
    zero_f32 = al.convert(0.0, al.f32)

    num_blocks = K // TILE_K
    k_off = al.convert(0, al.i32)
    for dummy in al.range(num_blocks):
        k_a0 = k_off + ty
        k_a1 = k_off + ty + TILE_N
        a_sh[tx, ty] = x[row, k_a0]
        a_sh[tx, ty + TILE_N] = x[row, k_a1]

        k_b0 = k_off + tx
        k_b1 = k_off + tx + TILE_M
        b_sh[ty, tx] = w[col, k_b0]
        b_sh[ty, tx + TILE_M] = w[col, k_b1]

        al.syncthreads()

        for kk in al.range(TILE_K):
            a_val = al.convert(a_sh[tx, kk], al.f32)
            b_val = al.convert(b_sh[ty, kk], al.f32)
            acc = acc + a_val * b_val

        al.syncthreads()

        k_off = k_off + TILE_K

    acc = acc + b[col]
    if acc < zero_f32:
        acc = zero_f32
    acc = al.convert(acc / DIVISOR, al.f32)
    out[row, col] = al.convert(acc, al.bf16)


def avelang_linear_relu_div(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    divisor: float,
) -> torch.Tensor:
    M, K_in = x.shape
    N, K_w = weight.shape
    assert K_in == K_w

    device = weight.device
    x_bf16 = x.to(device=device, dtype=torch.bfloat16).contiguous()
    w_bf16 = weight.to(device=device, dtype=torch.bfloat16).contiguous()
    b_f32 = bias.to(device=device, dtype=torch.float32).contiguous()

    out = torch.empty(M, N, dtype=torch.bfloat16, device=device)

    grid_m = (M + TILE_M - 1) // TILE_M
    grid_n = (N + TILE_N - 1) // TILE_N

    linear_relu_div_kernel[lambda: ((grid_m, grid_n, 1), (TILE_M, TILE_N, 1))](
        x_bf16, w_bf16, b_f32, out,
        M, K_in, N, divisor, TILE_M, TILE_N, TILE_K,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor

    def forward(self, x):
        w = self.linear.weight
        b = self.linear.bias
        return avelang_linear_relu_div(x, w, b, self.divisor)
