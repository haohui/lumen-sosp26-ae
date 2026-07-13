import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def fused_linear_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
    SUB_VAL: al.constexpr,
    MUL_VAL: al.constexpr,
):
    # Build tensor views from raw pointers.
    a = al.make_tensor(a_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    b = al.make_tensor(b_ptr, al.bf16, al.make_layout((N, K), (K, 1)))
    bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((N,), (1,)))
    c = al.make_tensor(c_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    block_m = al.block_id(1) * 64
    block_n = al.block_id(0) * 64

    tid = al.thread_id(0)
    tid_m = tid // 16
    tid_n = tid % 16

    # Shared memory tiles for A [64 x 32] and B [64 x 32].
    a_smem = al.make_shared((64, 32), al.bf16)
    b_smem = al.make_shared((64, 32), al.bf16)

    # Per-thread FP32 accumulator for 4x4 output sub-tile.
    acc = al.make_local((4, 4), al.f32)
    for i in al.range(0, 4):
        for j in al.range(0, 4):
            acc[i, j] = al.convert(0.0, al.f32)

    sub_f32 = al.convert(SUB_VAL, al.f32)
    mul_f32 = al.convert(MUL_VAL, al.f32)

    # Main K-loop.
    for k_block in al.range(0, K, 32):
        # Cooperative load of A tile [64, 32] and B tile [64, 32].
        for i in al.range(0, 8):
            linear_idx = tid + i * 256
            if linear_idx < 2048:
                row = linear_idx // 32
                col = linear_idx % 32
                a_smem[row, col] = a[block_m + row, k_block + col]
                b_smem[row, col] = b[block_n + row, k_block + col]

        al.syncthreads()

        # Inner product: each thread accumulates 4x4 output elements.
        for kk in al.range(0, 32):
            for mr in al.range(0, 4):
                row = tid_m + mr * 16
                a_val = al.convert(a_smem[row, kk], al.f32)
                for nc in al.range(0, 4):
                    col = tid_n + nc * 16
                    b_val = al.convert(b_smem[col, kk], al.f32)
                    acc[mr, nc] = acc[mr, nc] + a_val * b_val

        al.syncthreads()

    # Apply bias, subtract, multiply, ReLU and store to global memory.
    zero = al.convert(0.0, al.f32)
    for mr in al.range(0, 4):
        global_row = block_m + tid_m + mr * 16
        for nc in al.range(0, 4):
            global_col = block_n + tid_n + nc * 16
            val = acc[mr, nc] + al.convert(bias[global_col], al.f32)
            val = val - sub_f32
            val = val * mul_f32
            if val < zero:
                val = zero
            c[global_row, global_col] = al.convert(val, al.bf16)


def _run_fused_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    subtract_value: float,
    multiply_value: float,
) -> torch.Tensor:
    """Launch the fused linear kernel and return the BF16 result."""
    assert x.is_cuda, "Input must be on CUDA/HIP device."
    M, K_in = x.shape
    N, K_w = weight.shape
    assert K_in == K_w, f"Inner dimension mismatch: {K_in} vs {K_w}"

    a_bf16 = x.to(torch.bfloat16).contiguous()
    w_bf16 = weight.to(torch.bfloat16).contiguous()
    b_bf16 = bias.to(torch.bfloat16).contiguous()

    c = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)

    grid = (N // 64, M // 64, 1)
    block = (256, 1, 1)

    fused_linear_kernel[lambda: (grid, block)](
        a_bf16,
        w_bf16,
        b_bf16,
        c,
        M,
        N,
        K_in,
        subtract_value,
        multiply_value,
    )
    return c


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x):
        if not x.is_cuda:
            x = x.to(self.linear.weight.device)
        return _run_fused_linear(
            x,
            self.linear.weight,
            self.linear.bias,
            self.subtract_value,
            self.multiply_value,
        )
