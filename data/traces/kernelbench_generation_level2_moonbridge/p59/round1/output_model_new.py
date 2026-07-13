import torch
import torch.nn as nn
import avelang
import avelang.language as al


BF16_BYTES = 2


@avelang.jit
def linear_swish_scale_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    k: al.u32,
    scaling_factor: al.f32,
    BLOCK_M: al.constexpr,
    BLOCK_N: al.constexpr,
    BLOCK_K: al.constexpr,
):
    lane = al.thread_id(0)
    lane_col = lane & 31
    lane_group = lane >> 5

    block_m = al.block_id(1) * BLOCK_M
    block_n = al.block_id(0) * BLOCK_N

    x_bf16 = al.make_tensor(x_ptr, al.bf16, al.make_layout((m, k), (k, 1)))
    w_bf16 = al.make_tensor(w_ptr, al.bf16, al.make_layout((n, k), (k, 1)))
    bias_bf16 = al.make_tensor(bias_ptr, al.bf16, al.make_layout((n,), (1,)))
    out_bf16 = al.make_tensor(out_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    k_vecs = k >> 3
    packed_row_stride = k >> 1

    x_vec = al.view(
        x_bf16, al.i32,
        al.make_layout((m, k_vecs, 4), (packed_row_stride, 4, 1)),
    )
    w_vec = al.view(
        w_bf16, al.i32,
        al.make_layout((n, k_vecs, 4), (packed_row_stride, 4, 1)),
    )

    a_smem = al.make_shared(
        (BLOCK_M * (BLOCK_K >> 3), BLOCK_K >> 2), al.i32
    )
    b_smem = al.make_shared(
        (BLOCK_N * (BLOCK_K >> 3), BLOCK_K >> 2), al.i32
    )
    c_smem = al.make_shared((BLOCK_M, BLOCK_N), al.f32)
    acc = al.full((16,), 0.0, al.f32)

    k_tiles = k // BLOCK_K
    for kt in al.range(k_tiles):
        k_vec = kt * 2 + lane_group
        a_smem[lane] = x_vec[block_m + lane_col, k_vec]
        b_smem[lane] = w_vec[block_n + lane_col, k_vec]
        al.syncthreads()

        a_words = a_smem[lane]
        b_words = b_smem[lane]
        a_frag = al.view(a_words, al.Tensor((2, 2, 1), al.u32))
        b_frag = al.view(b_words, al.Tensor((2, 2, 1), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)
        al.syncthreads()

    # Remap accumulator to row-major shared memory
    for r in al.range(16):
        row_offset = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        c_smem[lane_col, row_offset] = acc[r]

    al.syncthreads()

    # Apply epilogue per (row, col) and write bf16 output
    one = al.convert(1.0, al.f32)
    store_row = lane >> 1
    store_col_base = (lane & 1) * (BLOCK_N >> 1)
    for v in al.range(BLOCK_N >> 1):
        store_col = store_col_base + v
        row = block_m + store_row
        col = block_n + store_col
        result = c_smem[store_row, store_col]
        result = result + al.convert(bias_bf16[col], al.f32)
        # Swish: x * sigmoid(x) where sigmoid(x) = 1 / (1 + exp(-x))
        sigmoid_val = one / (one + al.exp(al.convert(-result, al.f32)))
        result = result * sigmoid_val
        result = result * scaling_factor
        out_bf16[row, col] = al.convert(result, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_linear_swish_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scaling_factor: float,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight)
    bias_bf16 = _prepare_bf16_cuda_contiguous(bias)

    m, k = x_bf16.shape
    n, weight_k = weight_bf16.shape
    if weight_k != k:
        raise ValueError(
            f"Weight/input K mismatch: x has K={k}, weight has K={weight_k}"
        )

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 16

    out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)
    grid_x = (n + BLOCK_N - 1) // BLOCK_N
    grid_y = (m + BLOCK_M - 1) // BLOCK_M
    grid = (grid_x, grid_y, 1)

    linear_swish_scale_kernel[lambda: (grid, (64, 1, 1))](
        x_bf16,
        weight_bf16,
        bias_bf16,
        out,
        m,
        n,
        k,
        scaling_factor,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int, scaling_factor: float):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_linear_swish_scale(
            x, self.weight, self.bias, self.scaling_factor
        )


def get_inputs():
    return [torch.rand(128, 32768)]


def get_init_inputs():
    return [32768, 32768, 2.0]
