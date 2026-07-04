---
name: substrate-examples-gemm
description: >
  Verified Substrate DSL kernels for BF16 GEMM-style kernels on AMDGPU.
  Each example compiled and passed correctness checks on AMD MI210/MI300X (BF16).
  Reuse the tiling / shared-memory / MFMA structure; adapt only layout,
  shape contract, output epilogue, and math.
tags: [substrate, amd, kernel, gemm]
---

# Substrate Verified Examples: GEMM

Each kernel below compiled and passed correctness checks on AMD Instinct MI210
or MI300X (BF16).

Reuse strategy:
1. Copy the tile / block / thread structure verbatim.
2. Preserve MFMA lane swizzles and accumulator writeback invariants exactly.
3. Do NOT invent API calls absent from these examples or `substrate-language-spec`.


### fused_linear_relu_bf16_mfma: BF16 Fused GEMM Epilogue

**PyTorch reference:**
```python
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Fused reference: y = relu((x @ w.T + bias - subtract_value) * multiply_value)
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = self.linear(x)
        x = x - self.subtract_value
        x = x * self.multiply_value
        x = torch.relu(x)
        return x


BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
SUBTRACT_VALUE = 2.0
MULTIPLY_VALUE = 1.5


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_FEATURES)]


def get_init_inputs():
    return [IN_FEATURES, OUT_FEATURES, SUBTRACT_VALUE, MULTIPLY_VALUE]
```

**Verified Substrate kernel:**
```python
import torch
import torch.nn as nn
import substrate
import substrate.language as S


BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
SUBTRACT_VALUE = 2.0
MULTIPLY_VALUE = 1.5

WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS
GROUP_M = 128
GROUP_N = 128
GROUP_K = 16
MMA_M = 32
MMA_N = 32
VEC_ELEMS = 8
BF16_BYTES = 2
ACC_SIZE = 16
WARPS_M = 2
WARPS_N = 2
M_TILES_PER_WARP = GROUP_M // (WARPS_M * MMA_M)
N_TILES_PER_WARP = GROUP_N // (WARPS_N * MMA_N)
A_VECS_PER_ROW = GROUP_K // VEC_ELEMS
B_VECS_PER_ROW = GROUP_K // VEC_ELEMS
SHM_A_VECS = GROUP_M * A_VECS_PER_ROW
SHM_B_VECS = GROUP_N * B_VECS_PER_ROW
GLOBAL_LOADS_A = SHM_A_VECS // THREADS
GLOBAL_LOADS_B = SHM_B_VECS // THREADS
ROW_U32 = A_VECS_PER_ROW * 4


@substrate.jit
def _load_global_a_to_shm(
    shm_a: S.Tensor((SHM_A_VECS, 4), S.u32),
    x_rsrc: S.Tensor((4,), S.u32),
    block_m: S.u32,
    k_base: S.u32,
    k: S.u32,
    tid: S.u32,
):
    zero = S.convert(0, S.u32)
    idx = tid
    for _ in S.range(GLOBAL_LOADS_A):
        row = idx // A_VECS_PER_ROW
        col_vec = idx % A_VECS_PER_ROW
        off = ((block_m * GROUP_M + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
        shm_a[idx] = S.amdgpu.raw_buffer_load_x4(x_rsrc, zero, off, 0)
        idx += THREADS


@substrate.jit
def _load_global_b_to_shm(
    shm_b: S.Tensor((SHM_B_VECS, 4), S.u32),
    w_rsrc: S.Tensor((4,), S.u32),
    block_n: S.u32,
    k_base: S.u32,
    k: S.u32,
    tid: S.u32,
):
    zero = S.convert(0, S.u32)
    idx = tid
    for _ in S.range(GLOBAL_LOADS_B):
        row = idx // B_VECS_PER_ROW
        col_vec = idx % B_VECS_PER_ROW
        off = ((block_n * GROUP_N + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
        shm_b[idx] = S.amdgpu.raw_buffer_load_x4(w_rsrc, zero, off, 0)
        idx += THREADS


@substrate.jit
def _fetch_mfma_operand_32x32x16(
    ret: S.Tensor((2, 4), S.bf16),
    shm: S.Tensor((SHM_A_VECS, 4), S.u32),
    tile_idx: S.u32,
    lane: S.u32,
):
    ret_u32 = S.view(ret, S.Tensor((4,), S.u32))
    shm_u32 = S.view(shm, S.Tensor((SHM_A_VECS * 4,), S.u32))
    row = tile_idx * MMA_M + (lane % MMA_M)
    k_group_u32 = (lane // MMA_M) * 2
    row_base = row * ROW_U32

    ret_u32[0] = shm_u32[row_base + k_group_u32]
    ret_u32[1] = shm_u32[row_base + k_group_u32 + 1]
    ret_u32[2] = shm_u32[row_base + 4 + k_group_u32]
    ret_u32[3] = shm_u32[row_base + 5 + k_group_u32]


@substrate.jit
def linear_fused_relu_bf16_kernel(
    x_ptr: S.Pointer(S.bf16),
    w_ptr: S.Pointer(S.bf16),
    bias_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    m: S.u32,
    n: S.u32,
    k: S.u32,
):
    tid = S.thread_id(0)
    block_n = S.block_id(0)
    block_m = S.block_id(1)
    wid = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    warp_row = wid // WARPS_N
    warp_col = wid % WARPS_N

    x_memref = S.make_tensor(x_ptr, S.bf16, S.make_layout((m * k,), (1,)))
    w_memref = S.make_tensor(w_ptr, S.bf16, S.make_layout((n * k,), (1,)))
    g_bias = S.make_tensor(bias_ptr, S.bf16, S.make_layout((n,), (1,)))
    g_out = S.make_tensor(out_ptr, S.bf16, S.make_layout((m, n), (n, 1)))

    x_rsrc = S.amdgpu.make_rsrc(x_memref, m * k * BF16_BYTES)
    w_rsrc = S.amdgpu.make_rsrc(w_memref, n * k * BF16_BYTES)

    shm_a = S.make_shared((SHM_A_VECS, 4), S.u32)
    shm_b = S.make_shared((SHM_B_VECS, 4), S.u32)
    a_reg = S.make_local((M_TILES_PER_WARP, 2, 4), S.bf16)
    b_reg = S.make_local((N_TILES_PER_WARP, 2, 4), S.bf16)
    acc = S.make_local((M_TILES_PER_WARP * N_TILES_PER_WARP, ACC_SIZE), S.f32)

    for i in S.range(M_TILES_PER_WARP * N_TILES_PER_WARP):
        for j in S.range(ACC_SIZE):
            acc[i, j] = 0

    k_tiles = k // GROUP_K
    for kt in S.range(k_tiles):
        k_base = kt * GROUP_K
        _load_global_a_to_shm(shm_a, x_rsrc, block_m, k_base, k, tid)
        _load_global_b_to_shm(shm_b, w_rsrc, block_n, k_base, k, tid)
        S.syncthreads()

        for i in S.range(M_TILES_PER_WARP):
            _fetch_mfma_operand_32x32x16(a_reg[i], shm_a, warp_row * M_TILES_PER_WARP + i, lane)
        for j in S.range(N_TILES_PER_WARP):
            _fetch_mfma_operand_32x32x16(b_reg[j], shm_b, warp_col * N_TILES_PER_WARP + j, lane)

        for i in S.range(M_TILES_PER_WARP):
            for j in S.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                acc[acc_idx] = S.amdgpu.mfma_32x32x8_bf16_f32(a_reg[i, 0], b_reg[j, 0], acc[acc_idx])
                acc[acc_idx] = S.amdgpu.mfma_32x32x8_bf16_f32(a_reg[i, 1], b_reg[j, 1], acc[acc_idx])

        S.syncthreads()

    subtract_val = S.convert(SUBTRACT_VALUE, S.f32)
    multiply_val = S.convert(MULTIPLY_VALUE, S.f32)
    zero = S.convert(0.0, S.f32)
    lane_group = lane // MMA_N
    lane_col = lane % MMA_N
    block_row_base = block_m * GROUP_M
    block_col_base = block_n * GROUP_N

    for j in S.range(N_TILES_PER_WARP):
        col = block_col_base + (warp_col * N_TILES_PER_WARP + j) * MMA_N + lane_col
        bias = S.convert(g_bias[col], S.f32)
        for i in S.range(M_TILES_PER_WARP):
            acc_idx = i * N_TILES_PER_WARP + j
            row_base = block_row_base + (warp_row * M_TILES_PER_WARP + i) * MMA_M
            for t in S.range(ACC_SIZE):
                row = row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
                result = acc[acc_idx, t] + bias
                result = result - subtract_val
                result = result * multiply_val
                if result < zero:
                    result = zero
                g_out[row, col] = S.convert(result, S.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def substrate_linear_fused_relu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight)
    bias_bf16 = _prepare_bf16_cuda_contiguous(bias)

    m, k = x_bf16.shape
    n, weight_k = weight_bf16.shape
    if weight_k != k:
        raise ValueError(
            f"Weight/input K mismatch: x has K={k}, weight has K={weight_k}"
        )
    if m % GROUP_M != 0 or n % GROUP_N != 0 or k % GROUP_K != 0:
        raise ValueError(
            f"Expected m % {GROUP_M} == 0, n % {GROUP_N} == 0, k % {GROUP_K} == 0 "
            f"(got m={m}, n={n}, k={k})"
        )

    out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)
    grid = (n // GROUP_N, m // GROUP_M, 1)
    linear_fused_relu_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, weight_bf16, bias_bf16, out, m, n, k
    )
    return out


class ModelNew(nn.Module):
    """
    Fused BF16 linear + subtract + multiply + ReLU through a custom AMDGPU kernel.
    """

    def __init__(self, in_features: int, out_features: int, subtract_value: float, multiply_value: float):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return substrate_linear_fused_relu(x, self.weight, self.bias)


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_FEATURES)]


def get_init_inputs():
    return [IN_FEATURES, OUT_FEATURES, SUBTRACT_VALUE, MULTIPLY_VALUE]
```

Contract notes:
1. The kernel entry point is `substrate_linear_fused_relu(x, weight_t, bias)`.
2. `x` must be rank-2 BF16 CUDA with shape `(M, K)`.
3. The Substrate kernel expects `weight_t` in linear-module storage layout with shape `(N, K)`.
4. Accumulation happens in `f32`, MFMA uses `mfma_32x32x8_bf16_f32`, and the final output is written as BF16.
5. Preserve these translation invariants when porting similar kernels:
   - Keep the 4 warps as a `2 x 2` warp grid and add warp ownership only at operand tile selection and output writeback.
   - Stage A and B through LDS with `raw_buffer_load_x4` and build each lane operand as one 16-byte fragment viewed as `2 x (4, bf16)`.
   - For the `32x32x16` K-slice, fetch low-K and high-K halves from the same tile in natural order; do not add lane-dependent control flow to choose halves.
   - Preserve the accumulator writeback mapping `row = row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)` and `col = col_base + (lane % 32)`.
