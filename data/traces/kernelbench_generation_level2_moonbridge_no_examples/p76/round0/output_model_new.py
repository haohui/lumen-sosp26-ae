import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def gemm_bias_relu_mfma_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.f32),
    m: al.i32,
    n: al.i32,
    k: al.i32,
):
    TILE_M = 32
    TILE_N = 32
    TILE_K = 32
    K_GROUPS = TILE_K >> 3  # 4 groups of 8 bf16 each

    lane = al.thread_id(0)
    lane_col = lane & 31
    lane_group = lane >> 5

    block_m = al.block_id(1) * TILE_M
    block_n = al.block_id(0) * TILE_N

    # Create tensor views
    A_bf16 = al.make_tensor(A_ptr, al.bf16, al.make_layout((m, k), (k, 1)))
    B_bf16 = al.make_tensor(B_ptr, al.bf16, al.make_layout((n, k), (k, 1)))
    bias_bf16 = al.make_tensor(bias_ptr, al.bf16, al.make_layout((n,), (1,)))
    C = al.make_tensor(C_ptr, al.f32, al.make_layout((m, n), (n, 1)))

    # Packed i32 views
    k_vecs = k >> 3
    packed_row_stride = k >> 1

    A_vec = al.view(
        A_bf16, al.i32,
        al.make_layout((m, k_vecs, 4), (packed_row_stride, 4, 1)),
    )
    B_vec = al.view(
        B_bf16, al.i32,
        al.make_layout((n, k_vecs, 4), (packed_row_stride, 4, 1)),
    )

    # Shared memory: K_GROUPS rows per matrix row, 4 i32 wide
    a_smem = al.make_shared((TILE_M * K_GROUPS, 4), al.i32)
    b_smem = al.make_shared((TILE_N * K_GROUPS, 4), al.i32)
    c_smem = al.make_shared((TILE_M, TILE_N), al.f32)

    acc = al.full((16,), 0.0, al.f32)

    for kt in al.range(k // TILE_K):
        k_base = kt * K_GROUPS
        # Each lane loads 2 words (K_GROUPS=4, lane_group=0/1, 2 sub-iterations)
        for sub_k in al.range(K_GROUPS >> 1):
            k_vec = k_base + sub_k * 2 + lane_group
            smem_idx = lane + sub_k * 64
            a_smem[smem_idx] = A_vec[block_m + lane_col, k_vec]
            b_smem[smem_idx] = B_vec[block_n + lane_col, k_vec]

        al.syncthreads()

        for sub_k in al.range(K_GROUPS >> 1):
            smem_idx = lane + sub_k * 64
            a_words = a_smem[smem_idx]
            b_words = b_smem[smem_idx]
            a_frag = al.view(a_words, al.Tensor((2, 2, 1), al.u32))
            b_frag = al.view(b_words, al.Tensor((2, 2, 1), al.u32))
            acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
            acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)

        al.syncthreads()

    # Write accumulator to shared memory
    for r in al.range(16):
        row_offset = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        c_smem[lane_col, row_offset] = acc[r]

    al.syncthreads()

    # Apply bias + ReLU and write to global memory
    f32_zero = al.convert(0.0, al.f32)
    for i in al.range(16):
        idx = lane + i * 64
        local_row = idx // TILE_N
        local_col = idx % TILE_N
        global_row = block_m + local_row
        global_col = block_n + local_col
        if global_row < m:
            if global_col < n:
                val = c_smem[local_row, local_col]
                bias_val = al.convert(bias_bf16[global_col], al.f32)
                val = val + bias_val
                if val < f32_zero:
                    val = f32_zero
                C[global_row, global_col] = val


def avelang_gemm_bias_relu(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    M, K = x.shape
    N = weight.shape[0]

    x_bf16 = x.to(torch.bfloat16).contiguous()
    w_bf16 = weight.to(torch.bfloat16).contiguous()
    bias_bf16 = bias.to(torch.bfloat16).contiguous()
    out_f32 = torch.empty(M, N, dtype=torch.float32, device=x.device)

    grid_m = (M + 31) // 32
    grid_n = (N + 31) // 32

    gemm_bias_relu_mfma_kernel[lambda: ((grid_n, grid_m, 1), (64, 1, 1))](
        x_bf16.data_ptr(),
        w_bf16.data_ptr(),
        bias_bf16.data_ptr(),
        out_f32.data_ptr(),
        M, N, K,
    )

    return out_f32.to(x.dtype)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return avelang_gemm_bias_relu(x, self.gemm.weight, self.bias)
