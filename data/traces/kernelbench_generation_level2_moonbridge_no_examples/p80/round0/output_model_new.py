import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Tile / launch constants
BM = 64
BN = 64
BK = 16
TM = 16
TN = 16
THREADS = TM * TN                 # 256
ELEM_M = BM // TM                 # 4
ELEM_N = BN // TN                 # 4
NUM_ACC = ELEM_M * ELEM_N         # 16

REDUCE_THREADS = 256
ELEMS_PER_THREAD = 8192 // REDUCE_THREADS  # 32


@avelang.jit
def gemm_bf16_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M: al.constexpr,
    N: al.constexpr,
    K: al.constexpr,
):
    # A  M x K  row-major
    # B  K x N  row-major
    # C  M x N  row-major

    layout_a = al.make_layout((M, K), (K, 1))
    A = al.make_tensor(A_ptr, al.bf16, layout_a)

    layout_b = al.make_layout((K, N), (N, 1))
    B = al.make_tensor(B_ptr, al.bf16, layout_b)

    layout_bias = al.make_layout((N,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, layout_bias)

    layout_c = al.make_layout((M, N), (N, 1))
    C = al.make_tensor(C_ptr, al.bf16, layout_c)

    block_m = al.block_id(0)
    block_n = al.block_id(1)

    tid = al.thread_id(0)
    ti = tid // 16
    tj = tid % 16

    # F32 accumulators for this thread's 4x4 = 16 output elements
    acc = al.make_local((16,), al.f32)
    for i in al.range(16):
        acc[i] = al.convert(0.0, al.f32)

    # Shared-memory tiles
    As = al.make_shared((64, 16), al.bf16)
    Bs = al.make_shared((16, 64), al.bf16)

    m_start = block_m * 64
    n_start = block_n * 64

    num_k_blocks = K // 16

    for kb in al.range(num_k_blocks):
        k_start = kb * 16

        # Cooperative load A tile (64 x 16 = 1024 elems; 4 per thread)
        for i in al.range(4):
            idx = tid * 4 + i
            row = idx // 16
            col = idx % 16
            As[row, col] = A[m_start + row, k_start + col]

        # Cooperative load B tile (16 x 64 = 1024 elems; 4 per thread)
        for i in al.range(4):
            idx = tid * 4 + i
            row = idx // 64
            col = idx % 64
            Bs[row, col] = B[k_start + row, n_start + col]

        al.syncthreads()

        # Dot-product accumulation in F32
        for k in al.range(16):
            for emi in al.range(4):
                a_val = al.convert(As[ti * 4 + emi, k], al.f32)
                for eni in al.range(4):
                    b_val = al.convert(Bs[k, tj * 4 + eni], al.f32)
                    acc[emi * 4 + eni] = acc[emi * 4 + eni] + a_val * b_val

        al.syncthreads()

    # Write results + bias; demote to BF16
    for emi in al.range(4):
        for eni in al.range(4):
            m_idx = m_start + ti * 4 + emi
            n_idx = n_start + tj * 4 + eni
            val = acc[emi * 4 + eni] + al.convert(bias[n_idx], al.f32)
            C[m_idx, n_idx] = al.convert(val, al.bf16)


@avelang.jit
def row_max_reduce_kernel(
    src_ptr: al.Pointer(al.bf16),
    dst_ptr: al.Pointer(al.bf16),
    M: al.constexpr,
    N: al.constexpr,
):
    layout_src = al.make_layout((M, N), (N, 1))
    src = al.make_tensor(src_ptr, al.bf16, layout_src)

    layout_dst = al.make_layout((M, 1), (1, 1))
    dst = al.make_tensor(dst_ptr, al.bf16, layout_dst)

    row = al.block_id(0)
    tid = al.thread_id(0)

    # Local max (F32) over this thread's 32 columns
    first_col = tid * 32
    best = al.convert(src[row, first_col], al.f32)

    for i in al.range(1, 32):
        val = al.convert(src[row, tid * 32 + i], al.f32)
        if val > best:
            best = val

    # Shared-memory tree reduction
    smem = al.make_shared((256,), al.f32)
    smem[tid] = best
    al.syncthreads()

    if tid == 0:
        result = smem[0]
        for i in al.range(1, 256):
            v = smem[i]
            if v > result:
                result = v
        # result - result = 0  →  GELU(0) = 0
        dst[row, 0] = al.convert(0.0, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device

        # Extract weight and bias from nn.Linear
        # weight: (out_features, in_features) → B = weight.T: (in_features, out_features)
        W = self.gemm.weight.data.T.contiguous()
        bias = self.gemm.bias.data.contiguous()

        # Cast to BF16
        x_bf16 = x.to(torch.bfloat16).contiguous()
        W_bf16 = W.to(torch.bfloat16).contiguous()
        bias_bf16 = bias.to(torch.bfloat16).contiguous()

        # GEMM output: (batch_size, out_features) in BF16
        gemm_out = torch.empty(batch_size, self.gemm.out_features,
                               dtype=torch.bfloat16, device=device)

        grid_m = batch_size // BM
        grid_n = self.gemm.out_features // BN

        gemm_bf16_kernel[lambda: ((grid_m, grid_n, 1), (THREADS, 1, 1))](
            x_bf16.data_ptr(),
            W_bf16.data_ptr(),
            bias_bf16.data_ptr(),
            gemm_out.data_ptr(),
            batch_size,
            self.gemm.out_features,
            self.gemm.in_features,
        )

        # Reduction output: (batch_size, 1) in BF16
        reduce_out = torch.empty(batch_size, 1, dtype=torch.bfloat16, device=device)

        row_max_reduce_kernel[lambda: ((batch_size, 1, 1), (REDUCE_THREADS, 1, 1))](
            gemm_out.data_ptr(),
            reduce_out.data_ptr(),
            batch_size,
            self.gemm.out_features,
        )

        return reduce_out
