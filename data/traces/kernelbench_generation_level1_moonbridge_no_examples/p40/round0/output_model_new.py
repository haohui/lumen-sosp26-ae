import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── Compile-time constants ──────────────────────────────────────────────
BLOCK_SIZE = 256
NUM_CHUNKS = 256  # partitions per batch element
# norm_elems = 64 * 256 * 256 = 16384 * 256 = 4194304
# ELEMS_PER_CHUNK = 4194304 / 256 = 16384
ELEMS_PER_CHUNK = 16384


# ── Kernel 1: chunked partial reduction ─────────────────────────────────
@avelang.jit
def ln_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    partial_ptr: al.Pointer(al.f32),
    batch: al.i32,
    norm_elems: al.i32,
    NUM_CHUNKS: al.constexpr,
    ELEMS_PER_CHUNK: al.constexpr,
    BLOCK_SIZE: al.constexpr,
):
    bid = al.block_id(0)
    tid = al.thread_id(0)

    batch_idx = bid // NUM_CHUNKS
    chunk_idx = bid % NUM_CHUNKS

    # 1-D flat views
    flat_layout = al.make_layout((batch * norm_elems,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, flat_layout)

    partial_layout = al.make_layout((batch * NUM_CHUNKS * 2,), (1,))
    partial = al.make_tensor(partial_ptr, al.f32, partial_layout)

    base = batch_idx * norm_elems + chunk_idx * ELEMS_PER_CHUNK

    # Shared memory for block-level reduction
    smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
    smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

    # Each thread accumulates over its interleaved slice of the chunk
    my_sum = al.convert(0.0, al.f32)
    my_sq = al.convert(0.0, al.f32)

    for i in al.range(tid, ELEMS_PER_CHUNK, BLOCK_SIZE):
        val = al.convert(x[base + i], al.f32)
        my_sum = my_sum + val
        my_sq = my_sq + val * val

    smem_sum[tid] = my_sum
    smem_sq[tid] = my_sq
    al.syncthreads()

    # Tree reduction across the block (log2(256) = 8 steps)
    stride_val = BLOCK_SIZE // 2
    for _ in al.range(8):
        if tid < stride_val:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + stride_val]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + stride_val]
        stride_val = stride_val // 2
        al.syncthreads()

    # Thread 0 writes the partial (sum, sum_sq) to global memory
    if tid == 0:
        out_base = batch_idx * NUM_CHUNKS * 2 + chunk_idx * 2
        partial[out_base] = smem_sum[0]
        partial[out_base + 1] = smem_sq[0]


# ── Kernel 2: final reduction + normalization ───────────────────────────
@avelang.jit
def ln_normalize_kernel(
    x_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    partial_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    batch: al.i32,
    norm_elems: al.i32,
    eps: al.f32,
    NUM_CHUNKS: al.constexpr,
    ELEMS_PER_CHUNK: al.constexpr,
    BLOCK_SIZE: al.constexpr,
):
    bid = al.block_id(0)
    tid = al.thread_id(0)

    batch_idx = bid // NUM_CHUNKS
    chunk_idx = bid % NUM_CHUNKS

    # 1-D flat views
    flat_layout = al.make_layout((batch * norm_elems,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, flat_layout)
    out = al.make_tensor(out_ptr, al.bf16, flat_layout)

    norm_layout = al.make_layout((norm_elems,), (1,))
    gamma = al.make_tensor(gamma_ptr, al.bf16, norm_layout)
    beta = al.make_tensor(beta_ptr, al.bf16, norm_layout)

    partial_layout = al.make_layout((batch * NUM_CHUNKS * 2,), (1,))
    partial = al.make_tensor(partial_ptr, al.f32, partial_layout)

    # ── Phase A: reduce NUM_CHUNKS partial sums to global mean and var ──
    smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
    smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

    # Thread tid loads the partial sum for chunk tid
    p_base = batch_idx * NUM_CHUNKS * 2
    smem_sum[tid] = partial[p_base + tid * 2]
    smem_sq[tid] = partial[p_base + tid * 2 + 1]
    al.syncthreads()

    stride_val = BLOCK_SIZE // 2
    for _ in al.range(8):
        if tid < stride_val:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + stride_val]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + stride_val]
        stride_val = stride_val // 2
        al.syncthreads()

    total_sum = smem_sum[0]
    total_sq = smem_sq[0]
    n = al.convert(norm_elems, al.f32)
    mean = total_sum / n
    var = total_sq / n - mean * mean
    inv_std = al.convert(1.0, al.f32) / al.sqrt(var + eps)

    # ── Phase B: normalize this chunk and write output ──
    x_base = batch_idx * norm_elems + chunk_idx * ELEMS_PER_CHUNK
    gb_base = chunk_idx * ELEMS_PER_CHUNK  # offset into gamma/beta

    for i in al.range(tid, ELEMS_PER_CHUNK, BLOCK_SIZE):
        idx = x_base + i
        val = al.convert(x[idx], al.f32)
        g = al.convert(gamma[gb_base + i], al.f32)
        b = al.convert(beta[gb_base + i], al.f32)
        normed = (val - mean) * inv_std * g + b
        out[idx] = al.convert(normed, al.bf16)


# ── Host wrapper ────────────────────────────────────────────────────────
def avelang_layer_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA/HIP device."

    B = x.shape[0]
    norm_elems = gamma.numel()

    # Ensure BF16 contiguous storage
    x_bf16 = x.detach().contiguous().to(torch.bfloat16)
    gamma_bf16 = gamma.detach().contiguous().to(torch.bfloat16)
    beta_bf16 = beta.detach().contiguous().to(torch.bfloat16)
    out = torch.empty_like(x_bf16)

    # Intermediate buffer: [B, NUM_CHUNKS, 2] of f32
    partial = torch.empty(B, NUM_CHUNKS, 2, dtype=torch.float32, device=x.device)

    total_blocks = B * NUM_CHUNKS

    # Launch kernel 1: partial reduction
    ln_reduce_kernel[lambda: ((total_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, partial, B, norm_elems,
        NUM_CHUNKS, ELEMS_PER_CHUNK, BLOCK_SIZE,
    )

    # Launch kernel 2: final reduction + normalization
    ln_normalize_kernel[lambda: ((total_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, gamma_bf16, beta_bf16, partial, out,
        B, norm_elems, eps,
        NUM_CHUNKS, ELEMS_PER_CHUNK, BLOCK_SIZE,
    )

    return out


# ── ModelNew entry point ────────────────────────────────────────────────
class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        out = avelang_layer_norm(
            x,
            self.ln.weight,
            self.ln.bias,
            self.ln.eps,
        )
        return out.to(orig_dtype)
