import torch
import substrate
import substrate.language as S

BLOCK_SIZE: S.constexpr = 256


@substrate.jit
def triplet_loss_per_sample_kernel(
    anchor_ptr: S.Pointer(S.bf16),
    positive_ptr: S.Pointer(S.bf16),
    negative_ptr: S.Pointer(S.bf16),
    loss_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    dim: S.i32,
    margin_scaled: S.i32,
):
    """Compute per-sample triplet margin loss.

    margin_scaled: margin * 10000 (fixed-point representation)
    """
    tid = S.thread_id(0)
    bid = S.block_id(0)

    if bid >= batch_size:
        return

    sample_idx = bid

    # Create tensor views for input
    layout = S.make_layout((batch_size, dim), (dim, 1))
    anchor = S.make_tensor(anchor_ptr, S.bf16, layout)
    positive = S.make_tensor(positive_ptr, S.bf16, layout)
    negative = S.make_tensor(negative_ptr, S.bf16, layout)

    # Each thread accumulates partial sums for strided elements
    # Use f32 for accumulation to avoid precision loss
    sum_pos_sq = S.convert(0.0, S.f32)
    sum_neg_sq = S.convert(0.0, S.f32)

    # Process elements with stride BLOCK_SIZE
    for j in S.range(0, dim, BLOCK_SIZE):
        idx = j + tid
        if idx < dim:
            # Convert bf16 to f32 for computation
            a_val = S.convert(anchor[sample_idx, idx], S.f32)
            p_val = S.convert(positive[sample_idx, idx], S.f32)
            n_val = S.convert(negative[sample_idx, idx], S.f32)

            diff_pos = a_val - p_val
            diff_neg = a_val - n_val

            sum_pos_sq = sum_pos_sq + diff_pos * diff_pos
            sum_neg_sq = sum_neg_sq + diff_neg * diff_neg

    # Shared memory for block reduction
    shared_pos = S.make_shared((BLOCK_SIZE,), S.f32)
    shared_neg = S.make_shared((BLOCK_SIZE,), S.f32)

    shared_pos[tid] = sum_pos_sq
    shared_neg[tid] = sum_neg_sq

    S.syncthreads()

    # Tree reduction within block
    if tid < 128:
        shared_pos[tid] = shared_pos[tid] + shared_pos[tid + 128]
        shared_neg[tid] = shared_neg[tid] + shared_neg[tid + 128]
    S.syncthreads()

    if tid < 64:
        shared_pos[tid] = shared_pos[tid] + shared_pos[tid + 64]
        shared_neg[tid] = shared_neg[tid] + shared_neg[tid + 64]
    S.syncthreads()

    if tid < 32:
        shared_pos[tid] = shared_pos[tid] + shared_pos[tid + 32]
        shared_neg[tid] = shared_neg[tid] + shared_neg[tid + 32]
    S.syncthreads()

    if tid < 16:
        shared_pos[tid] = shared_pos[tid] + shared_pos[tid + 16]
        shared_neg[tid] = shared_neg[tid] + shared_neg[tid + 16]
    S.syncthreads()

    if tid < 8:
        shared_pos[tid] = shared_pos[tid] + shared_pos[tid + 8]
        shared_neg[tid] = shared_neg[tid] + shared_neg[tid + 8]
    S.syncthreads()

    if tid < 4:
        shared_pos[tid] = shared_pos[tid] + shared_pos[tid + 4]
        shared_neg[tid] = shared_neg[tid] + shared_neg[tid + 4]
    S.syncthreads()

    if tid < 2:
        shared_pos[tid] = shared_pos[tid] + shared_pos[tid + 2]
        shared_neg[tid] = shared_neg[tid] + shared_neg[tid + 2]
    S.syncthreads()

    # Thread 0 computes final loss for this sample
    if tid == 0:
        total_pos_sq = shared_pos[0] + shared_pos[1]
        total_neg_sq = shared_neg[0] + shared_neg[1]

        d_pos = S.sqrt(total_pos_sq)
        d_neg = S.sqrt(total_neg_sq)

        # Convert scaled margin back to float
        margin = S.convert(margin_scaled, S.f32) * S.convert(0.0001, S.f32)
        loss_val = d_pos - d_neg + margin

        # ReLU: max(0, loss_val)
        if loss_val < S.convert(0.0, S.f32):
            loss_val = S.convert(0.0, S.f32)

        loss_layout = S.make_layout((batch_size,), (1,))
        loss_tensor = S.make_tensor(loss_ptr, S.bf16, loss_layout)
        loss_tensor[sample_idx] = S.convert(loss_val, S.bf16)


def substrate_triplet_margin_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Compute Triplet Margin Loss using Substrate kernels."""
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "Tensors must be on CUDA/HIP device."

    batch_size = anchor.shape[0]
    dim = anchor.shape[1]

    # Ensure contiguous and BF16
    anchor_bf16 = anchor.to(dtype=torch.bfloat16).contiguous()
    positive_bf16 = positive.to(dtype=torch.bfloat16).contiguous()
    negative_bf16 = negative.to(dtype=torch.bfloat16).contiguous()

    # Allocate output for per-sample losses in BF16
    losses = torch.empty(batch_size, dtype=torch.bfloat16, device=anchor.device)

    # Scale margin to integer (fixed-point with 4 decimal places)
    margin_scaled = int(margin * 10000)

    # Launch kernel to compute per-sample losses
    triplet_loss_per_sample_kernel[lambda: ((batch_size, 1, 1), (BLOCK_SIZE, 1, 1))](
        anchor_bf16, positive_bf16, negative_bf16, losses, batch_size, dim, margin_scaled
    )

    # Compute mean - return as BF16 scalar to match reference
    result = losses.mean()

    return result


class ModelNew(torch.nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.
    Optimized with Substrate DSL kernels for AMD GPU.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """

    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return substrate_triplet_margin_loss(anchor, positive, negative, self.margin)
