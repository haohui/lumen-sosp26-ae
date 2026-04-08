import torch
import torch.nn as nn
import substrate
import substrate.language as S

BLOCK_SIZE = 256


@substrate.jit
def sum_reduce_kernel(
    input_ptr: S.Pointer(S.bf16),
    out_ptr: S.Pointer(S.bf16),
    batch_size: S.i32,
    out_features: S.i32,
):
    """
    Sum reduction over dim=1.

    Each block processes one batch element.
    Threads cooperate to sum all output features.
    """
    bid = S.block_id(0)
    tid = S.thread_id(0)

    if bid >= batch_size:
        return

    # Create tensor views
    input_layout = S.make_layout((batch_size, out_features), (out_features, 1))
    input_tensor = S.make_tensor(input_ptr, S.bf16, input_layout)

    out_layout = S.make_layout((batch_size, 1), (1, 1))
    out = S.make_tensor(out_ptr, S.bf16, out_layout)

    # Shared memory for partial sums
    shared = S.make_shared((BLOCK_SIZE,), S.f32)

    # Each thread accumulates over multiple elements
    thread_sum = S.convert(0.0, S.f32)

    # Strided accumulation over output features
    for j in S.range(tid, out_features, BLOCK_SIZE):
        val = S.convert(input_tensor[bid, j], S.f32)
        thread_sum = thread_sum + val

    # Store partial sum
    shared[tid] = thread_sum
    S.syncthreads()

    # Reduce (thread 0)
    if tid == 0:
        total = S.convert(0.0, S.f32)
        for t in S.range(BLOCK_SIZE):
            total = total + shared[t]
        out[bid, 0] = S.convert(total, S.bf16)


class ModelNew(nn.Module):
    """
    Model that performs linear + sum using a hybrid approach.

    Linear layer uses PyTorch's optimized implementation for numerical
    correctness. The reduction is implemented as a Substrate kernel.

    The operations after sum (max, mean, logsumexp x2) are identity
    operations on a single-element dimension.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        batch_size = x.shape[0]
        input_dtype = x.dtype

        # Compute linear output using PyTorch (numerically matches reference)
        linear_out = self.linear(x)

        # Apply reduction kernel
        linear_out_bf16 = linear_out.to(torch.bfloat16).contiguous()
        out = torch.empty(batch_size, 1, dtype=torch.bfloat16, device=x.device)

        grid = (batch_size, 1, 1)
        block = (BLOCK_SIZE, 1, 1)

        sum_reduce_kernel[lambda: (grid, block)](
            linear_out_bf16, out,
            batch_size, self.out_features
        )

        return out.to(input_dtype)
