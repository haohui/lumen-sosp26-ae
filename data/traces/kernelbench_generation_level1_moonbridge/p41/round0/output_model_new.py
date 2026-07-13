import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
SHM_ELEMS: al.constexpr = 277  # BLOCK_SIZE + (kernel_size - 1) * dilation = 256 + 21


@avelang.jit
def maxpool1d_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    features: al.i32,
    seq_len: al.i32,
    output_len: al.i32,
    kernel_size: al.i32,
    stride: al.i32,
    padding: al.i32,
    dilation: al.i32,
):
    tid = al.thread_id(0)
    bid_x = al.block_id(0)
    bid_y = al.block_id(1)

    block_start = bid_x * BLOCK_SIZE
    output_idx = block_start + tid
    bf_idx = bid_y

    batch_idx = bf_idx // features
    feature_idx = bf_idx - batch_idx * features

    total_in = batch_size * features * seq_len
    total_out = batch_size * features * output_len
    in_flat = al.make_tensor(input_ptr, al.bf16, al.make_layout((total_in,), (1,)))
    out_flat = al.make_tensor(output_ptr, al.bf16, al.make_layout((total_out,), (1,)))

    # All threads in the block participate in shared memory staging,
    # regardless of whether they have a valid output to compute.
    shm = al.make_shared((SHM_ELEMS,), al.bf16)

    neg_inf_bf16 = al.convert(-1.0e10, al.bf16)
    in_base = block_start * stride - padding
    row_offset = (batch_idx * features + feature_idx) * seq_len

    for load_i in al.range(2):
        load_idx = tid + load_i * BLOCK_SIZE
        if load_idx < SHM_ELEMS:
            in_pos = in_base + load_idx
            if in_pos >= 0:
                if in_pos < seq_len:
                    shm[load_idx] = in_flat[row_offset + in_pos]
                else:
                    shm[load_idx] = neg_inf_bf16
            else:
                shm[load_idx] = neg_inf_bf16

    al.syncthreads()

    # Only compute and write for valid output positions
    if batch_idx < batch_size:
        if feature_idx < features:
            if output_idx < output_len:
                neg_inf = al.convert(-1.0e10, al.f32)
                current_max = neg_inf

                local_idx = tid * stride
                for k in al.range(kernel_size):
                    shm_idx = local_idx + k * dilation
                    val = al.convert(shm[shm_idx], al.f32)
                    if val > current_max:
                        current_max = val

                out_offset = (batch_idx * features + feature_idx) * output_len
                out_flat[out_offset + output_idx] = al.convert(current_max, al.bf16)


def avelang_maxpool1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Input tensor must be bfloat16"

    batch_size, features, seq_len = x.shape

    output_len = (seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    x_contig = x.contiguous()
    out = torch.empty((batch_size, features, output_len), dtype=torch.bfloat16, device=x.device)

    num_blocks_seq = (output_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks_bf = batch_size * features

    maxpool1d_kernel[lambda: ((num_blocks_seq, num_blocks_bf, 1), (BLOCK_SIZE, 1, 1))](
        x_contig, out, batch_size, features, seq_len, output_len,
        kernel_size, stride, padding, dilation,
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 1D using AveLang DSL.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bf16 = x.to(dtype=torch.bfloat16, device=x.device).contiguous()
        result = avelang_maxpool1d(x_bf16, self.kernel_size, self.stride, self.padding, self.dilation)
        return result.to(x.dtype)
