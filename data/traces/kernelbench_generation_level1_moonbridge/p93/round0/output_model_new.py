import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def apply_mask_kernel(
    x_ptr: al.Pointer(al.bf16),
    mask_ptr: al.Pointer(al.u8),
    out_ptr: al.Pointer(al.bf16),
    total_elems: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    gid = bid * BLOCK_SIZE + tid

    if gid < total_elems:
        layout_1d = al.make_layout((total_elems,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_1d)
        mask = al.make_tensor(mask_ptr, al.u8, layout_1d)
        out = al.make_tensor(out_ptr, al.bf16, layout_1d)

        x_val = x[gid]
        m_val = mask[gid]

        # Multiply: if mask is non-zero, keep x, else zero
        f32_val = al.convert(x_val, al.f32)
        f32_mask = al.convert(m_val, al.f32)
        result = f32_val * f32_mask
        out[gid] = al.convert(result, al.bf16)


def avelang_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Masked cumulative sum using AveLang mask kernel + PyTorch cumsum."""
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.shape == mask.shape, "x and mask must have same shape."

    # Ensure BF16 precision
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    if mask.dtype != torch.uint8:
        mask = mask.to(torch.uint8)

    x = x.contiguous()
    mask = mask.contiguous()

    total_elems = x.numel()
    masked = torch.empty_like(x)

    num_blocks = (total_elems + BLOCK_SIZE - 1) // BLOCK_SIZE

    apply_mask_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x, mask, masked, total_elems,
    )

    # Use PyTorch's cumsum for the scan (matching reference semantics exactly)
    return torch.cumsum(masked, dim=dim)


class ModelNew(nn.Module):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.

        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        return avelang_masked_cumsum(x, mask, self.dim)
