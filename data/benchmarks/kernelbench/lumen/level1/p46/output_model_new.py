import torch
import substrate
import substrate.language as S


@substrate.jit
def avg_pool3d_kernel(
    x: S.Tensor((16, 32, 128, 128, 256), S.bf16),
    out: S.Tensor((16, 32, 64, 64, 128), S.bf16),
):
    """
    3D Average Pooling kernel with kernel_size=3, stride=2, padding=1.
    Each thread computes one output element by averaging over a 3x3x3 window.
    """
    # Get thread and block IDs
    bid = S.block_id(0)
    tid = S.thread_id(0)

    # Compute output position from linear ID
    # Output dimensions: batch=16, channels=32, od=64, oh=64, ow=128
    linear_id = bid * 256 + tid

    b = linear_id // 16777216  # 32 * 64 * 64 * 128
    rem = linear_id % 16777216
    c = rem // 524288  # 64 * 64 * 128
    rem = rem % 524288
    od = rem // 8192  # 64 * 128
    rem = rem % 8192
    oh = rem // 128
    ow = rem % 128

    # Constants for pooling
    kernel_size = 3
    stride = 2
    padding = 1
    kernel_volume = 27  # 3*3*3

    # Compute corresponding input position (without padding)
    id_base = od * stride - padding
    ih_base = oh * stride - padding
    iw_base = ow * stride - padding

    # Accumulate sum for valid elements (treating padding as zeros)
    sum_val = 0.0

    # Iterate over 3x3x3 kernel window
    for kd in S.range(kernel_size):
        id_idx = id_base + kd
        for kh in S.range(kernel_size):
            ih_idx = ih_base + kh
            for kw in S.range(kernel_size):
                iw_idx = iw_base + kw
                # Check if input position is valid (inside bounds)
                if id_idx >= 0 and id_idx < 128 and ih_idx >= 0 and ih_idx < 128 and iw_idx >= 0 and iw_idx < 256:
                    val = S.convert(x[b, c, id_idx, ih_idx, iw_idx], S.f32)
                    sum_val = sum_val + val

    # Compute average (dividing by kernel volume including padding)
    avg = sum_val / S.convert(kernel_volume, S.f32)
    out[b, c, od, oh, ow] = S.convert(avg, S.bf16)


def substrate_avg_pool3d(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for 3D Average Pooling using Substrate kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.shape == (16, 32, 128, 128, 256), f"Expected shape (16, 32, 128, 128, 256), got {x.shape}"

    # Ensure contiguous and correct dtype
    x = x.contiguous()
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    # Create output tensor
    out = torch.empty((16, 32, 64, 64, 128), dtype=torch.bfloat16, device=x.device)

    # Compute grid size: total output elements / threads per block
    output_elements = 16 * 32 * 64 * 64 * 128  # 268,435,456
    threads_per_block = 256
    grid_size = (output_elements + threads_per_block - 1) // threads_per_block  # 1,048,576

    # Launch kernel
    avg_pool3d_kernel[lambda: ((grid_size, 1, 1), (threads_per_block, 1, 1))](x, out)

    return out


class ModelNew(torch.nn.Module):
    """
    Optimized 3D Average Pooling model using Substrate DSL.
    """
    def __init__(self, kernel_size: int = 3, stride: int = 2, padding: int = 1):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the kernel to apply pooling. Must be 3 for this kernel.
            stride (int): Stride of the pooling operation. Must be 2 for this kernel.
            padding (int): Padding to apply before pooling. Must be 1 for this kernel.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

        # Validate parameters match the compiled kernel
        if kernel_size != 3 or self.stride != 2 or padding != 1:
            raise NotImplementedError(
                f"This optimized kernel only supports kernel_size=3, stride=2, padding=1. "
                f"Got kernel_size={kernel_size}, stride={self.stride}, padding={padding}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        # Move to CUDA if needed
        if not x.is_cuda:
            x = x.cuda()

        # Apply the optimized pooling kernel
        return substrate_avg_pool3d(x)
