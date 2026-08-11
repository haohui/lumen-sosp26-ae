import torch
import substrate
import substrate.language as S


@substrate.jit
def max_pool3d_kernel(
    x: S.Tensor((16, 32, 128, 128, 128), S.bf16),
    out: S.Tensor((16, 32, 62, 62, 62), S.bf16),
):
    """
    3D Max Pooling kernel with kernel_size=3, stride=2, padding=1, dilation=3.
    Each thread computes one output element by finding the maximum over a 3x3x3 dilated window.
    """
    # Get thread and block IDs
    bid = S.block_id(0)
    tid = S.thread_id(0)

    # Compute output position from linear ID
    # Output dimensions: batch=16, channels=32, od=62, oh=62, ow=62
    linear_id = bid * 256 + tid

    b = linear_id // 7626496  # 32 * 62 * 62 * 62
    rem = linear_id % 7626496
    c = rem // 238328  # 62 * 62 * 62
    rem = rem % 238328
    od = rem // 3844  # 62 * 62
    rem = rem % 3844
    oh = rem // 62
    ow = rem % 62

    # Constants for pooling
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 3

    # Compute corresponding input position (without padding)
    id_base = od * stride - padding
    ih_base = oh * stride - padding
    iw_base = ow * stride - padding

    # Initialize max value to a very small number
    # Use a large negative value for max pooling
    max_val = -1e30

    # Track if we've found any valid element
    found_valid = 0

    # Iterate over 3x3x3 kernel window with dilation
    for kd in S.range(kernel_size):
        id_idx = id_base + kd * dilation
        for kh in S.range(kernel_size):
            ih_idx = ih_base + kh * dilation
            for kw in S.range(kernel_size):
                iw_idx = iw_base + kw * dilation
                # Check if input position is valid (inside bounds)
                if id_idx >= 0 and id_idx < 128 and ih_idx >= 0 and ih_idx < 128 and iw_idx >= 0 and iw_idx < 128:
                    val = S.convert(x[b, c, id_idx, ih_idx, iw_idx], S.f32)
                    if found_valid == 0:
                        max_val = val
                        found_valid = 1
                    else:
                        if val > max_val:
                            max_val = val

    out[b, c, od, oh, ow] = S.convert(max_val, S.bf16)


def substrate_max_pool3d(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for 3D Max Pooling using Substrate kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.shape == (16, 32, 128, 128, 128), f"Expected shape (16, 32, 128, 128, 128), got {x.shape}"

    # Ensure contiguous and correct dtype
    x = x.contiguous()
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    # Create output tensor
    out = torch.empty((16, 32, 62, 62, 62), dtype=torch.bfloat16, device=x.device)

    # Compute grid size: total output elements / threads per block
    output_elements = 16 * 32 * 62 * 62 * 62  # 122,023,936
    threads_per_block = 256
    grid_size = (output_elements + threads_per_block - 1) // threads_per_block

    # Launch kernel
    max_pool3d_kernel[lambda: ((grid_size, 1, 1), (threads_per_block, 1, 1))](x, out)

    return out


class ModelNew(torch.nn.Module):
    """
    Optimized 3D Max Pooling model using Substrate DSL.
    """
    def __init__(self, kernel_size: int = 3, stride: int = 2, padding: int = 1, dilation: int = 3):
        """
        Initializes the Max Pooling layer.

        Args:
            kernel_size (int): Size of the kernel for the max pooling operation.
            stride (int, optional): Stride of the pooling operation.
            padding (int, optional): Padding applied to the input tensor.
            dilation (int, optional): Spacing between kernel elements.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation

        # Validate parameters match the compiled kernel
        if kernel_size != 3 or self.stride != 2 or padding != 1 or dilation != 3:
            raise NotImplementedError(
                f"This optimized kernel only supports kernel_size=3, stride=2, padding=1, dilation=3. "
                f"Got kernel_size={kernel_size}, stride={self.stride}, padding={padding}, dilation={dilation}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 3D applied.
        """
        # Move to CUDA if needed
        if not x.is_cuda:
            x = x.cuda()

        # Apply the optimized pooling kernel
        return substrate_max_pool3d(x)
