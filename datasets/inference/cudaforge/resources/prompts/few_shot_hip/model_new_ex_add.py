import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_hip(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);
    int size = a.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    auto stream = at::hip::getCurrentHIPStream().stream();
    elementwise_add_kernel<<<num_blocks, block_size, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size
    );
    return out;
}
"""

cpp_src = "torch::Tensor elementwise_add_hip(torch::Tensor a, torch::Tensor b);"

elementwise_add = load_inline(
    name="elementwise_add_hip",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["elementwise_add_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_hip(a, b)
