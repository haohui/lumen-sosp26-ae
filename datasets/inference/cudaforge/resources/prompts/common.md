CUDAForge HIP extension requirements:
1. Under ROCm, `torch.utils.cpp_extension.load_inline` must still pass HIP source through the `cuda_sources=` argument. Do not use `hip_sources=` or `extra_hip_cflags=`.
