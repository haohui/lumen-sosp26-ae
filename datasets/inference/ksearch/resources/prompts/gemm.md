[KSEARCH HIP HARD CONSTRAINTS]
- main.cpp MUST NOT contain kernel launch syntax `<<< >>>`.
- main.cpp MUST NOT call `hipLaunchKernelGGL`.
- main.cpp must call host launch wrappers declared in kernel.h with this pattern:
  `hipError_t ksearch_launch_<kernel>(dim3 grid, dim3 block, size_t shared_mem, hipStream_t stream, ...);`
- kernel.cu must implement those wrappers and perform actual `<<< >>>` launches there.
- Do NOT include private HIP headers under `hip/amd_detail/*`.
- Use only public HIP headers like `<hip/hip_runtime.h>`, `<hip/hip_bfloat16.h>`, `<hip/hip_fp8.h>` when needed.
