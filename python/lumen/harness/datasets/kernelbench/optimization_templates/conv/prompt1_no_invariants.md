## 1: Prompt 1

Implement the Conv2D path with AveLang kernels.

Replace scalar implicit-GEMM accumulation with an MFMA Conv2D kernel using
`al.amdgpu.mfma_32x32x8_bf16_f32`. Preserve the public behavior of the
reference `Model.forward`, including input and output layout, dtype behavior,
stride, padding, dilation, groups, and bias semantics when they are present in
the target architecture.

Build a tiled implicit-GEMM Conv2D mapping from the reference model's input,
weight, and output tensors. Map output spatial positions and output channels to
the GEMM M/N dimensions, map input channels and kernel spatial positions to the
GEMM K dimension, and accumulate in FP32 before returning the dtype expected by
the reference model.

Scope invariants:
  - This is an MFMA-only transformation.
  - Do not add LDS staging, vectorized loads, async copies, or a different tile
    shape as part of this change.
  - Preserve the supported behavior of the existing kernel.
  - Do not look at other commits in the repo.
