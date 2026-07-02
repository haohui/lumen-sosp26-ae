## 1: Prompt 1

Optimize the substrate Conv2D kernel in xxx. Replace the scalar implicit-GEMM accumulation in `_igemm_kernel` with a 4-wave MFMA kernel using `S.amdgpu.mfma_32x32x8_bf16_f32`. Keep the existing launch structure and public behavior intact. Rename the optimized entrypoint from `conv2d_asym_naive` to `conv2d_asym`, update `__all__`, and update the benchmark/test callsites to use `conv2d_asym`.

  Scope invariants:
  - This is an MFMA-only transformation.
  - Do not add LDS staging, vectorized loads, async copies, or a different tile shape as part of this change.
  - Preserve the supported behavior of the existing kernel.
  - Do not look at other commits in the repo.

  Update callsites:
  - /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py
  - /workspace/substrate/benchmark/conv2d/bench_conv2d.py

  Validation:
  - Run `python -m pytest /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py`

- Make the optimized path cudagraph-safe: never build descriptor / metadata device tensors inside `forward()`. Prebuild or cache them and reuse them; only rebuild if the underlying storage pointer changes.
- Do not use torch native compute anywhere in `output_model_new.py` to perform convolution, multiplication, or linear algebra, including fallback branches.
- The optimized kernel must actually issue MFMA instructions in the substrate kernel; a solution that does not use MFMA is not acceptable.
- Do not use git to try to find any old files!!!
