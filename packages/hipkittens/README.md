# hipkittens

`hipkittens` packages the pinned HIPKittens square BF16 GEMM kernels as one
Python extension. It is intended for ROCm systems and uses the current PyTorch
CUDA/HIP stream.

## Install

Install a ROCm-enabled PyTorch first. Then point `HIPKITTENS_ROOT` at an
external checkout of HipKittens pinned to
`7d58fa1026b4582a75ebdaf7ab5e45e3747a2b7b`:

```bash
HIPKITTENS_ROOT=/path/to/HipKittens uv pip install -e ./packages/hipkittens
```

The build uses `hipcc`, targets `gfx942` by default, and produces one native
`hipkittens._C` extension. Set `HIPKITTENS_ARCH` to select another architecture
or `HIPCC` to select a particular compiler.

## API

```python
import hipkittens

hipkittens.gemm(a, b, out)  # out = a @ b.T
```

`a`, `b`, and `out` must be contiguous CUDA BF16 tensors on the same device.
Only the square sizes 1024, 2048, 4096, 8192, and 16384 are supported. The
operation neither allocates output nor converts tensor layouts.

## Upstream attribution

The compiled kernels are from [HazyResearch/HipKittens](https://github.com/HazyResearch/HipKittens)
at commit `7d58fa1026b4582a75ebdaf7ab5e45e3747a2b7b`, which is licensed under
the MIT License (Copyright 2024 HazyResearch).
