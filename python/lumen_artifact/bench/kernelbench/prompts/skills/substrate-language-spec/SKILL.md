---
name: substrate-language-spec
description: Use when generating, reviewing, or debugging Substrate DSL kernels. Covers imports, @substrate.jit syntax, types, launch form, memory and layout primitives, supported control flow, host-wrapper rules, and the verified API surface used in this workspace.
---

Use this skill first whenever Substrate syntax or semantics are unclear.

## Import requirement
Insert following code block at the beginning of the file:
```python
import substrate
import substrate.language as S
```

## Public surface map

### Runtime-visible Python exports

These names exist as normal Python objects and are safe in annotations or host code:

- `substrate.jit`: JIT-kernel decorator. It returns a launchable object, not a normal Python callable.
- `substrate.language` as `S`: the standard namespace alias used throughout the repo.
- `S.i8`, `S.i16`, `S.i32`, `S.i64`: signed integer scalar dtypes.
- `S.u1`: 1-bit unsigned integer / bool-like scalar dtype. Treat it as a real 1-bit type, not as `u8`.
- `S.u8`, `S.u16`, `S.u32`, `S.u64`: unsigned integer scalar dtypes.
- `S.f16`, `S.bf16`, `S.f32`, `S.f64`: floating-point scalar dtypes.
- `S.constexpr`:
  - in annotations, marks a compile-time kernel parameter
  - as `S.constexpr(value)`, wraps a Python global/nonlocal compile-time value
- `S.dynamic`: exported sentinel for dynamic shapes, but repo code prefers `S.Pointer + S.make_tensor` for runtime sizes and strides.
- `S.Tensor(shape, dtype)`: static tensor type used in annotations and `S.view(..., S.Tensor(...))` casts.
- `S.Pointer(dtype)`: raw pointer type used for runtime-shaped buffers.
- `S.block_id(dim)`: block index along launched grid dimension `dim`.
- `S.block_dim(dim)`: block extent along launched block dimension `dim`.
- `S.thread_id(dim)`: thread index inside the current block/workgroup along dimension `dim`.
- `S.grid_dim(dim)`: total grid extent along dimension `dim`.
- `S.shuffle(value, offset, width)`: absolute-lane shuffle.
- `S.shuffle_up(value, offset, width)`: upward lane shuffle.
- `S.shuffle_down(value, offset, width)`: downward lane shuffle.
- `S.shuffle_xor(value, offset, width)`: XOR-lane shuffle.
- `S.min(lhs, rhs)`: integer minimum. Signed vs unsigned behavior comes from operand type info.
- `S.max(lhs, rhs)`: integer maximum. Signed vs unsigned behavior comes from operand type info.
- `S.exp2(x)`: floating-point exponential builtin computing `2**x`. 
- `S.exp(x)`: floating-point exponential builtin computing `e**x`. AMDGPU lowering maps `f32` to `llvm.amdgcn.exp`.
- `S.tanh(x)`: floating-point hyperbolic tangent builtin.
- `S.log(x)`: floating-point natural logarithm builtin. 
- `S.log2(x)`: floating-point base-2 logarithm builtin. 
- `S.erf(x)`: floating-point error-function builtin. Compiler-exposed, lightly tested — accuracy on gfx942 BF16 may be insufficient for GELU.
- `S.sqrt(x)`: floating-point square-root builtin.

### Compiler-recognized DSL symbols

These are valid inside `@substrate.jit` function bodies because the AST lowerer recognizes them. Many are not real Python functions on `substrate.language`, so never rely on them outside JIT code:

- `S.convert`
- `S.bitcast`
- `S.range`
- `S.syncthreads`
- `S.printf`
- `S.make_tensor`
- `S.make_local`
- `S.make_shared`
- `S.make_layout`
- `S.subview`
- `S.view`
- `S.full`
- `S.amdgpu.*`
- `S.nvvm.*`

### Tooling-only public APIs

`substrate.compiler.compile`, `substrate.compiler.make_backend`, `substrate.compiler.ASTSource`, and `substrate.compiler.CompiledKernel` are public, but they are testing/tooling APIs. Do not emit them in generated model code unless the task is explicitly about compiler internals.

## Required file and import shape

- Always start generated files with `import substrate` and `import substrate.language as S`.
- Missing `import substrate.language as S` causes launch-time failure; there is an explicit error test for this.
- `@substrate.jit` functions must live in a Python file on disk. The runtime rejects REPL/inline-only functions.
- The kernel body is parsed from Python AST. Body-only DSL names like `S.make_shared` are not executed as normal Python code.

## Kernel declaration rules

- Kernels are Python functions decorated with `@substrate.jit`.
- Kernel parameters may be:
  - Scalar dtypes, for example `x: S.i32`
  - Static tensors, for example `x: S.Tensor((128, 64), S.f16)`
  - Raw pointers, for example `x_ptr: S.Pointer(S.bf16)`
  - Compile-time parameters, for example `BLOCK_M: S.constexpr`
- Use `S.Tensor(...)` when the shape contract is known statically.
- Use `S.Pointer(dtype)` plus runtime shape/stride scalars and `S.make_tensor(...)` when sizes or strides are dynamic.
- Kernels cannot have explicit non-void return types. Write results into output buffers.
- `return` with no value is valid for early exit and is used in repo tests.
- Helper functions may also be `@substrate.jit` and can be called from kernels.
- Nested local helper functions inside a kernel are supported in tests.
- Verified tuple returns exist for helper functions, but tuple element types must be integer or index types.

Example:

```python
import substrate
import substrate.language as S

@substrate.jit
def saxpy(x: S.Tensor((64,), S.f32), y: S.Tensor((64,), S.f32), a: S.f32):
    tid = S.thread_id(0)
    y[tid] = a * x[tid] + y[tid]
```

## Launch and specialization behavior

- Launch syntax is:

```python
kernel[lambda: ((grid_x, grid_y, grid_z), (block_x, block_y, block_z))](...)
```

- The launch lambda is called with no arguments. Unlike Triton meta-launch patterns, do not expect meta-parameters to be passed into the lambda.
- Always provide explicit 3-tuples for grid and block.
- The only backend launch option verified in this repo is `num_warps=...`.
- `device`, `device_type`, and `stream` kwargs are explicitly rejected as deprecated.
- Compilation is cached per device and specialization.
- `S.constexpr` parameters and captured global `S.constexpr(...)` values contribute to specialization and cache keys.
- Calling an `@substrate.jit` function like a normal Python function from host code raises an error. Use launch syntax.

## Type system and scalar semantics

- Exported scalar types: `i8/i16/i32/i64`, `u1/u8/u16/u32/u64`, `f16/bf16/f32/f64`.
- `u1` is the bool-like integer type. Recent lowering work keeps 1-bit values as 1-bit instead of silently widening them to 8-bit storage.
- `S.Tensor((shape...), dtype)` means a statically typed memref-like tensor.
- `S.Pointer(dtype)` is a raw pointer-like argument. It is normally wrapped immediately with `S.make_tensor`.
- `S.constexpr` in parameter annotations marks compile-time parameters.
- `S.constexpr(value)` wraps a Python global/nonlocal value as compile-time data. The JIT currently infers compile-time types from Python `bool`, `int`, and `float`.
- `S.dynamic` is exported as a sentinel, but there are no repo examples using dynamic `S.Tensor` annotations. Prefer `S.Pointer + S.make_tensor` for runtime shapes.
- `void` exists internally in `core.py` but is not exported from `substrate.language`; do not use `S.void`.
- Use `S.convert` for explicit promotion or demotion. Do not rely on implicit dtype conversion.
- Assignment and store sites are stricter than `S.convert`: implicit demotion is still rejected in general, including augmented assignment such as `dst += value`.
- One narrow exception now exists for stores: a floating-point constant may be implicitly narrowed to a smaller floating-point destination type when the value round-trips exactly with no precision loss. Do not rely on this for non-constant values or for precision-losing literals; use `S.convert(...)` instead.

## Execution model primitives

- `S.thread_id(axis)`, `S.block_id(axis)`, `S.block_dim(axis)`, `S.grid_dim(axis)` are the core launch-space queries.
- Use axis values `0`, `1`, or `2`.
- Substrate does not provide Triton-style implicit program indexing. Compute flattened or tiled coordinates yourself.
- `S.syncthreads()` is the block/workgroup barrier and takes no arguments.
- `S.printf(fmt, *args)` is compiler-exposed. Constraints from the implementation:
  - first argument must be a non-empty string literal
  - at least one argument is required
  - no repo kernel currently relies on it, so treat it as compiler-exposed rather than verified

## Control flow and function-call rules

- `if` / `else` is supported.
- Integer conditions are coerced to boolean by comparing against zero.
- `for` loops must iterate over `S.range(...)`. Plain Python `range(...)` is not accepted.
- `S.range` supports the three Python-like forms:
  - `S.range(stop)`
  - `S.range(start, stop)`
  - `S.range(start, stop, step)`
- Loop targets must be simple names, not tuple unpacking or other complex targets.
- Python unary operators are supported with current compiler semantics:
  - `-x` for float scalars/vectors, integer scalars/vectors, and index values
  - `+x` as a no-op
  - `not x` via boolean coercion against zero
- The compiler has a `while` lowering path, but the repo has no end-to-end `while` examples. Do not generate `while` unless you intend to validate it directly.
- Top-level helper callees should be `@substrate.jit`. Calling arbitrary top-level Python helpers from kernels is not a repo-proven pattern.
- Tuple unpacking from helper returns is verified:

```python
id_m, id_n = tuple_add(x, y)
```

## Core primitive semantics

- `S.convert(value, dtype)`: explicit numeric conversion. Demotion is allowed because the user requested it explicitly.
- `S.bitcast(value, dtype)`: reinterpret bits without changing the bit pattern.
- `S.min(lhs, rhs)` / `S.max(lhs, rhs)`: integer extrema. Signedness comes from type information. Do not treat them as verified floating-point min/max helpers.
- `S.tanh(x)`: floating-point only. Lowers to `math.tanh`; AMDGPU lowering maps `f32` to `__ocml_tanh_f32`.
- `S.exp2(x)`: floating-point only. This is `2**x`. AMDGPU lowering maps `f32` to `llvm.amdgcn.exp2`.
- `S.exp(x)`: floating-point only. This is `e**x`. AMDGPU lowering maps `f32` to `llvm.amdgcn.exp`.
- `S.log(x)`: floating-point only. This is the natural logarithm.
- `S.log2(x)`: floating-point only. This is the base-2 logarithm.
- `S.erf(x)`: floating-point only. AMDGPU lowering maps `f32` to `__ocml_erf_f32`.
- `S.sqrt(x)`: floating-point only.
- `S.shuffle(value, offset, width)`: lane shuffle by absolute lane id.
- `S.shuffle_up(value, offset, width)`: upward lane shuffle.
- `S.shuffle_down(value, offset, width)`: downward lane shuffle.
- `S.shuffle_xor(value, offset, width)`: XOR-lane shuffle.
- Shuffle constraints:
  - `value` must be an int, float, or 1D vector of int/float
  - `offset` and `width` must be integer or index values

## Memory, layout, and view system

Substrate is not just a flat pointer DSL. Layout/view composition is a first-class part of the language and is intentionally close to CuTE/PyCuTe.

- `S.make_shared(shape, dtype)`
  - allocates workgroup/shared memory
  - shape must be compile-time static
  - row-major strides are synthesized automatically
  - dynamic shared memory is not supported
- `S.make_local(shape, dtype)`
  - allocates private/register memory
  - shape must be compile-time static
  - row-major strides are synthesized automatically
- `S.make_layout(dims, strides)`
  - creates a layout descriptor
  - `dims` and `strides` must be tuples of the same arity
  - nested tuples are supported and are used for swizzled/tiled layouts
- `S.make_tensor(ptr, dtype, layout)`
  - wraps a raw pointer or i8 memref pointer with explicit shape/stride semantics
  - the third argument must come from `S.make_layout(...)`
  - this is the standard way to give a `S.Pointer(...)` argument tensor semantics
- `S.subview(base, offsets, sizes, strides)`
  - all three metadata arguments must be tuples
  - tuple lengths must match the base memref rank
  - offsets/sizes/strides may be static or runtime expressions
  - dimensions whose `size == 1` are rank-reduced in the result
- `S.view(memref, dtype, layout)`
  - remaps an existing memref through an explicit layout
  - preserves subview offsets correctly
  - the first argument must be a memref
- `S.view(value, S.Tensor(...))`
  - reinterpret-casts a memref, scalar, or vector into a new tensor/vector view
  - for vector inputs, source and target must have the same total bitwidth
  - target shape must be static for vector inputs
  - constants are not accepted as the first argument
- `S.full(shape, fill_value, dtype)`
  - creates a private tensor filled with one scalar
  - shape must be a static tuple
  - fill value is converted to the target scalar dtype if needed

Indexing rules:

- Use tuple-style indexing, for example `buf[i, j]`.
- Direct indexing works on tensor arguments, shared memory, local memory, subviews, and layout views.
- Nested `make_layout` shapes support logical multi-indexing and also the repo-tested “nested-linear” pattern where a linear index expands across nested dimensions.

Example pattern:

```python
shared_words = S.make_shared((16,), S.u32)
left_words = S.subview(shared_words, (0,), (8,), (1,))
layout = S.make_layout((2, 2), (2, 1))
left = S.view(left_words, S.u32, layout)
left[0, 0] = S.convert(11, S.u32)
```

## AMDGPU namespace: `S.amdgpu.*`

Verified or compiler-exposed AMDGPU intrinsics:

- MFMA matmul instructions:
  - `S.amdgpu.mfma_16x16x16_f16_f32`
  - `S.amdgpu.mfma_16x16x16_bf16_f32`
  - `S.amdgpu.mfma_f32_16x16x16_bf16`
  - `S.amdgpu.mfma_32x32x8_bf16_f32`
  - `S.amdgpu.mfma_f32_32x32x8_bf16`
- Buffer/resource helpers:
  - `S.amdgpu.make_rsrc(tensor, range_bytes)`
  - `S.amdgpu.raw_buffer_load_x1(rsrc, vindex, soffset, aux)`
  - `S.amdgpu.raw_buffer_load_x2(rsrc, vindex, soffset, aux)`
  - `S.amdgpu.raw_buffer_load_x4(rsrc, vindex, soffset, aux)`
  - `S.amdgpu.raw_buffer_load_x1_lds(rsrc, lds_ptr, size, vindex, soffset, offset, aux)`
  - `S.amdgpu.raw_buffer_store_x1(vdata, rsrc, vindex, soffset, aux)`
  - `S.amdgpu.raw_buffer_store_x2(vdata, rsrc, vindex, soffset, aux)`
  - `S.amdgpu.raw_buffer_store_x4(vdata, rsrc, vindex, soffset, aux)`
- Scalar/control helpers:
  - `S.amdgpu.perm(hi, lo, selector)`
  - `S.amdgpu.rcp(x)`
  - `S.amdgpu.s_waitcnt(vmcnt, expcnt, lgkmcnt)`
  - `S.amdgpu.sched_group_barrier(mask, size, group_id)`

Important AMDGPU constraints from the implementation:

- MFMA operands are vector fragments. Use the exact fragment shapes already present in `substrate/python/substrate_kernels/amdgpu_gemm.py` and `substrate/python/substrate_kernels/flash_attn.py`; do not guess fragment packing.
- `make_rsrc` expects a tensor/memref and an integer/index byte range in `[0, 2^32 - 1]`.
- `raw_buffer_load_x{1,2,4}` expects `rsrc` to be `vector<4xi32>`.
- `raw_buffer_store_x1` expects an integer scalar payload; `x2/x4` expect `vector<2xi32>` / `vector<4xi32>`.
- `raw_buffer_load_x1_lds` is now used in the handwritten flash-attention kernels. It currently requires compile-time `size=4` and `aux=0`.
- `perm` expects three 32-bit integer arguments.
- `rcp` expects `f32`.
- `s_waitcnt` is now used in the handwritten flash-attention kernels. It requires compile-time integers with ranges `vmcnt=[0,63]`, `expcnt=[0,7]`, `lgkmcnt=[0,15]`.
- `sched_group_barrier` requires compile-time non-negative integers representable as `u32`.

For flash-attention MFMA mapping, use `references/amd-attention-mfma.md` as the source of truth.

## NVVM namespace: `S.nvvm.*`

Verified or compiler-exposed NVVM intrinsics:

- MMA:
  - `S.nvvm.mma_16x8x16_f16_f16`
  - `S.nvvm.mma_16x8x8_f16_f32`
- Load-matrix families:
  - `S.nvvm.ldmatrix_m8n8_x{1,2,4}_b{16,8}`
  - `S.nvvm.ldmatrix_m8n8_x{1,2,4}_b{16,8}_trans`
  - `S.nvvm.ldmatrix_m16n16_x{1,2,4}_b{16,8}`
  - `S.nvvm.ldmatrix_m16n16_x{1,2,4}_b{16,8}_trans`
- Store-matrix families:
  - `S.nvvm.stmatrix_m8n8_x{1,2,4}_b{16,8}`
  - `S.nvvm.stmatrix_m8n8_x{1,2,4}_b{16,8}_trans`
  - `S.nvvm.stmatrix_m16n16_x{1,2,4}_b{16,8}`
  - `S.nvvm.stmatrix_m16n16_x{1,2,4}_b{16,8}_trans`

Important NVVM constraints from the implementation:

- `ldmatrix_*` takes one memref argument in workgroup/shared memory.
- The memref must have shape `8x8` or `16x16` matching the intrinsic name.
- Element types must match bit width:
  - `_b16`: `f16`, `bf16`, or `i16`
  - `_b8`: `i8` or supported float8 types
- `stmatrix_*` takes `(ptr, source)` where source is `i32` for `x1` and `vector<num x i32>` for `x2/x4`.
- `mma_*` operands must be vector types. Follow the exact fragment loading/staging patterns in the NVIDIA GEMM examples rather than inventing new fragment layouts.

## Host-wrapper rules for KernelBench-style generation

- `ModelNew.forward` is the semantic entrypoint. Preserve the exact math, shapes, and output dtype semantics of the original `Model.forward`.
- Make tensors contiguous before launching Substrate kernels.
- Keep device moves explicit. If the wrapper accepts CPU inputs, move them to the active GPU backend before launch and move outputs back only when needed.
- Prefer static `S.Tensor(...)` signatures when the benchmark’s shape contract is fixed.
- Prefer `S.Pointer + S.make_tensor` when runtime lengths/strides come from the model input.
- Pass runtime metadata such as lengths, strides, block counts, or GQA ratios as scalar arguments.
- Use `num_warps=...` only when the kernel structure clearly depends on it or when existing examples do.
- Preserve reference accumulation semantics. For BF16 kernels, FP32 accumulation is common in AMD/NVIDIA matmul paths, but the output dtype should still match the model contract unless the benchmark explicitly expects a different dtype.

## Known failure patterns

These patterns have been observed to cause compilation hangs, MLIR lowering failures, HIP crashes,
or silent numerical errors in KernelBench runs. Avoid all of them.

### 1. `S.Tensor` with non-constexpr shape → compilation hang / timeout

`S.Tensor((shape), dtype)` requires every element of `shape` to be a **compile-time constant**
(a Python integer literal, or a value wrapped in `S.constexpr`).
Using Python module-level variables or expressions that reference runtime values causes the
Substrate AST lowerer to hang indefinitely while trying to resolve the shape.

```python
# ❌ M, K are Python variables — compiler hangs
@substrate.jit
def kernel(A: S.Tensor((M, K), S.f32), B: S.Tensor((K, N), S.f32)):
    ...

# ❌ BATCH_SIZE, DIM are module-level constants but not S.constexpr — same hang
@substrate.jit
def kernel(x: S.Tensor((BATCH_SIZE, DIM), S.f32)):
    ...

# ✅ Correct: use S.Pointer + S.make_tensor for any runtime-variable shape
@substrate.jit
def kernel(x_ptr: S.Pointer(S.bf16), m: S.i32, k: S.i32):
    layout = S.make_layout((m, k), (k, 1))
    x = S.make_tensor(x_ptr, S.bf16, layout)
    ...
```

**Rule**: In KernelBench, `get_inputs()` returns tensors of arbitrary runtime shape.
Use `S.Pointer + S.make_tensor` by default.
Reserve `S.Tensor(static_shape)` only for shapes that are genuinely fixed compile-time
constants smaller than ~65536 elements per dimension.

### 2. `S.Tensor` with very large static shapes → MLIR codegen explosion / timeout

Even when the shape is a valid integer literal, huge extents cause the MLIR code generator
to produce enormous code, exceeding the 180 s eval timeout.

```python
# ❌ 1.6B-element tensor — codegen hangs
x: S.Tensor((1610612736,), S.f32)

# ❌ 2B-element 2-D tensor
x: S.Tensor((32768, 65535), S.f32)
```

## Do not invent APIs

Stay inside the API surface above. Common hallucinations from Triton/TVM/CUDA ports that are not verified Substrate APIs here:

- `S.load`, `S.store`, `S.arange`, `S.program_id`, `S.atomic_add`
- `tl.*` APIs from Triton
- `T.*` / `@T.prim_func` APIs from TVM/TIR for user-facing Substrate Python kernels
- ad hoc backend namespaces or MFMA names not present in the compiler

