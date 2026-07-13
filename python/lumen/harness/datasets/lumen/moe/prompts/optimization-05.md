# Optimization 05: expose packed two-lane arithmetic

Optimize the existing AveLang fused FP8 MoE kernel by rewriting selected
scalar arithmetic as explicit packed two-lane operations. The input already
contains the Stage 1/Stage 2 pipelines and `v_setvskip` output masking;
preserve them.

This round is about making pairs of independent FP32 operations visible to the
compiler as `Tensor((2,), f32)` vectors. Apply the transformation only to the
four hot arithmetic regions below.

Before applying the arithmetic transformation, audit the inherited Stage 1
global-to-LDS waits. Every asynchronous activation/scale stage fetch must be
followed by `S.amdgpu.s_waitcnt(0, -1, -1)` before the workgroup barrier and LDS
read. If the input contains a partial `S.amdgpu.s_waitcnt(0, 7, 15)`, replace it
with the full wait. This is a correctness repair for the existing pipeline,
not a new optimization; do not otherwise change its buffers, overlap, or
dataflow. Packed FMA changes instruction timing and must not expose an
incomplete LDS transfer.

## Required transformation

### 1. MFMA accumulator scaling

- In both `_matmul_stage0` and `_matmul_stage1`, retain the existing MFMA calls,
  DPP scale selection, fragment mapping, and pair loop.
- For each pair, view the duplicated scale, two accumulator values, and two
  destination values as packed two-lane FP32 vectors.
- Replace the two scalar `scale * src + dst` updates with one packed
  `S.fma(scale_pair_vec, src_pair_vec, dst_pair_vec)`, then store its lanes back
  to the unchanged accumulator locations.
- Materialize `src_pair` and `dst_pair` as loop-local two-element tensors,
  create vector views of those locals, and explicitly write the result lanes
  back. Do not retain a vector view of the complete accumulator tensor across
  loops; later persistent execution relies on these pair temporaries having a
  short lexical lifetime.

### 2. SiLU-gated product

- Rewrite `_silu_dot` to process the four elements as two pairs.
- Build packed two-lane constants for `-log2(e)` and `1.0`; load each gate/up
  pair into packed FP32 vectors and perform pairwise multiply/add operations.
- `S.exp2` and `S.amdgpu.rcp` may remain per-lane where no packed intrinsic is
  available, but collect their results back into a packed vector before the
  final `gate * reciprocal * up` operation.
- Preserve the exact approximation formula and result ordering.
- Use loop-local two-element gate, up, exponent, and reciprocal tensors. Avoid
  long-lived vector aliases of the complete `gate`, `up`, or `ret` tensors.

### 3. Intermediate FP8 quantization

- In `_quantize_and_shuffle`, duplicate each row's scalar quantization scale
  into a packed FP32 pair.
- Load `h[i, 0:2]` and `h[i, 2:4]` as two packed pairs, multiply both pairs by
  the packed scale, then use two `S.amdgpu.cvt_pk_fp8_f32` calls to insert the
  first pair into half 0 and the second pair into half 1 of the same `u32`.
- Store that completed `u32` directly in `q[i]`; do not assemble it with a
  separate shift/OR of two temporary results.
- Preserve the existing LDS shuffle and quantization-scale calculation.
- Copy the four scalar `h` elements into loop-local `xy_in` and `zw_in` pair
  tensors before viewing them as vectors. Do not keep a direct vector alias of
  the full `h` fragment.

### 4. Route-weight multiplication

- In `_multiply_route_weights`, duplicate each route weight into a packed FP32
  pair and process each four-element row as two vector pairs.
- Multiply packed source pairs by the packed route weight and store the two
  lanes back to their original positions. Preserve route/row indexing.
- Build a fresh loop-local two-element source tensor for each pair and write
  the result lanes back explicitly; do not retain a vector view of the full
  `t` accumulator across the route loops.

The optimization is complete only if the generated code contains actual
two-lane vector views and packed arithmetic in all four regions; merely keeping
scalar loops with renamed temporaries is not sufficient.

## Correctness invariants

- Preserve routed fused-MoE semantics, FP8 block scaling and packing order,
  SiLU formula, route weights, BF16 rounding, and atomic accumulation.
- Preserve the public API, MFMA operations and fragment layout, both software
  pipelines, `v_setvskip` masking, workgroup/grid mapping, and all barriers.
- Do not reassociate operations across different MFMA fragments, rows, routes,
  or quantization scales.

## Scope boundary for this round

This round is only packed two-lane arithmetic. Do not introduce later
optimizations:

- no persistent thread-group scheduling or `num_persistent_tgs` argument;
- no explicit instruction-scheduling barriers.

The final implementation must pass correctness for token counts 1024, 2048,
4096, 8192, and 16384. Benchmark all five sizes and retain the optimized
implementation only if it does not materially regress the input kernel; do not
add fallback compute paths.
