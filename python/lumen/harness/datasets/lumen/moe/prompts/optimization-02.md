# Optimization 02: pipeline the first MoE GEMM stage

Optimize the existing AveLang fused FP8 MoE kernel by software-pipelining only
the first GEMM stage (the fused gate/up projection). The input implementation
serializes each 128-wide activation tile: it stages one activation/scale tile
in LDS, waits, loads it into registers, then separately loads and computes W1
and W3. Replace that loop with a two-stage ping-pong pipeline that overlaps
activation staging and weight loads with the MFMA work for the current tile.

## Required transformation

- Add two logical activation stages in shared memory. Each stage must contain
  one complete activation tile allocation and its activation-scale allocation.
  Keep the existing per-wave padding and layouts inside each stage.
- Generalize the direct global-to-LDS activation and activation-scale helpers
  to accept the LDS base of the destination stage instead of using one fixed
  shared-memory base.
- In `_stage1`, create two LDS views (`x0`/`scale_x0` and
  `x1`/`scale_x1`) and two corresponding register fragments. Move these views
  into `_stage1`; the outer kernel should pass the shared allocation rather
  than a single fixed activation view.
- Prologue: asynchronously stage activation tile 0 and its scales into stage
  0, load the first W1 register tiles and W1 scale, wait for the LDS loads,
  synchronize, and fetch stage 0 into the first activation registers.
- Process two 128-wide activation tiles per outer iteration. While computing
  the first tile from the stage-0 registers:
  - issue the activation and scale loads for the second tile into LDS stage 1;
  - load W3 for the first tile, run the W1 MFMA helpers, then prefetch W1 for
    the second tile before running the W3 MFMA helpers;
  - advance W1/W3 value and scale offsets incrementally rather than recomputing
    them from the loop index.
- After the stage-1 LDS loads complete, synchronize and fetch them into the
  second activation registers. While computing those registers, stage the next
  outer iteration's activation tile into LDS stage 0 and interleave the W3/W1
  loads and MFMA calls in the same manner.
- Preserve the early exit for a missing second tile. Use waits and workgroup
  barriers so an LDS stage is never overwritten while it is still being read.
- After issuing asynchronous activation and activation-scale loads for an LDS
  stage, use `S.amdgpu.s_waitcnt(0, -1, -1)` before the workgroup barrier and
  before reading that stage into registers. Do not use a partial wait such as
  `S.amdgpu.s_waitcnt(0, 7, 15)`; later arithmetic scheduling changes must not
  expose incomplete global-to-LDS transfers.
- Reuse the existing `_matmul_stage0` and `_matmul_stage1` helpers for both W1
  and W3. Do not inline or change their MFMA math in this round.

The intended steady-state ordering is therefore: prefetch activation, prefetch
weights, compute W1, prefetch the following W1 tile, compute W3, switch the LDS
stage, and repeat. The generated IR must expose this ordering; merely allocating
two buffers while retaining the old serialized loop is not sufficient.

## Correctness invariants

- Preserve routed fused-MoE semantics, FP8 block scaling, SiLU-gated product,
  intermediate quantization, W2 computation, route weighting, and BF16 output.
- Preserve the public `fused_moe_fp8_blockscale_g1u1` API and all supported
  input shapes.
- Preserve the route-group mapping, workgroup size, MFMA fragment mapping,
  quantization behavior, and Stage 2 implementation.
- Keep all global and LDS accesses in bounds, including the last pipeline
  iteration.

## Scope boundary for this round

This round is only the Stage 1 two-buffer software pipeline. Do not introduce
later optimizations:

- no Stage 2 ping-pong pipeline or extra Stage 2 result buffers;
- no `v_setvskip` invalid-route masking;
- no packed/vectorized SiLU, quantization, or route-weight arithmetic changes;
- no persistent thread-group scheduling or `num_persistent_tgs` argument;
- no explicit instruction-scheduling barriers.

The final implementation must pass correctness for token counts 1024, 2048,
4096, 8192, and 16384. Benchmark all five sizes and retain the optimized
implementation only if it does not materially regress the input kernel; do not
add fallback compute paths.
