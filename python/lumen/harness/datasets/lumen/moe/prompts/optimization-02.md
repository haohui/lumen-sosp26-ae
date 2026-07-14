# Optimization 02: pipeline the first MoE GEMM stage

Optimize the existing AveLang fused FP8 MoE kernel by software-pipelining only
the first GEMM stage (the fused gate/up projection). The input implementation
serializes each `GROUP_DIM`-wide activation tile: it stages one activation and
scale tile in LDS, waits, loads it into registers, then separately loads and
computes W1 and W3. Replace that loop with a two-stage ping-pong pipeline that
overlaps activation staging and weight loads with the MFMA work for the current
tile.

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
- Advance W1/W3 value and scale offsets incrementally rather than recomputing
  them from the loop index.
- After issuing asynchronous activation and activation-scale loads for an LDS
  stage, use `S.amdgpu.s_waitcnt(0, -1, -1)` before the workgroup barrier and
  before reading that stage into registers. Do not use a partial wait such as
  `S.amdgpu.s_waitcnt(0, 7, 15)`; later arithmetic scheduling changes must not
  expose incomplete global-to-LDS transfers.
- Reuse the existing `_matmul_stage0` and `_matmul_stage1` helpers for both W1
  and W3. Do not inline or change their MFMA math in this round.

## Required compiler-visible schedule

Use this loop shape and ordering. It is important for register allocation:

```python
# Prologue: fill LDS stage 0; load W1 tile 0; advance W1 offsets;
# fully wait, synchronize, and read stage 0 into x0 registers.

stage1_iters = (dim + 2 * GROUP_DIM - 1) // (2 * GROUP_DIM)
for iter_idx in S.range(stage1_iters):
    d = iter_idx * (2 * GROUP_DIM)
    S.syncthreads()

    # Fill LDS stage 1 for d + GROUP_DIM.
    # Load W3 for x0 and advance W3 offsets.
    # Compute W1 from x0.
    S.amdgpu.s_waitcnt(0, -1, -1)
    # Load the next W1 tile and advance W1 offsets.
    # Compute W3 from x0.
    if d + GROUP_DIM >= dim:
        break

    # Fully wait, synchronize, and read stage 1 into x1 registers.
    # Fill LDS stage 0 for d + 2 * GROUP_DIM.
    # Load W3 for x1 and advance W3 offsets.
    # Compute W1 from x1.
    S.amdgpu.s_waitcnt(0, -1, -1)
    # Load the next W1 tile and advance W1 offsets.
    # Compute W3 from x1.
    # Fully wait, synchronize, and read stage 0 into x0 registers.
```

Do not replace this with an outer `if num_tiles != 0`, a
`S.range((num_tiles + 1) // 2)` loop, a separate odd-tile epilogue, or returns
inside `_stage1`. Keep independent `x0` and `x1` register fragments. The full
wait immediately after each W1 computation is intentional: it limits live
ranges before loading the following W1 tile.

The generated IR must expose this ordering; merely allocating two buffers while
retaining the old serialized loop is not sufficient. A correct implementation
of this schedule should compile without scratch spills. If an attempt reaches
about 512 VGPRs and spills, revise it to match the skeleton above instead of
accepting it or restoring the input kernel.

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
