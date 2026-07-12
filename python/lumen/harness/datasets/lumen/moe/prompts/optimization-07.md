# Optimization 07: constrain hot-loop instruction scheduling

Optimize the existing AveLang fused FP8 MoE kernel by adding explicit AMD
instruction-scheduling barriers to the Stage 1 and Stage 2 software-pipeline
hot loops. The input already contains all earlier pipeline, masking,
vectorization, and persistent-route optimizations; preserve them exactly.

This round changes compiler scheduling only. Do not change data flow,
arithmetic, memory layouts, grid mapping, or loop trip counts.

## Required transformation

- Add two zero-argument `@avelang.jit` helpers named
  `_hot_loop_scheduler_stage1` and `_hot_loop_scheduler_stage2`.
- `_hot_loop_scheduler_stage1` must emit this exact ordered sequence:

  1. repeat six times:
     `sched_group_barrier(0x20, 1, 0)`, then
     `sched_group_barrier(0x8, 4, 0)`;
  2. repeat two times:
     `sched_group_barrier(0x1, 1, 0)`, then
     `sched_group_barrier(0x8, 4, 0)`;
  3. finish with `S.amdgpu.sched_barrier(0)`.

- `_hot_loop_scheduler_stage2` must emit this exact ordered sequence:

  1. `sched_group_barrier(0x200, 1, 0)`;
  2. `sched_group_barrier(0x8, 4, 0)`;
  3. repeat three times:
     `sched_group_barrier(0x20, 1, 0)`, then
     `sched_group_barrier(0x8, 4, 0)`;
  4. `sched_group_barrier(0x100, 1, 0)`;
  5. `sched_group_barrier(0x8, 4, 0)`;
  6. repeat two times:
     `sched_group_barrier(0x1, 1, 0)`, then
     `sched_group_barrier(0x8, 4, 0)`;
  7. finish with `S.amdgpu.sched_barrier(0)`.

- Call `_hot_loop_scheduler_stage1()` once at the beginning of each Stage 1
  outer pipeline iteration: after the workgroup synchronization that protects
  LDS reuse and immediately before issuing the next activation/weight loads and
  MFMA sequence.
- Call `_hot_loop_scheduler_stage2()` once for every Stage 2 output tile: after
  synchronization and accumulator/result initialization, immediately before
  the branch that prefetches W2 and executes the MFMA/result pipeline.
- The helpers must be called from the steady-state hot loops, not only from a
  launcher, prologue, epilogue, or dead code path. They must remain visible as
  explicit scheduler intrinsics in generated device IR.

## Correctness invariants

- Preserve the public API including `num_persistent_tgs`, and preserve default
  and persistent route coverage.
- Preserve all numeric operations, FP8/BF16 packing, route weights, atomics,
  LDS layouts, waits, synchronization, MFMA mapping, and output order.
- The scheduler helpers must have no data-dependent behavior or memory side
  effects.

## Scope boundary for this round

This round is only explicit instruction scheduling. Do not change tile sizes,
buffering, vectorization, route mapping, launch dimensions, or mathematical
operations, and do not add fallback compute paths.

The final implementation must pass correctness for token counts 1024, 2048,
4096, 8192, and 16384, and a nonzero persistent-mode equivalence check.
Benchmark all five standard sizes and retain the scheduled implementation only
if it does not materially regress the input kernel.
